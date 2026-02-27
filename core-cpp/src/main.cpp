#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sim/config.hpp"
#include "sim/error.hpp"
#include "sim/state.hpp"

namespace fs = std::filesystem;

struct Args {
    std::string case_path;
    std::string backend;
    int steps = 0;
    int seed = 0;
    std::string out_dir;
};

struct OutputContext {
    fs::path out_dir;
    std::string run_id;
    int steps_to_run = 0;
};

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (const char c : s) {
        switch (c) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << c;
            break;
        }
    }
    return out.str();
}

[[noreturn]] void emit_and_exit(ExitCode code, const std::string& symbol, const std::string& message) {
    std::cerr << "{\"code\":" << static_cast<int>(code)
              << ",\"symbol\":\"" << symbol
              << "\",\"message\":\"" << json_escape(message) << "\"}\n";
    std::exit(static_cast<int>(code));
}

int parse_positive_int_arg(const std::string& raw, const std::string& field) {
    try {
        size_t idx = 0;
        const int value = std::stoi(raw, &idx);
        if (idx != raw.size() || value <= 0) {
            emit_and_exit(ExitCode::E_ARG_INVALID, "E_ARG_INVALID", field + " must be a positive integer.");
        }
        return value;
    } catch (const std::exception&) {
        emit_and_exit(ExitCode::E_ARG_INVALID, "E_ARG_INVALID", field + " must be a positive integer.");
    }
}

Args parse_args(int argc, char** argv) {
    Args args;
    std::map<std::string, std::string> kv;

    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        if (token.rfind("--", 0) != 0) {
            emit_and_exit(ExitCode::E_ARG_INVALID, "E_ARG_INVALID", "Unexpected positional argument: " + token);
        }
        if (i + 1 >= argc) {
            emit_and_exit(ExitCode::E_ARG_INVALID, "E_ARG_INVALID", "Missing value for flag: " + token);
        }
        const std::string value = argv[++i];
        if (kv.find(token) != kv.end()) {
            emit_and_exit(ExitCode::E_ARG_INVALID, "E_ARG_INVALID", "Duplicate flag: " + token);
        }
        kv[token] = value;
    }

    const std::vector<std::string> required_flags = {"--case", "--backend", "--steps", "--seed", "--out"};
    for (const auto& flag : required_flags) {
        if (kv.find(flag) == kv.end()) {
            emit_and_exit(ExitCode::E_ARG_MISSING, "E_ARG_MISSING", "Missing required flag: " + flag);
        }
    }

    for (const auto& [flag, _] : kv) {
        if (std::find(required_flags.begin(), required_flags.end(), flag) == required_flags.end()) {
            emit_and_exit(ExitCode::E_ARG_INVALID, "E_ARG_INVALID", "Unknown flag: " + flag);
        }
    }

    args.case_path = kv["--case"];
    args.backend = kv["--backend"];
    args.steps = parse_positive_int_arg(kv["--steps"], "steps");

    try {
        size_t idx = 0;
        args.seed = std::stoi(kv["--seed"], &idx);
        if (idx != kv["--seed"].size()) {
            emit_and_exit(ExitCode::E_ARG_INVALID, "E_ARG_INVALID", "seed must be an integer.");
        }
    } catch (const std::exception&) {
        emit_and_exit(ExitCode::E_ARG_INVALID, "E_ARG_INVALID", "seed must be an integer.");
    }

    args.out_dir = kv["--out"];
    if (args.backend != "cpu" && args.backend != "gpu") {
        emit_and_exit(ExitCode::E_ARG_INVALID, "E_ARG_INVALID", "backend must be one of: cpu, gpu.");
    }
    if (args.case_path.empty() || args.out_dir.empty()) {
        emit_and_exit(ExitCode::E_ARG_INVALID, "E_ARG_INVALID", "case and out paths must be non-empty.");
    }

    return args;
}

void write_binary(std::ofstream& out, const void* data, size_t bytes, const fs::path& path) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "I/O failure while writing: " + path.string());
    }
}

void write_npy_f64(const fs::path& path, const std::vector<size_t>& shape, const std::vector<double>& data) {
    size_t expected = 1;
    for (const size_t dim : shape) {
        expected *= dim;
    }
    if (expected != data.size()) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "NPY shape/data mismatch for file: " + path.string());
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "Unable to write file: " + path.string());
    }

    std::string shape_str;
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (shape.size() == 1) {
            shape_str += ",";
            break;
        }
        if (i + 1 < shape.size()) {
            shape_str += ", ";
        }
    }

    std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + shape_str + "), }";
    const size_t prefix_len = 10;  // magic(6) + version(2) + header_len(2)
    const size_t total_no_newline = prefix_len + header.size();
    const size_t pad_len = (16 - ((total_no_newline + 1) % 16)) % 16;
    header.append(pad_len, ' ');
    header.push_back('\n');

    const unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    const unsigned char version[2] = {1, 0};
    const uint16_t header_len = static_cast<uint16_t>(header.size());

    write_binary(out, magic, sizeof(magic), path);
    write_binary(out, version, sizeof(version), path);
    write_binary(out, &header_len, sizeof(header_len), path);
    write_binary(out, header.data(), header.size(), path);
    if (!data.empty()) {
        write_binary(out, data.data(), data.size() * sizeof(double), path);
    }
    out.close();
    if (!out) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "I/O failure while writing: " + path.string());
    }
}

OutputContext prepare_output_context(const Args& args, int schedule_end_step) {
    std::error_code ec;
    const fs::path out_dir = fs::path(args.out_dir);
    fs::create_directories(out_dir, ec);
    if (ec) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "Unable to create output directory: " + out_dir.string());
    }

    OutputContext ctx;
    ctx.out_dir = out_dir;
    ctx.steps_to_run = std::min(args.steps, schedule_end_step);
    ctx.run_id = out_dir.filename().string();
    if (ctx.run_id.empty()) {
        ctx.run_id = "run";
    }
    return ctx;
}

void write_meta_json(const OutputContext& ctx, const Args& args, const SimulationConfig& cfg) {
    const fs::path meta_path = ctx.out_dir / "meta.json";
    std::ofstream meta(meta_path);
    if (!meta.is_open()) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "Unable to write file: " + meta_path.string());
    }

    meta << "{\n"
         << "  \"run_id\": \"" << json_escape(ctx.run_id) << "\",\n"
         << "  \"case_name\": \"" << json_escape(cfg.case_name) << "\",\n"
         << "  \"nx\": " << cfg.nx << ",\n"
         << "  \"ny\": " << cfg.ny << ",\n"
         << "  \"backend\": \"" << args.backend << "\",\n"
         << "  \"dt_policy\": \"" << json_escape(cfg.dt_policy) << "\",\n"
         << "  \"seed\": " << args.seed << ",\n"
         << "  \"units\": \"" << json_escape(cfg.units) << "\",\n"
         << "  \"version\": \"slice0\",\n"
         << "  \"steps_requested\": " << args.steps << ",\n"
         << "  \"schedule_end_step\": " << cfg.schedule_end_step << ",\n"
         << "  \"steps_to_run\": " << ctx.steps_to_run << "\n"
         << "}\n";
    meta.close();
    if (!meta) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "I/O failure while writing: " + meta_path.string());
    }
}

void write_logs(const OutputContext& ctx) {
    const fs::path log_path = ctx.out_dir / "logs.txt";
    std::ofstream logs(log_path);
    if (!logs.is_open()) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "Unable to write file: " + log_path.string());
    }
    logs << "slice0: validation-only run; no physics execution.\n"
         << "artifacts: initialized state arrays and placeholder timing/well rates emitted.\n";
    logs.close();
    if (!logs) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "I/O failure while writing: " + log_path.string());
    }
}

void write_state_outputs(const OutputContext& ctx, const ReservoirState& state) {
    const std::vector<size_t> state_shape = {
        1U,
        static_cast<size_t>(state.ny),
        static_cast<size_t>(state.nx),
    };
    write_npy_f64(ctx.out_dir / "state_pressure.npy", state_shape, state.pressure);
    write_npy_f64(ctx.out_dir / "state_sw.npy", state_shape, state.sw);
}

void write_well_rates(const OutputContext& ctx) {
    // Slice 0 has no well solve; keep a stable placeholder array shape for downstream readers.
    const std::vector<double> well_rates = {0.0, 0.0};
    write_npy_f64(ctx.out_dir / "well_rates.npy", {1U, 2U}, well_rates);
}

void write_timing_csv(const OutputContext& ctx) {
    const fs::path timing_path = ctx.out_dir / "timing.csv";
    std::ofstream timing(timing_path);
    if (!timing.is_open()) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "Unable to write file: " + timing_path.string());
    }
    timing << "run_id,row_type,step_idx,dt_days,pressure_time_s,transport_time_s,io_time_s,total_time_s\n";
    timing << ctx.run_id << ",aggregate,-1,0,0,0,0,0\n";
    timing.close();
    if (!timing) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "I/O failure while writing: " + timing_path.string());
    }
}

void write_required_outputs(const OutputContext& ctx, const Args& args, const SimulationConfig& cfg, const ReservoirState& state) {
    write_meta_json(ctx, args, cfg);
    write_logs(ctx);
    write_state_outputs(ctx, state);
    write_well_rates(ctx);
    write_timing_csv(ctx);
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const SimulationConfig cfg = load_simulation_config(args.case_path);
        const ReservoirState state = initialize_state(cfg);
        validate_state_invariants(state);
        const OutputContext ctx = prepare_output_context(args, cfg.schedule_end_step);
        write_required_outputs(ctx, args, cfg, state);
        return static_cast<int>(ExitCode::SUCCESS);
    } catch (const CliError& e) {
        emit_and_exit(e.code(), e.symbol(), e.what());
    }
}
