#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sim/config.hpp"
#include "sim/error.hpp"
#include "sim/pressure.hpp"
#include "sim/state.hpp"
#include "sim/transport.hpp"

namespace fs = std::filesystem;

struct Args {
    std::string case_path;
    std::string backend;
    int steps = 0;
    int output_every = 1;
    std::string out_dir;
};

struct OutputContext {
    fs::path out_dir;
    std::string run_id;
    int steps_to_run = 0;
};

struct RunDiagnostics {
    PressureSolveResult pressure_solve;
    double pressure_time_s = 0.0;
    TransportDiagnostics transport;
    double transport_time_s = 0.0;
    int retries_used = 0;
    double pressure_avg = 0.0;
    double pressure_min = 0.0;
    double pressure_max = 0.0;
    double sw_avg = 0.0;
    double sw_min = 0.0;
    double sw_max = 0.0;
    double simulation_day = 0.0;
    std::vector<double> well_rates_step;
    std::vector<double> well_bhp_step;
};

struct RunSummary {
    std::vector<RunDiagnostics> step_diagnostics;
    std::vector<double> pressure_history;
    std::vector<double> sw_history;
    std::vector<double> well_rates_history;
    std::vector<double> well_bhp_history;
    int checkpoint_count = 0;
    double mass_balance_rel_last = 0.0;
    double mass_balance_rel_max = 0.0;
    double mass_balance_rel_cumulative = 0.0;
    int step_retries_total = 0;
    int step_retries_max = 0;
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

    const std::vector<std::string> required_flags = {"--case", "--backend", "--steps", "--out"};
    const std::vector<std::string> known_flags = {"--case", "--backend", "--steps", "--output-every", "--out"};
    for (const auto& flag : required_flags) {
        if (kv.find(flag) == kv.end()) {
            emit_and_exit(ExitCode::E_ARG_MISSING, "E_ARG_MISSING", "Missing required flag: " + flag);
        }
    }

    for (const auto& [flag, _] : kv) {
        if (std::find(known_flags.begin(), known_flags.end(), flag) == known_flags.end()) {
            emit_and_exit(ExitCode::E_ARG_INVALID, "E_ARG_INVALID", "Unknown flag: " + flag);
        }
    }

    args.case_path = kv["--case"];
    args.backend = kv["--backend"];
    args.steps = parse_positive_int_arg(kv["--steps"], "steps");
    if (const auto it = kv.find("--output-every"); it != kv.end()) {
        args.output_every = parse_positive_int_arg(it->second, "output-every");
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

SimulationConfig apply_history_controls_for_day(const SimulationConfig& cfg, double simulation_day) {
    if (!cfg.history.enabled) {
        return cfg;
    }

    SimulationConfig step_cfg = cfg;
    step_cfg.wells.producer_pi = cfg.wells.producer_pi;
    for (const HistoryControlEntry& entry : cfg.history.controls) {
        if (entry.day > simulation_day) {
            continue;
        }
        if (entry.well == "injector") {
            if (entry.control_kind == "shut") {
                step_cfg.wells.injector_rate_stb_day = 0.0;
            } else if (entry.control_kind == "rate") {
                step_cfg.wells.injector_rate_stb_day = entry.target_value;
            }
        } else if (entry.well == "producer") {
            if (entry.control_kind == "shut") {
                step_cfg.wells.producer_pi = 0.0;
            } else if (entry.control_kind == "bhp") {
                step_cfg.wells.producer_bhp_psi = entry.target_value;
                step_cfg.wells.producer_pi = cfg.wells.producer_pi;
            }
        }
    }
    return step_cfg;
}

void write_meta_json(const OutputContext& ctx, const Args& args, const SimulationConfig& cfg, const RunSummary& summary) {
    const fs::path meta_path = ctx.out_dir / "meta.json";
    std::ofstream meta(meta_path);
    if (!meta.is_open()) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "Unable to write file: " + meta_path.string());
    }

    const RunDiagnostics last = summary.step_diagnostics.empty() ? RunDiagnostics{} : summary.step_diagnostics.back();

    meta << "{\n"
         << "  \"run_id\": \"" << json_escape(ctx.run_id) << "\",\n"
         << "  \"case_name\": \"" << json_escape(cfg.case_name) << "\",\n"
         << "  \"nx\": " << cfg.nx << ",\n"
         << "  \"ny\": " << cfg.ny << ",\n"
         << "  \"nz\": " << cfg.nz << ",\n"
         << "  \"backend\": \"" << args.backend << "\",\n"
         << "  \"dt_policy\": \"" << json_escape(cfg.dt_policy) << "\",\n"
         << "  \"units\": \"" << json_escape(cfg.units) << "\",\n"
         << "  \"version\": \"slice4\",\n"
         << "  \"steps_requested\": " << args.steps << ",\n"
         << "  \"output_every_steps\": " << args.output_every << ",\n"
         << "  \"schedule_end_step\": " << cfg.schedule_end_step << ",\n"
         << "  \"steps_to_run\": " << ctx.steps_to_run << ",\n"
         << "  \"steps_completed\": " << summary.step_diagnostics.size() << ",\n"
         << "  \"checkpoints_written\": " << summary.checkpoint_count << ",\n"
         << "  \"pressure_iterations\": " << last.pressure_solve.iterations << ",\n"
         << "  \"pressure_relative_residual\": " << last.pressure_solve.relative_residual << ",\n"
         << "  \"transport_dt_days\": " << last.transport.dt_days << ",\n"
         << "  \"transport_clip_count\": " << last.transport.clip_count << ",\n"
         << "  \"transport_mass_balance_rel_last\": " << summary.mass_balance_rel_last << ",\n"
         << "  \"transport_mass_balance_rel_max\": " << summary.mass_balance_rel_max << ",\n"
         << "  \"transport_mass_balance_rel_cumulative\": " << summary.mass_balance_rel_cumulative << ",\n"
         << "  \"step_retries_total\": " << summary.step_retries_total << ",\n"
         << "  \"step_retries_max\": " << summary.step_retries_max;
    if (cfg.history.enabled) {
        meta << ",\n"
             << "  \"run_kind\": \"history\",\n"
             << "  \"history_controls_csv\": \"" << json_escape(cfg.history.controls_csv) << "\",\n"
             << "  \"history_observations_csv\": \"" << json_escape(cfg.history.observations_csv) << "\",\n"
             << "  \"history_start_day\": " << cfg.history.start_day << ",\n"
             << "  \"history_end_day\": " << cfg.history.end_day;
    }
    meta << "\n}\n";
    meta.close();
    if (!meta) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "I/O failure while writing: " + meta_path.string());
    }
}

void write_logs(const OutputContext& ctx, const RunSummary& summary) {
    const fs::path log_path = ctx.out_dir / "logs.txt";
    std::ofstream logs(log_path);
    if (!logs.is_open()) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "Unable to write file: " + log_path.string());
    }
    const RunDiagnostics last = summary.step_diagnostics.empty() ? RunDiagnostics{} : summary.step_diagnostics.back();
    int total_pressure_iterations = 0;
    int total_transport_clips = 0;
    int total_step_retries = 0;
    int max_step_retries = 0;
    for (const RunDiagnostics& diagnostics : summary.step_diagnostics) {
        total_pressure_iterations += diagnostics.pressure_solve.iterations;
        total_transport_clips += diagnostics.transport.clip_count;
        total_step_retries += diagnostics.retries_used;
        max_step_retries = std::max(max_step_retries, diagnostics.retries_used);
    }
    logs << "slice4: pressure system assembled and solved with CPU CG + Jacobi.\n"
         << "steps_completed: " << summary.step_diagnostics.size() << "\n"
         << "checkpoints_written: " << summary.checkpoint_count << "\n"
         << "pressure_iterations_last: " << last.pressure_solve.iterations << "\n"
         << "pressure_iterations_total: " << total_pressure_iterations << "\n"
         << "pressure_relative_residual_last: " << last.pressure_solve.relative_residual << "\n"
         << "transport_dt_days_last: " << last.transport.dt_days << "\n"
         << "transport_clip_count_last: " << last.transport.clip_count << "\n"
         << "transport_clip_count_total: " << total_transport_clips << "\n"
         << "transport_mass_balance_rel_last: " << summary.mass_balance_rel_last << "\n"
         << "transport_mass_balance_rel_max: " << summary.mass_balance_rel_max << "\n"
         << "transport_mass_balance_rel_cumulative: " << summary.mass_balance_rel_cumulative << "\n"
         << "step_retries_total: " << total_step_retries << "\n"
         << "step_retries_max: " << max_step_retries << "\n";
    logs << "per_step_stats:\n";
    logs << "step retries p_residual   dt_days mass_bal clip inj_rate prod_rate inj_bhp prod_bhp p_avg  p_min  p_max  sw_avg  sw_min  sw_max\n";
    for (size_t step = 0; step < summary.step_diagnostics.size(); ++step) {
        const RunDiagnostics& d = summary.step_diagnostics[step];
        logs << std::setw(4) << step << " "
             << std::setw(7) << d.retries_used << " "
             << std::setw(10) << std::scientific << std::setprecision(3) << d.pressure_solve.relative_residual << " "
             << std::fixed << std::setprecision(4) << std::setw(7) << d.transport.dt_days << " "
             << std::scientific << std::setprecision(3) << std::setw(8) << d.transport.mass_balance_rel << " "
             << std::fixed << std::setprecision(0) << std::setw(4) << d.transport.clip_count << " "
             << std::fixed << std::setprecision(3) << std::setw(8) << d.well_rates_step[0] << " "
             << std::setw(9) << d.well_rates_step[1] << " "
             << std::setprecision(2) << std::setw(7) << d.well_bhp_step[0] << " "
             << std::setw(8) << d.well_bhp_step[1] << " "
             << std::fixed << std::setprecision(2) << std::setw(6) << d.pressure_avg << " "
             << std::setw(6) << d.pressure_min << " "
             << std::setw(6) << d.pressure_max << " "
             << std::setprecision(4) << std::setw(7) << d.sw_avg << " "
             << std::setw(7) << d.sw_min << " "
             << std::setw(7) << d.sw_max << "\n";
    }
    logs.close();
    if (!logs) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "I/O failure while writing: " + log_path.string());
    }
}

void write_state_outputs(const OutputContext& ctx, const ReservoirState& state, const RunSummary& summary) {
    std::vector<size_t> state_shape;
    if (state.nz > 1) {
        state_shape = {
            static_cast<size_t>(summary.checkpoint_count),
            static_cast<size_t>(state.nz),
            static_cast<size_t>(state.ny),
            static_cast<size_t>(state.nx),
        };
    } else {
        state_shape = {
            static_cast<size_t>(summary.checkpoint_count),
            static_cast<size_t>(state.ny),
            static_cast<size_t>(state.nx),
        };
    }
    write_npy_f64(ctx.out_dir / "state_pressure.npy", state_shape, summary.pressure_history);
    write_npy_f64(ctx.out_dir / "state_sw.npy", state_shape, summary.sw_history);
}

void write_well_outputs(const OutputContext& ctx, const RunSummary& summary) {
    write_npy_f64(
        ctx.out_dir / "well_rates.npy",
        {static_cast<size_t>(summary.checkpoint_count), 2U},
        summary.well_rates_history);
    write_npy_f64(
        ctx.out_dir / "well_bhp.npy",
        {static_cast<size_t>(summary.checkpoint_count), 2U},
        summary.well_bhp_history);
}

void write_timing_csv(const OutputContext& ctx, const RunSummary& summary) {
    const fs::path timing_path = ctx.out_dir / "timing.csv";
    std::ofstream timing(timing_path);
    if (!timing.is_open()) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "Unable to write file: " + timing_path.string());
    }
    timing << "run_id,row_type,step_idx,dt_days,pressure_time_s,transport_time_s,io_time_s,total_time_s\n";
    double dt_total = 0.0;
    double pressure_total = 0.0;
    double transport_total = 0.0;
    for (size_t step = 0; step < summary.step_diagnostics.size(); ++step) {
        const RunDiagnostics& diagnostics = summary.step_diagnostics[step];
        const double total_time_s = diagnostics.pressure_time_s + diagnostics.transport_time_s;
        dt_total += diagnostics.transport.dt_days;
        pressure_total += diagnostics.pressure_time_s;
        transport_total += diagnostics.transport_time_s;
        timing << ctx.run_id << ",step," << step << "," << diagnostics.transport.dt_days
               << "," << diagnostics.pressure_time_s << "," << diagnostics.transport_time_s
               << ",0," << total_time_s << "\n";
    }
    timing << ctx.run_id << ",aggregate,-1," << dt_total << "," << pressure_total
           << "," << transport_total << ",0," << (pressure_total + transport_total) << "\n";
    timing.close();
    if (!timing) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "I/O failure while writing: " + timing_path.string());
    }
}

void write_step_stats_csv(const OutputContext& ctx, const RunSummary& summary) {
    const fs::path stats_path = ctx.out_dir / "step_stats.csv";
    std::ofstream stats(stats_path);
    if (!stats.is_open()) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "Unable to write file: " + stats_path.string());
    }
    stats << "run_id,step_idx,retries_used,pressure_iterations,pressure_relative_residual,dt_days,mass_balance_rel,"
             "clip_count,simulation_day,pressure_avg,pressure_min,pressure_max,sw_avg,sw_min,sw_max,inj_rate,prod_rate,inj_bhp,prod_bhp,"
             "pressure_time_s,transport_time_s,total_time_s\n";
    for (size_t step = 0; step < summary.step_diagnostics.size(); ++step) {
        const RunDiagnostics& d = summary.step_diagnostics[step];
        const double total_time_s = d.pressure_time_s + d.transport_time_s;
        stats << ctx.run_id << "," << step << "," << d.retries_used << "," << d.pressure_solve.iterations
              << "," << d.pressure_solve.relative_residual << "," << d.transport.dt_days << "," << d.transport.mass_balance_rel
              << "," << d.transport.clip_count << "," << d.simulation_day << "," << d.pressure_avg << "," << d.pressure_min << "," << d.pressure_max
              << "," << d.sw_avg << "," << d.sw_min << "," << d.sw_max
              << "," << d.well_rates_step[0] << "," << d.well_rates_step[1]
              << "," << d.well_bhp_step[0] << "," << d.well_bhp_step[1] << "," << d.pressure_time_s
              << "," << d.transport_time_s << "," << total_time_s << "\n";
    }
    stats.close();
    if (!stats) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "I/O failure while writing: " + stats_path.string());
    }
}

void write_step_stats_pretty(const OutputContext& ctx, const RunSummary& summary) {
    const fs::path pretty_path = ctx.out_dir / "step_stats.txt";
    std::ofstream pretty(pretty_path);
    if (!pretty.is_open()) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "Unable to write file: " + pretty_path.string());
    }
    pretty << "Step-wise Simulation Stats\n";
    pretty << "==========================\n";
    pretty << "run_id: " << ctx.run_id << "\n\n";
    pretty << " step retries  p_iter   p_residual      dt_days      mass_bal  clip"
              "   inj_rate  prod_rate   inj_bhp  prod_bhp"
              "     p_avg     p_min     p_max    sw_avg    sw_min    sw_max  total_t[s]\n";
    for (size_t step = 0; step < summary.step_diagnostics.size(); ++step) {
        const RunDiagnostics& d = summary.step_diagnostics[step];
        const double total_time_s = d.pressure_time_s + d.transport_time_s;
        pretty << std::setw(5) << step
               << std::setw(8) << d.retries_used
               << std::setw(8) << d.pressure_solve.iterations
               << std::setw(14) << std::scientific << std::setprecision(3) << d.pressure_solve.relative_residual
               << std::setw(13) << std::fixed << std::setprecision(4) << d.transport.dt_days
               << std::setw(14) << std::scientific << std::setprecision(3) << d.transport.mass_balance_rel
               << std::setw(6) << std::fixed << std::setprecision(0) << d.transport.clip_count
               << std::setw(11) << std::fixed << std::setprecision(3) << d.well_rates_step[0]
               << std::setw(11) << d.well_rates_step[1]
               << std::setw(10) << std::setprecision(2) << d.well_bhp_step[0]
               << std::setw(10) << d.well_bhp_step[1]
               << std::setw(10) << std::fixed << std::setprecision(2) << d.pressure_avg
               << std::setw(10) << d.pressure_min
               << std::setw(10) << d.pressure_max
               << std::setw(10) << std::setprecision(4) << d.sw_avg
               << std::setw(10) << d.sw_min
               << std::setw(10) << d.sw_max
               << std::setw(12) << std::setprecision(6) << total_time_s << "\n";
    }
    pretty.close();
    if (!pretty) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "I/O failure while writing: " + pretty_path.string());
    }
}

void write_required_outputs(
    const OutputContext& ctx,
    const Args& args,
    const SimulationConfig& cfg,
    const ReservoirState& state,
    const RunSummary& summary) {
    write_meta_json(ctx, args, cfg, summary);
    write_logs(ctx, summary);
    write_state_outputs(ctx, state, summary);
    write_well_outputs(ctx, summary);
    write_timing_csv(ctx, summary);
    write_step_stats_csv(ctx, summary);
    write_step_stats_pretty(ctx, summary);
}

std::pair<std::vector<double>, std::vector<double>> compute_well_step_metrics(
    const SimulationConfig& cfg,
    const ReservoirState& state) {
    constexpr double kWellRateScale = 1.0e-3;
    std::vector<double> rates(2U, 0.0);
    std::vector<double> bhp(2U, 0.0);
    if (!cfg.wells.enabled) {
        return {rates, bhp};
    }
    const size_t injector_idx =
        (static_cast<size_t>(cfg.wells.injector_cell_z) * static_cast<size_t>(cfg.ny) + static_cast<size_t>(cfg.wells.injector_cell_y)) *
            static_cast<size_t>(cfg.nx) +
        static_cast<size_t>(cfg.wells.injector_cell_x);
    const size_t producer_idx =
        (static_cast<size_t>(cfg.wells.producer_cell_z) * static_cast<size_t>(cfg.ny) + static_cast<size_t>(cfg.wells.producer_cell_y)) *
            static_cast<size_t>(cfg.nx) +
        static_cast<size_t>(cfg.wells.producer_cell_x);

    const double q_inj = kWellRateScale * cfg.wells.injector_rate_stb_day;
    const double drawdown = std::max(state.pressure[producer_idx] - cfg.wells.producer_bhp_psi, 0.0);
    const double q_prod = -kWellRateScale * cfg.wells.producer_pi * drawdown;

    rates[0] = q_inj;
    rates[1] = q_prod;
    bhp[0] = state.pressure[injector_idx];
    bhp[1] = cfg.wells.producer_bhp_psi;
    return {rates, bhp};
}

RunDiagnostics run_step_with_retry_policy(const SimulationConfig& cfg, ReservoirState& state, const std::string& backend) {
    constexpr double kPressureResidualTol = 1.0e-8;
    constexpr int kPressureMaxIterations = 500;
    constexpr int kRetryBudget = 5;
    constexpr double kRetryDtScale = 0.5;
    constexpr double kMassBalanceTol = 1.0e-6;
    const bool force_step_acceptance_fail = []() {
        const char* raw = std::getenv("SIM_FORCE_STEP_ACCEPTANCE_FAIL");
        return raw != nullptr && std::string(raw) == "1";
    }();

    const ReservoirState state_start = state;
    const double cfl_dt_days = compute_transport_cfl_dt_days(cfg, state_start);
    double trial_dt_days = cfl_dt_days;
    RunDiagnostics last_attempt;

    for (int retry = 0; retry <= kRetryBudget; ++retry) {
        state = state_start;

        const auto pressure_start = std::chrono::steady_clock::now();
        PressureSystem system = assemble_pressure_system(cfg, state);
        apply_pressure_gauge(system, 0, state.pressure.front());
        const PressureSolveResult solve = (backend == "gpu")
                                              ? solve_pressure_cg_jacobi_gpu(system, state.pressure, kPressureResidualTol, kPressureMaxIterations)
                                              : solve_pressure_cg_jacobi(system, state.pressure, kPressureResidualTol, kPressureMaxIterations);
        state.pressure = solve.pressure;
        const auto pressure_end = std::chrono::steady_clock::now();

        const auto transport_start = std::chrono::steady_clock::now();
        const TransportDiagnostics transport = (backend == "gpu")
                                                  ? advance_saturation_impes_with_dt_gpu(cfg, state, trial_dt_days)
                                                  : advance_saturation_impes_with_dt(cfg, state, trial_dt_days);
        const auto transport_end = std::chrono::steady_clock::now();

        RunDiagnostics diagnostics;
        diagnostics.pressure_solve = solve;
        diagnostics.pressure_time_s = std::chrono::duration<double>(pressure_end - pressure_start).count();
        diagnostics.transport = transport;
        diagnostics.transport_time_s = std::chrono::duration<double>(transport_end - transport_start).count();
        diagnostics.retries_used = retry;
        const auto [pmin_it, pmax_it] = std::minmax_element(state.pressure.begin(), state.pressure.end());
        const auto [swmin_it, swmax_it] = std::minmax_element(state.sw.begin(), state.sw.end());
        double psum = 0.0;
        double swsum = 0.0;
        for (double v : state.pressure) {
            psum += v;
        }
        for (double v : state.sw) {
            swsum += v;
        }
        const double cell_count = static_cast<double>(state.pressure.size());
        diagnostics.pressure_avg = psum / cell_count;
        diagnostics.pressure_min = *pmin_it;
        diagnostics.pressure_max = *pmax_it;
        diagnostics.sw_avg = swsum / cell_count;
        diagnostics.sw_min = *swmin_it;
        diagnostics.sw_max = *swmax_it;
        const auto [well_rates_step, well_bhp_step] = compute_well_step_metrics(cfg, state);
        diagnostics.well_rates_step = std::move(well_rates_step);
        diagnostics.well_bhp_step = std::move(well_bhp_step);
        if (force_step_acceptance_fail) {
            diagnostics.transport.mass_balance_rel = kMassBalanceTol * 10.0;
        }
        last_attempt = diagnostics;

        const bool pressure_ok = diagnostics.pressure_solve.relative_residual <= kPressureResidualTol;
        const bool mass_ok = diagnostics.transport.mass_balance_rel <= kMassBalanceTol;
        if (pressure_ok && mass_ok) {
            return diagnostics;
        }

        trial_dt_days *= kRetryDtScale;
    }

    throw CliError(
        ExitCode::E_CASE_SCHEMA,
        "E_CASE_SCHEMA",
        "Step acceptance failed after retry budget; pressure_relative_residual=" +
            std::to_string(last_attempt.pressure_solve.relative_residual) +
            ", mass_balance_rel=" + std::to_string(last_attempt.transport.mass_balance_rel) + ".");
}

bool should_write_checkpoint(int step_idx, int steps_to_run, int output_every) {
    const int one_based = step_idx + 1;
    return (one_based % output_every == 0) || (one_based == steps_to_run);
}

RunSummary execute_time_loop(
    const SimulationConfig& cfg,
    ReservoirState& state,
    const OutputContext& ctx,
    int output_every,
    const std::string& backend) {
    RunSummary summary;
    const size_t cells_per_state = static_cast<size_t>(state.nx) * static_cast<size_t>(state.ny) * static_cast<size_t>(state.nz);
    const int progress_every = std::max(1, ctx.steps_to_run / 100); // ~1% cadence
    double simulation_day = cfg.history.enabled ? cfg.history.start_day : 0.0;

    for (int step = 0; step < ctx.steps_to_run; ++step) {
        const SimulationConfig step_cfg = apply_history_controls_for_day(cfg, simulation_day);
        RunDiagnostics diagnostics = run_step_with_retry_policy(step_cfg, state, backend);
        simulation_day += diagnostics.transport.dt_days;
        diagnostics.simulation_day = simulation_day;
        summary.step_diagnostics.push_back(diagnostics);
        summary.mass_balance_rel_last = diagnostics.transport.mass_balance_rel;
        summary.mass_balance_rel_max = std::max(summary.mass_balance_rel_max, diagnostics.transport.mass_balance_rel);
        summary.mass_balance_rel_cumulative += diagnostics.transport.mass_balance_rel;
        summary.step_retries_total += diagnostics.retries_used;
        summary.step_retries_max = std::max(summary.step_retries_max, diagnostics.retries_used);

        const bool checkpoint = should_write_checkpoint(step, ctx.steps_to_run, output_every);
        if (checkpoint) {
            summary.pressure_history.insert(summary.pressure_history.end(), state.pressure.begin(), state.pressure.end());
            summary.sw_history.insert(summary.sw_history.end(), state.sw.begin(), state.sw.end());
            summary.well_rates_history.insert(
                summary.well_rates_history.end(),
                diagnostics.well_rates_step.begin(),
                diagnostics.well_rates_step.end());
            summary.well_bhp_history.insert(
                summary.well_bhp_history.end(),
                diagnostics.well_bhp_step.begin(),
                diagnostics.well_bhp_step.end());
            ++summary.checkpoint_count;
        }

        const int one_based = step + 1;
        const bool print_progress = (one_based == 1) || (one_based == ctx.steps_to_run) || (one_based % progress_every == 0);
        if (print_progress) {
            std::cout << "[progress] step " << one_based << "/" << ctx.steps_to_run
                      << " checkpoint=" << (checkpoint ? "yes" : "no")
                      << " mass_balance_rel=" << diagnostics.transport.mass_balance_rel
                      << std::endl;
        }
    }

    if (summary.checkpoint_count == 0) {
        RunDiagnostics fallback_diagnostics;
        fallback_diagnostics.well_rates_step = {0.0, 0.0};
        fallback_diagnostics.well_bhp_step = {0.0, 0.0};
        if (!summary.step_diagnostics.empty()) {
            fallback_diagnostics = summary.step_diagnostics.back();
        }
        summary.pressure_history.insert(summary.pressure_history.end(), state.pressure.begin(), state.pressure.end());
        summary.sw_history.insert(summary.sw_history.end(), state.sw.begin(), state.sw.end());
        summary.well_rates_history.insert(
            summary.well_rates_history.end(),
            fallback_diagnostics.well_rates_step.begin(),
            fallback_diagnostics.well_rates_step.end());
        summary.well_bhp_history.insert(
            summary.well_bhp_history.end(),
            fallback_diagnostics.well_bhp_step.begin(),
            fallback_diagnostics.well_bhp_step.end());
        summary.checkpoint_count = 1;
    }

    const size_t expected_count = static_cast<size_t>(summary.checkpoint_count) * cells_per_state;
    if (summary.pressure_history.size() != expected_count || summary.sw_history.size() != expected_count) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "State checkpoint history has inconsistent shape.");
    }
    const size_t expected_well_count = static_cast<size_t>(summary.checkpoint_count) * 2U;
    if (summary.well_rates_history.size() != expected_well_count || summary.well_bhp_history.size() != expected_well_count) {
        emit_and_exit(ExitCode::E_IO, "E_IO", "Well checkpoint history has inconsistent shape.");
    }

    return summary;
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const SimulationConfig cfg = load_simulation_config(args.case_path);
        if (args.backend == "gpu" && !gpu_transport_enabled()) {
            emit_and_exit(
                ExitCode::E_ARG_INVALID,
                "E_ARG_INVALID",
                "backend=gpu requested but CUDA transport is not enabled in this build.");
        }
        if (args.backend == "gpu" && !gpu_pressure_enabled()) {
            emit_and_exit(
                ExitCode::E_ARG_INVALID,
                "E_ARG_INVALID",
                "backend=gpu requested but CUDA pressure backend is not enabled/available.");
        }
        const OutputContext ctx = prepare_output_context(args, cfg.schedule_end_step);
        ReservoirState state = initialize_state(cfg);
        validate_state_invariants(state);
        const RunSummary summary = execute_time_loop(cfg, state, ctx, args.output_every, args.backend);
        validate_state_invariants(state);
        write_required_outputs(ctx, args, cfg, state, summary);
        return static_cast<int>(ExitCode::SUCCESS);
    } catch (const CliError& e) {
        emit_and_exit(e.code(), e.symbol(), e.what());
    }
}
