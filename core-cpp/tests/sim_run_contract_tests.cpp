#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string fixture_case_path() {
    const char* root = std::getenv("RESERV_ML_REPO_ROOT");
    if (root == nullptr) {
        std::cerr << "RESERV_ML_REPO_ROOT not set\n";
        std::exit(1);
    }
    return std::string(root) + "/core-cpp/tests/fixtures/valid_case.yaml";
}

std::string shell_quote(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() + 2);
    out.push_back('\'');
    for (char c : raw) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

void expect_true(bool ok, const std::string& msg) {
    if (!ok) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

std::vector<int> parse_npy_shape(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    expect_true(in.is_open(), "open npy: " + path.string());

    const size_t probe_len = 256;
    std::string buffer(probe_len, '\0');
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize got = in.gcount();
    expect_true(got > 0, "read npy header bytes");
    buffer.resize(static_cast<size_t>(got));

    const std::regex shape_re("'shape': \\(([^\\)]*)\\)");
    std::smatch m;
    expect_true(std::regex_search(buffer, m, shape_re), "shape tuple present in npy header");

    std::vector<int> dims;
    std::stringstream ss(m[1].str());
    std::string token;
    while (std::getline(ss, token, ',')) {
        std::string trimmed;
        for (char c : token) {
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                trimmed.push_back(c);
            }
        }
        if (trimmed.empty()) {
            continue;
        }
        dims.push_back(std::stoi(trimmed));
    }
    return dims;
}

int count_timing_rows(const fs::path& timing_csv, const std::string& row_type) {
    std::ifstream in(timing_csv);
    expect_true(in.is_open(), "open timing.csv");
    std::string line;
    expect_true(static_cast<bool>(std::getline(in, line)), "timing.csv has header");
    int count = 0;
    while (std::getline(in, line)) {
        if (line.find("," + row_type + ",") != std::string::npos) {
            ++count;
        }
    }
    return count;
}

int run_command(const std::string& cmd) {
    return std::system(cmd.c_str());
}

int command_exit_code(int rc) {
    if (rc == -1) {
        return -1;
    }
    if (WIFEXITED(rc)) {
        return WEXITSTATUS(rc);
    }
    return -1;
}

std::string read_text_file(const fs::path& path) {
    std::ifstream in(path);
    expect_true(in.is_open(), "open file: " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void test_output_every_and_schema(const fs::path& build_dir) {
    const fs::path sim_run = build_dir / "sim_run";
    expect_true(fs::exists(sim_run), "sim_run executable exists");

    const fs::path out_dir = fs::temp_directory_path() / "reserv_ml_sim_run_contract_steps5_every2";
    std::error_code ec;
    fs::remove_all(out_dir, ec);
    fs::create_directories(out_dir, ec);
    expect_true(!ec, "create output directory");

    const std::string cmd =
        shell_quote(sim_run.string()) +
        " --case " + shell_quote(fixture_case_path()) +
        " --backend cpu --steps 5 --output-every 2 --seed 7 --out " + shell_quote(out_dir.string()) +
        " > /dev/null 2>&1";
    expect_true(command_exit_code(run_command(cmd)) == 0, "sim_run returns success");

    const std::vector<fs::path> required = {
        out_dir / "meta.json",
        out_dir / "logs.txt",
        out_dir / "state_pressure.npy",
        out_dir / "state_sw.npy",
        out_dir / "well_rates.npy",
        out_dir / "well_bhp.npy",
        out_dir / "timing.csv",
    };
    for (const fs::path& p : required) {
        expect_true(fs::exists(p), "required output exists: " + p.filename().string());
    }

    const std::vector<int> pressure_shape = parse_npy_shape(out_dir / "state_pressure.npy");
    const std::vector<int> sw_shape = parse_npy_shape(out_dir / "state_sw.npy");
    const std::vector<int> rates_shape = parse_npy_shape(out_dir / "well_rates.npy");
    const std::vector<int> bhp_shape = parse_npy_shape(out_dir / "well_bhp.npy");

    expect_true(pressure_shape.size() == 3, "state_pressure rank-3");
    expect_true(sw_shape.size() == 3, "state_sw rank-3");
    expect_true(pressure_shape[0] == 3, "state_pressure checkpoint count is 3");
    expect_true(sw_shape[0] == 3, "state_sw checkpoint count is 3");
    expect_true(rates_shape.size() == 2, "well_rates rank-2");
    expect_true(bhp_shape.size() == 2, "well_bhp rank-2");
    expect_true(rates_shape[0] == 3 && rates_shape[1] == 2, "well_rates shape is [3,2]");
    expect_true(bhp_shape[0] == 3 && bhp_shape[1] == 2, "well_bhp shape is [3,2]");

    const int step_rows = count_timing_rows(out_dir / "timing.csv", "step");
    const int aggregate_rows = count_timing_rows(out_dir / "timing.csv", "aggregate");
    expect_true(step_rows == 5, "timing.csv has one step row per executed step");
    expect_true(aggregate_rows == 1, "timing.csv has one aggregate row");

    fs::remove_all(out_dir, ec);
}

void test_output_every_larger_than_steps_writes_final_checkpoint(const fs::path& build_dir) {
    const fs::path sim_run = build_dir / "sim_run";
    expect_true(fs::exists(sim_run), "sim_run executable exists");

    const fs::path out_dir = fs::temp_directory_path() / "reserv_ml_sim_run_contract_steps3_every10";
    std::error_code ec;
    fs::remove_all(out_dir, ec);
    fs::create_directories(out_dir, ec);
    expect_true(!ec, "create output directory");

    const std::string cmd =
        shell_quote(sim_run.string()) +
        " --case " + shell_quote(fixture_case_path()) +
        " --backend cpu --steps 3 --output-every 10 --seed 7 --out " + shell_quote(out_dir.string()) +
        " > /dev/null 2>&1";
    expect_true(command_exit_code(run_command(cmd)) == 0, "sim_run returns success");

    const std::vector<int> pressure_shape = parse_npy_shape(out_dir / "state_pressure.npy");
    const std::vector<int> rates_shape = parse_npy_shape(out_dir / "well_rates.npy");
    const std::vector<int> bhp_shape = parse_npy_shape(out_dir / "well_bhp.npy");

    expect_true(pressure_shape.size() == 3, "state_pressure rank-3");
    expect_true(pressure_shape[0] == 1, "final-step checkpoint emitted when output_every > steps");
    expect_true(rates_shape.size() == 2 && rates_shape[0] == 1, "well_rates checkpoint count follows state");
    expect_true(bhp_shape.size() == 2 && bhp_shape[0] == 1, "well_bhp checkpoint count follows state");
    expect_true(count_timing_rows(out_dir / "timing.csv", "step") == 3, "timing rows still reflect executed steps");

    fs::remove_all(out_dir, ec);
}

void test_step_acceptance_failure_emits_json_error(const fs::path& build_dir) {
    const fs::path sim_run = build_dir / "sim_run";
    expect_true(fs::exists(sim_run), "sim_run executable exists");

    const fs::path out_dir = fs::temp_directory_path() / "reserv_ml_sim_run_contract_forced_fail";
    const fs::path err_path = fs::temp_directory_path() / "reserv_ml_sim_run_contract_forced_fail.stderr";
    std::error_code ec;
    fs::remove_all(out_dir, ec);
    fs::remove(err_path, ec);
    fs::create_directories(out_dir, ec);
    expect_true(!ec, "create output directory");

    const std::string cmd =
        "SIM_FORCE_STEP_ACCEPTANCE_FAIL=1 " + shell_quote(sim_run.string()) +
        " --case " + shell_quote(fixture_case_path()) +
        " --backend cpu --steps 2 --output-every 1 --seed 7 --out " + shell_quote(out_dir.string()) +
        " > /dev/null 2> " + shell_quote(err_path.string());
    const int exit_code = command_exit_code(run_command(cmd));
    expect_true(exit_code == 5, "forced acceptance failure returns E_CASE_SCHEMA (5)");

    const std::string stderr_text = read_text_file(err_path);
    expect_true(stderr_text.find("\"code\":5") != std::string::npos, "stderr JSON contains code");
    expect_true(stderr_text.find("\"symbol\":\"E_CASE_SCHEMA\"") != std::string::npos, "stderr JSON contains symbol");
    expect_true(
        stderr_text.find("Step acceptance failed after retry budget") != std::string::npos,
        "stderr JSON contains retry budget failure message");

    fs::remove_all(out_dir, ec);
    fs::remove(err_path, ec);
}

}  // namespace

int main(int argc, char** argv) {
    expect_true(argc > 0, "argv[0] available");
    const fs::path build_dir = fs::absolute(fs::path(argv[0])).parent_path();
    test_output_every_and_schema(build_dir);
    test_output_every_larger_than_steps_writes_final_checkpoint(build_dir);
    test_step_acceptance_failure_emits_json_error(build_dir);
    std::cout << "sim_run_contract_tests: PASS\n";
    return 0;
}
