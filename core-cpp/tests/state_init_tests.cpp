#include "sim/config.hpp"
#include "sim/error.hpp"
#include "sim/state.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

std::string fixture_path(const std::string& filename) {
    const char* root = std::getenv("RESERV_ML_REPO_ROOT");
    if (root == nullptr) {
        std::cerr << "RESERV_ML_REPO_ROOT not set\n";
        std::exit(1);
    }
    return std::string(root) + "/core-cpp/tests/fixtures/" + filename;
}

void expect_true(bool ok, const std::string& msg) {
    if (!ok) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
}

void expect_throws_schema(const SimulationConfig& cfg, const std::string& msg) {
    try {
        (void)initialize_state(cfg);
        expect_true(false, "expected schema exception: " + msg);
    } catch (const CliError& e) {
        expect_true(e.code() == ExitCode::E_CASE_SCHEMA, "schema code: " + msg);
        expect_true(e.symbol() == "E_CASE_SCHEMA", "schema symbol: " + msg);
    }
}

void test_valid_state_initialization() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    const ReservoirState state = initialize_state(cfg);

    const size_t expected = static_cast<size_t>(cfg.nx) * static_cast<size_t>(cfg.ny);
    expect_true(state.nx == cfg.nx, "nx copied");
    expect_true(state.ny == cfg.ny, "ny copied");
    expect_true(state.pressure.size() == expected, "pressure size");
    expect_true(state.sw.size() == expected, "sw size");
    expect_true(state.porosity.size() == expected, "porosity size");
    expect_true(state.permeability_md.size() == expected, "permeability size");
    expect_true(state.pressure.front() > 0.0, "pressure positive");
    expect_true(state.sw.front() >= cfg.fluid.swc, "sw lower bound");
    expect_true(state.sw.front() <= 1.0 - cfg.fluid.sor, "sw upper bound");
}

void test_invalid_state_configs() {
    SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));

    cfg.nx = 0;
    expect_throws_schema(cfg, "invalid nx");

    cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    cfg.rock.porosity = 1.1;
    expect_throws_schema(cfg, "invalid porosity");

    cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    cfg.rock.permeability_md = 0.0;
    expect_throws_schema(cfg, "invalid permeability");

    cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    cfg.fluid.swc = 0.6;
    cfg.fluid.sor = 0.4;
    expect_throws_schema(cfg, "invalid swc+sor");
}

void test_invariant_validation() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    ReservoirState state = initialize_state(cfg);
    state.sw[0] = 1.2;

    try {
        validate_state_invariants(state);
        expect_true(false, "expected invariant failure");
    } catch (const CliError& e) {
        expect_true(e.code() == ExitCode::E_CASE_SCHEMA, "invariant code");
        expect_true(e.symbol() == "E_CASE_SCHEMA", "invariant symbol");
    }
}

}  // namespace

int main() {
    test_valid_state_initialization();
    test_invalid_state_configs();
    test_invariant_validation();
    std::cout << "state_init_tests: PASS\n";
    return 0;
}

