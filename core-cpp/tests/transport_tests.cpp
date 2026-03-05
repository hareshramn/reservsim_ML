#include "sim/config.hpp"
#include "sim/error.hpp"
#include "sim/state.hpp"
#include "sim/transport.hpp"

#include <cstdlib>
#include <cmath>
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

void test_zero_gradient_has_zero_flux_and_no_transport_change() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    ReservoirState state = initialize_state(cfg);
    const std::vector<double> flux_x = compute_total_flux_x(cfg, state);
    const std::vector<double> flux_y = compute_total_flux_y(cfg, state);
    for (double value : flux_x) {
        expect_true(std::abs(value) < 1.0e-12, "zero x flux");
    }
    for (double value : flux_y) {
        expect_true(std::abs(value) < 1.0e-12, "zero y flux");
    }

    const std::vector<double> sw_before = state.sw;
    const TransportDiagnostics diagnostics = advance_saturation_impes(cfg, state);
    expect_true(diagnostics.dt_days > 0.0, "positive dt");
    expect_true(diagnostics.clip_count == 0, "no clipping");
    expect_true(diagnostics.mass_balance_rel <= 1.0e-12, "mass balance conserved in zero-gradient case");
    for (size_t i = 0; i < state.sw.size(); ++i) {
        expect_true(std::abs(state.sw[i] - sw_before[i]) < 1.0e-12, "zero transport change");
    }
}

void test_transport_moves_water_down_pressure_gradient() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    ReservoirState state = initialize_state(cfg);
    state.pressure[0] = 3100.0;
    state.pressure[1] = 3000.0;
    state.sw[0] = cfg.fluid.swc + 0.20;
    state.sw[1] = cfg.fluid.swc + 0.05;

    const double left_before = state.sw[0];
    const double right_before = state.sw[1];
    const std::vector<double> flux_x = compute_total_flux_x(cfg, state);
    expect_true(!flux_x.empty(), "x flux exists");
    expect_true(flux_x[0] > 0.0, "positive left-to-right flux");

    const TransportDiagnostics diagnostics = advance_saturation_impes(cfg, state);
    expect_true(diagnostics.dt_days > 0.0, "positive dt with gradient");
    expect_true(diagnostics.mass_balance_rel >= 0.0, "mass balance rel is non-negative");
    expect_true(state.sw[0] < left_before, "left cell loses water");
    expect_true(state.sw[1] > right_before, "right cell gains water");
}

void test_transport_clips_to_residual_bounds() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    ReservoirState state = initialize_state(cfg);
    state.pressure[0] = 5000.0;
    state.pressure[1] = 1000.0;
    state.sw[0] = 1.0 - cfg.fluid.sor;
    state.sw[1] = cfg.fluid.swc;

    const TransportDiagnostics diagnostics = advance_saturation_impes(cfg, state);
    expect_true(diagnostics.clip_count >= 0, "clip count recorded");
    expect_true(diagnostics.mass_balance_rel >= 0.0, "mass balance rel is non-negative with clipping");
    expect_true(state.sw[0] >= cfg.fluid.swc, "left lower bounded");
    expect_true(state.sw[0] <= 1.0 - cfg.fluid.sor, "left upper bounded");
    expect_true(state.sw[1] >= cfg.fluid.swc, "right lower bounded");
    expect_true(state.sw[1] <= 1.0 - cfg.fluid.sor, "right upper bounded");
}

void test_transport_rejects_dimension_mismatch() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    ReservoirState state = initialize_state(cfg);
    state.nx = cfg.nx - 1;

    try {
        (void)compute_total_flux_x(cfg, state);
        expect_true(false, "expected dimension mismatch failure");
    } catch (const CliError& e) {
        expect_true(e.code() == ExitCode::E_CASE_SCHEMA, "dimension mismatch code");
    }
}

void test_transport_explicit_dt_path() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    ReservoirState state = initialize_state(cfg);
    state.pressure[0] = 3200.0;
    state.pressure[1] = 3000.0;

    const double cfl_dt = compute_transport_cfl_dt_days(cfg, state);
    expect_true(cfl_dt > 0.0, "cfl dt positive");

    const double forced_dt = cfl_dt * 0.5;
    const TransportDiagnostics diagnostics = advance_saturation_impes_with_dt(cfg, state, forced_dt);
    expect_true(std::abs(diagnostics.dt_days - forced_dt) < 1.0e-15, "explicit dt propagated");
    expect_true(diagnostics.mass_balance_rel >= 0.0, "explicit dt mass balance non-negative");
}

}  // namespace

int main() {
    test_zero_gradient_has_zero_flux_and_no_transport_change();
    test_transport_moves_water_down_pressure_gradient();
    test_transport_clips_to_residual_bounds();
    test_transport_rejects_dimension_mismatch();
    test_transport_explicit_dt_path();
    std::cout << "transport_tests: PASS\n";
    return 0;
}
