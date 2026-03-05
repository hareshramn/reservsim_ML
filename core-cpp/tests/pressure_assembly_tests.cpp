#include "sim/config.hpp"
#include "sim/error.hpp"
#include "sim/pressure.hpp"
#include "sim/state.hpp"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

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

double apply_row(const PressureSystem& system, const std::vector<double>& x, int cell) {
    const int nx = system.nx;
    const int ny = system.ny;
    const int cx = cell % nx;
    const int cy = cell / nx;

    double value = system.diag[static_cast<size_t>(cell)] * x[static_cast<size_t>(cell)];
    if (cx > 0) {
        value += system.west[static_cast<size_t>(cell)] * x[static_cast<size_t>(cell - 1)];
    }
    if (cx + 1 < nx) {
        value += system.east[static_cast<size_t>(cell)] * x[static_cast<size_t>(cell + 1)];
    }
    if (cy > 0) {
        value += system.south[static_cast<size_t>(cell)] * x[static_cast<size_t>(cell - nx)];
    }
    if (cy + 1 < ny) {
        value += system.north[static_cast<size_t>(cell)] * x[static_cast<size_t>(cell + nx)];
    }
    return value;
}

void test_pressure_system_sign_structure() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    const ReservoirState state = initialize_state(cfg);
    const PressureSystem system = assemble_pressure_system(cfg, state);

    const size_t count = static_cast<size_t>(cfg.nx) * static_cast<size_t>(cfg.ny);
    expect_true(system.diag.size() == count, "diag size");
    expect_true(system.rhs.size() == count, "rhs size");

    for (size_t i = 0; i < count; ++i) {
        expect_true(system.diag[i] >= 0.0, "diag nonnegative");
        expect_true(system.west[i] <= 0.0, "west nonpositive");
        expect_true(system.east[i] <= 0.0, "east nonpositive");
        expect_true(system.south[i] <= 0.0, "south nonpositive");
        expect_true(system.north[i] <= 0.0, "north nonpositive");
        expect_true(std::abs(system.rhs[i]) < 1.0e-12, "rhs zero for closed box");
    }
}

void test_pressure_system_symmetry() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    ReservoirState state = initialize_state(cfg);
    state.permeability_md[0] *= 2.0;
    state.permeability_md[1] *= 0.5;

    const PressureSystem system = assemble_pressure_system(cfg, state);
    for (int y = 0; y < cfg.ny; ++y) {
        for (int x = 0; x < cfg.nx; ++x) {
            const int i = y * cfg.nx + x;
            if (x + 1 < cfg.nx) {
                const int j = i + 1;
                expect_true(std::abs(system.east[static_cast<size_t>(i)] - system.west[static_cast<size_t>(j)]) < 1.0e-12,
                            "east/west symmetry");
            }
            if (y + 1 < cfg.ny) {
                const int j = i + cfg.nx;
                expect_true(std::abs(system.north[static_cast<size_t>(i)] - system.south[static_cast<size_t>(j)]) < 1.0e-12,
                            "north/south symmetry");
            }
        }
    }
}

void test_constant_pressure_conservation() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    ReservoirState state = initialize_state(cfg);
    for (size_t i = 0; i < state.sw.size(); ++i) {
        state.sw[i] = cfg.fluid.swc + 0.05 * static_cast<double>(i % 3);
    }

    const PressureSystem system = assemble_pressure_system(cfg, state);
    const std::vector<double> constant_field(system.diag.size(), 1.0);
    for (size_t i = 0; i < constant_field.size(); ++i) {
        expect_true(std::abs(apply_row(system, constant_field, static_cast<int>(i))) < 1.0e-10,
                    "constant pressure nullspace");
    }
}

void test_invalid_state_rejected() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    ReservoirState state = initialize_state(cfg);
    state.permeability_md[0] = 0.0;

    try {
        (void)assemble_pressure_system(cfg, state);
        expect_true(false, "expected invalid state failure");
    } catch (const CliError& e) {
        expect_true(e.code() == ExitCode::E_CASE_SCHEMA, "invalid state error code");
    }
}

void test_pressure_gauge_application() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    const ReservoirState state = initialize_state(cfg);
    PressureSystem system = assemble_pressure_system(cfg, state);

    apply_pressure_gauge(system, 0, 3000.0);
    expect_true(std::abs(system.diag[0] - 1.0) < 1.0e-12, "gauge diagonal reset");
    expect_true(std::abs(system.rhs[0] - 3000.0) < 1.0e-12, "gauge rhs set");
    expect_true(std::abs(system.east[0]) < 1.0e-12, "gauge east zeroed");
    expect_true(std::abs(system.north[0]) < 1.0e-12, "gauge north zeroed");
    expect_true(system.rhs[1] > 0.0, "neighbor rhs adjusted");
}

void test_pressure_solver_converges_to_constant_reference() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    const ReservoirState state = initialize_state(cfg);
    PressureSystem system = assemble_pressure_system(cfg, state);
    apply_pressure_gauge(system, 0, 3000.0);

    const std::vector<double> initial_guess(system.diag.size(), 0.0);
    const PressureSolveResult result = solve_pressure_cg_jacobi(system, initial_guess, 1.0e-10, 500);
    expect_true(result.iterations > 0, "solver performed iterations");
    expect_true(result.relative_residual <= 1.0e-10, "solver residual target");

    const std::vector<double> ax = apply_pressure_system(system, result.pressure);
    double residual_sq = 0.0;
    double rhs_sq = 0.0;
    for (size_t i = 0; i < ax.size(); ++i) {
        const double residual = ax[i] - system.rhs[i];
        residual_sq += residual * residual;
        rhs_sq += system.rhs[i] * system.rhs[i];
        expect_true(std::abs(result.pressure[i] - 3000.0) < 1.0e-6, "constant pressure solution");
    }
    const double relative_linear_residual = std::sqrt(residual_sq / rhs_sq);
    expect_true(relative_linear_residual <= 1.0e-10, "solver satisfies linear system");
}

void test_pressure_solver_rejects_bad_input() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    const ReservoirState state = initialize_state(cfg);
    PressureSystem system = assemble_pressure_system(cfg, state);
    apply_pressure_gauge(system, 0, 3000.0);

    try {
        (void)solve_pressure_cg_jacobi(system, std::vector<double>(1, 0.0), 1.0e-8, 10);
        expect_true(false, "expected bad initial guess failure");
    } catch (const CliError& e) {
        expect_true(e.code() == ExitCode::E_CASE_SCHEMA, "bad initial guess error code");
    }
}

}  // namespace

int main() {
    test_pressure_system_sign_structure();
    test_pressure_system_symmetry();
    test_constant_pressure_conservation();
    test_invalid_state_rejected();
    test_pressure_gauge_application();
    test_pressure_solver_converges_to_constant_reference();
    test_pressure_solver_rejects_bad_input();
    std::cout << "pressure_assembly_tests: PASS\n";
    return 0;
}
