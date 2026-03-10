#pragma once

#include <vector>

#include "sim/config.hpp"
#include "sim/state.hpp"

struct PressureSystem {
    int nx = 0;
    int ny = 0;
    std::vector<double> diag;
    std::vector<double> west;
    std::vector<double> east;
    std::vector<double> south;
    std::vector<double> north;
    std::vector<double> rhs;
};

struct PressureSolveResult {
    std::vector<double> pressure;
    int iterations = 0;
    double relative_residual = 0.0;
};

PressureSystem assemble_pressure_system(const SimulationConfig& cfg, const ReservoirState& state);
void apply_pressure_gauge(PressureSystem& system, int gauge_cell, double gauge_value);
std::vector<double> apply_pressure_system(const PressureSystem& system, const std::vector<double>& x);
PressureSolveResult solve_pressure_cg_jacobi(
    const PressureSystem& system,
    const std::vector<double>& initial_guess,
    double relative_tolerance,
    int max_iterations);
PressureSolveResult solve_pressure_cg_jacobi_gpu(
    const PressureSystem& system,
    const std::vector<double>& initial_guess,
    double relative_tolerance,
    int max_iterations);
bool gpu_pressure_enabled();
