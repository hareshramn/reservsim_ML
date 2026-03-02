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

PressureSystem assemble_pressure_system(const SimulationConfig& cfg, const ReservoirState& state);
