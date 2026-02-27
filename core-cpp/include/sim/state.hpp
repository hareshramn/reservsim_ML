#pragma once

#include <string>
#include <vector>

#include "sim/config.hpp"
#include "sim/error.hpp"

struct ReservoirState {
    int nx = 0;
    int ny = 0;
    std::vector<double> pressure;
    std::vector<double> sw;
    std::vector<double> porosity;
    std::vector<double> permeability_md;
};

ReservoirState initialize_state(const SimulationConfig& cfg);
void validate_state_invariants(const ReservoirState& state);

