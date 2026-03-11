#pragma once

#include <vector>

#include "sim/config.hpp"
#include "sim/state.hpp"

struct TransportDiagnostics {
    double dt_days = 0.0;
    int clip_count = 0;
    double mass_balance_rel = 0.0;
};

std::vector<double> compute_total_flux_x(const SimulationConfig& cfg, const ReservoirState& state);
std::vector<double> compute_total_flux_y(const SimulationConfig& cfg, const ReservoirState& state);
std::vector<double> compute_total_flux_z(const SimulationConfig& cfg, const ReservoirState& state);
double compute_transport_cfl_dt_days(const SimulationConfig& cfg, const ReservoirState& state);
TransportDiagnostics advance_saturation_impes_with_dt(const SimulationConfig& cfg, ReservoirState& state, double dt_days);
TransportDiagnostics advance_saturation_impes(const SimulationConfig& cfg, ReservoirState& state);
TransportDiagnostics advance_saturation_impes_with_dt_gpu(const SimulationConfig& cfg, ReservoirState& state, double dt_days);
bool gpu_transport_enabled();
