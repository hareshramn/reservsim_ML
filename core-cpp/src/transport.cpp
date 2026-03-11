#include "sim/transport.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw CliError(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", message);
}

size_t cell_index(int x, int y, int z, int nx, int ny) {
    return (static_cast<size_t>(z) * static_cast<size_t>(ny) + static_cast<size_t>(y)) * static_cast<size_t>(nx) + static_cast<size_t>(x);
}

double harmonic_average(double a, double b) {
    if (a <= 0.0 || b <= 0.0) {
        fail("harmonic average inputs must be positive.");
    }
    return 2.0 * a * b / (a + b);
}

double effective_saturation(const SimulationConfig& cfg, double sw) {
    const double denom = 1.0 - cfg.fluid.swc - cfg.fluid.sor;
    if (denom <= 0.0) {
        fail("fluid residual saturations must satisfy swc + sor < 1.");
    }
    const double se = (sw - cfg.fluid.swc) / denom;
    return std::clamp(se, 0.0, 1.0);
}

double krw(const SimulationConfig& cfg, double sw) {
    return std::pow(effective_saturation(cfg, sw), cfg.fluid.nw);
}

double kro(const SimulationConfig& cfg, double sw) {
    return std::pow(1.0 - effective_saturation(cfg, sw), cfg.fluid.no);
}

double water_mobility(const SimulationConfig& cfg, double sw) {
    return krw(cfg, sw) / cfg.fluid.mu_w_cp;
}

double total_mobility(const SimulationConfig& cfg, double sw) {
    return water_mobility(cfg, sw) + (kro(cfg, sw) / cfg.fluid.mu_o_cp);
}

double fractional_flow_water(const SimulationConfig& cfg, double sw) {
    const double lambda_t = total_mobility(cfg, sw);
    if (!std::isfinite(lambda_t) || lambda_t <= 0.0) {
        fail("total mobility must be finite and positive.");
    }
    return water_mobility(cfg, sw) / lambda_t;
}

double face_transmissibility(double perm_i, double perm_j, double lambda_i, double lambda_j) {
    const double perm_face = harmonic_average(perm_i, perm_j);
    const double lambda_face = harmonic_average(lambda_i, lambda_j);
    return perm_face * lambda_face;
}

constexpr double kWellRateScale = 1.0e-3;

double cfl_dt_days(
    const ReservoirState& state,
    const std::vector<double>& flux_x,
    const std::vector<double>& flux_y,
    const std::vector<double>& flux_z) {
    constexpr double c_cfl = 0.5;
    constexpr double safety = 0.9;
    constexpr double epsilon = 1.0e-20;
    constexpr double dt_min = 1.0e-6;
    constexpr double dt_max = 1.0e-1;

    const int fx_nx = state.nx - 1;
    const int fy_ny = state.ny - 1;
    const int fz_nz = state.nz - 1;

    double min_dt = std::numeric_limits<double>::infinity();
    for (int z = 0; z < state.nz; ++z) {
        for (int y = 0; y < state.ny; ++y) {
            for (int x = 0; x < state.nx; ++x) {
                const size_t i = cell_index(x, y, z, state.nx, state.ny);
                double outgoing = 0.0;
                if (x > 0) {
                    outgoing += std::max(-flux_x[cell_index(x - 1, y, z, fx_nx, state.ny)], 0.0);
                }
                if (x + 1 < state.nx) {
                    outgoing += std::max(flux_x[cell_index(x, y, z, fx_nx, state.ny)], 0.0);
                }
                if (y > 0) {
                    outgoing += std::max(-flux_y[cell_index(x, y - 1, z, state.nx, fy_ny)], 0.0);
                }
                if (y + 1 < state.ny) {
                    outgoing += std::max(flux_y[cell_index(x, y, z, state.nx, fy_ny)], 0.0);
                }
                if (z > 0) {
                    outgoing += std::max(-flux_z[cell_index(x, y, z - 1, state.nx, state.ny)], 0.0);
                }
                if (z + 1 < state.nz) {
                    outgoing += std::max(flux_z[cell_index(x, y, z, state.nx, state.ny)], 0.0);
                }

                const double dt_cell = c_cfl * state.porosity[i] / (outgoing + epsilon);
                min_dt = std::min(min_dt, dt_cell);
            }
        }
    }

    if (!std::isfinite(min_dt) || min_dt <= 0.0) {
        return dt_max;
    }
    return std::clamp(safety * min_dt, dt_min, dt_max);
}

double total_water_mass(const ReservoirState& state) {
    const size_t count = static_cast<size_t>(state.nx) * static_cast<size_t>(state.ny) * static_cast<size_t>(state.nz);
    double mass = 0.0;
    for (size_t i = 0; i < count; ++i) {
        mass += state.porosity[i] * state.sw[i];
    }
    return mass;
}

void accumulate_well_source_terms(
    const SimulationConfig& cfg,
    const ReservoirState& state,
    std::vector<double>& q_water) {
    if (!cfg.wells.enabled) {
        return;
    }

    const size_t injector_idx = cell_index(
        cfg.wells.injector_cell_x, cfg.wells.injector_cell_y, cfg.wells.injector_cell_z, cfg.nx, cfg.ny);
    const size_t producer_idx = cell_index(
        cfg.wells.producer_cell_x, cfg.wells.producer_cell_y, cfg.wells.producer_cell_z, cfg.nx, cfg.ny);

    const double q_inj = kWellRateScale * cfg.wells.injector_rate_stb_day;
    q_water[injector_idx] += q_inj;

    const double drawdown = std::max(state.pressure[producer_idx] - cfg.wells.producer_bhp_psi, 0.0);
    const double q_prod_abs = kWellRateScale * cfg.wells.producer_pi * drawdown;
    const double q_prod_total = -q_prod_abs;
    q_water[producer_idx] += fractional_flow_water(cfg, state.sw[producer_idx]) * q_prod_total;
}

}  // namespace

std::vector<double> compute_total_flux_x(const SimulationConfig& cfg, const ReservoirState& state) {
    validate_state_invariants(state);
    if (state.nx != cfg.nx || state.ny != cfg.ny || state.nz != cfg.nz) {
        fail("state dimensions must match configuration.");
    }

    const int fx_nx = cfg.nx - 1;
    if (fx_nx <= 0) {
        return {};
    }

    std::vector<double> fluxes(static_cast<size_t>(fx_nx) * static_cast<size_t>(cfg.ny) * static_cast<size_t>(cfg.nz), 0.0);
    for (int z = 0; z < cfg.nz; ++z) {
        for (int y = 0; y < cfg.ny; ++y) {
            for (int x = 0; x < fx_nx; ++x) {
                const size_t i = cell_index(x, y, z, cfg.nx, cfg.ny);
                const size_t j = cell_index(x + 1, y, z, cfg.nx, cfg.ny);
                const double lambda_i = total_mobility(cfg, state.sw[i]);
                const double lambda_j = total_mobility(cfg, state.sw[j]);
                const double t = face_transmissibility(
                    state.permeability_md[i], state.permeability_md[j], lambda_i, lambda_j);
                fluxes[cell_index(x, y, z, fx_nx, cfg.ny)] = -t * (state.pressure[j] - state.pressure[i]);
            }
        }
    }
    return fluxes;
}

std::vector<double> compute_total_flux_y(const SimulationConfig& cfg, const ReservoirState& state) {
    validate_state_invariants(state);
    if (state.nx != cfg.nx || state.ny != cfg.ny || state.nz != cfg.nz) {
        fail("state dimensions must match configuration.");
    }

    const int fy_ny = cfg.ny - 1;
    if (fy_ny <= 0) {
        return {};
    }

    std::vector<double> fluxes(static_cast<size_t>(cfg.nx) * static_cast<size_t>(fy_ny) * static_cast<size_t>(cfg.nz), 0.0);
    for (int z = 0; z < cfg.nz; ++z) {
        for (int y = 0; y < fy_ny; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                const size_t i = cell_index(x, y, z, cfg.nx, cfg.ny);
                const size_t j = cell_index(x, y + 1, z, cfg.nx, cfg.ny);
                const double lambda_i = total_mobility(cfg, state.sw[i]);
                const double lambda_j = total_mobility(cfg, state.sw[j]);
                const double t = face_transmissibility(
                    state.permeability_md[i], state.permeability_md[j], lambda_i, lambda_j);
                fluxes[cell_index(x, y, z, cfg.nx, fy_ny)] = -t * (state.pressure[j] - state.pressure[i]);
            }
        }
    }
    return fluxes;
}

std::vector<double> compute_total_flux_z(const SimulationConfig& cfg, const ReservoirState& state) {
    validate_state_invariants(state);
    if (state.nx != cfg.nx || state.ny != cfg.ny || state.nz != cfg.nz) {
        fail("state dimensions must match configuration.");
    }

    const int fz_nz = cfg.nz - 1;
    if (fz_nz <= 0) {
        return {};
    }

    std::vector<double> fluxes(static_cast<size_t>(cfg.nx) * static_cast<size_t>(cfg.ny) * static_cast<size_t>(fz_nz), 0.0);
    for (int z = 0; z < fz_nz; ++z) {
        for (int y = 0; y < cfg.ny; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                const size_t i = cell_index(x, y, z, cfg.nx, cfg.ny);
                const size_t j = cell_index(x, y, z + 1, cfg.nx, cfg.ny);
                const double lambda_i = total_mobility(cfg, state.sw[i]);
                const double lambda_j = total_mobility(cfg, state.sw[j]);
                const double t = face_transmissibility(
                    state.permeability_md[i], state.permeability_md[j], lambda_i, lambda_j);
                fluxes[cell_index(x, y, z, cfg.nx, cfg.ny)] = -t * (state.pressure[j] - state.pressure[i]);
            }
        }
    }
    return fluxes;
}

TransportDiagnostics advance_saturation_impes(const SimulationConfig& cfg, ReservoirState& state) {
    const double dt_days = compute_transport_cfl_dt_days(cfg, state);
    return advance_saturation_impes_with_dt(cfg, state, dt_days);
}

double compute_transport_cfl_dt_days(const SimulationConfig& cfg, const ReservoirState& state) {
    validate_state_invariants(state);
    if (state.nx != cfg.nx || state.ny != cfg.ny || state.nz != cfg.nz) {
        fail("state dimensions must match configuration.");
    }

    const std::vector<double> flux_x = compute_total_flux_x(cfg, state);
    const std::vector<double> flux_y = compute_total_flux_y(cfg, state);
    const std::vector<double> flux_z = compute_total_flux_z(cfg, state);
    return cfl_dt_days(state, flux_x, flux_y, flux_z);
}

TransportDiagnostics advance_saturation_impes_with_dt(const SimulationConfig& cfg, ReservoirState& state, double dt_days) {
    validate_state_invariants(state);
    if (state.nx != cfg.nx || state.ny != cfg.ny || state.nz != cfg.nz) {
        fail("state dimensions must match configuration.");
    }
    if (!(dt_days > 0.0) || !std::isfinite(dt_days)) {
        fail("transport dt_days must be finite and positive.");
    }

    const std::vector<double> flux_x = compute_total_flux_x(cfg, state);
    const std::vector<double> flux_y = compute_total_flux_y(cfg, state);
    const std::vector<double> flux_z = compute_total_flux_z(cfg, state);
    const int fx_nx = cfg.nx - 1;
    const int fy_ny = cfg.ny - 1;
    const size_t count = static_cast<size_t>(cfg.nx) * static_cast<size_t>(cfg.ny) * static_cast<size_t>(cfg.nz);
    std::vector<double> q_water(count, 0.0);
    accumulate_well_source_terms(cfg, state, q_water);

    const double mass_before = total_water_mass(state);
    std::vector<double> next_sw = state.sw;
    int clip_count = 0;
    const double sw_min = cfg.fluid.swc;
    const double sw_max = 1.0 - cfg.fluid.sor;

    for (int z = 0; z < cfg.nz; ++z) {
        for (int y = 0; y < cfg.ny; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                const size_t i = cell_index(x, y, z, cfg.nx, cfg.ny);
                double water_flux_sum = 0.0;

                if (x > 0) {
                    const size_t face = cell_index(x - 1, y, z, fx_nx, cfg.ny);
                    const double outward_flux = -flux_x[face];
                    const size_t upwind = (outward_flux >= 0.0) ? i : cell_index(x - 1, y, z, cfg.nx, cfg.ny);
                    water_flux_sum += fractional_flow_water(cfg, state.sw[upwind]) * outward_flux;
                }
                if (x + 1 < cfg.nx) {
                    const size_t face = cell_index(x, y, z, fx_nx, cfg.ny);
                    const double outward_flux = flux_x[face];
                    const size_t upwind = (outward_flux >= 0.0) ? i : cell_index(x + 1, y, z, cfg.nx, cfg.ny);
                    water_flux_sum += fractional_flow_water(cfg, state.sw[upwind]) * outward_flux;
                }
                if (y > 0) {
                    const size_t face = cell_index(x, y - 1, z, cfg.nx, fy_ny);
                    const double outward_flux = -flux_y[face];
                    const size_t upwind = (outward_flux >= 0.0) ? i : cell_index(x, y - 1, z, cfg.nx, cfg.ny);
                    water_flux_sum += fractional_flow_water(cfg, state.sw[upwind]) * outward_flux;
                }
                if (y + 1 < cfg.ny) {
                    const size_t face = cell_index(x, y, z, cfg.nx, fy_ny);
                    const double outward_flux = flux_y[face];
                    const size_t upwind = (outward_flux >= 0.0) ? i : cell_index(x, y + 1, z, cfg.nx, cfg.ny);
                    water_flux_sum += fractional_flow_water(cfg, state.sw[upwind]) * outward_flux;
                }
                if (z > 0) {
                    const size_t face = cell_index(x, y, z - 1, cfg.nx, cfg.ny);
                    const double outward_flux = -flux_z[face];
                    const size_t upwind = (outward_flux >= 0.0) ? i : cell_index(x, y, z - 1, cfg.nx, cfg.ny);
                    water_flux_sum += fractional_flow_water(cfg, state.sw[upwind]) * outward_flux;
                }
                if (z + 1 < cfg.nz) {
                    const size_t face = cell_index(x, y, z, cfg.nx, cfg.ny);
                    const double outward_flux = flux_z[face];
                    const size_t upwind = (outward_flux >= 0.0) ? i : cell_index(x, y, z + 1, cfg.nx, cfg.ny);
                    water_flux_sum += fractional_flow_water(cfg, state.sw[upwind]) * outward_flux;
                }

                double sw_new = state.sw[i] - (dt_days / state.porosity[i]) * water_flux_sum
                                + (dt_days / state.porosity[i]) * q_water[i];
                const double clipped = std::clamp(sw_new, sw_min, sw_max);
                if (std::abs(clipped - sw_new) > 0.0) {
                    ++clip_count;
                }
                next_sw[i] = clipped;
            }
        }
    }

    state.sw = std::move(next_sw);
    validate_state_invariants(state);
    const double mass_after = total_water_mass(state);
    double expected_source_delta = 0.0;
    for (double qwi : q_water) {
        expected_source_delta += dt_days * qwi;
    }
    const double mass_delta_abs = std::abs((mass_after - mass_before) - expected_source_delta);
    const double mass_denom = std::max(std::abs(mass_before), 1.0e-20);
    const double mass_balance_rel = mass_delta_abs / mass_denom;
    return TransportDiagnostics{dt_days, clip_count, mass_balance_rel};
}

bool gpu_transport_enabled() {
#if SIM_ENABLE_CUDA
    return true;
#else
    return false;
#endif
}

#if !SIM_ENABLE_CUDA
TransportDiagnostics advance_saturation_impes_with_dt_gpu(const SimulationConfig&, ReservoirState&, double) {
    fail("GPU transport backend requested, but CUDA support is disabled in this build.");
}
#endif
