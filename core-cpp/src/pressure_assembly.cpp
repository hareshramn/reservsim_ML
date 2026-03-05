#include "sim/pressure.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw CliError(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", message);
}

size_t cell_index(int x, int y, int nx) {
    return static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
}

void validate_system_shape(const PressureSystem& system) {
    const size_t count = static_cast<size_t>(system.nx) * static_cast<size_t>(system.ny);
    if (system.nx <= 0 || system.ny <= 0) {
        fail("pressure system dimensions must be positive.");
    }
    if (system.diag.size() != count || system.west.size() != count || system.east.size() != count ||
        system.south.size() != count || system.north.size() != count || system.rhs.size() != count) {
        fail("pressure system arrays must match nx * ny.");
    }
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        fail("dot product inputs must have matching sizes.");
    }
    double value = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        value += a[i] * b[i];
    }
    return value;
}

double l2_norm(const std::vector<double>& x) {
    return std::sqrt(dot(x, x));
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

double total_mobility(const SimulationConfig& cfg, double sw) {
    const double se = effective_saturation(cfg, sw);
    const double krw = std::pow(se, cfg.fluid.nw);
    const double kro = std::pow(1.0 - se, cfg.fluid.no);
    return (krw / cfg.fluid.mu_w_cp) + (kro / cfg.fluid.mu_o_cp);
}

double face_transmissibility(double perm_i, double perm_j, double lambda_i, double lambda_j) {
    const double perm_face = harmonic_average(perm_i, perm_j);
    const double lambda_face = harmonic_average(lambda_i, lambda_j);
    return perm_face * lambda_face;
}

constexpr double kWellRateScale = 1.0e-3;

}  // namespace

PressureSystem assemble_pressure_system(const SimulationConfig& cfg, const ReservoirState& state) {
    validate_state_invariants(state);
    if (state.nx != cfg.nx || state.ny != cfg.ny) {
        fail("state dimensions must match configuration.");
    }

    const size_t count = static_cast<size_t>(cfg.nx) * static_cast<size_t>(cfg.ny);
    PressureSystem system;
    system.nx = cfg.nx;
    system.ny = cfg.ny;
    system.diag.assign(count, 0.0);
    system.west.assign(count, 0.0);
    system.east.assign(count, 0.0);
    system.south.assign(count, 0.0);
    system.north.assign(count, 0.0);
    system.rhs.assign(count, 0.0);

    std::vector<double> lambda_t(count, 0.0);
    for (size_t i = 0; i < count; ++i) {
        lambda_t[i] = total_mobility(cfg, state.sw[i]);
        if (!std::isfinite(lambda_t[i]) || lambda_t[i] <= 0.0) {
            fail("total mobility must be finite and positive.");
        }
    }

    for (int y = 0; y < cfg.ny; ++y) {
        for (int x = 0; x < cfg.nx; ++x) {
            const size_t i = cell_index(x, y, cfg.nx);

            if (x + 1 < cfg.nx) {
                const size_t j = cell_index(x + 1, y, cfg.nx);
                const double t = face_transmissibility(
                    state.permeability_md[i], state.permeability_md[j], lambda_t[i], lambda_t[j]);
                system.diag[i] += t;
                system.diag[j] += t;
                system.east[i] = -t;
                system.west[j] = -t;
            }

            if (y + 1 < cfg.ny) {
                const size_t j = cell_index(x, y + 1, cfg.nx);
                const double t = face_transmissibility(
                    state.permeability_md[i], state.permeability_md[j], lambda_t[i], lambda_t[j]);
                system.diag[i] += t;
                system.diag[j] += t;
                system.north[i] = -t;
                system.south[j] = -t;
            }
        }
    }

    if (cfg.wells.enabled) {
        const size_t injector_idx = cell_index(cfg.wells.injector_cell_x, cfg.wells.injector_cell_y, cfg.nx);
        const size_t producer_idx = cell_index(cfg.wells.producer_cell_x, cfg.wells.producer_cell_y, cfg.nx);

        const double q_inj = kWellRateScale * cfg.wells.injector_rate_stb_day;
        const double pi = kWellRateScale * cfg.wells.producer_pi;
        system.rhs[injector_idx] += q_inj;
        system.diag[producer_idx] += pi;
        system.rhs[producer_idx] += pi * cfg.wells.producer_bhp_psi;
    }

    return system;
}

std::vector<double> apply_pressure_system(const PressureSystem& system, const std::vector<double>& x) {
    validate_system_shape(system);
    const size_t count = static_cast<size_t>(system.nx) * static_cast<size_t>(system.ny);
    if (x.size() != count) {
        fail("pressure system apply input must match system size.");
    }

    std::vector<double> y(count, 0.0);
    for (int y_idx = 0; y_idx < system.ny; ++y_idx) {
        for (int x_idx = 0; x_idx < system.nx; ++x_idx) {
            const size_t i = cell_index(x_idx, y_idx, system.nx);
            double value = system.diag[i] * x[i];
            if (x_idx > 0) {
                value += system.west[i] * x[cell_index(x_idx - 1, y_idx, system.nx)];
            }
            if (x_idx + 1 < system.nx) {
                value += system.east[i] * x[cell_index(x_idx + 1, y_idx, system.nx)];
            }
            if (y_idx > 0) {
                value += system.south[i] * x[cell_index(x_idx, y_idx - 1, system.nx)];
            }
            if (y_idx + 1 < system.ny) {
                value += system.north[i] * x[cell_index(x_idx, y_idx + 1, system.nx)];
            }
            y[i] = value;
        }
    }
    return y;
}

void apply_pressure_gauge(PressureSystem& system, int gauge_cell, double gauge_value) {
    validate_system_shape(system);
    const int count = system.nx * system.ny;
    if (gauge_cell < 0 || gauge_cell >= count) {
        fail("pressure gauge cell index is out of range.");
    }

    const int gauge_x = gauge_cell % system.nx;
    const int gauge_y = gauge_cell / system.nx;
    const size_t g = static_cast<size_t>(gauge_cell);

    if (gauge_x > 0) {
        const size_t west_idx = cell_index(gauge_x - 1, gauge_y, system.nx);
        system.rhs[west_idx] -= system.east[west_idx] * gauge_value;
        system.east[west_idx] = 0.0;
    }
    if (gauge_x + 1 < system.nx) {
        const size_t east_idx = cell_index(gauge_x + 1, gauge_y, system.nx);
        system.rhs[east_idx] -= system.west[east_idx] * gauge_value;
        system.west[east_idx] = 0.0;
    }
    if (gauge_y > 0) {
        const size_t south_idx = cell_index(gauge_x, gauge_y - 1, system.nx);
        system.rhs[south_idx] -= system.north[south_idx] * gauge_value;
        system.north[south_idx] = 0.0;
    }
    if (gauge_y + 1 < system.ny) {
        const size_t north_idx = cell_index(gauge_x, gauge_y + 1, system.nx);
        system.rhs[north_idx] -= system.south[north_idx] * gauge_value;
        system.south[north_idx] = 0.0;
    }

    system.diag[g] = 1.0;
    system.west[g] = 0.0;
    system.east[g] = 0.0;
    system.south[g] = 0.0;
    system.north[g] = 0.0;
    system.rhs[g] = gauge_value;
}

PressureSolveResult solve_pressure_cg_jacobi(
    const PressureSystem& system,
    const std::vector<double>& initial_guess,
    double relative_tolerance,
    int max_iterations) {
    validate_system_shape(system);
    const size_t count = static_cast<size_t>(system.nx) * static_cast<size_t>(system.ny);
    if (initial_guess.size() != count) {
        fail("pressure solver initial guess must match system size.");
    }
    if (!(relative_tolerance > 0.0) || max_iterations <= 0) {
        fail("pressure solver tolerance and iteration budget must be positive.");
    }

    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(system.diag[i]) || system.diag[i] <= 0.0) {
            fail("pressure solver requires positive finite diagonal entries.");
        }
    }

    std::vector<double> x = initial_guess;
    std::vector<double> ax = apply_pressure_system(system, x);
    std::vector<double> r(count, 0.0);
    for (size_t i = 0; i < count; ++i) {
        r[i] = system.rhs[i] - ax[i];
    }

    const double rhs_norm = l2_norm(system.rhs);
    const double denom = std::max(rhs_norm, 1.0);
    double relative_residual = l2_norm(r) / denom;
    if (relative_residual <= relative_tolerance) {
        return PressureSolveResult{std::move(x), 0, relative_residual};
    }

    std::vector<double> z(count, 0.0);
    std::vector<double> p(count, 0.0);
    std::vector<double> ap(count, 0.0);
    for (size_t i = 0; i < count; ++i) {
        z[i] = r[i] / system.diag[i];
        p[i] = z[i];
    }
    double rz_old = dot(r, z);
    if (!(rz_old > 0.0)) {
        fail("pressure solver encountered non-positive preconditioned residual.");
    }

    for (int iter = 0; iter < max_iterations; ++iter) {
        ap = apply_pressure_system(system, p);
        const double p_ap = dot(p, ap);
        if (!(p_ap > 0.0) || !std::isfinite(p_ap)) {
            fail("pressure solver encountered a non-SPD search direction.");
        }

        const double alpha = rz_old / p_ap;
        for (size_t i = 0; i < count; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        relative_residual = l2_norm(r) / denom;
        if (relative_residual <= relative_tolerance) {
            return PressureSolveResult{std::move(x), iter + 1, relative_residual};
        }

        for (size_t i = 0; i < count; ++i) {
            z[i] = r[i] / system.diag[i];
        }
        const double rz_new = dot(r, z);
        if (!(rz_new >= 0.0) || !std::isfinite(rz_new)) {
            fail("pressure solver encountered an invalid preconditioned residual.");
        }
        const double beta = rz_new / rz_old;
        for (size_t i = 0; i < count; ++i) {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }

    return PressureSolveResult{std::move(x), max_iterations, relative_residual};
}
