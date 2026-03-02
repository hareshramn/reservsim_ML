#include "sim/pressure.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw CliError(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", message);
}

size_t cell_index(int x, int y, int nx) {
    return static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
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

    return system;
}
