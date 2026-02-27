#include "sim/state.hpp"

#include <algorithm>
#include <cmath>

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw CliError(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", message);
}

}  // namespace

ReservoirState initialize_state(const SimulationConfig& cfg) {
    if (cfg.nx <= 0 || cfg.ny <= 0) {
        fail("nx and ny must be positive for state initialization.");
    }
    if (cfg.rock.porosity <= 0.0 || cfg.rock.porosity > 1.0) {
        fail("rock.porosity must be in (0, 1] for state initialization.");
    }
    if (cfg.rock.permeability_md <= 0.0) {
        fail("rock.permeability_md must be positive for state initialization.");
    }
    if (cfg.fluid.swc < 0.0 || cfg.fluid.sor < 0.0 || cfg.fluid.swc + cfg.fluid.sor >= 1.0) {
        fail("fluid.swc and fluid.sor must satisfy swc + sor < 1.");
    }

    const size_t count = static_cast<size_t>(cfg.nx) * static_cast<size_t>(cfg.ny);
    const double sw_init = std::clamp(cfg.fluid.swc + 1.0e-4, cfg.fluid.swc, 1.0 - cfg.fluid.sor);
    const double p_init = 3000.0;

    ReservoirState state;
    state.nx = cfg.nx;
    state.ny = cfg.ny;
    state.pressure.assign(count, p_init);
    state.sw.assign(count, sw_init);
    state.porosity.assign(count, cfg.rock.porosity);
    state.permeability_md.assign(count, cfg.rock.permeability_md);

    validate_state_invariants(state);
    return state;
}

void validate_state_invariants(const ReservoirState& state) {
    if (state.nx <= 0 || state.ny <= 0) {
        fail("state grid dimensions must be positive.");
    }

    const size_t count = static_cast<size_t>(state.nx) * static_cast<size_t>(state.ny);
    if (state.pressure.size() != count || state.sw.size() != count ||
        state.porosity.size() != count || state.permeability_md.size() != count) {
        fail("state arrays must match nx * ny.");
    }

    for (size_t i = 0; i < count; ++i) {
        const double p = state.pressure[i];
        const double sw = state.sw[i];
        const double phi = state.porosity[i];
        const double perm = state.permeability_md[i];

        if (!std::isfinite(p) || p <= 0.0) {
            fail("pressure must be finite and positive at all cells.");
        }
        if (!std::isfinite(sw) || sw < 0.0 || sw > 1.0) {
            fail("water saturation must be finite and in [0, 1] at all cells.");
        }
        if (!std::isfinite(phi) || phi <= 0.0 || phi > 1.0) {
            fail("porosity must be finite and in (0, 1] at all cells.");
        }
        if (!std::isfinite(perm) || perm <= 0.0) {
            fail("permeability must be finite and positive at all cells.");
        }
    }
}
