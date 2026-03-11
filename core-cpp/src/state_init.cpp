#include "sim/state.hpp"

#include <algorithm>
#include <cmath>

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw CliError(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", message);
}

}  // namespace

ReservoirState initialize_state(const SimulationConfig& cfg) {
    if (cfg.nx <= 0 || cfg.ny <= 0 || cfg.nz <= 0) {
        fail("nx, ny, and nz must be positive for state initialization.");
    }
    if (cfg.rock.porosity <= 0.0 || cfg.rock.porosity > 1.0) {
        fail("rock.porosity must be in (0, 1] for state initialization.");
    }
    if (cfg.rock.permeability_md <= 0.0) {
        fail("rock.permeability_md must be positive for state initialization.");
    }
    if (cfg.rock.layer_count <= 0) {
        fail("rock.layer_count must be positive for state initialization.");
    }
    if (cfg.rock.layer_count > cfg.ny) {
        fail("rock.layer_count cannot exceed ny for y-layer mapping.");
    }
    if (cfg.fluid.swc < 0.0 || cfg.fluid.sor < 0.0 || cfg.fluid.swc + cfg.fluid.sor >= 1.0) {
        fail("fluid.swc and fluid.sor must satisfy swc + sor < 1.");
    }

    const size_t count = static_cast<size_t>(cfg.nx) * static_cast<size_t>(cfg.ny) * static_cast<size_t>(cfg.nz);
    const double sw_init = std::clamp(cfg.fluid.swc + 1.0e-4, cfg.fluid.swc, 1.0 - cfg.fluid.sor);
    const double p_init = 3000.0;

    ReservoirState state;
    state.nx = cfg.nx;
    state.ny = cfg.ny;
    state.nz = cfg.nz;
    state.pressure.assign(count, p_init);
    state.sw.assign(count, sw_init);
    state.porosity.assign(count, cfg.rock.porosity);
    state.permeability_md.assign(count, cfg.rock.permeability_md);

    if (cfg.rock.layer_count > 1) {
        if (cfg.rock.layer_porosity.size() != static_cast<size_t>(cfg.rock.layer_count) ||
            cfg.rock.layer_permeability_md.size() != static_cast<size_t>(cfg.rock.layer_count)) {
            fail("layer property vectors must match rock.layer_count.");
        }
        for (int z = 0; z < cfg.nz; ++z) {
            for (int y = 0; y < cfg.ny; ++y) {
                const int layer = (y * cfg.rock.layer_count) / cfg.ny;
                for (int x = 0; x < cfg.nx; ++x) {
                    const size_t idx =
                        (static_cast<size_t>(z) * static_cast<size_t>(cfg.ny) + static_cast<size_t>(y)) * static_cast<size_t>(cfg.nx) +
                        static_cast<size_t>(x);
                    state.porosity[idx] = cfg.rock.layer_porosity[static_cast<size_t>(layer)];
                    state.permeability_md[idx] = cfg.rock.layer_permeability_md[static_cast<size_t>(layer)];
                }
            }
        }
    }

    validate_state_invariants(state);
    return state;
}

void validate_state_invariants(const ReservoirState& state) {
    if (state.nx <= 0 || state.ny <= 0 || state.nz <= 0) {
        fail("state grid dimensions must be positive.");
    }

    const size_t count = static_cast<size_t>(state.nx) * static_cast<size_t>(state.ny) * static_cast<size_t>(state.nz);
    if (state.pressure.size() != count || state.sw.size() != count ||
        state.porosity.size() != count || state.permeability_md.size() != count) {
        fail("state arrays must match nx * ny * nz.");
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
