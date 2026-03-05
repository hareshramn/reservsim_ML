#include "sim/config.hpp"
#include "sim/error.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

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

void test_valid_case() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("valid_case.yaml"));
    expect_true(cfg.case_name == "fixture_valid", "case_name parse");
    expect_true(cfg.nx == 8, "nx parse");
    expect_true(cfg.ny == 6, "ny parse");
    expect_true(cfg.dt_policy == "cfl_adaptive_v1", "dt_policy parse");
    expect_true(cfg.units == "field", "units parse");
    expect_true(cfg.schedule_end_step == 12, "schedule_end_step parse");
    expect_true(cfg.physics.phases == "oil_water", "physics.phases parse");
    expect_true(cfg.physics.incompressible, "physics.incompressible parse");
    expect_true(!cfg.physics.gravity, "physics.gravity parse");
    expect_true(!cfg.physics.capillary, "physics.capillary parse");
    expect_true(cfg.rock.porosity > 0.21 && cfg.rock.porosity < 0.23, "rock.porosity parse");
    expect_true(cfg.rock.permeability_md > 149.0 && cfg.rock.permeability_md < 151.0, "rock.permeability parse");
    expect_true(cfg.fluid.mu_w_cp > 1.1 && cfg.fluid.mu_w_cp < 1.3, "fluid.mu_w_cp parse");
    expect_true(cfg.fluid.mu_o_cp > 2.6 && cfg.fluid.mu_o_cp < 2.8, "fluid.mu_o_cp parse");
    expect_true(cfg.fluid.swc > 0.14 && cfg.fluid.swc < 0.16, "fluid.swc parse");
    expect_true(cfg.fluid.sor > 0.19 && cfg.fluid.sor < 0.21, "fluid.sor parse");
    expect_true(cfg.fluid.nw > 2.0 && cfg.fluid.nw < 2.2, "fluid.nw parse");
    expect_true(cfg.fluid.no > 2.2 && cfg.fluid.no < 2.4, "fluid.no parse");
    expect_true(!cfg.wells.enabled, "wells disabled for base fixture");
}

void test_valid_wells_case() {
    const SimulationConfig cfg = load_simulation_config(fixture_path("wells_case.yaml"));
    expect_true(cfg.wells.enabled, "wells enabled");
    expect_true(cfg.wells.injector_cell_x == 0, "injector x parse");
    expect_true(cfg.wells.injector_cell_y == 0, "injector y parse");
    expect_true(cfg.wells.producer_cell_x == 7, "producer x parse");
    expect_true(cfg.wells.producer_cell_y == 5, "producer y parse");
    expect_true(cfg.wells.injector_rate_stb_day > 119.9 && cfg.wells.injector_rate_stb_day < 120.1, "injector rate parse");
    expect_true(cfg.wells.producer_bhp_psi > 2799.0 && cfg.wells.producer_bhp_psi < 2801.0, "producer bhp parse");
    expect_true(cfg.wells.producer_pi > 1.49 && cfg.wells.producer_pi < 1.51, "producer pi parse");
}

void expect_throws(const std::string& file, ExitCode code, const std::string& symbol) {
    try {
        (void)load_simulation_config(fixture_path(file));
        expect_true(false, "expected exception for " + file);
    } catch (const CliError& e) {
        expect_true(e.code() == code, "exit code for " + file);
        expect_true(e.symbol() == symbol, "symbol for " + file);
    }
}

void test_invalid_cases() {
    expect_throws("bad_yaml.yaml", ExitCode::E_CASE_PARSE, "E_CASE_PARSE");
    expect_throws("missing_required.yaml", ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA");
    expect_throws("bad_range.yaml", ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA");
}

}  // namespace

int main() {
    test_valid_case();
    test_valid_wells_case();
    test_invalid_cases();
    std::cout << "config_parser_tests: PASS\n";
    return 0;
}
