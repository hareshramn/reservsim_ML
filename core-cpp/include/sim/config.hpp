#pragma once

#include <string>
#include <vector>

struct PhysicsConfig {
    std::string phases;
    bool incompressible = true;
    bool gravity = false;
    bool capillary = false;
};

struct RockConfig {
    double porosity = 0.0;
    double permeability_md = 0.0;
};

struct FluidConfig {
    double mu_w_cp = 0.0;
    double mu_o_cp = 0.0;
    double swc = 0.0;
    double sor = 0.0;
    double nw = 0.0;
    double no = 0.0;
};

struct WellsConfig {
    bool enabled = false;
    int injector_cell_x = 0;
    int injector_cell_y = 0;
    int injector_cell_z = 0;
    double injector_rate_stb_day = 0.0;
    int producer_cell_x = 0;
    int producer_cell_y = 0;
    int producer_cell_z = 0;
    double producer_bhp_psi = 0.0;
    double producer_pi = 1.0;
};

struct HistoryControlEntry {
    double day = 0.0;
    std::string well;
    std::string control_kind;
    double target_value = 0.0;
    std::string phase;
};

struct HistoryConfig {
    bool enabled = false;
    std::string controls_csv;
    std::string observations_csv;
    double start_day = 0.0;
    double end_day = 0.0;
    double match_frequency_days = 0.0;
    std::vector<HistoryControlEntry> controls;
};

struct SimulationConfig {
    std::string case_name;
    int nx = 0;
    int ny = 0;
    int nz = 1;
    std::string dt_policy;
    std::string units;
    int schedule_end_step = 0;
    PhysicsConfig physics;
    RockConfig rock;
    FluidConfig fluid;
    WellsConfig wells;
    HistoryConfig history;
};

SimulationConfig load_simulation_config(const std::string& case_path);
