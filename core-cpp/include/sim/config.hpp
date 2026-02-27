#pragma once

#include <string>

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

struct SimulationConfig {
    std::string case_name;
    int nx = 0;
    int ny = 0;
    std::string dt_policy;
    std::string units;
    int schedule_end_step = 0;
    PhysicsConfig physics;
    RockConfig rock;
    FluidConfig fluid;
};

SimulationConfig load_simulation_config(const std::string& case_path);
