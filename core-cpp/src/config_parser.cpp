#include "sim/config.hpp"

#include "sim/error.hpp"

#include <cctype>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])) != 0) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        --end;
    }
    return s.substr(start, end - start);
}

std::string unquote(std::string s) {
    if (s.size() >= 2) {
        const bool dq = s.front() == '"' && s.back() == '"';
        const bool sq = s.front() == '\'' && s.back() == '\'';
        if (dq || sq) {
            return s.substr(1, s.size() - 2);
        }
    }
    return s;
}

std::string to_lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

[[noreturn]] void fail(ExitCode code, const std::string& symbol, const std::string& message) {
    throw CliError(code, symbol, message);
}

int parse_positive_int(const std::string& raw, const std::string& field) {
    try {
        size_t idx = 0;
        const int value = std::stoi(raw, &idx);
        if (idx != raw.size() || value <= 0) {
            fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", field + " must be a positive integer.");
        }
        return value;
    } catch (const std::exception&) {
        fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", field + " must be a positive integer.");
    }
}

int parse_nonnegative_int(const std::string& raw, const std::string& field) {
    try {
        size_t idx = 0;
        const int value = std::stoi(raw, &idx);
        if (idx != raw.size() || value < 0) {
            fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", field + " must be a non-negative integer.");
        }
        return value;
    } catch (const std::exception&) {
        fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", field + " must be a non-negative integer.");
    }
}

double parse_positive_double(const std::string& raw, const std::string& field) {
    try {
        size_t idx = 0;
        const double value = std::stod(raw, &idx);
        if (idx != raw.size() || value <= 0.0) {
            fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", field + " must be a positive number.");
        }
        return value;
    } catch (const std::exception&) {
        fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", field + " must be a positive number.");
    }
}

double parse_unit_interval_double(const std::string& raw, const std::string& field) {
    try {
        size_t idx = 0;
        const double value = std::stod(raw, &idx);
        if (idx != raw.size() || value < 0.0 || value > 1.0) {
            fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", field + " must be in [0, 1].");
        }
        return value;
    } catch (const std::exception&) {
        fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", field + " must be in [0, 1].");
    }
}

bool parse_bool(const std::string& raw, const std::string& field) {
    const std::string v = to_lower(raw);
    if (v == "true") {
        return true;
    }
    if (v == "false") {
        return false;
    }
    fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", field + " must be true or false.");
}

std::string require_field(const std::map<std::string, std::string>& fields, const std::string& key) {
    const auto it = fields.find(key);
    if (it == fields.end()) {
        fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", "Missing required case field: " + key);
    }
    return it->second;
}

}  // namespace

SimulationConfig load_simulation_config(const std::string& case_path) {
    std::ifstream in(case_path);
    if (!in.is_open()) {
        fail(ExitCode::E_CASE_PARSE, "E_CASE_PARSE", "Unable to open case file: " + case_path);
    }

    std::map<std::string, std::string> fields;
    const std::unordered_set<std::string> sections = {"physics", "rock", "fluid"};
    std::string active_section;
    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        size_t indent = 0;
        while (indent < line.size() && line[indent] == ' ') {
            ++indent;
        }
        const std::string t = trim(line);
        if (t.empty() || t[0] == '#') {
            continue;
        }
        const auto pos = t.find(':');
        if (pos == std::string::npos) {
            fail(ExitCode::E_CASE_PARSE, "E_CASE_PARSE",
                 "Invalid YAML line " + std::to_string(line_no) + ": expected key: value.");
        }
        const std::string key = trim(t.substr(0, pos));
        const std::string value = trim(t.substr(pos + 1));
        if (key.empty()) {
            fail(ExitCode::E_CASE_PARSE, "E_CASE_PARSE",
                 "Invalid YAML line " + std::to_string(line_no) + ": empty key/value.");
        }

        if (indent == 0) {
            active_section.clear();
            if (value.empty()) {
                if (sections.find(key) == sections.end()) {
                    fail(ExitCode::E_CASE_PARSE, "E_CASE_PARSE",
                         "Invalid section on line " + std::to_string(line_no) + ": " + key);
                }
                active_section = key;
                continue;
            }
            fields[key] = unquote(value);
            continue;
        }

        if (indent != 2) {
            fail(ExitCode::E_CASE_PARSE, "E_CASE_PARSE",
                 "Unsupported indentation on line " + std::to_string(line_no) + ".");
        }
        if (active_section.empty()) {
            fail(ExitCode::E_CASE_PARSE, "E_CASE_PARSE",
                 "Nested key without parent section on line " + std::to_string(line_no) + ".");
        }
        if (value.empty()) {
            fail(ExitCode::E_CASE_PARSE, "E_CASE_PARSE",
                 "Nested key has empty value on line " + std::to_string(line_no) + ".");
        }
        fields[active_section + "." + key] = unquote(value);
    }

    const std::vector<std::string> required = {
        "case_name", "nx", "ny", "dt_policy", "units", "schedule_end_step"};
    const std::vector<std::string> required_physics = {
        "physics.phases", "physics.incompressible", "physics.gravity", "physics.capillary",
        "rock.porosity", "rock.permeability_md",
        "fluid.mu_w_cp", "fluid.mu_o_cp", "fluid.swc", "fluid.sor", "fluid.nw", "fluid.no"};
    for (const auto& key : required) { (void)require_field(fields, key); }
    for (const auto& key : required_physics) { (void)require_field(fields, key); }

    SimulationConfig cfg;
    cfg.case_name = require_field(fields, "case_name");
    cfg.nx = parse_positive_int(require_field(fields, "nx"), "nx");
    cfg.ny = parse_positive_int(require_field(fields, "ny"), "ny");
    cfg.dt_policy = require_field(fields, "dt_policy");
    cfg.units = require_field(fields, "units");
    cfg.schedule_end_step = parse_positive_int(require_field(fields, "schedule_end_step"), "schedule_end_step");
    cfg.physics.phases = require_field(fields, "physics.phases");
    cfg.physics.incompressible = parse_bool(require_field(fields, "physics.incompressible"), "physics.incompressible");
    cfg.physics.gravity = parse_bool(require_field(fields, "physics.gravity"), "physics.gravity");
    cfg.physics.capillary = parse_bool(require_field(fields, "physics.capillary"), "physics.capillary");
    cfg.rock.porosity = parse_unit_interval_double(require_field(fields, "rock.porosity"), "rock.porosity");
    cfg.rock.permeability_md = parse_positive_double(require_field(fields, "rock.permeability_md"), "rock.permeability_md");
    cfg.fluid.mu_w_cp = parse_positive_double(require_field(fields, "fluid.mu_w_cp"), "fluid.mu_w_cp");
    cfg.fluid.mu_o_cp = parse_positive_double(require_field(fields, "fluid.mu_o_cp"), "fluid.mu_o_cp");
    cfg.fluid.swc = parse_unit_interval_double(require_field(fields, "fluid.swc"), "fluid.swc");
    cfg.fluid.sor = parse_unit_interval_double(require_field(fields, "fluid.sor"), "fluid.sor");
    cfg.fluid.nw = parse_positive_double(require_field(fields, "fluid.nw"), "fluid.nw");
    cfg.fluid.no = parse_positive_double(require_field(fields, "fluid.no"), "fluid.no");

    const bool has_inj_x = fields.find("injector_cell_x") != fields.end();
    const bool has_inj_y = fields.find("injector_cell_y") != fields.end();
    const bool has_inj_rate = fields.find("injector_rate_stb_day") != fields.end();
    const bool has_prod_x = fields.find("producer_cell_x") != fields.end();
    const bool has_prod_y = fields.find("producer_cell_y") != fields.end();
    const bool has_prod_bhp = fields.find("producer_bhp_psi") != fields.end();
    const bool has_any_well_field = has_inj_x || has_inj_y || has_inj_rate || has_prod_x || has_prod_y || has_prod_bhp;
    const bool has_all_well_fields = has_inj_x && has_inj_y && has_inj_rate && has_prod_x && has_prod_y && has_prod_bhp;
    if (has_any_well_field && !has_all_well_fields) {
        fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA",
             "Well configuration must define injector_cell_x, injector_cell_y, injector_rate_stb_day, "
             "producer_cell_x, producer_cell_y, and producer_bhp_psi together.");
    }

    if (has_all_well_fields) {
        cfg.wells.enabled = true;
        cfg.wells.injector_cell_x = parse_nonnegative_int(require_field(fields, "injector_cell_x"), "injector_cell_x");
        cfg.wells.injector_cell_y = parse_nonnegative_int(require_field(fields, "injector_cell_y"), "injector_cell_y");
        cfg.wells.injector_rate_stb_day = parse_positive_double(require_field(fields, "injector_rate_stb_day"), "injector_rate_stb_day");
        cfg.wells.producer_cell_x = parse_nonnegative_int(require_field(fields, "producer_cell_x"), "producer_cell_x");
        cfg.wells.producer_cell_y = parse_nonnegative_int(require_field(fields, "producer_cell_y"), "producer_cell_y");
        cfg.wells.producer_bhp_psi = parse_positive_double(require_field(fields, "producer_bhp_psi"), "producer_bhp_psi");
        if (const auto it = fields.find("producer_pi"); it != fields.end()) {
            cfg.wells.producer_pi = parse_positive_double(it->second, "producer_pi");
        }
    }

    if (cfg.case_name.empty() || cfg.dt_policy.empty() || cfg.units.empty()) {
        fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", "case_name, dt_policy, and units must be non-empty.");
    }
    if (cfg.physics.phases.empty()) {
        fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", "physics.phases must be non-empty.");
    }
    if (cfg.fluid.swc + cfg.fluid.sor >= 1.0) {
        fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", "fluid.swc + fluid.sor must be < 1.");
    }
    if (cfg.wells.enabled) {
        if (cfg.wells.injector_cell_x < 0 || cfg.wells.injector_cell_x >= cfg.nx ||
            cfg.wells.injector_cell_y < 0 || cfg.wells.injector_cell_y >= cfg.ny) {
            fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", "Injector cell coordinates are out of grid bounds.");
        }
        if (cfg.wells.producer_cell_x < 0 || cfg.wells.producer_cell_x >= cfg.nx ||
            cfg.wells.producer_cell_y < 0 || cfg.wells.producer_cell_y >= cfg.ny) {
            fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", "Producer cell coordinates are out of grid bounds.");
        }
        if (cfg.wells.injector_cell_x == cfg.wells.producer_cell_x &&
            cfg.wells.injector_cell_y == cfg.wells.producer_cell_y) {
            fail(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", "Injector and producer cells must be distinct.");
        }
    }

    return cfg;
}
