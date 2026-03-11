#include "sim/transport.hpp"

#if SIM_ENABLE_CUDA

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "sim/error.hpp"
#include "sim/state.hpp"

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw CliError(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", message);
}

[[noreturn]] void fail_io(const std::string& message) {
    throw CliError(ExitCode::E_IO, "E_IO", message);
}

void check_cuda(cudaError_t code, const char* where) {
    if (code == cudaSuccess) {
        return;
    }
    throw CliError(ExitCode::E_IO, "E_IO", std::string(where) + ": " + cudaGetErrorString(code));
}

size_t cell_index(int x, int y, int z, int nx, int ny) {
    return (static_cast<size_t>(z) * static_cast<size_t>(ny) + static_cast<size_t>(y)) * static_cast<size_t>(nx) + static_cast<size_t>(x);
}

double effective_saturation_host(const SimulationConfig& cfg, double sw) {
    const double denom = 1.0 - cfg.fluid.swc - cfg.fluid.sor;
    if (denom <= 0.0) {
        fail("fluid residual saturations must satisfy swc + sor < 1.");
    }
    const double se = (sw - cfg.fluid.swc) / denom;
    return std::clamp(se, 0.0, 1.0);
}

double fractional_flow_water_host(const SimulationConfig& cfg, double sw) {
    const double se = effective_saturation_host(cfg, sw);
    const double krw = std::pow(se, cfg.fluid.nw);
    const double kro = std::pow(1.0 - se, cfg.fluid.no);
    const double lambda_w = krw / cfg.fluid.mu_w_cp;
    const double lambda_o = kro / cfg.fluid.mu_o_cp;
    const double lambda_t = lambda_w + lambda_o;
    if (!std::isfinite(lambda_t) || lambda_t <= 0.0) {
        fail("total mobility must be finite and positive.");
    }
    return lambda_w / lambda_t;
}

double total_water_mass(const ReservoirState& state) {
    const size_t count = static_cast<size_t>(state.nx) * static_cast<size_t>(state.ny) * static_cast<size_t>(state.nz);
    double mass = 0.0;
    for (size_t i = 0; i < count; ++i) {
        mass += state.porosity[i] * state.sw[i];
    }
    return mass;
}

struct DeviceTransportParams {
    int nx;
    int ny;
    int nz;
    double dt_days;
    double swc;
    double sor;
    double nw;
    double no;
    double mu_w_cp;
    double mu_o_cp;
    double sw_min;
    double sw_max;
    int wells_enabled;
    int injector_cell_x;
    int injector_cell_y;
    int injector_cell_z;
    int producer_cell_x;
    int producer_cell_y;
    int producer_cell_z;
    double injector_rate_stb_day;
    double producer_bhp_psi;
    double producer_pi;
};

struct DeviceBuffers {
    double* sw = nullptr;
    double* pressure = nullptr;
    double* porosity = nullptr;
    double* permeability = nullptr;
    double* flux_x = nullptr;
    double* flux_y = nullptr;
    double* flux_z = nullptr;
    double* q_water = nullptr;
    double* next_sw = nullptr;
    int* clip_flags = nullptr;

    ~DeviceBuffers() {
        if (sw != nullptr) {
            (void)cudaFree(sw);
        }
        if (pressure != nullptr) {
            (void)cudaFree(pressure);
        }
        if (porosity != nullptr) {
            (void)cudaFree(porosity);
        }
        if (permeability != nullptr) {
            (void)cudaFree(permeability);
        }
        if (flux_x != nullptr) {
            (void)cudaFree(flux_x);
        }
        if (flux_y != nullptr) {
            (void)cudaFree(flux_y);
        }
        if (flux_z != nullptr) {
            (void)cudaFree(flux_z);
        }
        if (q_water != nullptr) {
            (void)cudaFree(q_water);
        }
        if (next_sw != nullptr) {
            (void)cudaFree(next_sw);
        }
        if (clip_flags != nullptr) {
            (void)cudaFree(clip_flags);
        }
    }
};

void free_device_ptr(double*& ptr) {
    if (ptr != nullptr) {
        (void)cudaFree(ptr);
        ptr = nullptr;
    }
}

void free_device_ptr(int*& ptr) {
    if (ptr != nullptr) {
        (void)cudaFree(ptr);
        ptr = nullptr;
    }
}

template <typename T>
void realloc_device(T*& ptr, size_t bytes, const char* where) {
    free_device_ptr(ptr);
    if (bytes == 0U) {
        return;
    }
    check_cuda(cudaMalloc(&ptr, bytes), where);
}

struct TransportGpuWorkspace {
    DeviceBuffers d;
    int nx = -1;
    int ny = -1;
    int nz = -1;
    size_t cell_count = 0;
    size_t flux_x_count = 0;
    size_t flux_y_count = 0;
    size_t flux_z_count = 0;
    const double* porosity_ptr = nullptr;
    const double* permeability_ptr = nullptr;
    bool constants_synced = false;

    void ensure_capacity(int in_nx, int in_ny, int in_nz) {
        const size_t wanted_cells = static_cast<size_t>(in_nx) * static_cast<size_t>(in_ny) * static_cast<size_t>(in_nz);
        const size_t wanted_fx = static_cast<size_t>(in_nx - 1) * static_cast<size_t>(in_ny) * static_cast<size_t>(in_nz);
        const size_t wanted_fy = static_cast<size_t>(in_nx) * static_cast<size_t>(in_ny - 1) * static_cast<size_t>(in_nz);
        const size_t wanted_fz = static_cast<size_t>(in_nx) * static_cast<size_t>(in_ny) * static_cast<size_t>(in_nz - 1);
        if (in_nx == nx && in_ny == ny && in_nz == nz && wanted_cells == cell_count &&
            wanted_fx == flux_x_count && wanted_fy == flux_y_count && wanted_fz == flux_z_count) {
            return;
        }

        nx = in_nx;
        ny = in_ny;
        nz = in_nz;
        cell_count = wanted_cells;
        flux_x_count = wanted_fx;
        flux_y_count = wanted_fy;
        flux_z_count = wanted_fz;
        constants_synced = false;
        porosity_ptr = nullptr;
        permeability_ptr = nullptr;

        realloc_device(d.sw, cell_count * sizeof(double), "cudaMalloc(d_sw)");
        realloc_device(d.pressure, cell_count * sizeof(double), "cudaMalloc(d_pressure)");
        realloc_device(d.porosity, cell_count * sizeof(double), "cudaMalloc(d_porosity)");
        realloc_device(d.permeability, cell_count * sizeof(double), "cudaMalloc(d_permeability)");
        realloc_device(d.q_water, cell_count * sizeof(double), "cudaMalloc(d_q_water)");
        realloc_device(d.next_sw, cell_count * sizeof(double), "cudaMalloc(d_next_sw)");
        realloc_device(d.clip_flags, cell_count * sizeof(int), "cudaMalloc(d_clip_flags)");
        realloc_device(d.flux_x, flux_x_count * sizeof(double), "cudaMalloc(d_flux_x)");
        realloc_device(d.flux_y, flux_y_count * sizeof(double), "cudaMalloc(d_flux_y)");
        realloc_device(d.flux_z, flux_z_count * sizeof(double), "cudaMalloc(d_flux_z)");
    }
};

TransportGpuWorkspace& transport_workspace() {
    static TransportGpuWorkspace ws;
    return ws;
}

__device__ int cell_index_device(int x, int y, int z, int nx, int ny) {
    return (z * ny + y) * nx + x;
}

__device__ double clamp_device(double v, double lo, double hi) {
    return fmin(fmax(v, lo), hi);
}

__device__ double effective_saturation_device(const DeviceTransportParams& p, double sw) {
    const double denom = 1.0 - p.swc - p.sor;
    const double se = (sw - p.swc) / denom;
    return clamp_device(se, 0.0, 1.0);
}

__device__ double fractional_flow_water_device(const DeviceTransportParams& p, double sw) {
    const double se = effective_saturation_device(p, sw);
    const double krw = pow(se, p.nw);
    const double kro = pow(1.0 - se, p.no);
    const double lambda_w = krw / p.mu_w_cp;
    const double lambda_o = kro / p.mu_o_cp;
    return lambda_w / (lambda_w + lambda_o);
}

__device__ double harmonic_average_device(double a, double b) {
    if (a <= 0.0 || b <= 0.0) {
        return 0.0;
    }
    return 2.0 * a * b / (a + b);
}

__device__ double total_mobility_device(const DeviceTransportParams& p, double sw) {
    const double se = effective_saturation_device(p, sw);
    const double krw = pow(se, p.nw);
    const double kro = pow(1.0 - se, p.no);
    const double lambda_w = krw / p.mu_w_cp;
    const double lambda_o = kro / p.mu_o_cp;
    return lambda_w + lambda_o;
}

__global__ void compute_flux_x_kernel(
    DeviceTransportParams p,
    const double* pressure,
    const double* sw,
    const double* permeability,
    double* flux_x) {
    const int fx_nx = p.nx - 1;
    const int count = fx_nx * p.ny * p.nz;
    const int face_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (face_idx >= count) {
        return;
    }

    const int z = face_idx / (fx_nx * p.ny);
    const int rem = face_idx % (fx_nx * p.ny);
    const int x = rem % fx_nx;
    const int y = rem / fx_nx;
    const int i = cell_index_device(x, y, z, p.nx, p.ny);
    const int j = cell_index_device(x + 1, y, z, p.nx, p.ny);
    const double lambda_i = total_mobility_device(p, sw[i]);
    const double lambda_j = total_mobility_device(p, sw[j]);
    const double perm_face = harmonic_average_device(permeability[i], permeability[j]);
    const double lambda_face = harmonic_average_device(lambda_i, lambda_j);
    const double transmissibility = perm_face * lambda_face;
    flux_x[face_idx] = -transmissibility * (pressure[j] - pressure[i]);
}

__global__ void compute_flux_y_kernel(
    DeviceTransportParams p,
    const double* pressure,
    const double* sw,
    const double* permeability,
    double* flux_y) {
    const int fy_ny = p.ny - 1;
    const int count = p.nx * fy_ny * p.nz;
    const int face_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (face_idx >= count) {
        return;
    }

    const int z = face_idx / (p.nx * fy_ny);
    const int rem = face_idx % (p.nx * fy_ny);
    const int x = rem % p.nx;
    const int y = rem / p.nx;
    const int i = cell_index_device(x, y, z, p.nx, p.ny);
    const int j = cell_index_device(x, y + 1, z, p.nx, p.ny);
    const double lambda_i = total_mobility_device(p, sw[i]);
    const double lambda_j = total_mobility_device(p, sw[j]);
    const double perm_face = harmonic_average_device(permeability[i], permeability[j]);
    const double lambda_face = harmonic_average_device(lambda_i, lambda_j);
    const double transmissibility = perm_face * lambda_face;
    flux_y[face_idx] = -transmissibility * (pressure[j] - pressure[i]);
}

__global__ void compute_flux_z_kernel(
    DeviceTransportParams p,
    const double* pressure,
    const double* sw,
    const double* permeability,
    double* flux_z) {
    const int fz_nz = p.nz - 1;
    const int count = p.nx * p.ny * fz_nz;
    const int face_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (face_idx >= count) {
        return;
    }

    const int z = face_idx / (p.nx * p.ny);
    const int rem = face_idx % (p.nx * p.ny);
    const int x = rem % p.nx;
    const int y = rem / p.nx;
    const int i = cell_index_device(x, y, z, p.nx, p.ny);
    const int j = cell_index_device(x, y, z + 1, p.nx, p.ny);
    const double lambda_i = total_mobility_device(p, sw[i]);
    const double lambda_j = total_mobility_device(p, sw[j]);
    const double perm_face = harmonic_average_device(permeability[i], permeability[j]);
    const double lambda_face = harmonic_average_device(lambda_i, lambda_j);
    const double transmissibility = perm_face * lambda_face;
    flux_z[face_idx] = -transmissibility * (pressure[j] - pressure[i]);
}

__global__ void build_well_sources_kernel(
    DeviceTransportParams p,
    const double* pressure,
    const double* sw,
    double* q_water) {
    if (threadIdx.x != 0 || blockIdx.x != 0 || p.wells_enabled == 0) {
        return;
    }

    constexpr double kWellRateScale = 1.0e-3;
    const int injector_idx = cell_index_device(p.injector_cell_x, p.injector_cell_y, p.injector_cell_z, p.nx, p.ny);
    const int producer_idx = cell_index_device(p.producer_cell_x, p.producer_cell_y, p.producer_cell_z, p.nx, p.ny);
    const double q_inj = kWellRateScale * p.injector_rate_stb_day;
    q_water[injector_idx] += q_inj;

    const double drawdown = fmax(pressure[producer_idx] - p.producer_bhp_psi, 0.0);
    const double q_prod_abs = kWellRateScale * p.producer_pi * drawdown;
    const double q_prod_total = -q_prod_abs;
    q_water[producer_idx] += fractional_flow_water_device(p, sw[producer_idx]) * q_prod_total;
}

__global__ void update_saturation_kernel(
    DeviceTransportParams p,
    const double* sw,
    const double* porosity,
    const double* flux_x,
    const double* flux_y,
    const double* flux_z,
    const double* q_water,
    double* next_sw,
    int* clip_flags) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = p.nx * p.ny * p.nz;
    if (idx >= count) {
        return;
    }

    const int cells_per_layer = p.nx * p.ny;
    const int z = idx / cells_per_layer;
    const int rem = idx % cells_per_layer;
    const int x = rem % p.nx;
    const int y = rem / p.nx;
    double water_flux_sum = 0.0;

    const int fx_nx = p.nx - 1;
    const int fy_ny = p.ny - 1;
    if (x > 0) {
        const int face = cell_index_device(x - 1, y, z, fx_nx, p.ny);
        const double outward_flux = -flux_x[face];
        const int upwind = (outward_flux >= 0.0) ? idx : cell_index_device(x - 1, y, z, p.nx, p.ny);
        water_flux_sum += fractional_flow_water_device(p, sw[upwind]) * outward_flux;
    }
    if (x + 1 < p.nx) {
        const int face = cell_index_device(x, y, z, fx_nx, p.ny);
        const double outward_flux = flux_x[face];
        const int upwind = (outward_flux >= 0.0) ? idx : cell_index_device(x + 1, y, z, p.nx, p.ny);
        water_flux_sum += fractional_flow_water_device(p, sw[upwind]) * outward_flux;
    }
    if (y > 0) {
        const int face = cell_index_device(x, y - 1, z, p.nx, fy_ny);
        const double outward_flux = -flux_y[face];
        const int upwind = (outward_flux >= 0.0) ? idx : cell_index_device(x, y - 1, z, p.nx, p.ny);
        water_flux_sum += fractional_flow_water_device(p, sw[upwind]) * outward_flux;
    }
    if (y + 1 < p.ny) {
        const int face = cell_index_device(x, y, z, p.nx, fy_ny);
        const double outward_flux = flux_y[face];
        const int upwind = (outward_flux >= 0.0) ? idx : cell_index_device(x, y + 1, z, p.nx, p.ny);
        water_flux_sum += fractional_flow_water_device(p, sw[upwind]) * outward_flux;
    }
    if (z > 0) {
        const int face = cell_index_device(x, y, z - 1, p.nx, p.ny);
        const double outward_flux = -flux_z[face];
        const int upwind = (outward_flux >= 0.0) ? idx : cell_index_device(x, y, z - 1, p.nx, p.ny);
        water_flux_sum += fractional_flow_water_device(p, sw[upwind]) * outward_flux;
    }
    if (z + 1 < p.nz) {
        const int face = cell_index_device(x, y, z, p.nx, p.ny);
        const double outward_flux = flux_z[face];
        const int upwind = (outward_flux >= 0.0) ? idx : cell_index_device(x, y, z + 1, p.nx, p.ny);
        water_flux_sum += fractional_flow_water_device(p, sw[upwind]) * outward_flux;
    }

    const double sw_new =
        sw[idx] - (p.dt_days / porosity[idx]) * water_flux_sum + (p.dt_days / porosity[idx]) * q_water[idx];
    const double clipped = clamp_device(sw_new, p.sw_min, p.sw_max);
    next_sw[idx] = clipped;
    clip_flags[idx] = (fabs(clipped - sw_new) > 0.0) ? 1 : 0;
}

}  // namespace

TransportDiagnostics advance_saturation_impes_with_dt_gpu(const SimulationConfig& cfg, ReservoirState& state, double dt_days) {
    validate_state_invariants(state);
    if (state.nx != cfg.nx || state.ny != cfg.ny || state.nz != cfg.nz) {
        fail("state dimensions must match configuration.");
    }
    if (!(dt_days > 0.0) || !std::isfinite(dt_days)) {
        fail("transport dt_days must be finite and positive.");
    }

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        fail_io("backend=gpu requested but no CUDA device is available.");
    }

    const size_t count = static_cast<size_t>(cfg.nx) * static_cast<size_t>(cfg.ny) * static_cast<size_t>(cfg.nz);

    const double mass_before = total_water_mass(state);
    std::vector<double> next_sw(count, 0.0);
    std::vector<int> clip_flags(count, 0);

    DeviceTransportParams params{};
    params.nx = cfg.nx;
    params.ny = cfg.ny;
    params.nz = cfg.nz;
    params.dt_days = dt_days;
    params.swc = cfg.fluid.swc;
    params.sor = cfg.fluid.sor;
    params.nw = cfg.fluid.nw;
    params.no = cfg.fluid.no;
    params.mu_w_cp = cfg.fluid.mu_w_cp;
    params.mu_o_cp = cfg.fluid.mu_o_cp;
    params.sw_min = cfg.fluid.swc;
    params.sw_max = 1.0 - cfg.fluid.sor;
    params.wells_enabled = cfg.wells.enabled ? 1 : 0;
    params.injector_cell_x = cfg.wells.injector_cell_x;
    params.injector_cell_y = cfg.wells.injector_cell_y;
    params.injector_cell_z = cfg.wells.injector_cell_z;
    params.producer_cell_x = cfg.wells.producer_cell_x;
    params.producer_cell_y = cfg.wells.producer_cell_y;
    params.producer_cell_z = cfg.wells.producer_cell_z;
    params.injector_rate_stb_day = cfg.wells.injector_rate_stb_day;
    params.producer_bhp_psi = cfg.wells.producer_bhp_psi;
    params.producer_pi = cfg.wells.producer_pi;

    TransportGpuWorkspace& ws = transport_workspace();
    ws.ensure_capacity(cfg.nx, cfg.ny, cfg.nz);

    if (!ws.constants_synced || ws.porosity_ptr != state.porosity.data() || ws.permeability_ptr != state.permeability_md.data()) {
        check_cuda(
            cudaMemcpy(ws.d.porosity, state.porosity.data(), count * sizeof(double), cudaMemcpyHostToDevice),
            "cudaMemcpy(porosity)");
        check_cuda(
            cudaMemcpy(ws.d.permeability, state.permeability_md.data(), count * sizeof(double), cudaMemcpyHostToDevice),
            "cudaMemcpy(permeability)");
        ws.constants_synced = true;
        ws.porosity_ptr = state.porosity.data();
        ws.permeability_ptr = state.permeability_md.data();
    }

    check_cuda(
        cudaMemcpy(ws.d.sw, state.sw.data(), count * sizeof(double), cudaMemcpyHostToDevice),
        "cudaMemcpy(sw)");
    check_cuda(
        cudaMemcpy(ws.d.pressure, state.pressure.data(), count * sizeof(double), cudaMemcpyHostToDevice),
        "cudaMemcpy(pressure)");
    check_cuda(cudaMemset(ws.d.q_water, 0, count * sizeof(double)), "cudaMemset(q_water)");
    check_cuda(cudaMemset(ws.d.clip_flags, 0, count * sizeof(int)), "cudaMemset(clip_flags)");

    const int threads_per_block = 256;
    if (ws.flux_x_count > 0) {
        const int flux_x_blocks = static_cast<int>((ws.flux_x_count + static_cast<size_t>(threads_per_block) - 1U) / static_cast<size_t>(threads_per_block));
        compute_flux_x_kernel<<<flux_x_blocks, threads_per_block>>>(params, ws.d.pressure, ws.d.sw, ws.d.permeability, ws.d.flux_x);
        check_cuda(cudaGetLastError(), "compute_flux_x_kernel launch");
    }
    if (ws.flux_y_count > 0) {
        const int flux_y_blocks = static_cast<int>((ws.flux_y_count + static_cast<size_t>(threads_per_block) - 1U) / static_cast<size_t>(threads_per_block));
        compute_flux_y_kernel<<<flux_y_blocks, threads_per_block>>>(params, ws.d.pressure, ws.d.sw, ws.d.permeability, ws.d.flux_y);
        check_cuda(cudaGetLastError(), "compute_flux_y_kernel launch");
    }
    if (ws.flux_z_count > 0) {
        const int flux_z_blocks = static_cast<int>((ws.flux_z_count + static_cast<size_t>(threads_per_block) - 1U) / static_cast<size_t>(threads_per_block));
        compute_flux_z_kernel<<<flux_z_blocks, threads_per_block>>>(params, ws.d.pressure, ws.d.sw, ws.d.permeability, ws.d.flux_z);
        check_cuda(cudaGetLastError(), "compute_flux_z_kernel launch");
    }
    build_well_sources_kernel<<<1, 1>>>(params, ws.d.pressure, ws.d.sw, ws.d.q_water);
    check_cuda(cudaGetLastError(), "build_well_sources_kernel launch");

    const int blocks = static_cast<int>((count + static_cast<size_t>(threads_per_block) - 1U) / static_cast<size_t>(threads_per_block));
    update_saturation_kernel<<<blocks, threads_per_block>>>(
        params, ws.d.sw, ws.d.porosity, ws.d.flux_x, ws.d.flux_y, ws.d.flux_z, ws.d.q_water, ws.d.next_sw, ws.d.clip_flags);
    check_cuda(cudaGetLastError(), "update_saturation_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(update_saturation_kernel)");

    check_cuda(
        cudaMemcpy(next_sw.data(), ws.d.next_sw, count * sizeof(double), cudaMemcpyDeviceToHost),
        "cudaMemcpy(next_sw)");
    check_cuda(
        cudaMemcpy(clip_flags.data(), ws.d.clip_flags, count * sizeof(int), cudaMemcpyDeviceToHost),
        "cudaMemcpy(clip_flags)");

    int clip_count = 0;
    for (int clipped : clip_flags) {
        clip_count += clipped;
    }

    double expected_source_delta = 0.0;
    if (cfg.wells.enabled) {
        constexpr double kWellRateScale = 1.0e-3;
        const size_t producer_idx = cell_index(
            cfg.wells.producer_cell_x, cfg.wells.producer_cell_y, cfg.wells.producer_cell_z, cfg.nx, cfg.ny);
        const double q_inj = kWellRateScale * cfg.wells.injector_rate_stb_day;
        const double drawdown = std::max(state.pressure[producer_idx] - cfg.wells.producer_bhp_psi, 0.0);
        const double q_prod_abs = kWellRateScale * cfg.wells.producer_pi * drawdown;
        const double q_prod_total = -q_prod_abs;
        const double q_prod_w = fractional_flow_water_host(cfg, state.sw[producer_idx]) * q_prod_total;
        expected_source_delta = dt_days * (q_inj + q_prod_w);
    }
    state.sw = std::move(next_sw);
    validate_state_invariants(state);
    const double mass_after = total_water_mass(state);
    const double mass_delta_abs = std::abs((mass_after - mass_before) - expected_source_delta);
    const double mass_denom = std::max(std::abs(mass_before), 1.0e-20);
    const double mass_balance_rel = mass_delta_abs / mass_denom;
    return TransportDiagnostics{dt_days, clip_count, mass_balance_rel};
}

#endif
