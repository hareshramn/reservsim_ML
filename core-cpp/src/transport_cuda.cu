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

void check_cuda(cudaError_t code, const char* where) {
    if (code == cudaSuccess) {
        return;
    }
    throw CliError(ExitCode::E_IO, "E_IO", std::string(where) + ": " + cudaGetErrorString(code));
}

size_t cell_index(int x, int y, int nx) {
    return static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
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

void accumulate_well_source_terms_host(
    const SimulationConfig& cfg,
    const ReservoirState& state,
    std::vector<double>& q_water) {
    if (!cfg.wells.enabled) {
        return;
    }

    constexpr double kWellRateScale = 1.0e-3;
    const size_t injector_idx = cell_index(cfg.wells.injector_cell_x, cfg.wells.injector_cell_y, cfg.nx);
    const size_t producer_idx = cell_index(cfg.wells.producer_cell_x, cfg.wells.producer_cell_y, cfg.nx);

    const double q_inj = kWellRateScale * cfg.wells.injector_rate_stb_day;
    q_water[injector_idx] += q_inj;

    const double drawdown = std::max(state.pressure[producer_idx] - cfg.wells.producer_bhp_psi, 0.0);
    const double q_prod_abs = kWellRateScale * cfg.wells.producer_pi * drawdown;
    const double q_prod_total = -q_prod_abs;
    q_water[producer_idx] += fractional_flow_water_host(cfg, state.sw[producer_idx]) * q_prod_total;
}

double total_water_mass(const ReservoirState& state) {
    const size_t count = static_cast<size_t>(state.nx) * static_cast<size_t>(state.ny);
    double mass = 0.0;
    for (size_t i = 0; i < count; ++i) {
        mass += state.porosity[i] * state.sw[i];
    }
    return mass;
}

struct DeviceTransportParams {
    int nx;
    int ny;
    double dt_days;
    double swc;
    double sor;
    double nw;
    double no;
    double mu_w_cp;
    double mu_o_cp;
    double sw_min;
    double sw_max;
};

__device__ int cell_index_device(int x, int y, int nx) {
    return y * nx + x;
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

__global__ void update_saturation_kernel(
    DeviceTransportParams p,
    const double* sw,
    const double* porosity,
    const double* flux_x,
    const double* flux_y,
    const double* q_water,
    double* next_sw,
    int* clip_flags) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = p.nx * p.ny;
    if (idx >= count) {
        return;
    }

    const int x = idx % p.nx;
    const int y = idx / p.nx;
    double water_flux_sum = 0.0;

    if (x > 0) {
        const int face = cell_index_device(x - 1, y, p.nx - 1);
        const double outward_flux = -flux_x[face];
        const int upwind = (outward_flux >= 0.0) ? idx : cell_index_device(x - 1, y, p.nx);
        water_flux_sum += fractional_flow_water_device(p, sw[upwind]) * outward_flux;
    }
    if (x + 1 < p.nx) {
        const int face = cell_index_device(x, y, p.nx - 1);
        const double outward_flux = flux_x[face];
        const int upwind = (outward_flux >= 0.0) ? idx : cell_index_device(x + 1, y, p.nx);
        water_flux_sum += fractional_flow_water_device(p, sw[upwind]) * outward_flux;
    }
    if (y > 0) {
        const int face = cell_index_device(x, y - 1, p.nx);
        const double outward_flux = -flux_y[face];
        const int upwind = (outward_flux >= 0.0) ? idx : cell_index_device(x, y - 1, p.nx);
        water_flux_sum += fractional_flow_water_device(p, sw[upwind]) * outward_flux;
    }
    if (y + 1 < p.ny) {
        const int face = cell_index_device(x, y, p.nx);
        const double outward_flux = flux_y[face];
        const int upwind = (outward_flux >= 0.0) ? idx : cell_index_device(x, y + 1, p.nx);
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
    if (state.nx != cfg.nx || state.ny != cfg.ny) {
        fail("state dimensions must match configuration.");
    }
    if (!(dt_days > 0.0) || !std::isfinite(dt_days)) {
        fail("transport dt_days must be finite and positive.");
    }

    const std::vector<double> flux_x = compute_total_flux_x(cfg, state);
    const std::vector<double> flux_y = compute_total_flux_y(cfg, state);
    const size_t count = static_cast<size_t>(cfg.nx) * static_cast<size_t>(cfg.ny);
    std::vector<double> q_water(count, 0.0);
    accumulate_well_source_terms_host(cfg, state, q_water);

    const double mass_before = total_water_mass(state);
    std::vector<double> next_sw(count, 0.0);
    std::vector<int> clip_flags(count, 0);

    DeviceTransportParams params{};
    params.nx = cfg.nx;
    params.ny = cfg.ny;
    params.dt_days = dt_days;
    params.swc = cfg.fluid.swc;
    params.sor = cfg.fluid.sor;
    params.nw = cfg.fluid.nw;
    params.no = cfg.fluid.no;
    params.mu_w_cp = cfg.fluid.mu_w_cp;
    params.mu_o_cp = cfg.fluid.mu_o_cp;
    params.sw_min = cfg.fluid.swc;
    params.sw_max = 1.0 - cfg.fluid.sor;

    double* d_sw = nullptr;
    double* d_porosity = nullptr;
    double* d_flux_x = nullptr;
    double* d_flux_y = nullptr;
    double* d_q_water = nullptr;
    double* d_next_sw = nullptr;
    int* d_clip_flags = nullptr;

    check_cuda(cudaMalloc(&d_sw, count * sizeof(double)), "cudaMalloc(d_sw)");
    check_cuda(cudaMalloc(&d_porosity, count * sizeof(double)), "cudaMalloc(d_porosity)");
    check_cuda(cudaMalloc(&d_q_water, count * sizeof(double)), "cudaMalloc(d_q_water)");
    check_cuda(cudaMalloc(&d_next_sw, count * sizeof(double)), "cudaMalloc(d_next_sw)");
    check_cuda(cudaMalloc(&d_clip_flags, count * sizeof(int)), "cudaMalloc(d_clip_flags)");
    check_cuda(
        cudaMalloc(&d_flux_x, flux_x.size() * sizeof(double)),
        "cudaMalloc(d_flux_x)");
    check_cuda(
        cudaMalloc(&d_flux_y, flux_y.size() * sizeof(double)),
        "cudaMalloc(d_flux_y)");

    check_cuda(
        cudaMemcpy(d_sw, state.sw.data(), count * sizeof(double), cudaMemcpyHostToDevice),
        "cudaMemcpy(sw)");
    check_cuda(
        cudaMemcpy(d_porosity, state.porosity.data(), count * sizeof(double), cudaMemcpyHostToDevice),
        "cudaMemcpy(porosity)");
    check_cuda(
        cudaMemcpy(d_q_water, q_water.data(), count * sizeof(double), cudaMemcpyHostToDevice),
        "cudaMemcpy(q_water)");
    if (!flux_x.empty()) {
        check_cuda(
            cudaMemcpy(d_flux_x, flux_x.data(), flux_x.size() * sizeof(double), cudaMemcpyHostToDevice),
            "cudaMemcpy(flux_x)");
    }
    if (!flux_y.empty()) {
        check_cuda(
            cudaMemcpy(d_flux_y, flux_y.data(), flux_y.size() * sizeof(double), cudaMemcpyHostToDevice),
            "cudaMemcpy(flux_y)");
    }

    const int threads_per_block = 256;
    const int blocks = static_cast<int>((count + static_cast<size_t>(threads_per_block) - 1U) / static_cast<size_t>(threads_per_block));
    update_saturation_kernel<<<blocks, threads_per_block>>>(params, d_sw, d_porosity, d_flux_x, d_flux_y, d_q_water, d_next_sw, d_clip_flags);
    check_cuda(cudaGetLastError(), "update_saturation_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(update_saturation_kernel)");

    check_cuda(
        cudaMemcpy(next_sw.data(), d_next_sw, count * sizeof(double), cudaMemcpyDeviceToHost),
        "cudaMemcpy(next_sw)");
    check_cuda(
        cudaMemcpy(clip_flags.data(), d_clip_flags, count * sizeof(int), cudaMemcpyDeviceToHost),
        "cudaMemcpy(clip_flags)");

    check_cuda(cudaFree(d_sw), "cudaFree(d_sw)");
    check_cuda(cudaFree(d_porosity), "cudaFree(d_porosity)");
    check_cuda(cudaFree(d_flux_x), "cudaFree(d_flux_x)");
    check_cuda(cudaFree(d_flux_y), "cudaFree(d_flux_y)");
    check_cuda(cudaFree(d_q_water), "cudaFree(d_q_water)");
    check_cuda(cudaFree(d_next_sw), "cudaFree(d_next_sw)");
    check_cuda(cudaFree(d_clip_flags), "cudaFree(d_clip_flags)");

    int clip_count = 0;
    for (int clipped : clip_flags) {
        clip_count += clipped;
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

#endif
