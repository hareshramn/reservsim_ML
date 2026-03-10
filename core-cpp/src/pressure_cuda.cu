#include "sim/pressure.hpp"

#if SIM_ENABLE_CUDA

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "sim/error.hpp"

namespace {

[[noreturn]] void fail_schema(const std::string& message) {
    throw CliError(ExitCode::E_CASE_SCHEMA, "E_CASE_SCHEMA", message);
}

void check_cuda(cudaError_t code, const char* where) {
    if (code == cudaSuccess) {
        return;
    }
    throw CliError(ExitCode::E_IO, "E_IO", std::string(where) + ": " + cudaGetErrorString(code));
}

template <typename T>
void free_device(T*& ptr) {
    if (ptr != nullptr) {
        (void)cudaFree(ptr);
        ptr = nullptr;
    }
}

template <typename T>
void realloc_device(T*& ptr, size_t count, const char* where) {
    free_device(ptr);
    if (count == 0U) {
        return;
    }
    check_cuda(cudaMalloc(&ptr, count * sizeof(T)), where);
}

struct PressureGpuWorkspace {
    int nx = 0;
    int ny = 0;
    size_t count = 0;
    size_t partial_count = 0;

    double* diag = nullptr;
    double* west = nullptr;
    double* east = nullptr;
    double* south = nullptr;
    double* north = nullptr;
    double* rhs = nullptr;

    double* x = nullptr;
    double* r = nullptr;
    double* z = nullptr;
    double* p = nullptr;
    double* ap = nullptr;
    double* partial_a = nullptr;
    double* partial_b = nullptr;

    ~PressureGpuWorkspace() {
        free_device(diag);
        free_device(west);
        free_device(east);
        free_device(south);
        free_device(north);
        free_device(rhs);
        free_device(x);
        free_device(r);
        free_device(z);
        free_device(p);
        free_device(ap);
        free_device(partial_a);
        free_device(partial_b);
    }

    void ensure_capacity(int in_nx, int in_ny) {
        const size_t wanted_count = static_cast<size_t>(in_nx) * static_cast<size_t>(in_ny);
        constexpr int threads = 256;
        const size_t wanted_partial = (wanted_count + static_cast<size_t>(threads) - 1U) / static_cast<size_t>(threads);
        if (nx == in_nx && ny == in_ny && count == wanted_count && partial_count == wanted_partial) {
            return;
        }

        nx = in_nx;
        ny = in_ny;
        count = wanted_count;
        partial_count = wanted_partial;

        realloc_device(diag, count, "cudaMalloc(diag)");
        realloc_device(west, count, "cudaMalloc(west)");
        realloc_device(east, count, "cudaMalloc(east)");
        realloc_device(south, count, "cudaMalloc(south)");
        realloc_device(north, count, "cudaMalloc(north)");
        realloc_device(rhs, count, "cudaMalloc(rhs)");
        realloc_device(x, count, "cudaMalloc(x)");
        realloc_device(r, count, "cudaMalloc(r)");
        realloc_device(z, count, "cudaMalloc(z)");
        realloc_device(p, count, "cudaMalloc(p)");
        realloc_device(ap, count, "cudaMalloc(ap)");
        realloc_device(partial_a, partial_count, "cudaMalloc(partial_a)");
        realloc_device(partial_b, partial_count, "cudaMalloc(partial_b)");
    }
};

PressureGpuWorkspace& pressure_workspace() {
    static PressureGpuWorkspace ws;
    return ws;
}

__device__ int cell_index_device(int x, int y, int nx) {
    return y * nx + x;
}

__global__ void apply_system_kernel(
    int nx,
    int ny,
    const double* diag,
    const double* west,
    const double* east,
    const double* south,
    const double* north,
    const double* x,
    double* y) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = nx * ny;
    if (idx >= count) {
        return;
    }

    const int cx = idx % nx;
    const int cy = idx / nx;
    double value = diag[idx] * x[idx];
    if (cx > 0) {
        value += west[idx] * x[cell_index_device(cx - 1, cy, nx)];
    }
    if (cx + 1 < nx) {
        value += east[idx] * x[cell_index_device(cx + 1, cy, nx)];
    }
    if (cy > 0) {
        value += south[idx] * x[cell_index_device(cx, cy - 1, nx)];
    }
    if (cy + 1 < ny) {
        value += north[idx] * x[cell_index_device(cx, cy + 1, nx)];
    }
    y[idx] = value;
}

__global__ void residual_kernel(const double* rhs, const double* ax, double* r, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    r[idx] = rhs[idx] - ax[idx];
}

__global__ void jacobi_precond_kernel(const double* r, const double* diag, double* z, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    z[idx] = r[idx] / diag[idx];
}

__global__ void copy_kernel(const double* src, double* dst, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    dst[idx] = src[idx];
}

__global__ void axpy_kernel(double* y, const double* x, double alpha, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    y[idx] += alpha * x[idx];
}

__global__ void update_search_dir_kernel(double* p, const double* z, double beta, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    p[idx] = z[idx] + beta * p[idx];
}

__global__ void dot_partial_kernel(const double* a, const double* b, double* partial, int n) {
    extern __shared__ double ssum[];
    const int tid = static_cast<int>(threadIdx.x);
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    double value = 0.0;
    if (idx < n) {
        value = a[idx] * b[idx];
    }
    ssum[tid] = value;
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x / 2U); stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = ssum[0];
    }
}

__global__ void reduce_sum_kernel(const double* in, double* out, int n) {
    extern __shared__ double ssum[];
    const int tid = static_cast<int>(threadIdx.x);
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    double value = 0.0;
    if (idx < n) {
        value = in[idx];
    }
    ssum[tid] = value;
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x / 2U); stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = ssum[0];
    }
}

double dot_device(const double* a, const double* b, size_t n, PressureGpuWorkspace& ws) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + static_cast<size_t>(threads) - 1U) / static_cast<size_t>(threads));
    dot_partial_kernel<<<blocks, threads, static_cast<size_t>(threads) * sizeof(double)>>>(a, b, ws.partial_a, static_cast<int>(n));
    check_cuda(cudaGetLastError(), "dot_partial_kernel launch");

    int current_n = blocks;
    double* in = ws.partial_a;
    double* out = ws.partial_b;
    while (current_n > 1) {
        const int reduce_blocks = (current_n + threads - 1) / threads;
        reduce_sum_kernel<<<reduce_blocks, threads, static_cast<size_t>(threads) * sizeof(double)>>>(in, out, current_n);
        check_cuda(cudaGetLastError(), "reduce_sum_kernel launch");
        current_n = reduce_blocks;
        std::swap(in, out);
    }

    double sum = 0.0;
    check_cuda(cudaMemcpy(&sum, in, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(dot sum)");
    return sum;
}

void validate_system_shape(const PressureSystem& system) {
    const size_t count = static_cast<size_t>(system.nx) * static_cast<size_t>(system.ny);
    if (system.nx <= 0 || system.ny <= 0) {
        fail_schema("pressure system dimensions must be positive.");
    }
    if (system.diag.size() != count || system.west.size() != count || system.east.size() != count ||
        system.south.size() != count || system.north.size() != count || system.rhs.size() != count) {
        fail_schema("pressure system arrays must match nx * ny.");
    }
}

}  // namespace

bool gpu_pressure_enabled() {
    int device_count = 0;
    const cudaError_t rc = cudaGetDeviceCount(&device_count);
    if (rc != cudaSuccess) {
        return false;
    }
    return device_count > 0;
}

PressureSolveResult solve_pressure_cg_jacobi_gpu(
    const PressureSystem& system,
    const std::vector<double>& initial_guess,
    double relative_tolerance,
    int max_iterations) {
    validate_system_shape(system);
    const size_t count = static_cast<size_t>(system.nx) * static_cast<size_t>(system.ny);
    if (initial_guess.size() != count) {
        fail_schema("pressure solver initial guess must match system size.");
    }
    if (!(relative_tolerance > 0.0) || max_iterations <= 0) {
        fail_schema("pressure solver tolerance and iteration budget must be positive.");
    }

    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(system.diag[i]) || system.diag[i] <= 0.0) {
            fail_schema("pressure solver requires positive finite diagonal entries.");
        }
    }

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        fail_schema("GPU pressure backend requested, but no CUDA device is available.");
    }

    PressureGpuWorkspace& ws = pressure_workspace();
    ws.ensure_capacity(system.nx, system.ny);

    check_cuda(cudaMemcpy(ws.diag, system.diag.data(), count * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(diag)");
    check_cuda(cudaMemcpy(ws.west, system.west.data(), count * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(west)");
    check_cuda(cudaMemcpy(ws.east, system.east.data(), count * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(east)");
    check_cuda(cudaMemcpy(ws.south, system.south.data(), count * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(south)");
    check_cuda(cudaMemcpy(ws.north, system.north.data(), count * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(north)");
    check_cuda(cudaMemcpy(ws.rhs, system.rhs.data(), count * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(rhs)");
    check_cuda(cudaMemcpy(ws.x, initial_guess.data(), count * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy(x)");

    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + static_cast<size_t>(threads) - 1U) / static_cast<size_t>(threads));

    apply_system_kernel<<<blocks, threads>>>(system.nx, system.ny, ws.diag, ws.west, ws.east, ws.south, ws.north, ws.x, ws.ap);
    check_cuda(cudaGetLastError(), "apply_system_kernel(initial) launch");
    residual_kernel<<<blocks, threads>>>(ws.rhs, ws.ap, ws.r, static_cast<int>(count));
    check_cuda(cudaGetLastError(), "residual_kernel(initial) launch");

    const double rhs_norm = std::sqrt(dot_device(ws.rhs, ws.rhs, count, ws));
    const double denom = std::max(rhs_norm, 1.0);
    double relative_residual = std::sqrt(dot_device(ws.r, ws.r, count, ws)) / denom;

    if (relative_residual <= relative_tolerance) {
        std::vector<double> x_out(count, 0.0);
        check_cuda(cudaMemcpy(x_out.data(), ws.x, count * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(x out)");
        return PressureSolveResult{std::move(x_out), 0, relative_residual};
    }

    jacobi_precond_kernel<<<blocks, threads>>>(ws.r, ws.diag, ws.z, static_cast<int>(count));
    check_cuda(cudaGetLastError(), "jacobi_precond_kernel(initial) launch");
    copy_kernel<<<blocks, threads>>>(ws.z, ws.p, static_cast<int>(count));
    check_cuda(cudaGetLastError(), "copy_kernel(initial p) launch");

    double rz_old = dot_device(ws.r, ws.z, count, ws);
    if (!(rz_old > 0.0) || !std::isfinite(rz_old)) {
        fail_schema("GPU pressure solver encountered non-positive preconditioned residual.");
    }

    int iters = 0;
    for (int iter = 0; iter < max_iterations; ++iter) {
        apply_system_kernel<<<blocks, threads>>>(system.nx, system.ny, ws.diag, ws.west, ws.east, ws.south, ws.north, ws.p, ws.ap);
        check_cuda(cudaGetLastError(), "apply_system_kernel(iter) launch");

        const double p_ap = dot_device(ws.p, ws.ap, count, ws);
        if (!(p_ap > 0.0) || !std::isfinite(p_ap)) {
            fail_schema("GPU pressure solver encountered a non-SPD search direction.");
        }

        const double alpha = rz_old / p_ap;
        axpy_kernel<<<blocks, threads>>>(ws.x, ws.p, alpha, static_cast<int>(count));
        check_cuda(cudaGetLastError(), "axpy_kernel(x update) launch");
        axpy_kernel<<<blocks, threads>>>(ws.r, ws.ap, -alpha, static_cast<int>(count));
        check_cuda(cudaGetLastError(), "axpy_kernel(r update) launch");

        relative_residual = std::sqrt(dot_device(ws.r, ws.r, count, ws)) / denom;
        iters = iter + 1;
        if (relative_residual <= relative_tolerance) {
            break;
        }

        jacobi_precond_kernel<<<blocks, threads>>>(ws.r, ws.diag, ws.z, static_cast<int>(count));
        check_cuda(cudaGetLastError(), "jacobi_precond_kernel(iter) launch");
        const double rz_new = dot_device(ws.r, ws.z, count, ws);
        if (!(rz_new >= 0.0) || !std::isfinite(rz_new)) {
            fail_schema("GPU pressure solver encountered invalid preconditioned residual.");
        }

        const double beta = rz_new / rz_old;
        update_search_dir_kernel<<<blocks, threads>>>(ws.p, ws.z, beta, static_cast<int>(count));
        check_cuda(cudaGetLastError(), "update_search_dir_kernel launch");
        rz_old = rz_new;
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(pressure solve)");

    std::vector<double> x_out(count, 0.0);
    check_cuda(cudaMemcpy(x_out.data(), ws.x, count * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(x out)");
    return PressureSolveResult{std::move(x_out), iters, relative_residual};
}

#endif
