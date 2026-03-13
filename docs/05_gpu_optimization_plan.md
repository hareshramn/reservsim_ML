# 05 GPU Optimization Plan

## Optimization Objective

Show defensible speedup of GPU backend versus CPU baseline while preserving numerical parity.

## Execution Sequencing

- This plan is intentionally scheduled after CPU correctness, artifact reproducibility, and history-match ML baseline evidence are stable.
- GPU implementation may exist earlier for parity and interface validation, but optimization/profiling work is deferred to the final execution phase.
- Optimization decisions must not force interface or artifact schema changes already locked in `docs/04_software_architecture.md` and `docs/14_artifact_spec.md`.

## Kernel Candidates (Priority Order)

1. Property update kernel:
   - mobility, fractional flow, relperm from saturation.
2. Face flux kernel:
   - transmissibility and flux per face.
3. Saturation update kernel:
   - divergence accumulation and explicit transport update.
4. Well/source application kernel.
5. Pressure solve support kernels (SpMV, vector ops, residual updates).

## Optimization Checklist

- Ensure contiguous layout and coalesced global memory accesses.
- Reduce host-device transfers:
  - keep state resident on GPU across timesteps.
- Tune block/grid sizes by occupancy and memory bandwidth behavior.
- Use shared memory only where reuse is clear and bank conflicts are managed.
- Minimize branch divergence in upwind logic (predication where practical).

## Profiling and Diagnostics

Tools:
- Nsight Systems for timeline and transfer overhead.
- Nsight Compute for kernel metrics.

Required metrics:
- kernel runtime distribution,
- achieved occupancy,
- memory throughput,
- warp execution efficiency,
- H2D/D2H transfer time share.

## Benchmark Matrix

| Case | Grid | Steps | Warmup | Repeats |
|---|---:|---:|---:|---:|
| sanity | 16x16 | 500 | 1 | 5 |
| default | 64x64 | 1000 | 1 | 5 |
| stress | 128x128 | 1000 | 1 | 3 |

## Acceptance Criteria

1. GPU speedup > 1x on default and stress cases.
2. Profiling evidence stored with run artifacts.
3. CPU/GPU parity metrics remain within tolerances.
4. Optimization work does not regress CPU baseline reproducibility or history-match ML data-generation contracts.
