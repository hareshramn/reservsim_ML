# 10 Risks and Mitigations

## Risk 1: Solver Instability

- Symptom:
  - oscillatory saturation, repeated clipping, high mass residual.
- Early warning:
  - CFL violations,
  - retry count increasing.
- Mitigation:
  - adaptive dt backoff,
  - safer upwind handling,
  - tighter convergence guards.
- Scope fallback:
  - reduce max horizon per run,
  - keep default grid at `64x64`.

## Risk 2: CUDA Performance Below Target

- Symptom:
  - GPU speedup near 1x or worse.
- Early warning:
  - transfer overhead dominates timeline.
- Mitigation:
  - defer deep profiling until CPU outputs and surrogate data contracts are frozen,
  - keep state on device,
  - kernel fusion where valid,
  - launch parameter tuning by occupancy data.
- Scope fallback:
  - ship with documented GPU parity only and treat optimization as the final stretch goal.

## Risk 3: Surrogate Rollout Drift

- Symptom:
  - long-horizon error blow-up.
- Early warning:
  - gap between one-step and rollout metrics.
- Mitigation:
  - increase physics/mass penalties,
  - scheduled sampling style training,
  - scenario-balanced data generation.
- Scope fallback:
  - position surrogate as short-horizon accelerator.

## Risk 4: Schedule Compression

- Symptom:
  - documentation or implementation backlog by Day 5.
- Mitigation:
  - freeze non-essential extensions,
  - use predefined acceptance thresholds,
  - narrow to default scenario first.
- Scope fallback:
  - preserve CPU correctness, surrogate evidence, and reproducible artifacts first; move GPU optimization to the end of the schedule.
