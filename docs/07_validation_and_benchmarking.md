# 07 Validation and Benchmarking

## Validation Categories

1. Numerical correctness.
2. History-run replay and mismatch quality.
3. Surrogate accuracy and throughput.
4. CPU/GPU parity.
5. Performance speedup.

## Validation Execution Order

1. Lock CPU numerical correctness and artifact schema.
2. Validate history-match ML training/evaluation outputs against the CPU baseline.
3. Run CPU/GPU parity checks once the baseline dataset and outputs are stable.
4. Treat GPU speedup benchmarking as the final optimization-stage validation pass.

## Numerical Validation

- Global mass-balance error per step and cumulative:
  - threshold target: `< 1e-4` relative (configurable).
- Saturation physical bounds:
  - no persistent out-of-range values.
- Well trend sanity:
  - expected water-cut evolution shape.

## History-Run Validation

- Control replay correctness:
  - active well constraints match the declared control schedule.
- Observation alignment:
  - simulated and observed rows align on declared compare times.
- Mismatch metrics:
  - weighted RMSE/MAE per observable and aggregate objective.
- Per-well breakdown:
  - identify which well/observable dominates mismatch.

## CPU-GPU Parity Metrics

- `L2` norm difference for pressure and saturation fields.
- `L_inf` difference per checkpoint.
- Tolerance recorded per case in benchmark config.

## Performance Validation

Record:
- total wall-clock,
- pressure solve time,
- transport update time,
- transfer overhead time.

Publish:
- speedup table CPU vs GPU by scenario,
- variance across repeated runs.

Execution note:
- Performance speedup evidence is collected after correctness, history-match ML, and parity evidence are stable enough to avoid chasing profiling noise from moving baselines.

## Surrogate Validation

- One-step MAE/RMSE for `p` and `sw`.
- Rollout MAE at horizons 20/50/100.
- Trend comparison:
  - water-cut curve,
  - average reservoir pressure.
- Throughput:
  - predictions per second,
  - speedup relative to solver stepping.

## Reporting Table Template

| scenario | backend | total_time_s | speedup_vs_cpu | mass_err_rel | l2_sw | l2_p |
|---|---|---:|---:|---:|---:|---:|

| scenario | model | horizon | rmse_sw | rmse_p | mass_penalty | infer_steps_per_s |
|---|---|---:|---:|---:|---:|---:|

| run_id | well | observable | rmse | mae | weighted_misfit |
|---|---|---:|---:|---:|---:|
