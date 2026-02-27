# 07 Validation and Benchmarking

## Validation Categories

1. Numerical correctness.
2. CPU/GPU parity.
3. Performance speedup.
4. Surrogate accuracy and throughput.

## Numerical Validation

- Global mass-balance error per step and cumulative:
  - threshold target: `< 1e-4` relative (configurable).
- Saturation physical bounds:
  - no persistent out-of-range values.
- Well trend sanity:
  - expected water-cut evolution shape.

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

