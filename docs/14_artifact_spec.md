# 14 Artifact Specification

## Portfolio Artifact Set

1. Benchmark summary CSV.
2. Numerical validation CSV.
3. Surrogate evaluation CSV.
4. Required PNG figures.
5. MP4 animations.
6. Technical report (`report.pdf`).

## CSV Schemas

### `benchmark_summary.csv`

Columns:
- `run_id`
- `scenario`
- `backend`
- `grid_nx`
- `grid_ny`
- `steps`
- `total_time_s`
- `pressure_time_s`
- `transport_time_s`
- `transfer_time_s`
- `speedup_vs_cpu`
- `mass_err_rel`
- `l2_sw`
- `l2_p`

### `timing.csv`

Columns:
- `run_id`
- `row_type` (`step` or `aggregate`)
- `step_idx` (`0..T-1` for step rows, `-1` for aggregate row)
- `dt_days`
- `pressure_time_s`
- `transport_time_s`
- `io_time_s`
- `total_time_s`

### `surrogate_eval.csv`

Columns:
- `run_id`
- `model_name`
- `scenario`
- `horizon`
- `rmse_sw`
- `rmse_p`
- `mae_sw`
- `mae_p`
- `mass_penalty`
- `infer_steps_per_s`

## Figure Naming Rules

- Prefix all figures with `fig_`.
- Format: `fig_<nn>_<topic>_<scenario>.png`.
- Topic examples:
  - `pressure_snapshot_t050`
  - `sw_snapshot_t050`
  - `watercut_curve`
  - `speedup_bar`

## MP4 Naming Rules

- Format:
  - `anim_<scenario>_<backend>_pressure.mp4`
  - `anim_<scenario>_<backend>_sw.mp4`
  - `anim_<scenario>_sim_vs_surrogate.mp4`

## Report Structure

1. Executive summary.
2. Problem and assumptions.
3. Physics and numerics.
4. Software and GPU optimization.
5. Surrogate method.
6. Validation and benchmark results.
7. Limits and future work.
8. Reproducibility appendix (commands + seeds).

## Submission Bundle Checklist

- `README.md`
- `docs/` complete
- `benchmark_summary.csv`
- `surrogate_eval.csv`
- required PNG figures
- at least two MP4 files
- final `report.pdf`
