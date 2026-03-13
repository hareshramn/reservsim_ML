# 14 Artifact Specification

## Project Artifact Set

1. Benchmark summary CSV.
2. Numerical validation CSV.
3. History-match / mismatch CSV and JSON artifacts.
4. Surrogate evaluation CSV.
5. Required PNG figures.
6. MP4 animations.
7. Technical report (`report.pdf`).

## Output Layout

- Auto-managed simulation runs live under `cases/<model>/outputs/<purpose>/<run_id>/`.
- Purpose buckets:
  - `benchmark`
  - `ml-data`
  - `history`

State array shape contract:
- `state_pressure.npy` and `state_sw.npy` use:
  - `[T, ny, nx]` for 2D (`nz=1`),
  - `[T, nz, ny, nx]` for 3D (`nz>1`).
- `meta.json` must include `nz`.

## CSV Schemas

### `benchmarks/benchmark_summary.csv`

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

### `history_match.csv`

Columns:
- `run_id`
- `day`
- `well`
- `observable`
- `observed_value`
- `simulated_value`
- `abs_error`
- `squared_error`
- `weight`
- `weighted_error`

### `well_observed_vs_simulated.csv`

Columns:
- `run_id`
- `well`
- `observable`
- `rmse`
- `mae`
- `weighted_misfit`

### `history_mismatch.json`

Required top-level fields:
- `run_id`
- `objective_name`
- `objective_value`
- `compare_count`
- `per_well`
- `per_observable`

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
4. Software and CPU baseline implementation.
5. Surrogate method and evaluation.
6. CPU/GPU parity and GPU optimization.
7. Validation and benchmark results.
8. Limits and future work.
9. Reproducibility appendix (commands + seeds).

## Submission Bundle Checklist

- `README.md`
- `docs/` complete
- `benchmarks/benchmark_summary.csv`
- `history_match.csv` or equivalent exported mismatch artifact for the main history run
- `surrogate_eval.csv`
- required PNG figures
- at least two MP4 files
- final `report.pdf`
