# 13 Runbook Commands (Spec-Only)

This document defines command contracts. It does not imply implementation already exists.

## Global Reproducibility Policy

- Seed is mandatory for every run.
- Auto run ID format: `YYYYMMDD_HHMMSS_mmm__<model>__<backend>__s<seed>__n<steps>__oe<k>[__<tag>]`.
- Auto-managed outputs must be saved under `outputs/<purpose>/<run_id>/`.
- Purpose buckets are:
  - `adhoc` for manual/exploratory runs,
  - `benchmark` for performance/parity matrices,
  - `ml-data` for runs intended to feed surrogate dataset generation.

## Command Contracts

### Simulator Run

```bash
sim_run --case <path> --backend cpu|gpu --steps <N> --output-every <k> --seed <int> --out <dir>
tools/model_run.sh --model-dir <dir> --backend cpu|gpu --steps <N> --output-every <k> --seed <int> --purpose adhoc|benchmark|ml-data --tag <label> --out auto|<dir>
```

Expected behavior:
- Parse case config.
- Run until `min(N, schedule_end_step)`.
- Persist schema-defined arrays and metadata.
- Emit `timing.csv`.
- If `--out auto`, place the run under the selected purpose bucket.
- On error, exit with stable code and emit one-line JSON on `stderr`:
  - `2` (`E_ARG_MISSING`), `3` (`E_ARG_INVALID`), `4` (`E_CASE_PARSE`), `5` (`E_CASE_SCHEMA`), `6` (`E_IO`).

### Batch Benchmark

```bash
sim_bench --cases <list> --backends cpu,gpu --repeats <N> --out <dir>
```

Expected behavior:
- Execute matrix across scenarios/backends.
- Aggregate into benchmark summary CSV.

### Surrogate Training

```bash
python python/ml/train_surrogate.py --data <outputs_dir> --config <yaml> --seed <int> --out <dir>
```

Expected behavior:
- Build datasets per split policy.
- Train and save checkpoint(s) + train log.

### Surrogate Evaluation

```bash
python python/ml/eval_surrogate.py --checkpoint <ckpt> --case <path> --horizons 20,50,100 --out <dir>
```

Expected behavior:
- Run one-step and rollout metrics.
- Save metrics CSV and comparison plots.

### Visualization Build

```bash
tools/plot_run.sh --run <run_id_or_run_dir> --out <dir>
tools/plot_run.sh --run <run_id_or_run_dir> --check-only
python python/viz/make_figures.py --run <run_id_or_run_dir> --out <dir>
python python/viz/make_animation.py --run <run_id> --field pressure|sw --fps 12 --out <dir>
```

Expected behavior:
- Generate required figures and MP4 with naming conventions.
- `--check-only` must fail clearly when required output files are missing or shape contracts are violated.

## Minimum Output Folder Contract

For each run directory:
- `meta.json`
- `state_pressure.npy`
- `state_sw.npy`
- `well_rates.npy`
- `well_bhp.npy`
- `timing.csv`
- `logs.txt`

## Checklist Before Accepting Any Run

1. Seed present in metadata.
2. Run directory matches naming policy when `--out auto` is used.
3. Required files exist.
4. Metrics parse without schema errors.
