# 13 Runbook Commands (Spec-Only)

This document defines command contracts. It does not imply implementation already exists.

## Global Reproducibility Policy

- Auto run ID format: `YYYYMMDD_HHMMSS_mmm__<model>__<backend>__n<steps>__oe<k>[__<tag>]`.
- Auto-managed outputs must be saved under `outputs/<purpose>/<run_id>/`.
- Purpose buckets are:
  - `adhoc` for manual/exploratory runs,
  - `benchmark` for performance/parity matrices,
  - `ml-data` for runs intended to feed surrogate dataset generation,
  - `history` for history-run replay and mismatch artifacts.

## Command Contracts

### Simulator Run

```bash
sim_run --case <path> --backend cpu|gpu --steps <N> --output-every <k> --out <dir>
tools/model_run.sh --model-dir <dir> --backend cpu|gpu --steps <N> --output-every <k> --tag <label> --out auto|<dir> [--case-file <path>]
```

Expected behavior:
- Parse case config.
- Accept optional top-level `nz` (default `1`) for 3D runs.
- Run until `min(N, schedule_end_step)`.
- Persist schema-defined arrays and metadata.
- Emit `timing.csv`.
- If `--out auto`, place the run under the selected purpose bucket.
- Purpose bucket is selected by caller contract (`workflow run`=adhoc, `workflow ml-data-gen`=ml-data, `workflow bench`=benchmark).
- If `--case-file` is provided, that case YAML is used for the run instead of `<model-dir>/model.yaml`.
- On error, exit with stable code and emit one-line JSON on `stderr`:
  - `2` (`E_ARG_MISSING`), `3` (`E_ARG_INVALID`), `4` (`E_CASE_PARSE`), `5` (`E_CASE_SCHEMA`), `6` (`E_IO`).

### History Run

```bash
sim_history_run --case <path> --backend cpu|gpu --steps <N> --output-every <k> --out <dir>
tools/history_run.sh --model-dir <dir> --backend cpu|gpu --steps <N> --output-every <k> --tag <label> --out auto|<dir> [--case-file <path>]
./workflow history-run --model <modelN> [--backend cpu|gpu] [--steps <N>] [--output-every <k>] [--tag <label>] [--out auto|<dir>] [--case-file <path>]
```

Expected behavior:
- Parse case config and require `history.controls_csv` and `history.observations_csv`.
- Replay time-varying well controls over the declared history window.
- Align simulator outputs with observation rows according to the locked compare policy.
- Persist standard run artifacts plus mismatch artifacts under `outputs/history/<run_id>/` when `--out auto` is used.
- Emit `history_match.csv`, `history_mismatch.json`, and `well_observed_vs_simulated.csv`.
- Purpose bucket is selected by caller contract (`workflow history-run` = `history`).
- On error, use the same stable exit-code family as `sim_run` unless a new code is explicitly locked later.

### ML Data Generation

```bash
tools/ml_data_generate.sh --model-dir <dir> --plan <scenario_csv> --steps <N> --output-every <k>
./workflow ml-data-gen --model <modelN> --plan cases/<modelN>/ml_scenarios.csv --steps <N>
```

Expected behavior:
- Read base `model.yaml`.
- Generate temporary variant YAMLs from scenario CSV overrides.
- Run each scenario into `outputs/ml-data/<run_id>/`.
- Persist the exact case used for that run as `case_input.yaml` in the run directory.
- Remove temporary generated YAML files when finished.

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

1. Run directory matches naming policy when `--out auto` is used.
2. Required files exist.
3. Metrics parse without schema errors.
4. For history runs, controls and observations resolve to existing files and mismatch artifacts are present.

## Web UI Contract (History Mode)

- `./webui` is the primary entrypoint for end users.
- The Web UI must expose `history-run` as a top-level runnable mode, not as a hidden CLI-only workflow.
- Minimum Web UI inputs for `history-run`:
  - `model`
  - `backend`
  - `steps`
  - `output_every`
  - optional `case_file`
  - read-only display of the resolved `history.controls_csv` and `history.observations_csv` from the selected model
- Minimum Web UI result summary for `history-run`:
  - output directory
  - aggregate mismatch/objective value
  - per-well mismatch table or link to exported artifact
