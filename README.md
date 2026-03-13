# Reservoir History Matching Mini-Stack

A browser-first reservoir simulation project focused on `history-run` replay, mismatch analysis, and ML-assisted candidate screening for history matching.

## What This Project Is

This repo implements a compact synthetic oil-water reservoir workflow with:
- an IMPES-based simulator,
- prescribed historical control replay,
- observed-vs-simulated mismatch artifacts,
- a Web UI as the primary user surface,
- an ML ranking workflow that scores candidate parameter sets before expensive simulator runs.

The current project is intentionally centered on `history mode`, not forecast mode.

## Why It Is Interesting

- It connects reservoir physics, numerical methods, and reproducible workflow design in one codebase.
- It frames ML around a realistic inverse problem: screening parameter candidates for history matching.
- It is structured as a small but coherent scientific-computing system rather than an isolated simulator demo.

## Current Scope

Implemented now:
- browser-first `history-run` workflow,
- `10x10x3` synthetic `model1` with one injector and one producer,
- history controls and observations,
- mismatch artifacts:
  - `history_match.csv`
  - `history_mismatch.json`
  - `well_observed_vs_simulated.csv`
- ML candidate data generation,
- ML train/evaluate/score flow for history-match ranking.

Not implemented yet:
- iterative parameter-optimization loop for full automatic `history-match`,
- prediction-mode scenario forecasting,
- final GPU optimization pass.

## Main Workflow

### 1. Launch the Web UI

```bash
./webui
```

Equivalent shortcut:

```bash
./workflow
```

This is the primary way to use the project.

### 2. Compile the simulator

```bash
./workflow compile --mode debug --cuda off
```

For an optimized CPU build:

```bash
./workflow compile --mode release --cuda off
```

### 3. Run history mode

```bash
./workflow history-run --model model1 --mode release --backend cpu --steps 200 --output-every 20
```

This writes outputs under:

```text
cases/model1/outputs/history/<run_id>/
```

### 4. Generate ML candidate data

```bash
./workflow ml-data-gen --model model1 --steps 200 --output-every 20
```

This runs a matrix of history-mode candidate cases from [ml_scenarios.csv](cases/model1/ml_scenarios.csv) and writes:

```text
cases/model1/outputs/ml-data/<run_id>/
cases/model1/outputs/ml-data/history_ml_dataset.csv
```

### 5. Train the history-match ML ranker

```bash
./workflow ml-train --model model1
```

Checkpoint output:

```text
cases/model1/outputs/ml-train/latest/history_match_checkpoint.npz
```

### 6. Evaluate the ranker

```bash
./workflow ml-eval --model model1 \
  --checkpoint cases/model1/outputs/ml-train/latest/history_match_checkpoint.npz
```

Evaluation output:

```text
cases/model1/outputs/ml-eval/latest/history_ml_eval.csv
```

### 7. Score candidate parameter sets

```bash
./workflow ml-score --model model1 \
  --checkpoint cases/model1/outputs/ml-train/latest/history_match_checkpoint.npz
```

Ranking output:

```text
cases/model1/outputs/ml-score/latest/candidate_scores.csv
```

## Project Structure

```text
.
├── core-cpp/      C++/CUDA simulator and tests
├── python/        ML and visualization scripts
├── tools/         workflow scripts, Web UI, MCP server
├── cases/model1/  active history-mode case
├── docs/          project specs and runbook
└── benchmarks/    benchmark summaries
```

Key files:
- [workflow](/mnt/c/Users/hares/Documents/Natarajan/reserv_ML/workflow)
- [cases/model1/model.yaml](/mnt/c/Users/hares/Documents/Natarajan/reserv_ML/cases/model1/model.yaml)
- [cases/model1/history_controls.csv](/mnt/c/Users/hares/Documents/Natarajan/reserv_ML/cases/model1/history_controls.csv)
- [cases/model1/history_observations.csv](/mnt/c/Users/hares/Documents/Natarajan/reserv_ML/cases/model1/history_observations.csv)
- [cases/model1/ml_scenarios.csv](/mnt/c/Users/hares/Documents/Natarajan/reserv_ML/cases/model1/ml_scenarios.csv)
- [cases/model1/history_ml_config.yaml](/mnt/c/Users/hares/Documents/Natarajan/reserv_ML/cases/model1/history_ml_config.yaml)

## Visualization

Generate figures for a history run:

```bash
tools/plot_run.sh --run cases/model1/outputs/history/<run_id> --out figs
```

Validate artifact structure only:

```bash
tools/plot_run.sh --run cases/model1/outputs/history/<run_id> --check-only
```

## Environment Notes

- `cmake` is required.
- `--cuda on` requires `nvcc`.
- Python ML and plotting scripts work best from the project venv.

Minimal setup:

```bash
python3 -m venv .venv
./.venv/bin/pip install numpy matplotlib
```

## Technical Positioning

This project is best described as:
- a history-mode reservoir simulation workflow,
- with mismatch reporting,
- plus ML-assisted candidate ranking for future history matching.

It is not yet a full automatic history-matching system, and it is not currently focused on forecast-mode surrogate simulation.

## Docs

Start here:

1. [Project Charter](docs/00_project_charter.md)
2. [Problem Statement](docs/01_problem_statement.md)
3. [Numerical Methods](docs/03_numerical_methods.md)
4. [Software Architecture](docs/04_software_architecture.md)
5. [History-Match ML Plan](docs/06_surrogate_ml_plan.md)
6. [Runbook Commands](docs/13_runbook_commands.md)
7. [Artifact Specification](docs/14_artifact_spec.md)

## MCP Server

The repo also exposes workflow actions through an MCP server:

```bash
./.venv/bin/pip install mcp
python3 tools/mcp_server.py
```

Available MCP tools:
- `compile_code`
- `run_model`
- `plot_run`
- `clean_outputs`
- `all_in_one`
