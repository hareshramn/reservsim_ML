# Reservoir History-Mode Simulation with Web UI + AI

Replay two-phase reservoir behavior under historical controls, validate the simulator response, and prepare for ML-assisted history matching through a browser-first workflow.

## Goal

Build a SPE1-inspired synthetic two-phase (oil-water) reservoir workflow that shows:
- numerically credible IMPES simulation,
- history-run style replay under prescribed controls,
- a path toward ML-assisted history matching,
- browser-first reproducible operation with advanced CLI fallback.

## At a Glance

- Models how water and oil move in a synthetic reservoir.
- Focuses first on history-mode style replay and validation.
- Keeps GPU optimization as a later-stage performance pass.
- Uses an AI surrogate plan aimed at accelerating history matching rather than forecast-first rollout.
- Produces reproducible artifacts for transparent validation.

## Why It Matters

- Reservoir physics and numerical methods (Darcy flow + saturation transport).
- History-run and history-matching workflow design.
- ML-assisted calibration strategy linked to physical constraints.
- Reproducible engineering workflow suitable for technical evaluation.

## Keywords

- Reservoir Simulation
- CUDA
- GPU Acceleration
- High-Performance Computing (HPC)
- Numerical Methods
- IMPES
- Finite Volume / TPFA
- Physics-Informed ML
- History Matching
- Surrogate Modeling
- Scientific Computing
- Performance Optimization
- Profiling (Nsight)
- Parallel Computing
- C++ and Python
- Reproducible Benchmarks

## Repo Map

```text
.
├── AGENTS.md
├── README.md
├── Project_deets
├── docs/
│   ├── 00_project_charter.md
│   ├── 01_problem_statement.md
│   ├── 02_physics_math.md
│   ├── 03_numerical_methods.md
│   ├── 04_software_architecture.md
│   ├── 05_gpu_optimization_plan.md
│   ├── 06_surrogate_ml_plan.md
│   ├── 07_validation_and_benchmarking.md
│   ├── 08_visualization_and_reporting.md
│   ├── 09_execution_timeline_2weeks.md
│   ├── 10_risks_and_mitigations.md
│   ├── 11_definition_of_done.md
│   ├── 12_agent_task_board.md
│   ├── 13_runbook_commands.md
│   └── 14_artifact_spec.md
├── core-cpp/      (C++ simulation core, builds, tests)
├── python/        (ML and visualization scripts)
├── cases/         (model configs and scenario inputs)
├── tools/         (workflow scripts, MCP server, web UI)
├── benchmarks/    (benchmark artifacts and summaries)
└── outputs/       (generated run artifacts)
```

## Start Here

1. [Project Charter](docs/00_project_charter.md)
2. [Problem Statement](docs/01_problem_statement.md)
3. [Physics and Math](docs/02_physics_math.md)
4. [Numerical Methods](docs/03_numerical_methods.md)
5. [Software Architecture](docs/04_software_architecture.md)
6. [GPU Optimization Plan](docs/05_gpu_optimization_plan.md)
7. [Surrogate ML Plan](docs/06_surrogate_ml_plan.md)
8. [Validation and Benchmarking](docs/07_validation_and_benchmarking.md)
9. [Visualization and Reporting](docs/08_visualization_and_reporting.md)
10. [2-Week Timeline](docs/09_execution_timeline_2weeks.md)

## Final Expected Outputs

- history-run validation artifacts and mismatch summaries.
- `benchmarks/benchmark_summary.csv` with CPU/GPU performance and parity metrics.
- Pressure/saturation plots and time-series charts.
- MP4 animations of field evolution.
- Technical report with physics, numerics, history-mode workflow, and ML findings.

## Workflow Paths

Primary user entrypoint (recommended for most users):

```bash
./webui
```

Equivalent shortcut:

```bash
./workflow
```

Advanced and manual workflows:

```bash
./workflow --help
```

The browser UI is the default product surface. Use the CLI for manual runs, debugging, contract checks, and advanced argument control.

Current implementation note:
- The repository already supports forward simulation, validation, and ML utilities.
- Full history-run and history-match features are the active direction and are not fully implemented yet.

Manual path (advanced examples from repo root):

```bash
./workflow compile --mode debug --cuda off
./workflow run --model model1 --steps 10 --mode release
./workflow plot --model model1
./workflow clean --model model1 --keep 3 --apply
./workflow all --model model1 --steps 10
```

Model convention:
- each model lives in its own folder,
- required file name: `model.yaml`,
- defaults live in `run.env`,
- run with local `./run`,
- outputs are written to `<model_dir>/outputs/<purpose>/<run_id>/`.

Examples:
- `cases/model1/model.yaml`

Notes:
- Requires `cmake`.
- `--cuda on` requires `nvcc`.
- Output binaries are placed under `core-cpp/build/<mode>-<cpu|cuda>/`.

## Plotting Workflow

Create project venv and install plotting dependencies:

```bash
python3 -m venv .venv
./.venv/bin/pip install numpy matplotlib
```

Generate figures from a run:

```bash
tools/plot_run.sh --run cases/model1/outputs/<run_id> --out figs
```

Validate run artifacts only (no figure generation):

```bash
tools/plot_run.sh --run cases/model1/outputs/<run_id> --check-only
```

## MCP Server (LLM Tooling)

This repo now includes a real MCP server that exposes compile/run/plot/clean tools:

```bash
.venv/bin/pip install mcp
python3 tools/mcp_server.py
```

Exposed MCP tools:
- `compile_code`
- `run_model`
- `plot_run`
- `clean_outputs`
- `all_in_one`

Example MCP-style intent:
- "Compile and run model1 for 10 steps, then plot" -> call `all_in_one(model=\"model1\", steps=10)`
