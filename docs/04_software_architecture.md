# 04 Software Architecture

## Module Boundaries

- `core-cpp/`: C++17 + CUDA simulation engine.
- `python/`: history-match ML training, evaluation, candidate scoring, visualization, report helpers.
- `cases/`: YAML case configurations.
- `outputs/`: generated run artifacts, organized by purpose bucket (`benchmark`, `ml-data`, `history`) when auto-managed by workflow tooling.
- `docs/`: specs and governance.

## Data Flow

1. Case YAML -> simulator config object.
2. Optional history control/observation tables -> replay schedule + comparison targets.
3. Simulator run -> checkpoint tensors + metrics under `outputs/<purpose>/<run_id>/`.
4. `outputs/ml-data/<run_id>/` history-run artifacts -> tabular ML dataset.
5. Trained ML ranker -> evaluation and candidate scoring.
6. Evaluation outputs -> ranking tables and visuals.

## Locked CLI Contracts

## Primary Interaction Model

- Primary user interaction path is the browser-based Web UI launched via `./webui`.
- CLI contracts remain stable and are retained for advanced/manual runs, automation, and debugging.
- New end-user workflows should be surfaced in the Web UI first unless they are strictly low-level developer operations.

### Simulator

```bash
sim_run --case <path> --backend cpu|gpu --steps <N> --out <dir>
```

### History Run

```bash
sim_history_run --case <path> --backend cpu|gpu --steps <N> --out <dir>
```

### Simulator Error Codes (Locked v1)

`sim_run` exits with stable machine-readable codes:
- `0`: success.
- `2` (`E_ARG_MISSING`): required CLI flag missing.
- `3` (`E_ARG_INVALID`): invalid CLI flag value or enum.
- `4` (`E_CASE_PARSE`): YAML parse failure.
- `5` (`E_CASE_SCHEMA`): case parsed but violates schema/range rules.
- `6` (`E_IO`): output directory or file write failure.

Error output contract:
- Print one single-line JSON object to `stderr` with fields:
  - `code` (integer exit code),
  - `symbol` (stable string token),
  - `message` (human-readable explanation).

### History-Match ML Training

```bash
python python/ml/train_history_matcher.py --data <dir> --config <yaml> --seed <int> --out <dir>
```

### History-Match ML Evaluation

```bash
python python/ml/eval_history_matcher.py --checkpoint <ckpt> --data <dir> --out <dir>
```

### Candidate Scoring

```bash
python python/ml/score_history_match.py --checkpoint <ckpt> --candidates <csv> --out <dir>
```

## Case Schema Contract (3D Grid v1)

Case YAML keeps existing required keys and supports 3D grids via:

- `nz` (integer, optional at top-level):
  - defaults to `1` when omitted.
  - when `nz > 1`, simulator runs a 3D Cartesian grid.

- Rock properties remain scalar in v1 (`rock.porosity`, `rock.permeability_md`) and are applied uniformly across all cells.

## History-Mode Data Contract

Optional `history` section in case YAML:
- `controls_csv` (string path, required for history mode)
- `observations_csv` (string path, required for history mode)
- `start_day` (float, required for history mode)
- `end_day` (float, required for history mode)
- `match_frequency_days` (float, optional, default determined by observations cadence)

Controls CSV contract:
- required columns: `day`, `well`, `control_kind`, `target_value`
- optional column: `phase`
- rows are ordered by nondecreasing `day` per well

Observations CSV contract:
- required columns: `day`, `well`, `observable`, `value`
- optional columns: `weight`, `source`
- duplicate `(day, well, observable)` rows are invalid unless explicitly aggregated before ingestion

## Output Schema Contract

Each run writes:
- `meta.json`
- `state_pressure.npy` shaped:
  - `[T, ny, nx]` for `nz=1`,
  - `[T, nz, ny, nx]` for `nz>1`.
- `state_sw.npy` shaped:
  - `[T, ny, nx]` for `nz=1`,
  - `[T, nz, ny, nx]` for `nz>1`.
- `well_rates.npy` shaped `[T, nwells]`
- `well_bhp.npy` shaped `[T, nwells]`
- `timing.csv` with both per-step and aggregate runtime metrics

History-run outputs add:
- `history_match.csv`
- `history_mismatch.json`
- `well_observed_vs_simulated.csv`

`meta.json` minimum fields:
- `case_name`, `nx`, `ny`, `nz`, `backend`, `dt_policy`, `units`, `version`.

For history runs, `meta.json` additionally records:
- `run_kind` = `history`
- `history_controls_csv`
- `history_observations_csv`
- `history_start_day`
- `history_end_day`


## Interface Stability Policy

- Any change to CLI flags or schema must update:
  - this file,
  - `docs/13_runbook_commands.md`,
  - `docs/14_artifact_spec.md`.

## C++ Simulator Incremental Build Plan (Post-Gate)

This section is planning-only and becomes executable after the pre-implementation gate in `docs/11_definition_of_done.md` is approved.

### Slice 0: CLI Shell and Config Validation

- Unit boundary:
  - `main.cpp` argument handling for `sim_run --case --backend --steps --out`.
  - Config validation with fail-fast errors and no physics compute.
- Tests:
  - valid minimal case returns success and writes `meta.json` skeleton.
  - missing required flags return non-zero exit code.
  - invalid backend enum is rejected.
- Exit criteria:
  - locked CLI contract from this document and `docs/13_runbook_commands.md` is enforced exactly.

### Slice 1: Case Parsing and Domain Model

- Unit boundary:
  - parse YAML into strongly typed `SimulationConfig`.
  - normalize units and defaults defined in case spec docs.
- Tests:
  - parser fixture tests for required fields and default filling.
  - schema rejection tests for malformed YAML and illegal ranges.
- Exit criteria:
  - parser output is deterministic and serializable into `meta.json` fields.

### Slice 2: Grid/Rock/Fluid State Initialization

- Unit boundary:
  - allocate pressure and water saturation arrays with `[nx, ny]`.
  - initialize static rock/fluid property fields.
- Tests:
  - array shape checks and boundary indexing checks.
  - initialization invariant tests (`Sw` bounds, pressure positivity where required).
- Exit criteria:
  - initial state passes invariant suite and is ready for first timestep.

### Slice 3: Pressure Operator Assembly

- Unit boundary:
  - transmissibility/flow coefficient assembly for pressure solve.
  - well source-term injection into linear system RHS.
- Tests:
  - coefficient symmetry and sign-structure checks.
  - conservation sanity test on a no-well closed box case.
- Exit criteria:
  - matrix assembly matches the discretization policy in `docs/03_numerical_methods.md`.

### Slice 4: Pressure Solve Step

- Unit boundary:
  - linear solver wrapper and pressure update for one timestep.
- Tests:
  - residual norm threshold checks.
  - regression fixture comparing against a locked tiny-case reference.
- Exit criteria:
  - pressure solver convergence policy is deterministic under fixed inputs.

### Slice 5: Saturation Transport Step

- Unit boundary:
  - explicit/IMPES saturation update using computed fluxes.
  - clipping or limiter policy as documented in numerics.
- Tests:
  - `Sw` bound-preservation checks.
  - mass-balance drift check within configured tolerance.
- Exit criteria:
  - transport update satisfies stability and boundedness criteria.

### Slice 6: Well Model Coupling

- Unit boundary:
  - rate/BHP calculations and per-well controls for each timestep.
- Tests:
  - producer/injector sign conventions.
  - control switching logic under edge conditions.
- Exit criteria:
  - `well_rates` and `well_bhp` align with output schema and conventions.

### Slice 7: Time Loop and Scheduler

- Unit boundary:
  - full `for step in N` orchestration, dt policy hook, and checkpoint cadence.
- Tests:
  - deterministic replay with identical configuration and backend.
  - early-stop behavior at schedule end.
- Exit criteria:
  - stable loop semantics independent of backend setting.

### Slice 8: Artifact Writer and Timing

- Unit boundary:
  - write `state_pressure.npy`, `state_sw.npy`, `well_rates.npy`, `well_bhp.npy`, `meta.json`, `timing.csv`.
- Tests:
  - file presence and shape schema checks.
  - timing CSV column schema checks.
- Exit criteria:
  - output contract matches this file and `docs/14_artifact_spec.md`.

### Slice 9: CPU/GPU Parity Harness

- Unit boundary:
  - shared reference comparison utility for CPU vs GPU trajectories.
- Tests:
  - max-abs/max-rel thresholds per field over selected timesteps.
  - parity report generation for validation docs.
- Exit criteria:
  - parity thresholds required by `docs/07_validation_and_benchmarking.md` are met.

## Assumptions (For Planning)

- Main simulator entrypoint remains `sim_run` with currently locked flags.
- History-mode replay is the primary end-user workflow for model execution.
- State arrays remain dense 2D fields indexed as `[nx, ny]` in docs-facing contracts.
- First implementation target is correctness and reproducibility on CPU, then history-match ML dataset/training flow, then CPU/GPU parity, with GPU optimization deferred to the final performance pass.

## Resolved v1 Decisions

- `--steps <N>` is an upper bound; run stops at `min(N, schedule_end_step)`.
- `timing.csv` must include both per-step rows and an aggregate summary row.
- CPU baseline pressure solver is fixed to `CG + Jacobi` for reproducibility.
- Parser/config failures use stable machine-readable exit codes and single-line JSON stderr output.
- New history-mode workflows must be exposed in the Web UI as first-class modes once implemented.

## Open Questions (Implementation Planning)

None.
