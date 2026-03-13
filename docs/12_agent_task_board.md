# 12 Agent Task Board

## Roles

- Agent A: physics + numerics docs.
- Agent B: software architecture + interfaces.
- Agent C: GPU optimization specification.
- Agent D: history-match ML specification.
- Agent E: validation, visualization, artifact and runbook packaging.

## Task Cards

### DOC-01 (Agent A)
- Objective: finalize problem statement and assumptions.
- Inputs: `00_project_charter.md`, `Project_deets`.
- Outputs: `01_problem_statement.md`.
- Dependencies: none.
- Done criteria: schema + scenarios + assumptions complete.

### DOC-02 (Agent A)
- Objective: formalize governing equations and symbols.
- Inputs: `01_problem_statement.md`.
- Outputs: `02_physics_math.md`.
- Dependencies: DOC-01.
- Done criteria: equations, definitions, BC/IC complete.

### DOC-03 (Agent A)
- Objective: define discretization and solver behavior.
- Inputs: `02_physics_math.md`.
- Outputs: `03_numerical_methods.md`.
- Dependencies: DOC-02.
- Done criteria: IMPES, TPFA, upwind, CFL, failure handling complete.

### DOC-04 (Agent B)
- Objective: lock architecture and interface contracts.
- Inputs: DOC-01 to DOC-03.
- Outputs: `04_software_architecture.md`.
- Dependencies: DOC-03.
- Done criteria: CLI + schema stable and unambiguous.

### DOC-05 (Agent C)
- Objective: define GPU kernel and profiling plan.
- Inputs: `03_numerical_methods.md`, `04_software_architecture.md`.
- Outputs: `05_gpu_optimization_plan.md`.
- Dependencies: DOC-04.
- Done criteria: kernel order + metrics + benchmark matrix fixed, with optimization deferred until after CPU and history-match ML baselines are stable.

### DOC-06 (Agent D)
- Objective: define history-match ML model and training protocol.
- Inputs: `02_physics_math.md`, `04_software_architecture.md`.
- Outputs: `06_surrogate_ml_plan.md`.
- Dependencies: DOC-04.
- Done criteria: IO tensors, loss decomposition, eval horizons fixed.

### DOC-07 (Agent E)
- Objective: define pass/fail validation and benchmark tables.
- Inputs: DOC-04 to DOC-06.
- Outputs: `07_validation_and_benchmarking.md`.
- Dependencies: DOC-05, DOC-06.
- Done criteria: metrics and threshold policy complete.

### DOC-08 (Agent E)
- Objective: define visualization/report standards.
- Inputs: DOC-07.
- Outputs: `08_visualization_and_reporting.md`.
- Dependencies: DOC-07.
- Done criteria: required figures and naming conventions complete.

### DOC-09/10/11 (Agent E)
- Objective: schedule, risks, and definition of done.
- Inputs: prior docs.
- Outputs: `09`, `10`, `11`.
- Dependencies: DOC-07.
- Done criteria: decision gates + mitigation + completion gates clear.

### DOC-13/14 (Agent E + B)
- Objective: runbook command contracts and artifact schema.
- Inputs: `04`, `07`, `08`.
- Outputs: `13_runbook_commands.md`, `14_artifact_spec.md`.
- Dependencies: DOC-04, DOC-07.
- Done criteria: commands and file formats implementation-ready.

## Parallelization and Merge Order

1. Serial: DOC-01 -> DOC-02 -> DOC-03 -> DOC-04.
2. Parallel: DOC-05 and DOC-06.
3. Serial: DOC-07 -> DOC-08 -> DOC-09/10/11.
4. Parallel: DOC-13 and DOC-14.
5. Final: README and AGENTS consistency check.

## Implementation Sequencing Note

- Documentation lock order remains unchanged.
- Execution order after approval is:
  1. CPU correctness and reproducibility.
  2. Surrogate data generation, training, and evaluation.
  3. CPU/GPU parity on frozen outputs.
  4. GPU optimization and profiling as the final performance pass.
