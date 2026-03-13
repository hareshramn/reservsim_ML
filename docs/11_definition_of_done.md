# 11 Definition of Done

## Must-Have Checklist

1. Documentation pack complete and internally consistent.
2. Reproducible CPU and GPU runs defined via runbook commands.
3. Benchmark summary table schema defined and populated in final stage.
4. Surrogate training/evaluation protocol defined and executed in final stage.
5. Technical report structure complete with required figures.
6. IMPES v1 discrete specification is locked in `docs/03_numerical_methods.md` with:
   - pressure equation form \(A(S_w^n)p^{n+1}=b\),
   - explicit saturation update equation,
   - TPFA total flux formula and upwinded water flux coupling,
   - explicit CFL formula and `clamp(safety * dt_cfl, dt_min, dt_max)` policy,
   - numeric tolerances for pressure residual, mass-balance, and retry budget.
7. Execution sequencing is respected:
   - CPU correctness and artifact reproducibility first,
   - history-match ML data/training/evaluation second,
   - GPU parity and optimization evidence last.

## Quality Bar by Artifact

- Case spec:
  - all required YAML fields documented,
  - defaults and units explicit.
- Numerics:
  - IMPES algorithm and stability rules explicitly stated with numeric constants and equations.
- GPU plan:
  - kernel priorities and profiler metrics fixed, with optimization explicitly treated as a final-stage activity.
- ML plan:
  - model input/output and loss terms fixed.
- Validation:
  - pass/fail thresholds documented.

## Not Done If

- CLI contracts are ambiguous.
- Output schema is incomplete.
- Mass-balance/parity metrics are not specified.
- IMPES equations exist only in prose without fully discrete forms.
- CFL policy, retry policy, or solver tolerances are unspecified.
- Benchmark and figure naming conventions are inconsistent.
- P0 open questions block implementation decisions.

## Pre-Implementation Gate

Implementation can start only when:
1. `docs/00` through `docs/14` are approved.
2. `AGENTS.md` protocol is accepted.
3. No unresolved P0 decisions remain.
4. `docs/03_numerical_methods.md` contains a dedicated `Open Questions (P0 Only)` section and all listed P0s are closed or explicitly deferred by approver.

## Gate Approval Record

- Status: Approved for implementation start.
- Approval date: 2026-02-25.
- Approved by: project owner (interactive session approval).
- Scope released: Slice 0 (`sim_run` CLI shell and config validation).
