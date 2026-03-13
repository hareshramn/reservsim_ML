# 09 Execution Timeline (2 Weeks)

## Week 1

### Day 1
- Finalize charter, problem, and interface contracts.
- Output: approved docs `00` to `04`.

### Day 2
- Complete physics/math and numerics pseudocode.
- Output: `02`, `03` locked.

### Day 3
- Define GPU kernel plan and benchmark matrix.
- Output: `05` locked.

### Day 4
- Define surrogate architecture/loss/dataset strategy.
- Output: `06` locked.

### Day 5
- Lock validation metrics and reporting templates.
- Output: `07`, `08` draft complete.
- Decision gate: confirm scope still feasible.

### Day 6
- Prepare runbook command contracts and artifact naming.
- Output: `13`, `14` draft complete.

### Day 7
- Consolidation and cross-doc consistency pass.
- Output: all docs aligned and reviewed.

## Week 2

### Day 8
- Start CPU baseline implementation (post-approval stage).

### Day 9
- Complete CPU baseline artifacts, validation, and reproducibility checks.

### Day 10
- Start dataset generation and surrogate training baseline.
- Decision gate: confirm CPU baseline and ML data pipeline are stable enough to freeze outputs.

### Day 11
- Surrogate rollout evaluation and comparison visuals.

### Day 12
- CPU/GPU parity harness and benchmark automation on frozen baseline outputs.

### Day 13
- GPU optimization pass and profiling instrumentation.
- Decision gate: evaluate speedup trajectory only after parity and reporting inputs are stable.

### Day 14
- Final QA, GPU benchmark table finalization, and submission bundle.

## Critical Path

1. Interface lock (`04`) -> runbook (`13`) -> implementation.
2. Numerics lock (`03`) -> CPU validation (`07`) -> surrogate evidence (`06`,`07`) -> GPU parity and optimization evidence (`05`,`07`).
3. Output schema lock (`04`,`14`) -> visualization (`08`) -> report.

## Parallelizable Work

- GPU plan (`05`) and ML plan (`06`) can proceed in parallel after `04`.
- Visualization (`08`) and artifact spec (`14`) can proceed in parallel after `07`.
- During implementation, surrogate training/evaluation can proceed before final GPU optimization as long as output schemas remain frozen.
