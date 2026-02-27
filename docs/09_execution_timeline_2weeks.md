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
- Add initial GPU kernels and parity harness.

### Day 10
- Optimization pass and profiling instrumentation.
- Decision gate: evaluate speedup trajectory.

### Day 11
- Dataset generation and surrogate training baseline.

### Day 12
- Surrogate rollout evaluation and comparison visuals.

### Day 13
- Benchmark table finalization and report drafting.

### Day 14
- Final QA, reproducibility check, and submission bundle.

## Critical Path

1. Interface lock (`04`) -> runbook (`13`) -> implementation.
2. Numerics lock (`03`) -> validation (`07`) -> benchmark evidence.
3. Output schema lock (`04`,`14`) -> visualization (`08`) -> report.

## Parallelizable Work

- GPU plan (`05`) and ML plan (`06`) can proceed in parallel after `04`.
- Visualization (`08`) and artifact spec (`14`) can proceed in parallel after `07`.

