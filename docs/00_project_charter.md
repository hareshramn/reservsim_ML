# 00 Project Charter

## Objective

Deliver a portfolio-grade mini-project in 2 weeks that demonstrates:
- two-phase reservoir simulation competence,
- GPU optimization competence on RTX 3060,
- ML surrogate integration with physics-aware constraints.

## In Scope

- SPE1-inspired synthetic quarter-five-spot case.
- 2D structured-grid IMPES formulation (oil-water, immiscible, incompressible).
- CPU baseline and GPU path comparison.
- Physics-informed surrogate for next-step field prediction.
- Reproducible documentation and benchmark artifacts.

## Out of Scope

- Full commercial deck parsing.
- Black-oil compositional complexity.
- Full implicit Newton framework.
- Production-grade UI.

## Constraints

- Duration: 14 days.
- Compute: single RTX 3060 workstation.
- Primary languages: C++/CUDA and Python.

## Success Criteria

1. Speedup evidence: GPU runtime faster than CPU baseline on target grids.
2. Numerical credibility:
   - bounded mass-balance error,
   - CPU/GPU parity within defined tolerances.
3. Surrogate evidence:
   - rollout accuracy metrics,
   - inference throughput metrics.
4. Portfolio packaging:
   - benchmark table,
   - plots and MP4,
   - technical report.

## Primary Audience

- Reservoir simulation interview panels.
- GPU/HPC engineering interviewers.
- Applied ML hiring teams.

