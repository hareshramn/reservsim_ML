# 00 Project Charter

## Objective

Deliver a focused mini-project in 2 weeks that demonstrates:
- two-phase reservoir simulation competence,
- history-mode workflow competence under prescribed controls,
- ML-assisted history-matching design with physics-aware constraints,
- browser-first reproducible workflow design,
- GPU optimization competence on RTX 3060 as a final-stage stretch goal.

## In Scope

- SPE1-inspired synthetic quarter-five-spot case.
- 2D structured-grid IMPES formulation (oil-water, immiscible, incompressible).
- CPU baseline for history-run style replay under known controls.
- Observed-vs-simulated response comparison and mismatch reporting.
- ML ranker plan aimed at accelerating history matching.
- Reproducible documentation and benchmark artifacts.
- Browser-first workflow entrypoint with CLI fallback for advanced/manual runs.

## Out of Scope

- Full commercial deck parsing.
- Black-oil compositional complexity.
- Full implicit Newton framework.
- Production-grade UI.
- Production-scale history matching over real field datasets in v1.
- Prediction-mode scenario forecasting in v1.

## Constraints

- Duration: 14 days.
- Compute: single RTX 3060 workstation.
- Primary languages: C++/CUDA and Python.

## Success Criteria

1. Numerical credibility:
   - bounded mass-balance error,
   - stable replay under prescribed controls.
2. History-mode evidence:
   - observed-vs-simulated comparison artifacts defined,
   - mismatch metrics reproducible.
3. ML-assisted history-matching evidence:
   - clearly scoped role in history matching,
   - measurable potential to reduce expensive simulator evaluations.
4. Portfolio packaging:
   - benchmark table,
   - plots and MP4,
   - technical report.
5. Stretch goal:
   - GPU runtime faster than CPU baseline on target grids after the baseline workflow is frozen.

## Primary Audience

- Reservoir simulation practitioners and researchers.
- GPU/HPC engineering practitioners.
- Applied ML practitioners in scientific computing.
- Engineering and product stakeholders evaluating simulation workflow outcomes.
