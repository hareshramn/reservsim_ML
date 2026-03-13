# 06 History-Match ML Plan

## Objective

Train a history-match ML model that predicts mismatch from candidate parameter sets so the workflow can screen promising models before expensive simulator runs.

## Modeling Target

Preferred v1 target:
\[
(\theta, u_{0:T}) \rightarrow \mathcal{J}
\]

where:
- \(\theta\) denotes uncertain model parameters,
- \(u_{0:T}\) denotes prescribed historical controls over the replay window,
- \(\mathcal{J}\) denotes mismatch metrics against observed response.

## Baseline Model Family

- Lightweight tabular regressor for parameter-to-mismatch prediction.
- Ranking-oriented evaluation is preferred over field-state reconstruction.
- Field-to-field transition surrogates are out of scope for the current codebase.

## Physics-Informed Loss

Total loss:
\[
\mathcal{L} = \lambda_d \mathcal{L}_{data} + \lambda_r \mathcal{L}_{residual} + \lambda_m \mathcal{L}_{mass}
\]

- `L_data`: supervised error to simulator-derived mismatch targets.
- `L_residual`: optional regularizer for parameter smoothness or monotonic priors.
- `L_mass`: not required in the current tabular ranker implementation.

## Dataset Generation Plan

- Generate datasets from repeated history-run simulations across uncertain parameter samples.
- Store parameter vectors, mismatch scores, and per-observable misfit summaries.
- Split by parameter realization groups or scenario families to reduce leakage.

## Split Strategy

- Train: 70%
- Validation: 15%
- Test: 15%
- Hold out at least one unseen permeability realization.

## Evaluation Protocol

1. Mismatch prediction error metrics (MAE, RMSE) on held-out parameter samples.
2. Ranking quality:
   - can the model identify promising low-mismatch candidates?
3. Throughput:
   - candidate evaluations per second vs full simulator history run.

## Acceptance Criteria

1. ML objective is directly useful for history matching.
2. Held-out mismatch accuracy is stable enough to rank candidate models.
3. Measured evaluation speed advantage over full simulator history runs.
