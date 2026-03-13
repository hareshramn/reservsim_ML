# 06 Surrogate ML Plan

## Objective

Train a physics-informed surrogate that supports history matching, with emphasis on accelerating mismatch evaluation or candidate screening rather than forecast-first next-step rollout.

## Modeling Target

Preferred v1 target:
\[
(\theta, u_{0:T}) \rightarrow \mathcal{J}
\]

where:
- \(\theta\) denotes uncertain model parameters,
- \(u_{0:T}\) denotes prescribed historical controls over the replay window,
- \(\mathcal{J}\) denotes mismatch metrics against observed response.

Fallback baseline target:
\[
(\theta, u_{0:T}) \rightarrow y_{0:T}
\]

where \(y_{0:T}\) is a compact response summary such as rates, water cut, and pressure trends.

## Baseline Model Family

- MLP or lightweight sequence model for parameter-to-mismatch regression.
- Optional compact response surrogate for parameter-to-production summary mapping.
- Defer full field-to-field rollout surrogates until history-mode workflow is stable.

## Physics-Informed Loss

Total loss:
\[
\mathcal{L} = \lambda_d \mathcal{L}_{data} + \lambda_r \mathcal{L}_{residual} + \lambda_m \mathcal{L}_{mass}
\]

- `L_data`: supervised error to simulator-derived mismatch or response targets.
- `L_residual`: optional consistency penalty if response surrogates predict state-derived quantities.
- `L_mass`: optional conservation-informed regularization when state summaries are reconstructed.

## Dataset Generation Plan

- Generate datasets from repeated history-run style simulations across uncertain parameter samples.
- Store controls, parameter vectors, simulated response summaries, and mismatch scores.
- Split by parameter realization groups or scenario families to reduce leakage.

## Split Strategy

- Train: 70%
- Validation: 15%
- Test: 15%
- Hold out at least one unseen permeability realization.

## Evaluation Protocol

1. Mismatch prediction error metrics (MAE, RMSE) on held-out parameter samples.
2. Ranking quality:
   - can the surrogate identify promising low-mismatch candidates?
3. Optional response-summary fidelity:
   - rates, water cut, average pressure.
4. Throughput:
   - candidate evaluations per second vs full simulator history run.

## Acceptance Criteria

1. Surrogate objective is directly useful for history matching.
2. Held-out mismatch or response-summary accuracy is stable enough to rank candidate models.
3. Measured evaluation speed advantage over full simulator history runs.
