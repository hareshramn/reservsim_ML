# 06 Surrogate ML Plan

## Objective

Train a physics-informed surrogate to predict next-step pressure and saturation fields, then evaluate rollout quality and throughput.

## Modeling Target

\[
(p_t, S_{w,t}, u_t) \rightarrow (p_{t+1}, S_{w,t+1})
\]

where \(u_t\) includes control/context features (well controls, dt, static rock maps).

## Baseline Model Family

- CNN/UNet-style encoder-decoder.
- Input channels:
  - `p_t`, `sw_t`, permeability map, porosity map, optional well mask channels.
- Output channels:
  - `p_{t+1}`, `sw_{t+1}`.

## Physics-Informed Loss

Total loss:
\[
\mathcal{L} = \lambda_d \mathcal{L}_{data} + \lambda_r \mathcal{L}_{residual} + \lambda_m \mathcal{L}_{mass}
\]

- `L_data`: supervised MSE/MAE to simulator targets.
- `L_residual`: discrete PDE residual penalty using predicted fields.
- `L_mass`: mismatch in global water/oil material balance.

## Dataset Generation Plan

- Generate trajectory datasets from simulator runs over multiple seeds and well settings.
- Store train/val/test splits by scenario, not random frame mixing, to reduce leakage.
- Include both short and long horizon sequences.

## Split Strategy

- Train: 70%
- Validation: 15%
- Test: 15%
- Hold out at least one unseen permeability realization.

## Evaluation Protocol

1. One-step prediction error metrics (MAE, RMSE).
2. Multi-step rollout drift at 20, 50, 100-step horizons.
3. Physical plausibility checks:
   - saturation bounds,
   - mass-balance trend.
4. Throughput:
   - surrogate inference time vs solver step time.

## Acceptance Criteria

1. Stable rollout on default case horizon targets.
2. Better conservation metric than pure data-only baseline.
3. Measured inference speed advantage over numerical step path.

