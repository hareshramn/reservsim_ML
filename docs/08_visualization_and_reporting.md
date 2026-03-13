# 08 Visualization and Reporting

## Required Figures

1. Pressure field snapshots at key timesteps.
2. Water saturation front snapshots at key timesteps.
3. Time-series:
   - producer water-cut,
   - average reservoir pressure.
4. Comparison panels:
   - CPU vs GPU fields,
   - simulator vs surrogate fields.
5. Performance visuals:
   - runtime bars,
   - speedup bars,
   - kernel time breakdown pie/bar.

## MP4 Requirements

- Resolution: `1280x720` minimum.
- Frame rate: `12` fps default.
- Naming convention:
  - `anim_<scenario>_<backend>_pressure.mp4`
  - `anim_<scenario>_<backend>_sw.mp4`
  - `anim_<scenario>_sim_vs_surrogate.mp4`

## Plot Naming Convention

- `fig_<nn>_<topic>_<scenario>.png`
- Example: `fig_03_sw_front_default.png`

## Final Technical Report Figure List

- Problem setup diagram.
- Physics/numerics summary figure.
- Surrogate rollout and throughput summary.
- CPU/GPU parity/error summary.
- CPU/GPU speedup summary (final optimization section).

## Reporting Sequence

- Interim reviews may omit final GPU speedup visuals if optimization is still deferred.
- Before final submission, add the GPU optimization section after numerical and surrogate results so performance claims are presented against a stable baseline.

## Acceptance Criteria

1. All required figures generated for default case.
2. At least one animation for solver and one for surrogate comparison.
3. Figure naming is consistent with artifact specification.
4. If GPU optimization is still in progress, interim reports must label CPU/GPU speedup visuals as pending rather than implying final performance closure.
