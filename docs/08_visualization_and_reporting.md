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
- CPU/GPU speedup summary.
- Parity/error summary.
- Surrogate rollout and throughput summary.

## Acceptance Criteria

1. All required figures generated for default case.
2. At least one animation for solver and one for surrogate comparison.
3. Figure naming is consistent with artifact specification.

