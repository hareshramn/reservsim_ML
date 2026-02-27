# 03 Numerical Discretization and Methods

## Chosen Method

Use IMPES:
1. Solve pressure with saturation-dependent mobilities at time \(n\).
2. Compute total fluxes.
3. Update water saturation explicitly (or semi-implicitly if needed).
4. Advance to \(n+1\), recompute properties.

## Fully Discrete IMPES (Locked v1)

At each time step \(n \rightarrow n+1\), solve:
\[
A(S_w^n) p^{n+1} = b(S_w^n, q_t^{n+1})
\]

Then update water saturation cell-wise:
\[
S_{w,i}^{n+1} = S_{w,i}^{n}
- \frac{\Delta t^n}{\phi_i V_i}\sum_{f \in \partial i} F_{w,f}^{n+1}
+ \frac{\Delta t^n}{\phi_i V_i} q_{w,i}^{n+1}
\]

Water flux is defined from total flux and upwind fractional flow:
\[
F_{w,f}^{n+1} = f_{w,\text{up},f}^{n} F_{t,f}^{n+1}
\]

Notation:
- \(i\): control volume index.
- \(f\): face index on \(\partial i\).
- \(V_i\): cell bulk volume.
- \(q_w > 0\): injection, \(q_w < 0\): production.

## Spatial Discretization

- Grid: structured 2D Cartesian.
- Pressure equation: TPFA with face transmissibilities.
- Transport equation: upwinded fractional flow for intercell fluxes.

## TPFA Transmissibility

For face \(i\leftrightarrow j\):
\[
T_{ij} = \frac{A_{ij}}{\Delta x_{ij}}\left(\mathbf{n}_{ij}\cdot\mathbf{K}\mathbf{n}_{ij}\right)\lambda_{t,ij}
\]

Flux:
\[
F_{t,ij} = -T_{ij}(p_j - p_i)
\]

Face mobility policy (locked):
- \(\lambda_{t,ij}\) uses harmonic averaging of cell total mobilities.
- For equal-distance Cartesian neighbors:
\[
\lambda_{t,ij} = \frac{2\lambda_{t,i}\lambda_{t,j}}{\lambda_{t,i}+\lambda_{t,j}}
\]

## Upwind Logic

- Determine upwind cell by sign of face total flux.
- If \(F_{t,ij} \ge 0\), upwind is cell \(i\); else upwind is cell \(j\).
- Evaluate \(f_w\) from the upwind saturation only.
- Apply source/sink well terms to local balances.
- Sign convention is locked:
  - injection terms are positive,
  - production terms are negative.

## Time-Step Policy

CFL-controlled adaptive \(\Delta t\) is:
\[
\Delta t_{\mathrm{CFL}}^n
= C_{\mathrm{CFL}}
\min_i
\left(
\frac{\phi_i V_i}
{\sum_{f \in \partial i}\max(F_{t,f}^{n+1}, 0) + \epsilon}
\right)
\]

Final step size:
\[
\Delta t^n = \mathrm{clamp}(s_f \Delta t_{\mathrm{CFL}}^n, dt_{\min}, dt_{\max})
\]

Locked constants:
- \(C_{\mathrm{CFL}} = 0.5\)
- \(s_f = 0.9\)
- \(\epsilon = 10^{-20}\) (division safety)
- \(dt_{\min} = 10^{-6}\) days
- \(dt_{\max} = 10^{-1}\) days

## Solver Strategy

- CPU baseline:
  - sparse assembly and iterative linear solve for pressure.
  - locked v1 pressure solver: Conjugate Gradient (CG) with Jacobi preconditioner.
- GPU path:
  - CUDA kernels for property update, transmissibility/flux, and saturation step,
  - pressure solve path with GPU-compatible sparse iteration.

## Stability and Failure Handling

Locked acceptance/retry policy per time step:
- Pressure linear residual target:
  - \(\|r\|_2 / \|b\|_2 \le 10^{-8}\).
- Pressure solve max iterations:
  - 500.
- Step retry budget:
  - 5 retries with \(\Delta t \leftarrow 0.5\Delta t\) each retry.
- Saturation bound handling:
  - hard clip to \([S_{wc}, 1-S_{or}]\),
  - record clip-count diagnostic per step.
- Mass-balance criterion:
  - relative imbalance \(\le 10^{-6}\) per step.
- Failure condition:
  - if either residual or mass-balance criterion is not met after retry budget, stop run and emit diagnostics.

## Acceptance Criteria

1. Mass-balance residual remains below tolerance.
2. CPU/GPU state mismatch within predefined norms.
3. No persistent saturation bound violations.

## Open Questions (P0 Only)

None.
