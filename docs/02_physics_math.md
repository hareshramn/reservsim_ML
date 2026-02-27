# 02 Physics and Mathematical Formulation

## Derivation Note

See [02A Governing Equation Derivation (Compact)](./02a_governing_equation_derivation.md) for a short derivation from phase conservation and Darcy laws to the v1 pressure-saturation system used below.

## Governing Equations

### 1) Total Darcy Velocity

\[
\mathbf{u}_t = - \lambda_t \mathbf{K}\nabla p
\]

where \(\lambda_t = \lambda_w + \lambda_o\), and \(\lambda_\alpha = k_{r\alpha}/\mu_\alpha\).

### 2) Pressure Equation (Elliptic)

\[
\nabla \cdot \mathbf{u}_t = q_t
\]

Substituting Darcy velocity:
\[
\nabla \cdot \left(-\lambda_t \mathbf{K}\nabla p \right)=q_t
\]

### 3) Water Saturation Transport

\[
\phi \frac{\partial S_w}{\partial t} + \nabla \cdot \left(f_w \mathbf{u}_t\right) = q_w
\]

with fractional flow:
\[
f_w = \frac{\lambda_w}{\lambda_t}
\]

## Relative Permeability (Corey)

Define effective saturation:
\[
S_{we}=\frac{S_w-S_{wc}}{1-S_{wc}-S_{or}}
\]

\[
k_{rw}=S_{we}^{n_w}, \quad k_{ro}=(1-S_{we})^{n_o}
\]

## Definitions Table

| Symbol | Meaning | Units | Typical Range |
|---|---|---|---|
| \(p\) | pressure | Pa (or psi) | \(10^6\) to \(10^7\) Pa |
| \(\mathbf{u}_t\) | total Darcy velocity | m/s | \(10^{-8}\) to \(10^{-4}\) |
| \(S_w\) | water saturation | - | 0 to 1 |
| \(\phi\) | porosity | - | 0.05 to 0.35 |
| \(\mathbf{K}\) | absolute permeability | m\(^2\) (or mD) | 1 to 1000 mD |
| \(\mu_w,\mu_o\) | viscosities | Pa.s (or cP) | 0.3 to 10 cP |
| \(q_t,q_w\) | source terms | m\(^3\)/s per bulk volume | case-dependent |

## Assumptions and Implications

- Incompressible fluids and rock:
  - pressure equation is elliptic per step,
  - no storage term from compressibility.
- Immiscible two-phase system:
  - water and oil only.
- No capillary pressure in v1:
  - simpler flux formulation,
  - less realistic near front details.

## Boundary and Initial Conditions

- Boundary: no-flow on outer domain.
- Wells: source/sink terms via control constraints.
- Initial \(p\): uniform reference.
- Initial \(S_w\): uniform \(S_{wi}\).

## v2 Extensions (Excluded from v1)

- Capillary pressure \(p_c(S_w)\).
- Slight compressibility for fluids/rock.
- Gravity terms.
- 3D extension.
