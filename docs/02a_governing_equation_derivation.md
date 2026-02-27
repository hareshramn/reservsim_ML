# 02A Governing Equation Derivation (Compact)

## Purpose

Provide a short derivation from phase-wise mass conservation + Darcy flow to the v1 IMPES-style pressure-saturation system used in [02 Physics and Mathematical Formulation](./02_physics_math.md).

## Assumptions (v1)

- Two immiscible phases: water (`w`) and oil (`o`).
- Incompressible fluids and rock.
- No capillary pressure (`p_w = p_o = p`).
- Gravity neglected.
- Porosity `\phi` constant in time.

## Step 1: Phase Conservation

For each phase `\alpha \in \{w,o\}`:

\[
\phi \frac{\partial S_\alpha}{\partial t} + \nabla \cdot \mathbf{u}_\alpha = q_\alpha
\]

with saturation constraint:

\[
S_w + S_o = 1
\]

## Step 2: Darcy Law Per Phase

\[
\mathbf{u}_\alpha = - \lambda_\alpha \mathbf{K}\nabla p, \quad
\lambda_\alpha = \frac{k_{r\alpha}}{\mu_\alpha}
\]

Define total mobility and total velocity:

\[
\lambda_t = \lambda_w + \lambda_o, \quad
\mathbf{u}_t = \mathbf{u}_w + \mathbf{u}_o
\]

Thus:

\[
\mathbf{u}_t = -\lambda_t \mathbf{K}\nabla p
\]

## Step 3: Pressure Equation

Add water and oil conservation equations:

\[
\phi \frac{\partial (S_w + S_o)}{\partial t}
+ \nabla \cdot (\mathbf{u}_w + \mathbf{u}_o)
= q_w + q_o
\]

Use `S_w + S_o = 1` and incompressibility:

\[
\nabla \cdot \mathbf{u}_t = q_t, \quad q_t = q_w + q_o
\]

Substitute total Darcy velocity:

\[
\nabla \cdot \left(-\lambda_t \mathbf{K}\nabla p\right) = q_t
\]

This is the elliptic pressure equation solved at each IMPES step.

## Step 4: Water Transport Equation

From phase Darcy fluxes:

\[
\mathbf{u}_w = \frac{\lambda_w}{\lambda_t}\mathbf{u}_t = f_w \mathbf{u}_t,
\quad
f_w = \frac{\lambda_w}{\lambda_t}
\]

Insert into water conservation:

\[
\phi \frac{\partial S_w}{\partial t} + \nabla \cdot (f_w \mathbf{u}_t) = q_w
\]

This is the saturation transport equation (hyperbolic-dominant).

## Final v1 Coupled System

\[
\nabla \cdot \left(-\lambda_t \mathbf{K}\nabla p\right)=q_t
\]
\[
\phi \frac{\partial S_w}{\partial t} + \nabla \cdot \left(f_w \mathbf{u}_t\right)=q_w
\]
\[
\mathbf{u}_t = -\lambda_t \mathbf{K}\nabla p, \quad
f_w=\lambda_w/\lambda_t
\]

## Open Questions

- Confirm well-source partition policy (`q_w` vs `q_o`) for producer cells in v1.
- Confirm whether transmissibility-based discretization details should remain only in `docs/03_numerical_methods.md`.
