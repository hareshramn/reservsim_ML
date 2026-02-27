# 01 Problem Statement

## Case Definition

Use a SPE1-inspired synthetic quarter-five-spot waterflood mini-case for a 2D Cartesian reservoir.

## Domain and Resolution

- Default grid: `64 x 64`.
- Stress grid: `128 x 128`.
- Sanity grid: `16 x 16`.
- Boundary condition: no-flow outer boundaries.

## Wells and Controls

- Injector: rate-controlled water injection.
- Producer: BHP-controlled production.
- Controls fixed per scenario for reproducible comparisons.

## Rock and Fluid Assumptions

- Rock:
  - heterogeneous permeability (synthetic log-normal field),
  - constant porosity.
- Fluids:
  - immiscible oil-water,
  - incompressible in v1,
  - Corey relative permeability.
- Capillary pressure: excluded in v1.

## Initial Conditions

- Initial pressure: uniform reference field.
- Initial saturation: low uniform water saturation (`Swi`).

## v1 Case Input Schema (Draft)

```yaml
case_name: spe1_mini_default
grid:
  nx: 64
  ny: 64
  dx: 10.0
  dy: 10.0
rock:
  porosity: 0.2
  perm:
    type: lognormal
    mean_md: 100.0
    std_log: 0.8
fluid:
  mu_w_cp: 1.0
  mu_o_cp: 3.0
  swc: 0.2
  sor: 0.2
  nw: 2.0
  no: 2.0
wells:
  injector:
    i: 2
    j: 2
    control: rate
    value: 200.0
  producer:
    i: 62
    j: 62
    control: bhp
    value: 250.0
schedule:
  total_days: 200
  dt_days_initial: 0.1
  output_every_n_steps: 10
numerics:
  cfl_max: 0.5
  backend: cpu
```

## Acceptance for Problem Spec

1. Input schema covers grid, rock, fluid, wells, schedule, numerics.
2. Three predefined scenarios exist: sanity/default/stress.
3. All assumptions and exclusions are explicit.

