# 01 Problem Statement

## Case Definition

Use a SPE1-inspired synthetic quarter-five-spot waterflood mini-case for a 2D Cartesian reservoir.

## Current Focus

- Primary workflow target: history-run style replay under prescribed well controls.
- Future workflow target: prediction-mode scenario runs after the history-mode path is stable.
- In v1 planning, the synthetic case stands in for historical controls and observations until explicit observation tables are added.

## History-Mode Inputs

- Historical controls are treated as known over the replay window.
- Observed response data is treated as the comparison target.
- A history run replays controls once and reports mismatch.
- A future history-match loop may repeat history runs with different uncertain parameters.

## Domain and Resolution

- Default grid: `64 x 64`.
- Stress grid: `128 x 128`.
- Sanity grid: `16 x 16`.
- Boundary condition: no-flow outer boundaries.

## Wells and Controls

- Injector: rate-controlled water injection.
- Producer: BHP-controlled production.
- Controls are treated as known inputs for replay-style runs.
- Comparison target is the reservoir response, not the controls themselves.

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

## History-Mode Schema Additions (Planned)

The base case schema is extended with explicit history data references:

```yaml
history:
  controls_csv: cases/model1/history_controls.csv
  observations_csv: cases/model1/history_observations.csv
  start_day: 0.0
  end_day: 200.0
  match_frequency_days: 1.0
```

`history.controls_csv` minimum columns:
- `day`
- `well`
- `control_kind` (`rate`, `bhp`, `shut`)
- `target_value`
- `phase` (`water`, `oil`, `liquid`, optional for `bhp`)

`history.observations_csv` minimum columns:
- `day`
- `well`
- `observable` (`oil_rate`, `water_rate`, `liquid_rate`, `water_cut`, `bhp`)
- `value`
- `weight` (optional, default `1.0`)

Planning assumptions:
- Controls are piecewise constant between declared `day` values.
- Observations may be sparser than simulator timesteps and must be aligned by interpolation or nearest declared policy in implementation.
- If `history` is omitted, the case remains a forward/manual run case.

## Acceptance for Problem Spec

1. Input schema covers grid, rock, fluid, wells, schedule, numerics.
2. Three predefined scenarios exist: sanity/default/stress.
3. All assumptions and exclusions are explicit.
4. The distinction between known controls and matched response is explicit for history-mode planning.
5. History-mode data contracts for controls and observations are explicitly defined.
