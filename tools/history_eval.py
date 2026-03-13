#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def fail(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def parse_scalar(raw: str):
    text = raw.strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_simple_yaml(path: Path) -> dict[str, object]:
    cfg: dict[str, object] = {}
    section = ""
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        content = line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in content:
            fail(f"invalid config line at {path}:{idx}")
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0:
            section = ""
            if value == "":
                section = key
                continue
            cfg[key] = parse_scalar(value)
            continue
        if indent != 2 or not section:
            fail(f"unsupported nested YAML at {path}:{idx}")
        cfg[f"{section}.{key}"] = parse_scalar(value)
    return cfg


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def nearest_step(day: float, rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        fail("step_stats.csv is empty")
    best = rows[0]
    best_dist = abs(float(best.get("simulation_day", "0") or 0.0) - day)
    for row in rows[1:]:
        dist = abs(float(row.get("simulation_day", "0") or 0.0) - day)
        if dist < best_dist:
            best = row
            best_dist = dist
    return best


def observable_value(obs: str, row: dict[str, str]) -> float:
    mapping = {
        "injector_rate": float(row.get("inj_rate", "0") or 0.0),
        "producer_rate": abs(float(row.get("prod_rate", "0") or 0.0)),
        "producer_bhp": float(row.get("prod_bhp", "0") or 0.0),
        "injector_bhp": float(row.get("inj_bhp", "0") or 0.0),
    }
    if obs not in mapping:
        fail(f"unsupported observable for v1 history_eval: {obs}")
    return mapping[obs]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate history-run mismatch artifacts.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--case", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    case_path = Path(args.case).expanduser().resolve()
    if not run_dir.exists():
        fail(f"run directory not found: {run_dir}")
    if not case_path.exists():
        fail(f"case file not found: {case_path}")

    cfg = parse_simple_yaml(case_path)
    controls_csv = cfg.get("history.controls_csv")
    observations_csv = cfg.get("history.observations_csv")
    if not controls_csv or not observations_csv:
        fail("case file does not define history.controls_csv and history.observations_csv")

    controls_path = Path(str(controls_csv))
    observations_path = Path(str(observations_csv))
    if not controls_path.is_absolute():
        controls_path = (case_path.parent / controls_path).resolve()
    if not observations_path.is_absolute():
        observations_path = (case_path.parent / observations_path).resolve()
    if not controls_path.exists():
        fail(f"history controls CSV not found: {controls_path}")
    if not observations_path.exists():
        fail(f"history observations CSV not found: {observations_path}")

    step_rows = load_csv(run_dir / "step_stats.csv")
    observations = load_csv(observations_path)
    if not observations:
        fail("history observations CSV is empty")

    history_rows: list[dict[str, object]] = []
    per_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    weighted_total = 0.0

    for obs in observations:
        day = float(obs.get("day", "0") or 0.0)
        well = (obs.get("well", "") or "").strip()
        observable = (obs.get("observable", "") or "").strip()
        observed_value = float(obs.get("value", "0") or 0.0)
        weight = float(obs.get("weight", "1") or 1.0)
        step = nearest_step(day, step_rows)
        simulated_value = observable_value(observable, step)
        abs_error = abs(simulated_value - observed_value)
        squared_error = (simulated_value - observed_value) ** 2
        weighted_error = weight * squared_error
        weighted_total += weighted_error
        per_group[(well, observable)].append(squared_error)
        history_rows.append(
            {
                "run_id": run_dir.name,
                "day": day,
                "well": well,
                "observable": observable,
                "observed_value": observed_value,
                "simulated_value": simulated_value,
                "abs_error": abs_error,
                "squared_error": squared_error,
                "weight": weight,
                "weighted_error": weighted_error,
            }
        )

    history_match_path = run_dir / "history_match.csv"
    with history_match_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "day",
                "well",
                "observable",
                "observed_value",
                "simulated_value",
                "abs_error",
                "squared_error",
                "weight",
                "weighted_error",
            ],
        )
        writer.writeheader()
        writer.writerows(history_rows)

    summary_rows: list[dict[str, object]] = []
    per_well: dict[str, float] = defaultdict(float)
    per_observable: dict[str, float] = defaultdict(float)
    for (well, observable), errs in sorted(per_group.items()):
        rmse = (sum(errs) / len(errs)) ** 0.5
        mae = sum((e**0.5 for e in errs)) / len(errs)
        weighted_misfit = sum(errs)
        summary_rows.append(
            {
                "run_id": run_dir.name,
                "well": well,
                "observable": observable,
                "rmse": rmse,
                "mae": mae,
                "weighted_misfit": weighted_misfit,
            }
        )
        per_well[well] += weighted_misfit
        per_observable[observable] += weighted_misfit

    summary_path = run_dir / "well_observed_vs_simulated.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "well", "observable", "rmse", "mae", "weighted_misfit"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    mismatch_payload = {
        "run_id": run_dir.name,
        "objective_name": "weighted_squared_error",
        "objective_value": weighted_total,
        "compare_count": len(history_rows),
        "per_well": dict(sorted(per_well.items())),
        "per_observable": dict(sorted(per_observable.items())),
    }
    (run_dir / "history_mismatch.json").write_text(json.dumps(mismatch_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
