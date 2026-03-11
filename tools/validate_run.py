#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import sys

# Prefer the project virtualenv interpreter for numpy availability.
if os.environ.get("RESERV_USE_VENV_PY") != "1":
    _venv_py = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
    if _venv_py.exists():
        _env = dict(os.environ)
        _env["RESERV_USE_VENV_PY"] = "1"
        os.execve(str(_venv_py), [str(_venv_py), str(Path(__file__).resolve()), *sys.argv[1:]], _env)

import numpy as np


REQUIRED_FILES = [
    "meta.json",
    "state_pressure.npy",
    "state_sw.npy",
    "well_rates.npy",
    "well_bhp.npy",
    "timing.csv",
    "logs.txt",
]

TIMING_COLUMNS = [
    "run_id",
    "row_type",
    "step_idx",
    "dt_days",
    "pressure_time_s",
    "transport_time_s",
    "io_time_s",
    "total_time_s",
]


def resolve_run(root: Path, run: str) -> Path:
    p = Path(run)
    if not p.is_absolute():
        p = (root / p).resolve()
    if p.is_dir():
        return p
    candidates = list((root / "cases").glob(f"*/outputs/**/{run}"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not resolve run path: {run}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a simulator run directory")
    ap.add_argument("--run", required=True, help="Run id or run directory path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    run_dir = resolve_run(root, args.run)

    errors: list[str] = []

    for f in REQUIRED_FILES:
        if not (run_dir / f).exists():
            errors.append(f"missing file: {f}")

    if errors:
        for e in errors:
            print(f"[fail] {e}")
        return 1

    meta = json.loads((run_dir / "meta.json").read_text())
    for k in ["run_id", "backend", "nx", "ny", "steps_completed", "transport_mass_balance_rel_max"]:
        if k not in meta:
            errors.append(f"meta missing key: {k}")

    with (run_dir / "timing.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        timing_cols = reader.fieldnames or []
        if timing_cols != TIMING_COLUMNS:
            errors.append(f"timing.csv columns mismatch: got={timing_cols}")
        rows = list(reader)
        if not any(r.get("row_type") == "aggregate" for r in rows):
            errors.append("timing.csv missing aggregate row")

    p = np.load(run_dir / "state_pressure.npy")
    sw = np.load(run_dir / "state_sw.npy")
    if p.shape != sw.shape:
        errors.append(f"state shape mismatch: pressure={p.shape} sw={sw.shape}")
    if p.ndim not in (3, 4):
        errors.append(f"state arrays expected ndim=3 or 4, got pressure ndim={p.ndim}")
    if p.ndim == 4 and "nz" not in meta:
        errors.append("meta missing key: nz for 3D state arrays")

    if errors:
        for e in errors:
            print(f"[fail] {e}")
        return 1

    print(f"[ok] validated run: {run_dir}")
    print(f"[ok] backend={meta['backend']} steps_completed={meta['steps_completed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
