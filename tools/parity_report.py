#!/usr/bin/env python3
import argparse
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


def resolve_run(root: Path, model: str, seed: int, backend: str, run_hint: str | None) -> Path:
    if run_hint:
        p = Path(run_hint)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.is_dir():
            return p
        candidates = list((root / "cases").glob(f"*/outputs/{run_hint}"))
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(f"Could not resolve run: {run_hint}")

    outputs = root / "cases" / model / "outputs"
    if not outputs.is_dir():
        raise FileNotFoundError(f"Model outputs not found: {outputs}")
    matches = sorted(outputs.glob(f"*_{{backend}}_{seed}".format(backend=backend)))
    if not matches:
        raise FileNotFoundError(f"No {backend} runs found for seed={seed} under {outputs}")
    return matches[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU/GPU parity report for a model seed")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--cpu-run")
    ap.add_argument("--gpu-run")
    ap.add_argument("--out", help="Optional output JSON path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cpu_run = resolve_run(root, args.model, args.seed, "cpu", args.cpu_run)
    gpu_run = resolve_run(root, args.model, args.seed, "gpu", args.gpu_run)

    cp = np.load(cpu_run / "state_pressure.npy")
    gp = np.load(gpu_run / "state_pressure.npy")
    cs = np.load(cpu_run / "state_sw.npy")
    gs = np.load(gpu_run / "state_sw.npy")

    cpu_meta = json.loads((cpu_run / "meta.json").read_text())
    gpu_meta = json.loads((gpu_run / "meta.json").read_text())

    report = {
        "cpu_run": cpu_run.name,
        "gpu_run": gpu_run.name,
        "l2_p": float(np.linalg.norm(cp - gp)),
        "l2_sw": float(np.linalg.norm(cs - gs)),
        "linf_p": float(np.max(np.abs(cp - gp))),
        "linf_sw": float(np.max(np.abs(cs - gs))),
        "rel_l2_p": float(np.linalg.norm(cp - gp) / (np.linalg.norm(cp) + 1.0e-20)),
        "rel_l2_sw": float(np.linalg.norm(cs - gs) / (np.linalg.norm(cs) + 1.0e-20)),
        "mass_cpu_max": float(cpu_meta.get("transport_mass_balance_rel_max", 0.0)),
        "mass_gpu_max": float(gpu_meta.get("transport_mass_balance_rel_max", 0.0)),
    }

    for k, v in report.items():
        print(f"{k}={v}" if isinstance(v, float) else f"{k}={v}")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = (root / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote={out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
