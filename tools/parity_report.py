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


def list_backend_runs(outputs: Path, backend: str) -> list[Path]:
    runs = list(outputs.glob(f"**/*__{backend}__*"))
    if not runs:
        runs = list(outputs.glob(f"**/*_{backend}_*"))
    runs = [p for p in runs if p.is_dir() and (p / "meta.json").exists()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def resolve_run(root: Path, model: str, backend: str, run_hint: str | None) -> Path:
    if run_hint:
        p = Path(run_hint)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.is_dir():
            return p
        candidates = list((root / "cases").glob(f"*/outputs/**/{run_hint}"))
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(f"Could not resolve run: {run_hint}")

    outputs = root / "cases" / model / "outputs"
    if not outputs.is_dir():
        raise FileNotFoundError(f"Model outputs not found: {outputs}")
    matches = list_backend_runs(outputs, backend)
    if not matches:
        raise FileNotFoundError(f"No {backend} runs found under {outputs}")
    return matches[0]


def find_shape_matched_run(outputs: Path, backend: str, reference_shape: tuple[int, ...], exclude: Path | None = None) -> Path | None:
    for candidate in list_backend_runs(outputs, backend):
        if exclude is not None and candidate == exclude:
            continue
        try:
            candidate_shape = tuple(np.load(candidate / "state_pressure.npy").shape)
        except Exception:
            continue
        if candidate_shape == reference_shape:
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU/GPU parity report for a model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cpu-run")
    ap.add_argument("--gpu-run")
    ap.add_argument("--out", help="Optional output JSON path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    outputs = root / "cases" / args.model / "outputs"
    cpu_run = resolve_run(root, args.model, "cpu", args.cpu_run)
    gpu_run = resolve_run(root, args.model, "gpu", args.gpu_run)

    # Without seed filtering, latest CPU/GPU runs may not align in step count.
    # When no explicit run hint is provided for one side, auto-match by pressure tensor shape.
    cpu_shape = tuple(np.load(cpu_run / "state_pressure.npy").shape)
    gpu_shape = tuple(np.load(gpu_run / "state_pressure.npy").shape)
    if cpu_shape != gpu_shape:
        if args.gpu_run is None:
            matched_gpu = find_shape_matched_run(outputs, "gpu", cpu_shape, exclude=gpu_run)
            if matched_gpu is not None:
                gpu_run = matched_gpu
                gpu_shape = tuple(np.load(gpu_run / "state_pressure.npy").shape)
        if cpu_shape != gpu_shape and args.cpu_run is None:
            matched_cpu = find_shape_matched_run(outputs, "cpu", gpu_shape, exclude=cpu_run)
            if matched_cpu is not None:
                cpu_run = matched_cpu
                cpu_shape = tuple(np.load(cpu_run / "state_pressure.npy").shape)
        if cpu_shape != gpu_shape:
            raise RuntimeError(
                f"Could not auto-pair CPU/GPU runs with matching shapes: cpu={cpu_run.name}{cpu_shape}, "
                f"gpu={gpu_run.name}{gpu_shape}. Pass --cpu-run/--gpu-run explicitly."
            )

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
