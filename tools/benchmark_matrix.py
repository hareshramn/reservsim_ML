#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

# Prefer the project virtualenv interpreter for numpy availability.
if os.environ.get("RESERV_USE_VENV_PY") != "1":
    _venv_py = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
    if _venv_py.exists():
        _env = dict(os.environ)
        _env["RESERV_USE_VENV_PY"] = "1"
        os.execve(str(_venv_py), [str(_venv_py), str(Path(__file__).resolve()), *sys.argv[1:]], _env)

import numpy as np


COLUMNS = [
    "run_id",
    "scenario",
    "backend",
    "grid_nx",
    "grid_ny",
    "steps",
    "total_time_s",
    "pressure_time_s",
    "transport_time_s",
    "transfer_time_s",
    "speedup_vs_cpu",
    "mass_err_rel",
    "l2_sw",
    "l2_p",
]


def parse_output_dir(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("Output directory:"):
            return Path(line.split(":", 1)[1].strip())
    raise RuntimeError("Could not parse output directory from workflow output")


def run_case(root: Path, model: str, backend: str, repeat_idx: int, steps: int, output_every: int, gpu_init_retries: int) -> Path:
    model_dir = root / "cases" / model
    cmd = [
        str(root / "tools" / "model_run.sh"),
        "--model-dir",
        str(model_dir),
        "--mode",
        "release",
        "--backend",
        backend,
        "--steps",
        str(steps),
        "--output-every",
        str(output_every),
        "--tag",
        f"benchmark-matrix-r{repeat_idx}",
    ]
    if backend == "gpu":
        cmd += ["--gpu-init-retries", str(gpu_init_retries)]
    env = dict(os.environ)
    env["RESERV_OUTPUT_BUCKET"] = "benchmark"
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"benchmark model execution failed for backend={backend} repeat={repeat_idx}")
    return parse_output_dir(proc.stdout)


def read_aggregate_timing(run_dir: Path) -> dict[str, float]:
    with (run_dir / "timing.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    agg = next((r for r in rows if r.get("row_type") == "aggregate"), None)
    if agg is None:
        raise RuntimeError(f"aggregate row missing in {run_dir / 'timing.csv'}")
    return {
        "total_time_s": float(agg["total_time_s"]),
        "pressure_time_s": float(agg["pressure_time_s"]),
        "transport_time_s": float(agg["transport_time_s"]),
    }


def read_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description="Run CPU/GPU benchmark matrix and emit benchmarks/benchmark_summary.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--output-every", type=int, default=10)
    ap.add_argument("--gpu-init-retries", type=int, default=2)
    ap.add_argument("--out", default="benchmarks/benchmark_summary.csv")
    args = ap.parse_args()

    if args.repeats <= 0:
        raise SystemExit("error: --repeats must be >= 1")
    root = Path(__file__).resolve().parent.parent
    rows: list[dict] = []

    for repeat_idx in range(1, args.repeats + 1):
      cpu_dir = run_case(root, args.model, "cpu", repeat_idx, args.steps, args.output_every, args.gpu_init_retries)
      gpu_dir = run_case(root, args.model, "gpu", repeat_idx, args.steps, args.output_every, args.gpu_init_retries)

      cpu_meta = read_meta(cpu_dir)
      gpu_meta = read_meta(gpu_dir)
      cpu_t = read_aggregate_timing(cpu_dir)
      gpu_t = read_aggregate_timing(gpu_dir)

      cpu_p = np.load(cpu_dir / "state_pressure.npy")
      gpu_p = np.load(gpu_dir / "state_pressure.npy")
      cpu_sw = np.load(cpu_dir / "state_sw.npy")
      gpu_sw = np.load(gpu_dir / "state_sw.npy")
      l2_p = float(np.linalg.norm(cpu_p - gpu_p))
      l2_sw = float(np.linalg.norm(cpu_sw - gpu_sw))
      speedup = cpu_t["total_time_s"] / max(gpu_t["total_time_s"], 1.0e-20)

      rows.append({
          "run_id": cpu_meta["run_id"],
          "scenario": cpu_meta.get("case_name", args.model),
          "backend": "cpu",
          "grid_nx": cpu_meta["nx"],
          "grid_ny": cpu_meta["ny"],
          "steps": cpu_meta.get("steps_completed", args.steps),
          "total_time_s": cpu_t["total_time_s"],
          "pressure_time_s": cpu_t["pressure_time_s"],
          "transport_time_s": cpu_t["transport_time_s"],
          "transfer_time_s": 0.0,
          "speedup_vs_cpu": 1.0,
          "mass_err_rel": cpu_meta.get("transport_mass_balance_rel_max", 0.0),
          "l2_sw": 0.0,
          "l2_p": 0.0,
      })
      rows.append({
          "run_id": gpu_meta["run_id"],
          "scenario": gpu_meta.get("case_name", args.model),
          "backend": "gpu",
          "grid_nx": gpu_meta["nx"],
          "grid_ny": gpu_meta["ny"],
          "steps": gpu_meta.get("steps_completed", args.steps),
          "total_time_s": gpu_t["total_time_s"],
          "pressure_time_s": gpu_t["pressure_time_s"],
          "transport_time_s": gpu_t["transport_time_s"],
          "transfer_time_s": 0.0,
          "speedup_vs_cpu": speedup,
          "mass_err_rel": gpu_meta.get("transport_mass_balance_rel_max", 0.0),
          "l2_sw": l2_sw,
          "l2_p": l2_p,
      })
      print(f"completed repeat={repeat_idx} cpu_run={cpu_meta['run_id']} gpu_run={gpu_meta['run_id']}")

    out = Path(args.out)
    if not out.is_absolute():
      out = (root / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=COLUMNS)
      writer.writeheader()
      writer.writerows(rows)
    print(f"wrote={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
