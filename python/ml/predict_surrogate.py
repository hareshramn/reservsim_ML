#!/usr/bin/env python3
"""Run one-step surrogate inference and write predicted t+1 fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def fail(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict next-step pressure/saturation from a surrogate checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to surrogate checkpoint (.npz).")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--pressure", help="Path to input pressure field (.npy, 2D or 3D).")
    parser.add_argument("--sw", help="Path to input water saturation field (.npy, 2D or 3D).")
    parser.add_argument(
        "--run",
        help="Run directory path, run id, or case yaml. Used if --pressure/--sw are not provided.",
    )
    parser.add_argument("--step", type=int, default=0, help="Checkpoint index to read from --run (default: 0).")
    return parser.parse_args()


def normalize_field(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float64, copy=False)
    if arr.ndim == 3:
        if arr.shape[0] < 1:
            fail(f"{name} has empty first dimension")
        return arr[0].astype(np.float64, copy=False)
    fail(f"{name} must be 2D or 3D, got shape {arr.shape}")


def resolve_run_arg(run_arg: str) -> Path:
    p = Path(run_arg).expanduser().resolve()
    if p.is_dir() and (p / "meta.json").exists():
        return p
    if p.is_file() and p.suffix in {".yaml", ".yml"}:
        outputs_dir = p.parent / "outputs"
        if not outputs_dir.exists():
            fail(f"outputs directory not found next to case file: {outputs_dir}")
        runs = sorted([d for d in outputs_dir.glob("*") if d.is_dir() and (d / "meta.json").exists()])
        if not runs:
            fail(f"no run directories found under: {outputs_dir}")
        return runs[-1]

    # Try run-id lookup under cases/*/outputs.
    repo_root = Path(__file__).resolve().parent
    matches = sorted(repo_root.glob(f"cases/*/outputs/{run_arg}"))
    if len(matches) == 1 and (matches[0] / "meta.json").exists():
        return matches[0].resolve()
    if len(matches) > 1:
        fail(f"run id is ambiguous ({len(matches)} matches). Pass explicit run directory path.")
    fail(f"could not resolve --run: {run_arg}")


def load_from_run(run_dir: Path, step_idx: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    meta_path = run_dir / "meta.json"
    p_path = run_dir / "state_pressure.npy"
    sw_path = run_dir / "state_sw.npy"
    if not meta_path.exists() or not p_path.exists() or not sw_path.exists():
        fail(f"run directory missing required files: {run_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pressure = np.load(p_path)
    sw = np.load(sw_path)
    if pressure.ndim == 4 and sw.ndim == 4:
        pressure = pressure[:, pressure.shape[1] // 2, :, :]
        sw = sw[:, sw.shape[1] // 2, :, :]
    if pressure.ndim != 3 or sw.ndim != 3:
        fail("run arrays must be 3D/4D [T, ny, nx] or [T, nx, ny] (4D uses middle-z slice)")
    if pressure.shape != sw.shape:
        fail(f"pressure/sw shape mismatch in run: {pressure.shape} vs {sw.shape}")
    if step_idx < 0 or step_idx >= pressure.shape[0]:
        fail(f"--step out of range: {step_idx} not in [0, {pressure.shape[0]-1}]")

    p_t = pressure[step_idx].astype(np.float64, copy=False)
    sw_t = sw[step_idx].astype(np.float64, copy=False)
    return p_t, sw_t, meta


def load_checkpoint(path: Path) -> Dict[str, object]:
    if not path.exists():
        fail(f"checkpoint not found: {path}")
    ckpt = np.load(path, allow_pickle=False)

    model_type = str(ckpt["model_type"].item()) if "model_type" in ckpt else "delta_mean_v1"
    model_name = str(ckpt["model_name"].item()) if "model_name" in ckpt else model_type
    params: Dict[str, object] = {
        "model_type": model_type,
        "model_name": model_name,
        "sw_clip_min": float(ckpt["sw_clip_min"]),
        "sw_clip_max": float(ckpt["sw_clip_max"]),
    }
    if model_type == "delta_mean_v1":
        params["delta_p"] = float(ckpt["delta_p"])
        params["delta_sw"] = float(ckpt["delta_sw"])
    elif model_type == "cell_mlp_v1":
        params["w1"] = ckpt["w1"]
        params["b1"] = ckpt["b1"]
        params["w2"] = ckpt["w2"]
        params["b2"] = ckpt["b2"]
        params["x_mean"] = ckpt["x_mean"]
        params["x_std"] = ckpt["x_std"]
        params["y_mean"] = ckpt["y_mean"]
        params["y_std"] = ckpt["y_std"]
    else:
        fail(f"unsupported model_type in checkpoint: {model_type}")
    return params


def predict_step(p: np.ndarray, sw: np.ndarray, params: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    if p.shape != sw.shape:
        fail(f"input pressure/sw shape mismatch: {p.shape} vs {sw.shape}")

    sw_min = float(params["sw_clip_min"])
    sw_max = float(params["sw_clip_max"])
    model_type = str(params["model_type"])

    if model_type == "delta_mean_v1":
        p_next = p + float(params["delta_p"])
        sw_next = np.clip(sw + float(params["delta_sw"]), sw_min, sw_max)
        return p_next, sw_next

    x = np.stack([p.reshape(-1), sw.reshape(-1)], axis=1)
    x_n = (x - np.asarray(params["x_mean"])) / np.asarray(params["x_std"])
    z1 = x_n @ np.asarray(params["w1"]) + np.asarray(params["b1"])
    a1 = np.tanh(z1)
    y_n = a1 @ np.asarray(params["w2"]) + np.asarray(params["b2"])
    delta = y_n * np.asarray(params["y_std"]) + np.asarray(params["y_mean"])
    p_next = p.reshape(-1) + delta[:, 0]
    sw_next = np.clip(sw.reshape(-1) + delta[:, 1], sw_min, sw_max)
    return p_next.reshape(p.shape), sw_next.reshape(sw.shape)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pressure and args.sw:
        p_t = normalize_field(np.load(Path(args.pressure).expanduser().resolve()), "pressure")
        sw_t = normalize_field(np.load(Path(args.sw).expanduser().resolve()), "sw")
        source_meta: Dict[str, object] = {"source": "manual_fields"}
        source_step = None
        source_run = None
    elif args.run:
        run_dir = resolve_run_arg(args.run)
        p_t, sw_t, source_meta = load_from_run(run_dir, args.step)
        source_step = args.step
        source_run = str(run_dir)
    else:
        fail("provide either (--pressure and --sw) or --run")

    params = load_checkpoint(ckpt_path)
    p_next, sw_next = predict_step(p_t, sw_t, params)

    p_out = out_dir / "pred_pressure_t1.npy"
    sw_out = out_dir / "pred_sw_t1.npy"
    np.save(p_out, p_next)
    np.save(sw_out, sw_next)

    summary = {
        "checkpoint": str(ckpt_path),
        "model_name": str(params["model_name"]),
        "model_type": str(params["model_type"]),
        "input_shape": list(p_t.shape),
        "output_pressure_path": str(p_out),
        "output_sw_path": str(sw_out),
        "source_run": source_run,
        "source_step": source_step,
        "source_meta_case_name": source_meta.get("case_name"),
    }
    (out_dir / "predict_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Prediction complete. Files: {p_out}, {sw_out}")


if __name__ == "__main__":
    main()
