#!/usr/bin/env python3
"""Evaluate surrogate checkpoint with one-step and rollout metrics."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def fail(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def parse_horizons(raw: str) -> List[int]:
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise SystemExit(f"error: invalid horizon value: {token}") from exc
        if value <= 0:
            fail(f"horizon must be positive: {value}")
        out.append(value)
    if not out:
        fail("empty --horizons list")
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a surrogate checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to surrogate checkpoint (.npz).")
    parser.add_argument("--case", required=True, help="Run directory path or case YAML path.")
    parser.add_argument("--horizons", default="20,50,100", help="Comma-separated rollout horizons.")
    parser.add_argument("--out", required=True, help="Output directory for evaluation artifacts.")
    return parser.parse_args()


@dataclass
class RunData:
    run_id: str
    scenario: str
    pressure: np.ndarray  # [T, ny, nx]
    sw: np.ndarray  # [T, ny, nx]


def normalize_state_shape(arr: np.ndarray, nx: int, ny: int, name: str) -> np.ndarray:
    if arr.ndim == 4:
        # For 3D simulator runs, evaluate on the middle-z slice.
        if arr.shape[2:] == (ny, nx):
            return arr[:, arr.shape[1] // 2, :, :]
        if arr.shape[2:] == (nx, ny):
            return arr[:, arr.shape[1] // 2, :, :].transpose(0, 2, 1)
        fail(f"{name} 4D shape {arr.shape} incompatible with nx={nx}, ny={ny}")
    if arr.ndim != 3:
        fail(f"{name} must be 3D/4D, got shape {arr.shape}")
    if arr.shape[1:] == (ny, nx):
        return arr
    if arr.shape[1:] == (nx, ny):
        return np.transpose(arr, (0, 2, 1))
    fail(f"{name} shape {arr.shape} incompatible with nx={nx}, ny={ny}")


def resolve_run_from_case(case_arg: str) -> Path:
    case_path = Path(case_arg).expanduser().resolve()
    if case_path.is_dir() and (case_path / "meta.json").exists():
        return case_path
    if case_path.is_file() and case_path.name.endswith(".yaml"):
        outputs_dir = case_path.parent / "outputs"
        if not outputs_dir.exists():
            fail(f"outputs directory not found next to case file: {outputs_dir}")
        runs = sorted([p for p in outputs_dir.glob("*") if p.is_dir() and (p / "meta.json").exists()])
        if not runs:
            fail(f"no run directories found under: {outputs_dir}")
        return runs[-1]
    fail("--case must be a run directory or a case YAML file")


def load_run(run_dir: Path) -> RunData:
    meta_path = run_dir / "meta.json"
    p_path = run_dir / "state_pressure.npy"
    sw_path = run_dir / "state_sw.npy"
    if not meta_path.exists() or not p_path.exists() or not sw_path.exists():
        fail(f"run directory missing required files: {run_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    nx = int(meta["nx"])
    ny = int(meta["ny"])
    pressure = normalize_state_shape(np.load(p_path), nx=nx, ny=ny, name="state_pressure.npy")
    sw = normalize_state_shape(np.load(sw_path), nx=nx, ny=ny, name="state_sw.npy")
    if pressure.shape != sw.shape:
        fail("pressure/sw shape mismatch in run data")
    if pressure.shape[0] < 2:
        fail("run does not contain enough checkpoints for evaluation")
    return RunData(
        run_id=str(meta.get("run_id", run_dir.name)),
        scenario=str(meta.get("case_name", "default")),
        pressure=pressure.astype(np.float64, copy=False),
        sw=sw.astype(np.float64, copy=False),
    )


def load_checkpoint(path: Path) -> Dict[str, object]:
    if not path.exists():
        fail(f"checkpoint not found: {path}")
    ckpt = np.load(path, allow_pickle=False)

    model_type = str(ckpt["model_type"].item()) if "model_type" in ckpt else "delta_mean_v1"
    model_name = str(ckpt["model_name"].item()) if "model_name" in ckpt else model_type

    out: Dict[str, object] = {
        "model_type": model_type,
        "model_name": model_name,
        "sw_clip_min": float(ckpt["sw_clip_min"]),
        "sw_clip_max": float(ckpt["sw_clip_max"]),
    }

    if model_type == "delta_mean_v1":
        out.update(
            {
                "delta_p": float(ckpt["delta_p"]),
                "delta_sw": float(ckpt["delta_sw"]),
            }
        )
    elif model_type == "cell_mlp_v1":
        out.update(
            {
                "w1": ckpt["w1"],
                "b1": ckpt["b1"],
                "w2": ckpt["w2"],
                "b2": ckpt["b2"],
                "x_mean": ckpt["x_mean"],
                "x_std": ckpt["x_std"],
                "y_mean": ckpt["y_mean"],
                "y_std": ckpt["y_std"],
            }
        )
    else:
        fail(f"unsupported model_type in checkpoint: {model_type}")

    return out


def predict_step(p: np.ndarray, sw: np.ndarray, params: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    sw_min = float(params["sw_clip_min"])
    sw_max = float(params["sw_clip_max"])
    model_type = str(params["model_type"])

    if model_type == "delta_mean_v1":
        p_next = p + float(params["delta_p"])
        sw_next = np.clip(sw + float(params["delta_sw"]), sw_min, sw_max)
        return p_next, sw_next

    # cell_mlp_v1
    x = np.stack([p.reshape(-1), sw.reshape(-1)], axis=1)
    x_mean = np.asarray(params["x_mean"])  # [2]
    x_std = np.asarray(params["x_std"])  # [2]
    y_mean = np.asarray(params["y_mean"])  # [2]
    y_std = np.asarray(params["y_std"])  # [2]

    x_n = (x - x_mean) / x_std
    z1 = x_n @ np.asarray(params["w1"]) + np.asarray(params["b1"])
    a1 = np.tanh(z1)
    y_n = a1 @ np.asarray(params["w2"]) + np.asarray(params["b2"])
    delta = y_n * y_std + y_mean

    p_next = p.reshape(-1) + delta[:, 0]
    sw_next = np.clip(sw.reshape(-1) + delta[:, 1], sw_min, sw_max)
    return p_next.reshape(p.shape), sw_next.reshape(sw.shape)


def metrics(p_pred: np.ndarray, sw_pred: np.ndarray, p_true: np.ndarray, sw_true: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    err_p = p_pred - p_true
    err_sw = sw_pred - sw_true
    mae_p = float(np.mean(np.abs(err_p)))
    mae_sw = float(np.mean(np.abs(err_sw)))
    rmse_p = float(np.sqrt(np.mean(err_p**2)))
    rmse_sw = float(np.sqrt(np.mean(err_sw**2)))
    true_mass = float(np.sum(sw_true))
    pred_mass = float(np.sum(sw_pred))
    mass_penalty = float(abs(pred_mass - true_mass) / max(abs(true_mass), eps))
    return {
        "mae_p": mae_p,
        "mae_sw": mae_sw,
        "rmse_p": rmse_p,
        "rmse_sw": rmse_sw,
        "mass_penalty": mass_penalty,
    }


def one_step_eval(run: RunData, params: Dict[str, object]) -> Dict[str, float]:
    p_preds = np.empty_like(run.pressure[1:])
    sw_preds = np.empty_like(run.sw[1:])
    for i in range(run.pressure.shape[0] - 1):
        p_next, sw_next = predict_step(run.pressure[i], run.sw[i], params)
        p_preds[i] = p_next
        sw_preds[i] = sw_next
    return metrics(p_preds, sw_preds, run.pressure[1:], run.sw[1:])


def rollout_eval(run: RunData, params: Dict[str, object], horizon: int) -> Dict[str, float]:
    t_count = run.pressure.shape[0]
    starts = t_count - horizon
    if starts <= 0:
        fail(f"horizon {horizon} is too large for available checkpoints: {t_count}")

    preds_p = []
    preds_sw = []
    trues_p = []
    trues_sw = []

    t0 = time.perf_counter()
    for start in range(starts):
        p_cur = run.pressure[start].copy()
        sw_cur = run.sw[start].copy()
        for _ in range(horizon):
            p_cur, sw_cur = predict_step(p_cur, sw_cur, params)
        preds_p.append(p_cur)
        preds_sw.append(sw_cur)
        trues_p.append(run.pressure[start + horizon])
        trues_sw.append(run.sw[start + horizon])
    elapsed = max(time.perf_counter() - t0, 1e-12)

    p_pred = np.stack(preds_p, axis=0)
    sw_pred = np.stack(preds_sw, axis=0)
    p_true = np.stack(trues_p, axis=0)
    sw_true = np.stack(trues_sw, axis=0)

    m = metrics(p_pred, sw_pred, p_true, sw_true)
    infer_steps = float(starts * horizon)
    m["infer_steps_per_s"] = infer_steps / elapsed
    return m


def write_surrogate_eval(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "model_name",
                "scenario",
                "horizon",
                "rmse_sw",
                "rmse_p",
                "mae_sw",
                "mae_p",
                "mass_penalty",
                "infer_steps_per_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_horizons(args.horizons)

    run_dir = resolve_run_from_case(args.case)
    run = load_run(run_dir)
    params = load_checkpoint(ckpt_path)
    model_name = str(params["model_name"])

    rows: List[Dict[str, object]] = []

    one = one_step_eval(run, params)
    rows.append(
        {
            "run_id": run.run_id,
            "model_name": model_name,
            "scenario": run.scenario,
            "horizon": 1,
            "rmse_sw": one["rmse_sw"],
            "rmse_p": one["rmse_p"],
            "mae_sw": one["mae_sw"],
            "mae_p": one["mae_p"],
            "mass_penalty": one["mass_penalty"],
            "infer_steps_per_s": 0.0,
        }
    )

    for h in horizons:
        roll = rollout_eval(run, params, horizon=h)
        rows.append(
            {
                "run_id": run.run_id,
                "model_name": model_name,
                "scenario": run.scenario,
                "horizon": h,
                "rmse_sw": roll["rmse_sw"],
                "rmse_p": roll["rmse_p"],
                "mae_sw": roll["mae_sw"],
                "mae_p": roll["mae_p"],
                "mass_penalty": roll["mass_penalty"],
                "infer_steps_per_s": roll["infer_steps_per_s"],
            }
        )

    out_csv = out_dir / "surrogate_eval.csv"
    write_surrogate_eval(out_csv, rows)

    summary = {
        "checkpoint": str(ckpt_path),
        "run_dir": str(run_dir),
        "model_name": model_name,
        "model_type": str(params["model_type"]),
        "horizons": [1] + horizons,
        "rows": rows,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Evaluation complete. Metrics CSV: {out_csv}")


if __name__ == "__main__":
    main()
