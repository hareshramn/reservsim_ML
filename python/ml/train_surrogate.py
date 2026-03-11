#!/usr/bin/env python3
"""Train surrogate models from simulator outputs.

Supports:
- delta_mean_v1: constant per-step deltas (pipeline baseline)
- cell_mlp_v1: tiny neural model with backprop (shared cell-wise MLP)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


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
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        if not body:
            return []
        return [parse_scalar(part.strip()) for part in body.split(",")]
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_simple_yaml(path: Path) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        content = line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        if content.startswith(" ") or content.startswith("\t"):
            fail(f"unsupported nested YAML at {path}:{idx}")
        if ":" not in content:
            fail(f"invalid config line at {path}:{idx}")
        key, value = content.split(":", 1)
        key = key.strip()
        if not key:
            fail(f"empty key at {path}:{idx}")
        cfg[key] = parse_scalar(value.strip())
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surrogate from run outputs.")
    parser.add_argument("--data", required=True, help="Path to outputs root or a single run directory.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seed", required=True, type=int, help="Deterministic seed.")
    parser.add_argument("--out", required=True, help="Output directory for checkpoints and logs.")
    return parser.parse_args()


@dataclass
class Trajectory:
    run_id: str
    scenario: str
    pressure: np.ndarray  # [T, ny, nx]
    sw: np.ndarray  # [T, ny, nx]


def normalize_state_shape(arr: np.ndarray, nx: int, ny: int, name: str) -> np.ndarray:
    if arr.ndim == 4:
        # For 3D simulator runs, train surrogate on the middle-z slice.
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


def discover_run_dirs(data_path: Path) -> List[Path]:
    if (data_path / "meta.json").exists():
        return [data_path]
    if not data_path.exists():
        fail(f"--data path does not exist: {data_path}")
    candidates = sorted(p for p in data_path.glob("*") if p.is_dir() and (p / "meta.json").exists())
    if not candidates:
        fail(f"no run directories with meta.json found under: {data_path}")
    return candidates


def load_trajectories(data_path: Path) -> List[Trajectory]:
    runs: List[Trajectory] = []
    for run_dir in discover_run_dirs(data_path):
        meta_path = run_dir / "meta.json"
        pressure_path = run_dir / "state_pressure.npy"
        sw_path = run_dir / "state_sw.npy"
        if not pressure_path.exists() or not sw_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        nx = int(meta["nx"])
        ny = int(meta["ny"])
        pressure = normalize_state_shape(np.load(pressure_path), nx=nx, ny=ny, name="state_pressure.npy")
        sw = normalize_state_shape(np.load(sw_path), nx=nx, ny=ny, name="state_sw.npy")
        if pressure.shape != sw.shape:
            fail(f"pressure/sw shape mismatch in {run_dir}")
        if pressure.shape[0] < 2:
            continue
        runs.append(
            Trajectory(
                run_id=str(meta.get("run_id", run_dir.name)),
                scenario=str(meta.get("case_name", "default")),
                pressure=pressure.astype(np.float64, copy=False),
                sw=sw.astype(np.float64, copy=False),
            )
        )
    if not runs:
        fail(f"no usable trajectories found in: {data_path}")
    return runs


def build_dataset(runs: List[Trajectory]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_p: List[np.ndarray] = []
    x_sw: List[np.ndarray] = []
    y_p: List[np.ndarray] = []
    y_sw: List[np.ndarray] = []
    for run in runs:
        x_p.append(run.pressure[:-1])
        x_sw.append(run.sw[:-1])
        y_p.append(run.pressure[1:])
        y_sw.append(run.sw[1:])
    return (
        np.concatenate(x_p, axis=0),
        np.concatenate(x_sw, axis=0),
        np.concatenate(y_p, axis=0),
        np.concatenate(y_sw, axis=0),
    )


def split_indices(n: int, train_frac: float, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 3:
        fail("dataset too small; need at least 3 transition samples")
    if train_frac <= 0 or val_frac < 0 or (train_frac + val_frac) >= 1.0:
        fail("invalid split fractions in config")
    rng = np.random.default_rng(seed)
    perm = np.arange(n, dtype=int)
    rng.shuffle(perm)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
    i_train = perm[:n_train]
    i_val = perm[n_train : n_train + n_val]
    i_test = perm[n_train + n_val :]
    return i_train, i_val, i_test


def compute_metrics(p_pred: np.ndarray, sw_pred: np.ndarray, p_true: np.ndarray, sw_true: np.ndarray, eps: float) -> Dict[str, float]:
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


def write_split_metrics(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "samples", "mae_sw", "mae_p", "rmse_sw", "rmse_p", "mass_penalty"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ----- baseline model -----

def train_delta_mean(
    x_p: np.ndarray,
    x_sw: np.ndarray,
    y_p: np.ndarray,
    y_sw: np.ndarray,
    i_train: np.ndarray,
    i_val: np.ndarray,
    i_test: np.ndarray,
    sw_clip: Tuple[float, float],
    eps: float,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]], Dict[str, object]]:
    delta_p = float(np.mean(y_p[i_train] - x_p[i_train]))
    delta_sw = float(np.mean(y_sw[i_train] - x_sw[i_train]))

    def predict(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p_pred = x_p[idx] + delta_p
        sw_pred = np.clip(x_sw[idx] + delta_sw, sw_clip[0], sw_clip[1])
        return p_pred, sw_pred

    split_rows: List[Dict[str, object]] = []
    for split_name, idx in [("train", i_train), ("val", i_val), ("test", i_test)]:
        p_pred, sw_pred = predict(idx)
        m = compute_metrics(p_pred, sw_pred, y_p[idx], y_sw[idx], eps=eps)
        split_rows.append(
            {
                "split": split_name,
                "samples": int(idx.shape[0]),
                "mae_sw": m["mae_sw"],
                "mae_p": m["mae_p"],
                "rmse_sw": m["rmse_sw"],
                "rmse_p": m["rmse_p"],
                "mass_penalty": m["mass_penalty"],
            }
        )

    ckpt = {
        "model_type": np.array("delta_mean_v1"),
        "delta_p": np.array(delta_p),
        "delta_sw": np.array(delta_sw),
        "sw_clip_min": np.array(sw_clip[0]),
        "sw_clip_max": np.array(sw_clip[1]),
    }
    extra = {"num_parameters": 2}
    return ckpt, split_rows, extra


# ----- tiny NN model (shared per-cell MLP) -----

def init_mlp(rng: np.random.Generator, hidden_dim: int) -> Dict[str, np.ndarray]:
    w1 = rng.normal(0.0, 0.1, size=(2, hidden_dim))
    b1 = np.zeros((hidden_dim,), dtype=np.float64)
    w2 = rng.normal(0.0, 0.1, size=(hidden_dim, 2))
    b2 = np.zeros((2,), dtype=np.float64)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def mlp_forward(x: np.ndarray, params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    z1 = x @ params["w1"] + params["b1"]
    a1 = np.tanh(z1)
    y = a1 @ params["w2"] + params["b2"]
    cache = {"x": x, "z1": z1, "a1": a1}
    return y, cache


def mlp_backward(
    grad_y: np.ndarray,
    cache: Dict[str, np.ndarray],
    params: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    a1 = cache["a1"]
    x = cache["x"]

    grad_w2 = a1.T @ grad_y
    grad_b2 = np.sum(grad_y, axis=0)

    grad_a1 = grad_y @ params["w2"].T
    grad_z1 = grad_a1 * (1.0 - np.tanh(cache["z1"]) ** 2)

    grad_w1 = x.T @ grad_z1
    grad_b1 = np.sum(grad_z1, axis=0)

    return {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}


def apply_grads(params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], lr: float) -> None:
    for k in params:
        params[k] -= lr * grads[k]


def predict_transition_mlp(
    p: np.ndarray,
    sw: np.ndarray,
    params: Dict[str, np.ndarray],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    sw_clip: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    orig_shape = p.shape
    x = np.stack([p.reshape(-1), sw.reshape(-1)], axis=1)
    x_n = (x - x_mean) / x_std
    y_n, _ = mlp_forward(x_n, params)
    delta = y_n * y_std + y_mean
    p_next = p.reshape(-1) + delta[:, 0]
    sw_next = np.clip(sw.reshape(-1) + delta[:, 1], sw_clip[0], sw_clip[1])
    return p_next.reshape(orig_shape), sw_next.reshape(orig_shape)


def eval_split_mlp(
    x_p: np.ndarray,
    x_sw: np.ndarray,
    y_p: np.ndarray,
    y_sw: np.ndarray,
    idx: np.ndarray,
    params: Dict[str, np.ndarray],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    sw_clip: Tuple[float, float],
    eps: float,
) -> Dict[str, float]:
    p_preds = np.empty_like(y_p[idx])
    sw_preds = np.empty_like(y_sw[idx])
    for j, k in enumerate(idx):
        p_next, sw_next = predict_transition_mlp(
            x_p[k], x_sw[k], params, x_mean, x_std, y_mean, y_std, sw_clip
        )
        p_preds[j] = p_next
        sw_preds[j] = sw_next
    return compute_metrics(p_preds, sw_preds, y_p[idx], y_sw[idx], eps=eps)


def train_cell_mlp(
    x_p: np.ndarray,
    x_sw: np.ndarray,
    y_p: np.ndarray,
    y_sw: np.ndarray,
    i_train: np.ndarray,
    i_val: np.ndarray,
    i_test: np.ndarray,
    sw_clip: Tuple[float, float],
    eps: float,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    batch_transitions: int,
    loss_weight_p: float,
    loss_weight_sw: float,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]], Dict[str, object]]:
    if hidden_dim <= 0:
        fail("hidden_dim must be > 0")
    if epochs <= 0:
        fail("epochs must be > 0")
    if lr <= 0:
        fail("learning_rate must be > 0")
    if batch_transitions <= 0:
        fail("batch_transitions must be > 0")

    # Train on delta targets to handle scale differences naturally.
    delta_p = y_p - x_p
    delta_sw = y_sw - x_sw

    x_train = np.stack([x_p[i_train], x_sw[i_train]], axis=-1).reshape(-1, 2)
    y_train = np.stack([delta_p[i_train], delta_sw[i_train]], axis=-1).reshape(-1, 2)

    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)

    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)
    y_std = np.where(y_std < 1e-12, 1.0, y_std)

    rng = np.random.default_rng(seed)
    params = init_mlp(rng, hidden_dim=hidden_dim)

    n_train_transitions = i_train.shape[0]
    weight = np.array([loss_weight_p, loss_weight_sw], dtype=np.float64).reshape(1, 2)
    if np.any(weight <= 0):
        fail("loss_weight_p and loss_weight_sw must be > 0")

    best_val = float("inf")
    best_params = {k: v.copy() for k, v in params.items()}

    for _ in range(epochs):
        perm = np.arange(n_train_transitions)
        rng.shuffle(perm)

        for start in range(0, n_train_transitions, batch_transitions):
            batch_ids = i_train[perm[start : start + batch_transitions]]

            x_batch = np.stack([x_p[batch_ids], x_sw[batch_ids]], axis=-1).reshape(-1, 2)
            y_batch = np.stack([delta_p[batch_ids], delta_sw[batch_ids]], axis=-1).reshape(-1, 2)

            x_n = (x_batch - x_mean) / x_std
            y_n = (y_batch - y_mean) / y_std

            y_pred, cache = mlp_forward(x_n, params)
            err = y_pred - y_n
            grad_y = (2.0 / err.shape[0]) * err * weight

            grads = mlp_backward(grad_y, cache, params)
            apply_grads(params, grads, lr=lr)

        # Track best by validation RMSE sum.
        m_val = eval_split_mlp(
            x_p, x_sw, y_p, y_sw, i_val, params, x_mean, x_std, y_mean, y_std, sw_clip, eps
        )
        val_score = m_val["rmse_p"] + m_val["rmse_sw"]
        if val_score < best_val:
            best_val = val_score
            best_params = {k: v.copy() for k, v in params.items()}

    params = best_params

    split_rows: List[Dict[str, object]] = []
    for split_name, idx in [("train", i_train), ("val", i_val), ("test", i_test)]:
        m = eval_split_mlp(
            x_p, x_sw, y_p, y_sw, idx, params, x_mean, x_std, y_mean, y_std, sw_clip, eps
        )
        split_rows.append(
            {
                "split": split_name,
                "samples": int(idx.shape[0]),
                "mae_sw": m["mae_sw"],
                "mae_p": m["mae_p"],
                "rmse_sw": m["rmse_sw"],
                "rmse_p": m["rmse_p"],
                "mass_penalty": m["mass_penalty"],
            }
        )

    ckpt = {
        "model_type": np.array("cell_mlp_v1"),
        "w1": params["w1"],
        "b1": params["b1"],
        "w2": params["w2"],
        "b2": params["b2"],
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "sw_clip_min": np.array(sw_clip[0]),
        "sw_clip_max": np.array(sw_clip[1]),
    }
    num_params = int(params["w1"].size + params["b1"].size + params["w2"].size + params["b2"].size)
    extra = {
        "num_parameters": num_params,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_transitions": batch_transitions,
        "loss_weight_p": loss_weight_p,
        "loss_weight_sw": loss_weight_sw,
    }
    return ckpt, split_rows, extra


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        fail(f"config not found: {config_path}")
    cfg = parse_simple_yaml(config_path)

    model_type = str(cfg.get("model_type", cfg.get("model_name", "delta_mean_v1")))
    model_name = str(cfg.get("model_name", model_type))
    train_frac = float(cfg.get("split_train", 0.7))
    val_frac = float(cfg.get("split_val", 0.15))
    eps = float(cfg.get("mass_epsilon", 1e-12))

    runs = load_trajectories(data_path)
    x_p, x_sw, y_p, y_sw = build_dataset(runs)
    i_train, i_val, i_test = split_indices(x_p.shape[0], train_frac, val_frac, args.seed)
    sw_clip = (float(np.min(y_sw)), float(np.max(y_sw)))

    if model_type == "delta_mean_v1":
        ckpt, split_rows, extra = train_delta_mean(
            x_p, x_sw, y_p, y_sw, i_train, i_val, i_test, sw_clip=sw_clip, eps=eps
        )
    elif model_type == "cell_mlp_v1":
        ckpt, split_rows, extra = train_cell_mlp(
            x_p,
            x_sw,
            y_p,
            y_sw,
            i_train,
            i_val,
            i_test,
            sw_clip=sw_clip,
            eps=eps,
            seed=args.seed,
            hidden_dim=int(cfg.get("hidden_dim", 32)),
            epochs=int(cfg.get("epochs", 60)),
            lr=float(cfg.get("learning_rate", 1e-3)),
            batch_transitions=int(cfg.get("batch_transitions", 8)),
            loss_weight_p=float(cfg.get("loss_weight_p", 1.0)),
            loss_weight_sw=float(cfg.get("loss_weight_sw", 1.0)),
        )
    else:
        fail(f"unsupported model_type: {model_type}")

    checkpoint_path = out_dir / "surrogate_checkpoint.npz"
    np.savez(
        checkpoint_path,
        model_name=np.array(model_name),
        train_seed=np.array(args.seed),
        runs=np.array([r.run_id for r in runs]),
        **ckpt,
    )

    write_split_metrics(out_dir / "train_log.csv", split_rows)

    summary = {
        "model_name": model_name,
        "model_type": model_type,
        "seed": args.seed,
        "num_runs": len(runs),
        "num_samples": int(x_p.shape[0]),
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "splits": split_rows,
        **extra,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training complete. Checkpoint: {checkpoint_path}")
    print(f"Train log: {out_dir / 'train_log.csv'}")


if __name__ == "__main__":
    main()
