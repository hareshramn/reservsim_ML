#!/usr/bin/env python3
"""Train a lightweight model to predict history-match mismatch from candidate parameters."""

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
        cfg[key.strip()] = parse_scalar(value.strip())
    return cfg


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_dataset(path_arg: str) -> Path:
    path = Path(path_arg).expanduser().resolve()
    if path.is_file():
        return path
    dataset = path / "history_ml_dataset.csv"
    if dataset.exists():
        return dataset
    fail(f"dataset CSV not found: {path}")


@dataclass
class Dataset:
    run_ids: List[str]
    features: np.ndarray
    targets: np.ndarray
    target_raw: np.ndarray


def load_dataset(path: Path, feature_keys: List[str], target_transform: str, target_eps: float) -> Dataset:
    rows = load_rows(path)
    if len(rows) < 3:
        fail("dataset too small; need at least 3 rows")
    run_ids: List[str] = []
    features = np.zeros((len(rows), len(feature_keys)), dtype=np.float64)
    target_raw = np.zeros((len(rows),), dtype=np.float64)
    for i, row in enumerate(rows):
        run_ids.append(row["run_id"])
        target_raw[i] = float(row["objective_value"])
        for j, key in enumerate(feature_keys):
            if key not in row or row[key] == "":
                fail(f"missing feature column in dataset: {key}")
            features[i, j] = float(row[key])
    if target_transform == "log10":
        targets = np.log10(target_raw + target_eps)
    elif target_transform == "identity":
        targets = target_raw.copy()
    else:
        fail(f"unsupported target_transform: {target_transform}")
    return Dataset(run_ids=run_ids, features=features, targets=targets, target_raw=target_raw)


def split_indices(n: int, train_frac: float, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_frac <= 0 or val_frac < 0 or (train_frac + val_frac) >= 1.0:
        fail("invalid split fractions")
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
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def inverse_target(y: np.ndarray, transform: str, eps: float) -> np.ndarray:
    if transform == "log10":
        return np.power(10.0, y) - eps
    return y


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape[0] < 2:
        return 0.0
    r_true = rankdata(y_true)
    r_pred = rankdata(y_pred)
    true_std = float(np.std(r_true))
    pred_std = float(np.std(r_pred))
    if true_std == 0.0 or pred_std == 0.0:
        return 0.0
    return float(np.corrcoef(r_true, r_pred)[0, 1])


def top_k_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    k = max(1, min(k, y_true.shape[0]))
    best_true = set(np.argsort(y_true)[:k].tolist())
    best_pred = set(np.argsort(y_pred)[:k].tolist())
    return float(len(best_true & best_pred)) / float(k)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    return {
        "mae_objective": float(np.mean(np.abs(err))),
        "rmse_objective": float(np.sqrt(np.mean(err**2))),
        "spearman_rank": spearman(y_true, y_pred),
        "top3_overlap": top_k_overlap(y_true, y_pred, k=3),
    }


def train_ridge(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std = np.where(x_std <= 1e-12, 1.0, x_std)
    x_n = (x - x_mean) / x_std
    y_mean = float(np.mean(y))
    y_center = y - y_mean
    lhs = x_n.T @ x_n + ridge_lambda * np.eye(x_n.shape[1], dtype=np.float64)
    rhs = x_n.T @ y_center
    weights = np.linalg.solve(lhs, rhs)
    return weights, x_mean, x_std, y_mean


def predict(x: np.ndarray, weights: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, y_mean: float) -> np.ndarray:
    x_n = (x - x_mean) / x_std
    return x_n @ weights + y_mean


def write_metrics(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "samples", "mae_objective", "rmse_objective", "spearman_rank", "top3_overlap"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a history-match mismatch regressor.")
    ap.add_argument("--data", required=True, help="Dataset CSV path or directory containing history_ml_dataset.csv.")
    ap.add_argument("--config", required=True, help="Path to history_ml_config.yaml.")
    ap.add_argument("--seed", required=True, type=int, help="Deterministic split seed.")
    ap.add_argument("--out", required=True, help="Output directory for checkpoint and logs.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset(args.data)
    cfg = parse_simple_yaml(Path(args.config).expanduser().resolve())
    feature_keys = list(cfg.get("feature_keys", []))
    if not feature_keys:
        fail("config must define feature_keys")
    if not all(isinstance(k, str) for k in feature_keys):
        fail("feature_keys must be a list of strings")

    target_transform = str(cfg.get("target_transform", "log10"))
    target_eps = float(cfg.get("target_epsilon", 1.0))
    train_frac = float(cfg.get("split_train", 0.7))
    val_frac = float(cfg.get("split_val", 0.15))
    ridge_lambda = float(cfg.get("ridge_lambda", 1e-6))
    model_name = str(cfg.get("model_name", "history_match_ridge_v1"))

    data = load_dataset(dataset_path, feature_keys=feature_keys, target_transform=target_transform, target_eps=target_eps)
    i_train, i_val, i_test = split_indices(data.features.shape[0], train_frac, val_frac, args.seed)
    weights, x_mean, x_std, y_mean = train_ridge(data.features[i_train], data.targets[i_train], ridge_lambda=ridge_lambda)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_rows: List[Dict[str, object]] = []
    predictions_rows: List[Dict[str, object]] = []
    for split_name, idx in [("train", i_train), ("val", i_val), ("test", i_test)]:
        pred_trans = predict(data.features[idx], weights, x_mean, x_std, y_mean)
        pred_objective = inverse_target(pred_trans, target_transform, target_eps)
        true_objective = data.target_raw[idx]
        m = metrics(true_objective, pred_objective)
        split_rows.append({"split": split_name, "samples": int(idx.shape[0]), **m})
        for row_idx, pred_val, true_val in zip(idx.tolist(), pred_objective.tolist(), true_objective.tolist()):
            predictions_rows.append(
                {
                    "run_id": data.run_ids[row_idx],
                    "split": split_name,
                    "objective_actual": true_val,
                    "objective_predicted": pred_val,
                    "abs_error": abs(pred_val - true_val),
                }
            )

    checkpoint_path = out_dir / "history_match_checkpoint.npz"
    np.savez(
        checkpoint_path,
        model_type=np.array("ridge_regressor_v1"),
        model_name=np.array(model_name),
        feature_names=np.asarray(feature_keys),
        weights=weights,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=np.array(y_mean),
        target_transform=np.array(target_transform),
        target_epsilon=np.array(target_eps),
        train_run_ids=np.asarray([data.run_ids[i] for i in i_train]),
        val_run_ids=np.asarray([data.run_ids[i] for i in i_val]),
        test_run_ids=np.asarray([data.run_ids[i] for i in i_test]),
    )

    write_metrics(out_dir / "train_metrics.csv", split_rows)
    with (out_dir / "train_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "split", "objective_actual", "objective_predicted", "abs_error"])
        writer.writeheader()
        writer.writerows(predictions_rows)

    summary = {
        "dataset": str(dataset_path),
        "model_name": model_name,
        "model_type": "ridge_regressor_v1",
        "feature_keys": feature_keys,
        "target_transform": target_transform,
        "target_epsilon": target_eps,
        "seed": args.seed,
        "ridge_lambda": ridge_lambda,
        "rows": split_rows,
        "checkpoint": str(checkpoint_path),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Train checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
