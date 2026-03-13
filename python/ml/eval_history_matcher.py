#!/usr/bin/env python3
"""Evaluate a history-match ML ranker on held-out candidate rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def fail(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def resolve_dataset(path_arg: str) -> Path:
    path = Path(path_arg).expanduser().resolve()
    if path.is_file():
        return path
    dataset = path / "history_ml_dataset.csv"
    if dataset.exists():
        return dataset
    fail(f"dataset CSV not found: {path}")


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
    if float(np.std(r_true)) == 0.0 or float(np.std(r_pred)) == 0.0:
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
        "top3_overlap": top_k_overlap(y_true, y_pred, 3),
    }


def load_checkpoint(path: Path) -> Dict[str, object]:
    if not path.exists():
        fail(f"checkpoint not found: {path}")
    ckpt = np.load(path, allow_pickle=False)
    return {
        "model_type": str(ckpt["model_type"].item()),
        "model_name": str(ckpt["model_name"].item()),
        "feature_names": [str(x) for x in ckpt["feature_names"].tolist()],
        "weights": ckpt["weights"],
        "x_mean": ckpt["x_mean"],
        "x_std": ckpt["x_std"],
        "y_mean": float(ckpt["y_mean"]),
        "target_transform": str(ckpt["target_transform"].item()),
        "target_epsilon": float(ckpt["target_epsilon"]),
        "train_run_ids": [str(x) for x in ckpt["train_run_ids"].tolist()],
        "val_run_ids": [str(x) for x in ckpt["val_run_ids"].tolist()],
        "test_run_ids": [str(x) for x in ckpt["test_run_ids"].tolist()],
    }


def inverse_target(y: np.ndarray, transform: str, eps: float) -> np.ndarray:
    if transform == "log10":
        return np.power(10.0, y) - eps
    return y


def predict(x: np.ndarray, params: Dict[str, object]) -> np.ndarray:
    x_n = (x - np.asarray(params["x_mean"])) / np.asarray(params["x_std"])
    y = x_n @ np.asarray(params["weights"]) + float(params["y_mean"])
    return inverse_target(y, str(params["target_transform"]), float(params["target_epsilon"]))


def select_rows(rows: List[Dict[str, str]], run_ids: List[str], feature_keys: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    selected = [row for row in rows if row["run_id"] in set(run_ids)]
    if not selected:
        return np.zeros((0, len(feature_keys))), np.zeros((0,)), []
    x = np.zeros((len(selected), len(feature_keys)), dtype=np.float64)
    y = np.zeros((len(selected),), dtype=np.float64)
    ids: List[str] = []
    for i, row in enumerate(selected):
        ids.append(row["run_id"])
        y[i] = float(row["objective_value"])
        for j, key in enumerate(feature_keys):
            x[i, j] = float(row[key])
    return x, y, ids


def write_metrics(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "samples", "mae_objective", "rmse_objective", "spearman_rank", "top3_overlap"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a history-match ML model.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="Dataset CSV path or directory containing history_ml_dataset.csv.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = load_checkpoint(Path(args.checkpoint).expanduser().resolve())
    rows = load_rows(resolve_dataset(args.data))
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    for split_name, run_ids in [
        ("train", ckpt["train_run_ids"]),
        ("val", ckpt["val_run_ids"]),
        ("test", ckpt["test_run_ids"]),
    ]:
        x, y, ids = select_rows(rows, list(run_ids), list(ckpt["feature_names"]))
        if x.shape[0] == 0:
            continue
        pred = predict(x, ckpt)
        m = metrics(y, pred)
        split_rows.append({"split": split_name, "samples": int(x.shape[0]), **m})
        actual_rank = np.argsort(np.argsort(y)) + 1
        pred_rank = np.argsort(np.argsort(pred)) + 1
        for run_id, actual, pred_val, a_rank, p_rank in zip(ids, y.tolist(), pred.tolist(), actual_rank.tolist(), pred_rank.tolist()):
            pred_rows.append(
                {
                    "run_id": run_id,
                    "split": split_name,
                    "objective_actual": actual,
                    "objective_predicted": pred_val,
                    "actual_rank": int(a_rank),
                    "predicted_rank": int(p_rank),
                }
            )

    write_metrics(out_dir / "history_ml_eval.csv", split_rows)
    with (out_dir / "ranked_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "split", "objective_actual", "objective_predicted", "actual_rank", "predicted_rank"],
        )
        writer.writeheader()
        writer.writerows(pred_rows)

    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "dataset": str(resolve_dataset(args.data)),
        "model_name": ckpt["model_name"],
        "model_type": ckpt["model_type"],
        "rows": split_rows,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Evaluation CSV: {out_dir / 'history_ml_eval.csv'}")


if __name__ == "__main__":
    main()
