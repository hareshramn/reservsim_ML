#!/usr/bin/env python3
"""Score candidate parameter sets with a trained history-match ML model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def fail(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def load_checkpoint(path: Path) -> Dict[str, object]:
    if not path.exists():
        fail(f"checkpoint not found: {path}")
    ckpt = np.load(path, allow_pickle=False)
    return {
        "model_name": str(ckpt["model_name"].item()),
        "feature_names": [str(x) for x in ckpt["feature_names"].tolist()],
        "weights": ckpt["weights"],
        "x_mean": ckpt["x_mean"],
        "x_std": ckpt["x_std"],
        "y_mean": float(ckpt["y_mean"]),
        "target_transform": str(ckpt["target_transform"].item()),
        "target_epsilon": float(ckpt["target_epsilon"]),
    }


def inverse_target(y: np.ndarray, transform: str, eps: float) -> np.ndarray:
    if transform == "log10":
        return np.power(10.0, y) - eps
    return y


def predict(x: np.ndarray, params: Dict[str, object]) -> np.ndarray:
    x_n = (x - np.asarray(params["x_mean"])) / np.asarray(params["x_std"])
    y = x_n @ np.asarray(params["weights"]) + float(params["y_mean"])
    return inverse_target(y, str(params["target_transform"]), float(params["target_epsilon"]))


def load_candidates(path: Path, feature_keys: List[str]) -> List[Dict[str, str]]:
    filtered: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            filtered.append(raw)
    rows = list(csv.DictReader(filtered))
    if not rows:
        fail(f"candidate CSV is empty: {path}")
    for key in ["tag", *feature_keys]:
        if key not in rows[0]:
            fail(f"candidate CSV missing required column: {key}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Score candidate parameter sets for history matching.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidates", required=True, help="CSV with tag + feature columns matching the checkpoint feature set.")
    ap.add_argument("--out", required=True, help="Output directory.")
    args = ap.parse_args()

    ckpt = load_checkpoint(Path(args.checkpoint).expanduser().resolve())
    candidate_path = Path(args.candidates).expanduser().resolve()
    rows = load_candidates(candidate_path, list(ckpt["feature_names"]))
    x = np.zeros((len(rows), len(ckpt["feature_names"])), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, key in enumerate(ckpt["feature_names"]):
            x[i, j] = float(row[key])
    pred = predict(x, ckpt)

    scored: List[Dict[str, object]] = []
    for row, pred_val in zip(rows, pred.tolist()):
        scored.append(
            {
                "tag": row["tag"],
                "predicted_objective": pred_val,
                **{key: row[key] for key in ckpt["feature_names"]},
            }
        )
    scored.sort(key=lambda r: float(r["predicted_objective"]))
    for i, row in enumerate(scored, start=1):
        row["predicted_rank"] = i

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "candidate_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["predicted_rank", "tag", "predicted_objective", *ckpt["feature_names"]],
        )
        writer.writeheader()
        writer.writerows(scored)

    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "candidates": str(candidate_path),
        "model_name": ckpt["model_name"],
        "top_candidate": scored[0]["tag"] if scored else None,
        "rows": len(scored),
    }
    (out_dir / "score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Candidate scores: {out_dir / 'candidate_scores.csv'}")


if __name__ == "__main__":
    main()
