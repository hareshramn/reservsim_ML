#!/usr/bin/env python3
"""Build a tabular ML dataset from history-run outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


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
    section = ""
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        content = line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in content:
            fail(f"invalid config line at {path}:{idx}")
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0:
            section = ""
            if value == "":
                section = key
                continue
            cfg[key] = parse_scalar(value)
            continue
        if indent != 2 or not section:
            fail(f"unsupported nested YAML at {path}:{idx}")
        cfg[f"{section}.{key}"] = parse_scalar(value)
    return cfg


def plan_feature_keys(plan_path: Path) -> List[str]:
    with plan_path.open("r", encoding="utf-8", newline="") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            headers = [part.strip() for part in raw.rstrip("\n").split(",")]
            return [h for h in headers if h and h not in {"tag", "case_name"}]
    fail(f"plan file has no header: {plan_path}")


def discover_run_dirs(root: Path) -> List[Path]:
    if not root.exists():
        fail(f"runs root not found: {root}")
    latest_by_tag: Dict[str, Path] = {}
    for p in root.iterdir():
        if not (
            p.is_dir()
            and (p / "history_mismatch.json").exists()
            and (p / "case_input.yaml").exists()
            and (p / "meta.json").exists()
        ):
            continue
        tag = run_tag(p)
        prev = latest_by_tag.get(tag)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            latest_by_tag[tag] = p
    return sorted(latest_by_tag.values())


def run_tag(run_dir: Path) -> str:
    parts = run_dir.name.split("__")
    if len(parts) >= 6:
        return parts[-1]
    return run_dir.name


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a tabular dataset from history-match candidate runs.")
    ap.add_argument("--runs-root", required=True, help="Directory containing ml-data run subdirectories.")
    ap.add_argument("--plan", required=True, help="Scenario CSV used to generate the candidate runs.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    feature_keys = plan_feature_keys(plan_path)
    run_dirs = discover_run_dirs(runs_root)
    if not run_dirs:
        fail(f"no history-match runs found under: {runs_root}")

    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        cfg = parse_simple_yaml(run_dir / "case_input.yaml")
        mismatch = json.loads((run_dir / "history_mismatch.json").read_text(encoding="utf-8"))
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

        row: Dict[str, object] = {
            "run_id": run_dir.name,
            "tag": run_tag(run_dir),
            "case_name": meta.get("case_name", cfg.get("case_name", "")),
            "objective_name": mismatch.get("objective_name", ""),
            "objective_value": float(mismatch.get("objective_value", 0.0)),
            "compare_count": int(mismatch.get("compare_count", 0)),
        }

        per_obs = mismatch.get("per_observable", {}) or {}
        if isinstance(per_obs, dict):
            for key, value in sorted(per_obs.items()):
                row[f"misfit_{key}"] = float(value)

        for key in feature_keys:
            if key not in cfg:
                fail(f"missing feature key in case_input.yaml: {key} ({run_dir})")
            value = cfg[key]
            if isinstance(value, bool):
                row[key] = int(value)
            elif isinstance(value, (int, float)):
                row[key] = value
            else:
                fail(f"feature key must resolve to numeric scalar: {key} ({run_dir})")
        rows.append(row)

    fieldnames = [
        "run_id",
        "tag",
        "case_name",
        "objective_name",
        "objective_value",
        "compare_count",
    ]
    fieldnames.extend([f"misfit_{k}" for k in sorted((json.loads((run_dirs[0] / 'history_mismatch.json').read_text(encoding='utf-8')).get('per_observable', {}) or {}).keys())])
    fieldnames.extend(feature_keys)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "runs_root": str(runs_root),
        "plan": str(plan_path),
        "rows": len(rows),
        "feature_keys": feature_keys,
        "output_csv": str(out_path),
    }
    (out_path.parent / "history_ml_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Dataset CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
