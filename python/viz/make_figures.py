#!/usr/bin/env python3
"""Generate required figures from a simulator run directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PNG figures from run outputs.")
    parser.add_argument("--run", required=True, help="Run ID (e.g., 20260226_190356_cpu_7) or run directory path.")
    parser.add_argument(
        "--out",
        default="figs",
        help="Output directory for PNG files. Relative paths are resolved under the run directory.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate required files and array shapes, then exit without writing figures.",
    )
    return parser.parse_args()


def fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def sanitize_name(raw: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower()).strip("_") or "default"


def resolve_run_dir(run_arg: str, repo_root: Path) -> Path:
    candidate = Path(run_arg).expanduser()
    if candidate.exists():
        run_dir = candidate.resolve()
        if not (run_dir / "meta.json").exists():
            fail(f"{run_dir} does not contain meta.json")
        return run_dir

    matches = sorted(repo_root.glob(f"cases/*/outputs/{run_arg}"))
    if not matches:
        fail(f"run id not found under cases/*/outputs: {run_arg}")
    if len(matches) > 1:
        fail(f"run id is ambiguous ({len(matches)} matches). Pass an explicit directory path.")
    return matches[0].resolve()


def resolve_out_dir(out_arg: str, run_dir: Path) -> Path:
    out = Path(out_arg).expanduser()
    if out.is_absolute():
        return out
    return run_dir / out


def import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        print(
            "error: missing dependency 'matplotlib'. "
            "Install it in your environment (for example: pip install matplotlib).",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return plt


def load_meta(run_dir: Path) -> Dict[str, object]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        fail(f"missing required file: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_state(run_dir: Path, filename: str, nx: int, ny: int, nz: int) -> np.ndarray:
    path = run_dir / filename
    if not path.exists():
        fail(f"missing required file: {path}")
    arr = np.load(path)
    if arr.ndim == 2:
        if arr.shape == (ny, nx):
            return arr[None, :, :]
        if arr.shape == (nx, ny):
            return arr.T[None, :, :]
    if arr.ndim == 4:
        if arr.shape[1:] == (nz, ny, nx):
            return arr
        if arr.shape[1:] == (nz, nx, ny):
            return arr.transpose(0, 1, 3, 2)
        fail(f"4D array shape for {filename} does not match nx/ny/nz from meta: {arr.shape} vs ({nx},{ny},{nz})")
    if arr.ndim != 3:
        fail(f"expected 3D/4D array for {filename}, got shape {arr.shape}")

    # Accept both [T, ny, nx] and [T, nx, ny]. Normalize to [T, ny, nx].
    if arr.shape[1:] == (ny, nx):
        return arr
    if arr.shape[1:] == (nx, ny):
        return np.transpose(arr, (0, 2, 1))
    fail(f"array shape for {filename} does not match nx/ny from meta: {arr.shape} vs ({nx},{ny})")


def load_well_rates(run_dir: Path) -> np.ndarray:
    path = run_dir / "well_rates.npy"
    if not path.exists():
        fail(f"missing required file: {path}")
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        fail(f"expected 2D array for well_rates.npy, got shape {arr.shape}")
    return arr


def load_timing(run_dir: Path) -> List[Dict[str, str]]:
    path = run_dir / "timing.csv"
    if not path.exists():
        fail(f"missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def figure_path(out_dir: Path, idx: int, topic: str, scenario: str) -> Path:
    return out_dir / f"fig_{idx:02d}_{topic}_{scenario}.png"


def choose_snapshot_steps(t_count: int) -> List[int]:
    if t_count <= 1:
        return [0]
    mid = t_count // 2
    return sorted({0, mid, t_count - 1})


def plot_snapshots(data: np.ndarray, title_prefix: str, cmap: str, out_path: Path) -> None:
    plt = import_matplotlib_pyplot()
    steps = choose_snapshot_steps(data.shape[0])
    fig, axes = plt.subplots(1, len(steps), figsize=(5 * len(steps), 4), constrained_layout=True)
    if len(steps) == 1:
        axes = [axes]

    for ax, step in zip(axes, steps):
        field = data[step, :, :]
        im = ax.imshow(field, origin="lower", cmap=cmap, aspect="auto")
        ax.set_title(f"{title_prefix} t={step:03d}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def choose_z_slices(nz: int) -> List[int]:
    if nz <= 1:
        return [0]
    return sorted({0, nz // 2, nz - 1})


def plot_snapshots_3d(data: np.ndarray, title_prefix: str, cmap: str, out_path: Path) -> None:
    # data shape: [T, nz, ny, nx]
    plt = import_matplotlib_pyplot()
    steps = choose_snapshot_steps(data.shape[0])
    z_slices = choose_z_slices(data.shape[1])
    fig, axes = plt.subplots(
        len(steps),
        len(z_slices),
        figsize=(4.5 * len(z_slices), 3.6 * len(steps)),
        constrained_layout=True,
    )
    if len(steps) == 1:
        axes = np.array([axes])
    if len(z_slices) == 1:
        axes = axes[:, None]

    for r, step in enumerate(steps):
        for c, z_idx in enumerate(z_slices):
            ax = axes[r, c]
            field = data[step, z_idx, :, :]
            im = ax.imshow(field, origin="lower", cmap=cmap, aspect="auto")
            ax.set_title(f"{title_prefix} t={step:03d} z={z_idx}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_depth_profile(sw: np.ndarray, pressure: np.ndarray, out_path: Path) -> None:
    # sw/pressure shape: [T, nz, ny, nx]
    plt = import_matplotlib_pyplot()
    z_axis = np.arange(sw.shape[1], dtype=int)
    sw_profile = sw[-1].mean(axis=(1, 2))
    p_profile = pressure[-1].mean(axis=(1, 2))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, constrained_layout=True)
    ax1.plot(z_axis, sw_profile, marker="o", linewidth=1.8, color="tab:blue")
    ax1.set_ylabel("avg sw [-]")
    ax1.grid(alpha=0.3)
    ax1.set_title("Final-Step Depth Profiles")

    ax2.plot(z_axis, p_profile, marker="o", linewidth=1.8, color="tab:red")
    ax2.set_xlabel("z index")
    ax2.set_ylabel("avg pressure")
    ax2.grid(alpha=0.3)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_timeseries(sw: np.ndarray, pressure: np.ndarray, well_rates: np.ndarray, out_path: Path) -> None:
    plt = import_matplotlib_pyplot()
    t_count = pressure.shape[0]
    x = np.arange(t_count, dtype=int)
    avg_pressure = pressure.reshape(t_count, -1).mean(axis=1)

    # Placeholder-friendly proxy: positive producer fraction from last well column.
    if well_rates.shape[1] > 0:
        producer_rate = np.maximum(well_rates[:, -1], 0.0)
        total_pos = np.maximum(well_rates, 0.0).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            water_cut = np.where(total_pos > 0.0, producer_rate / total_pos, 0.0)
    else:
        water_cut = np.zeros(t_count, dtype=float)

    if water_cut.shape[0] != t_count:
        # Broadcast/truncate to state time axis if source length differs in Slice 0.
        padded = np.zeros(t_count, dtype=float)
        n = min(t_count, water_cut.shape[0])
        padded[:n] = water_cut[:n]
        water_cut = padded

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    ax1.plot(x, water_cut, marker="o", linewidth=1.8)
    ax1.set_ylabel("producer water-cut [-]")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(alpha=0.3)

    ax2.plot(x, avg_pressure, marker="o", linewidth=1.8, color="tab:red")
    ax2.set_xlabel("time step")
    ax2.set_ylabel("avg pressure")
    ax2.grid(alpha=0.3)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def aggregate_timing(timing_rows: Iterable[Dict[str, str]]) -> Dict[str, float]:
    numeric_cols = ("pressure_time_s", "transport_time_s", "io_time_s", "total_time_s")
    agg = {k: 0.0 for k in numeric_cols}

    rows = list(timing_rows)
    agg_row = next((r for r in rows if r.get("row_type") == "aggregate"), None)
    src = [agg_row] if agg_row is not None else rows

    for row in src:
        for col in numeric_cols:
            raw = row.get(col, "0")
            try:
                agg[col] += float(raw)
            except (TypeError, ValueError):
                pass
    return agg


def plot_performance(agg: Dict[str, float], out_runtime: Path, out_breakdown: Path, out_speedup: Path) -> None:
    plt = import_matplotlib_pyplot()
    runtime_labels = ["pressure", "transport", "io", "total"]
    runtime_values = [
        agg["pressure_time_s"],
        agg["transport_time_s"],
        agg["io_time_s"],
        agg["total_time_s"],
    ]

    fig1, ax1 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax1.bar(runtime_labels, runtime_values, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#444444"])
    ax1.set_ylabel("time [s]")
    ax1.set_title("Runtime Components")
    ax1.grid(axis="y", alpha=0.3)
    fig1.savefig(out_runtime, dpi=140)
    plt.close(fig1)

    breakdown_labels = ["pressure", "transport", "io"]
    breakdown_values = [max(agg["pressure_time_s"], 0.0), max(agg["transport_time_s"], 0.0), max(agg["io_time_s"], 0.0)]
    if sum(breakdown_values) <= 0.0:
        breakdown_values = [1.0, 0.0, 0.0]

    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    ax2.bar(breakdown_labels, breakdown_values, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    ax2.set_ylabel("time [s]")
    ax2.set_title("Kernel Time Breakdown (Bar)")
    ax2.grid(axis="y", alpha=0.3)
    ax3.pie(breakdown_values, labels=breakdown_labels, autopct="%1.1f%%")
    ax3.set_title("Kernel Time Breakdown (Pie)")
    fig2.savefig(out_breakdown, dpi=140)
    plt.close(fig2)

    # Single-run baseline until CPU/GPU pair data is supplied.
    fig3, ax4 = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax4.bar(["speedup_vs_cpu"], [1.0], color="#9467bd")
    ax4.set_ylabel("x")
    ax4.set_ylim(0.0, 1.2)
    ax4.set_title("Speedup (single-run baseline)")
    ax4.grid(axis="y", alpha=0.3)
    fig3.savefig(out_speedup, dpi=140)
    plt.close(fig3)


def validate_sanity(pressure: np.ndarray, sw: np.ndarray, well_rates: np.ndarray, timing_rows: List[Dict[str, str]]) -> None:
    if pressure.shape != sw.shape:
        fail(f"pressure/sw shape mismatch: {pressure.shape} vs {sw.shape}")
    if pressure.shape[0] == 0:
        fail("state arrays must contain at least one time slice")
    if not np.isfinite(pressure).all():
        fail("state_pressure.npy contains non-finite values")
    if not np.isfinite(sw).all():
        fail("state_sw.npy contains non-finite values")

    if pressure.ndim not in (3, 4):
        fail(f"state arrays must be 3D or 4D, got ndim={pressure.ndim}")
    t_count = pressure.shape[0]
    if well_rates.shape[0] not in (1, t_count):
        fail(
            "well_rates.npy first dimension must be 1 or match state time dimension: "
            f"{well_rates.shape[0]} vs {t_count}"
        )

    required_timing_cols = {"run_id", "row_type", "step_idx", "dt_days", "pressure_time_s", "transport_time_s", "io_time_s", "total_time_s"}
    if timing_rows:
        missing = required_timing_cols - set(timing_rows[0].keys())
        if missing:
            fail(f"timing.csv missing required columns: {sorted(missing)}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    run_dir = resolve_run_dir(args.run, repo_root)
    out_dir = resolve_out_dir(args.out, run_dir).resolve()

    meta = load_meta(run_dir)
    nx = int(meta.get("nx", 0))
    ny = int(meta.get("ny", 0))
    nz = int(meta.get("nz", 1))
    if nx <= 0 or ny <= 0 or nz <= 0:
        fail("meta.json must contain positive nx, ny, and nz")

    scenario = sanitize_name(str(meta.get("case_name", "default")))
    pressure = load_state(run_dir, "state_pressure.npy", nx, ny, nz)
    sw = load_state(run_dir, "state_sw.npy", nx, ny, nz)
    well_rates = load_well_rates(run_dir)
    timing_rows = load_timing(run_dir)
    validate_sanity(pressure, sw, well_rates, timing_rows)

    if args.check_only:
        print(f"Sanity check passed: {run_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    timing_agg = aggregate_timing(timing_rows)

    if pressure.ndim == 4:
        plot_snapshots_3d(pressure, "Pressure", "viridis", figure_path(out_dir, 1, "pressure_slices_3d", scenario))
        plot_snapshots_3d(sw, "Water Saturation", "Blues", figure_path(out_dir, 2, "sw_slices_3d", scenario))
        pressure_mid = pressure[:, pressure.shape[1] // 2, :, :]
        sw_mid = sw[:, sw.shape[1] // 2, :, :]
        plot_timeseries(sw_mid, pressure_mid, well_rates, figure_path(out_dir, 3, "timeseries_watercut_pressure", scenario))
        plot_depth_profile(sw, pressure, figure_path(out_dir, 4, "depth_profile", scenario))
        plot_performance(
            timing_agg,
            out_runtime=figure_path(out_dir, 5, "runtime_bar", scenario),
            out_breakdown=figure_path(out_dir, 6, "kernel_breakdown", scenario),
            out_speedup=figure_path(out_dir, 7, "speedup_bar", scenario),
        )
    else:
        plot_snapshots(pressure, "Pressure", "viridis", figure_path(out_dir, 1, "pressure_snapshot", scenario))
        plot_snapshots(sw, "Water Saturation", "Blues", figure_path(out_dir, 2, "sw_front", scenario))
        plot_timeseries(sw, pressure, well_rates, figure_path(out_dir, 3, "timeseries_watercut_pressure", scenario))
        plot_performance(
            timing_agg,
            out_runtime=figure_path(out_dir, 4, "runtime_bar", scenario),
            out_breakdown=figure_path(out_dir, 5, "kernel_breakdown", scenario),
            out_speedup=figure_path(out_dir, 6, "speedup_bar", scenario),
        )

    print(f"Generated figures in: {out_dir}")
    print(f"Run source: {run_dir}")


if __name__ == "__main__":
    main()
