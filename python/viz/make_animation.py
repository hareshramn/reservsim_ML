#!/usr/bin/env python3
"""Generate field animation (pressure or sw) from a simulator run."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np


def fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MP4/GIF animation from run outputs.")
    parser.add_argument("--run", required=True, help="Run ID or run directory path.")
    parser.add_argument("--field", required=True, choices=("pressure", "sw"), help="Field to animate.")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second (default: 12).")
    parser.add_argument("--out", required=True, help="Output path (file) or directory.")
    return parser.parse_args()


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


def sanitize_name(raw: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower()).strip("_") or "default"


def load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        fail(f"missing required file: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_state(run_dir: Path, field: str, nx: int, ny: int, nz: int) -> np.ndarray:
    filename = "state_pressure.npy" if field == "pressure" else "state_sw.npy"
    path = run_dir / filename
    if not path.exists():
        fail(f"missing required file: {path}")
    arr = np.load(path)
    if arr.ndim == 4:
        if arr.shape[1:] == (nz, ny, nx):
            return arr
        if arr.shape[1:] == (nz, nx, ny):
            return arr.transpose(0, 1, 3, 2)
        fail(f"array shape for {filename} does not match meta nx/ny/nz: {arr.shape} vs ({nx},{ny},{nz})")
    if arr.ndim != 3:
        fail(f"expected 3D/4D array for {filename}, got shape {arr.shape}")
    if arr.shape[1:] == (ny, nx):
        return arr
    if arr.shape[1:] == (nx, ny):
        return np.transpose(arr, (0, 2, 1))
    fail(f"array shape for {filename} does not match meta nx/ny: {arr.shape} vs ({nx},{ny})")
    return arr


def resolve_out_path(out_arg: str, run_dir: Path, field: str, scenario: str) -> Path:
    out = Path(out_arg).expanduser()
    if out.suffix.lower() in (".mp4", ".gif"):
        return out if out.is_absolute() else (Path.cwd() / out).resolve()

    base_dir = out if out.is_absolute() else (run_dir / out)
    base_dir = base_dir.resolve()
    return base_dir / f"anim_{scenario}_{field}.mp4"


def build_animation(data: np.ndarray, field: str, out_path: Path, fps: int) -> Path:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # type: ignore
    except ModuleNotFoundError:
        fail("missing dependency matplotlib (and pillow for gif fallback).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vmin = float(np.min(data))
    vmax = float(np.max(data))
    cmap = "viridis" if field == "pressure" else "Blues"
    title = "Pressure" if field == "pressure" else "Water Saturation"

    if data.ndim == 4:
        # Render an isometric-style 3D view using a few representative depth planes.
        nz, ny, nx = data.shape[1], data.shape[2], data.shape[3]
        z_slices = sorted({0, nz // 2, nz - 1})
        x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny))
        z_scale = max(1.0, 0.25 * max(nx, ny) / max(1, nz))
        fig = plt.figure(figsize=(8.0, 6.0), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.08, label=title)

        def _draw_frame(frame: int) -> tuple:
            ax.cla()
            for z_idx in z_slices:
                z_plane = np.full_like(x_grid, z_idx * z_scale, dtype=float)
                face_colors = plt.get_cmap(cmap)(norm(data[frame, z_idx]))
                ax.plot_surface(
                    x_grid,
                    y_grid,
                    z_plane,
                    facecolors=face_colors,
                    rstride=1,
                    cstride=1,
                    shade=False,
                    linewidth=0,
                    antialiased=False,
                    alpha=0.88,
                )
            ax.set_title(f"{title} 3D Isometric t={frame}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(0, max(nx - 1, 1))
            ax.set_ylim(0, max(ny - 1, 1))
            ax.set_zlim(0, max((nz - 1) * z_scale, 1))
            ax.view_init(elev=28, azim=-55)
            ax.set_box_aspect((max(nx, 1), max(ny, 1), max(nz * z_scale, 1)))
            return ()

        _draw_frame(0)

        def _update(frame: int):
            return _draw_frame(frame)
    else:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(data[0], origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(title)
        label = ax.set_title(f"{title} t=0")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        def _update(frame: int):
            im.set_data(data[frame])
            label.set_text(f"{title} t={frame}")
            return (im,)

    writer_used = ""
    if shutil.which("ffmpeg"):
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        writer_used = "ffmpeg"
        final_path = out_path
    else:
        writer = PillowWriter(fps=fps)
        writer_used = "pillow"
        final_path = out_path.with_suffix(".gif")

    ani = FuncAnimation(fig, _update, frames=data.shape[0], blit=False)
    ani.save(final_path, writer=writer)
    plt.close(fig)
    print(f"Animation writer: {writer_used}")
    return final_path


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        fail("--fps must be positive")

    repo_root = Path(__file__).resolve().parent
    run_dir = resolve_run_dir(args.run, repo_root)
    meta = load_meta(run_dir)
    nx = int(meta.get("nx", 0))
    ny = int(meta.get("ny", 0))
    nz = int(meta.get("nz", 1))
    if nx <= 0 or ny <= 0 or nz <= 0:
        fail("meta.json must contain positive nx, ny, and nz")

    scenario = sanitize_name(str(meta.get("case_name", "default")))
    state = load_state(run_dir, args.field, nx, ny, nz)
    out_path = resolve_out_path(args.out, run_dir, args.field, scenario)
    final_path = build_animation(state, args.field, out_path, args.fps)
    print(f"Generated animation: {final_path}")
    print(f"Run source: {run_dir}")


if __name__ == "__main__":
    main()
