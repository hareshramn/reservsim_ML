#!/usr/bin/env python3
"""MCP server exposing repository workflow tools."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    print(
        "Missing dependency: install the MCP Python package before running this server.\n"
        "Example: ./.venv/bin/pip install mcp",
        file=sys.stderr,
    )
    raise


ROOT_DIR = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT_DIR / "workflow"

mcp = FastMCP("reserv-ml-workflow")


def _run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "command": " ".join(cmd),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


@mcp.tool()
def compile_code(
    mode: str = "debug",
    cuda: str = "off",
    clean: bool = False,
    tests: bool = False,
) -> dict[str, Any]:
    """Compile simulator binaries."""
    cmd = [str(WORKFLOW), "compile", "--mode", mode, "--cuda", cuda]
    if clean:
        cmd.append("--clean")
    if tests:
        cmd.append("--tests")
    return _run(cmd)


@mcp.tool()
def run_model(
    model: str,
    mode: str = "release",
    backend: str = "cpu",
    steps: int = 10,
    output_every: int = 1,
    seed: int = 7,
    out: str = "auto",
) -> dict[str, Any]:
    """Run one case model."""
    cmd = [
        str(WORKFLOW),
        "run",
        "--model",
        model,
        "--mode",
        mode,
        "--backend",
        backend,
        "--steps",
        str(steps),
        "--output-every",
        str(output_every),
        "--seed",
        str(seed),
        "--out",
        out,
    ]
    return _run(cmd)


@mcp.tool()
def plot_run(
    model: str,
    run: str = "",
    out: str = "figs",
    check_only: bool = False,
) -> dict[str, Any]:
    """Generate figures for a run (latest by default)."""
    cmd = [str(WORKFLOW), "plot", "--model", model, "--out", out]
    if run:
        cmd.extend(["--run", run])
    if check_only:
        cmd.append("--check-only")
    return _run(cmd)


@mcp.tool()
def clean_outputs(
    model: str = "",
    all_models: bool = False,
    keep: int = 5,
    apply: bool = False,
) -> dict[str, Any]:
    """Clean old case output folders."""
    cmd = [str(WORKFLOW), "clean", "--keep", str(keep)]
    if all_models:
        cmd.append("--all")
    elif model:
        cmd.extend(["--model", model])
    else:
        return {
            "ok": False,
            "exit_code": 2,
            "command": "",
            "stdout": "",
            "stderr": "Provide model or set all_models=true.",
        }
    if apply:
        cmd.append("--apply")
    return _run(cmd)


@mcp.tool()
def all_in_one(
    model: str,
    steps: int = 10,
    seed: int = 7,
    mode: str = "release",
    backend: str = "cpu",
    output_every: int = 1,
) -> dict[str, Any]:
    """Compile, run, and plot from one call."""
    cmd = [
        str(WORKFLOW),
        "all",
        "--model",
        model,
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--mode",
        mode,
        "--backend",
        backend,
        "--output-every",
        str(output_every),
    ]
    return _run(cmd)


if __name__ == "__main__":
    mcp.run()
