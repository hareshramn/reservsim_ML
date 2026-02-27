#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
FIG_SCRIPT="$ROOT_DIR/make_figures.py"

RUN_ARG=""
OUT_ARG="figs"
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  tools/plot_run.sh --run <run_id_or_run_dir> [--out <dir>] [--check-only]

Description:
  Wrapper for make_figures.py using the project virtualenv (.venv).
  Relative --out paths are resolved under the run directory.

Examples:
  tools/plot_run.sh --run 20260227_045706_cpu_7
  tools/plot_run.sh --run cases/model1/outputs/20260227_045706_cpu_7 --out figs
  tools/plot_run.sh --run cases/model1/outputs/20260227_045706_cpu_7 --check-only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_ARG="${2:-}"; shift 2 ;;
    --out)
      OUT_ARG="${2:-}"; shift 2 ;;
    --check-only)
      CHECK_ONLY=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "$RUN_ARG" ]]; then
  echo "Missing required argument: --run" >&2
  usage
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Project virtualenv not found: $PYTHON_BIN" >&2
  echo "Create it and install deps first:" >&2
  echo "  python3 -m venv .venv && ./.venv/bin/pip install numpy matplotlib" >&2
  exit 2
fi

cmd=("$PYTHON_BIN" "$FIG_SCRIPT" --run "$RUN_ARG" --out "$OUT_ARG")
if [[ "$CHECK_ONLY" -eq 1 ]]; then
  cmd+=(--check-only)
fi

"${cmd[@]}"
