#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

usage() {
  cat <<'USAGE'
Usage:
  tools/history_run.sh --model-dir <dir> [--mode debug|release] [--backend cpu|gpu] [--steps N] [--output-every N] [--gpu-init-retries N] [--tag name] [--out path] [--case-file path]

Runs the simulator through the existing model runner using the history output bucket,
then computes mismatch artifacts from the configured history observations.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

set +e
run_output="$(
  RESERV_OUTPUT_BUCKET=history "$ROOT_DIR/tools/model_run.sh" "$@" 2>&1
)"
rc=$?
set -e
printf '%s\n' "$run_output"
if [[ $rc -ne 0 ]]; then
  exit $rc
fi

run_dir="$(printf '%s\n' "$run_output" | awk -F': ' '/^Output directory: /{print $2}' | tail -n 1)"
case_file=""
args=("$@")
idx=0
while [[ $idx -lt ${#args[@]} ]]; do
  arg="${args[$idx]}"
  if [[ "$arg" == "--case-file" || "$arg" == "--case" ]]; then
    idx=$((idx + 1))
    case_file="${args[$idx]:-}"
    break
  fi
  idx=$((idx + 1))
done
if [[ -z "$run_dir" ]]; then
  echo "Unable to determine output directory from history run." >&2
  exit 2
fi

if [[ -n "$case_file" && "$case_file" != /* ]]; then
  if [[ -f "$case_file" ]]; then
    case_file="$(pwd)/$case_file"
  fi
fi

if [[ -z "$case_file" ]]; then
  model_dir=""
  idx=0
  while [[ $idx -lt ${#args[@]} ]]; do
    arg="${args[$idx]}"
    if [[ "$arg" == "--model-dir" ]]; then
      idx=$((idx + 1))
      model_dir="${args[$idx]:-}"
      break
    fi
    idx=$((idx + 1))
  done
  if [[ -z "$model_dir" ]]; then
    echo "Unable to determine model directory for history evaluation." >&2
    exit 2
  fi
  if [[ "$model_dir" != /* ]]; then
    model_dir="$(pwd)/$model_dir"
  fi
  case_file="$model_dir/model.yaml"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" "$ROOT_DIR/tools/history_eval.py" --run-dir "$run_dir" --case "$case_file"
echo "History artifacts generated."
