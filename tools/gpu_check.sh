#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR=""
MODE="release"
STEPS="1"
OUTPUT_EVERY="1"
GPU_INIT_RETRIES="2"

usage() {
  cat <<'USAGE'
Usage:
  tools/gpu_check.sh --model-dir <path> [--mode debug|release] [--steps N] [--output-every N] [--gpu-init-retries N]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    --output-every) OUTPUT_EVERY="${2:-}"; shift 2 ;;
    --gpu-init-retries) GPU_INIT_RETRIES="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MODEL_DIR" ]]; then
  echo "Missing required argument: --model-dir <path>" >&2
  exit 2
fi

if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  /usr/lib/wsl/lib/nvidia-smi >/dev/null || true
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >/dev/null || true
fi

echo "Running GPU preflight probe..."
exec "$ROOT_DIR/tools/model_run.sh" \
  --model-dir "$MODEL_DIR" \
  --mode "$MODE" \
  --backend gpu \
  --steps "$STEPS" \
  --output-every "$OUTPUT_EVERY" \
  --gpu-init-retries "$GPU_INIT_RETRIES"
