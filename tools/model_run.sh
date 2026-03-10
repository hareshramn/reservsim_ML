#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$(pwd)"

MODE="release"
BACKEND="cpu"
STEPS="10"
OUTPUT_EVERY="1"
GPU_INIT_RETRIES="0"
PURPOSE="adhoc"
TAG=""
OUT_DIR="auto"

usage() {
  cat <<'USAGE'
Usage:
  ./run [--model-dir path] [--mode debug|release] [--backend cpu|gpu] [--steps N] [--output-every N] [--gpu-init-retries N] [--purpose adhoc|benchmark|ml-data] [--tag name] [--out path]

Defaults come from run.env in the selected model folder.
CLI arguments override run.env values.
USAGE
}

sanitize_component() {
  local raw="$1"
  local sanitized
  sanitized="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
  if [[ -z "$sanitized" ]]; then
    sanitized="run"
  fi
  printf '%s\n' "$sanitized"
}

# Pass 1: only discover --model-dir so we can load the right run.env.
pass1=("$@")
idx=0
while [[ $idx -lt ${#pass1[@]} ]]; do
  arg="${pass1[$idx]}"
  if [[ "$arg" == "--model-dir" ]]; then
    idx=$((idx + 1))
    if [[ $idx -ge ${#pass1[@]} ]]; then
      echo "Missing value for --model-dir" >&2
      exit 2
    fi
    MODEL_DIR="${pass1[$idx]}"
  fi
  idx=$((idx + 1))
done

if [[ "$MODEL_DIR" != /* ]]; then
  MODEL_DIR="$(pwd)/$MODEL_DIR"
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 2
fi
if [[ ! -f "$MODEL_DIR/model.yaml" ]]; then
  echo "model.yaml not found in $MODEL_DIR" >&2
  exit 2
fi

if [[ -f "$MODEL_DIR/run.env" ]]; then
  # shellcheck disable=SC1091
  source "$MODEL_DIR/run.env"
  MODE="${SIM_MODE:-$MODE}"
  BACKEND="${SIM_BACKEND:-$BACKEND}"
  STEPS="${SIM_STEPS:-$STEPS}"
  OUTPUT_EVERY="${SIM_OUTPUT_EVERY:-$OUTPUT_EVERY}"
  GPU_INIT_RETRIES="${SIM_GPU_INIT_RETRIES:-$GPU_INIT_RETRIES}"
  PURPOSE="${SIM_RUN_PURPOSE:-$PURPOSE}"
  TAG="${SIM_RUN_TAG:-$TAG}"
  OUT_DIR="${SIM_OUT_DIR:-$OUT_DIR}"
fi

# Pass 2: apply all CLI options (override run.env/defaults).
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --backend) BACKEND="${2:-}"; shift 2 ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    --output-every) OUTPUT_EVERY="${2:-}"; shift 2 ;;
    --gpu-init-retries) GPU_INIT_RETRIES="${2:-}"; shift 2 ;;
    --purpose) PURPOSE="${2:-}"; shift 2 ;;
    --tag) TAG="${2:-}"; shift 2 ;;
    --out) OUT_DIR="${2:-}"; shift 2 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$MODEL_DIR" != /* ]]; then
  MODEL_DIR="$(pwd)/$MODEL_DIR"
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 2
fi
if [[ ! -f "$MODEL_DIR/model.yaml" ]]; then
  echo "model.yaml not found in $MODEL_DIR" >&2
  exit 2
fi

if [[ "$MODE" != "debug" && "$MODE" != "release" ]]; then
  echo "Invalid mode: $MODE (debug|release)" >&2
  exit 2
fi
if [[ "$BACKEND" != "cpu" && "$BACKEND" != "gpu" ]]; then
  echo "Invalid backend: $BACKEND (cpu|gpu)" >&2
  exit 2
fi
if ! [[ "$GPU_INIT_RETRIES" =~ ^[0-9]+$ ]]; then
  echo "Invalid --gpu-init-retries: $GPU_INIT_RETRIES (non-negative integer required)" >&2
  exit 2
fi
if [[ "$PURPOSE" != "adhoc" && "$PURPOSE" != "benchmark" && "$PURPOSE" != "ml-data" ]]; then
  echo "Invalid --purpose: $PURPOSE (adhoc|benchmark|ml-data)" >&2
  exit 2
fi

BIN_FLAVOR="cpu"
if [[ "$BACKEND" == "gpu" ]]; then
  BIN_FLAVOR="cuda"
fi
BIN="$ROOT_DIR/core-cpp/build/${MODE}-${BIN_FLAVOR}/sim_run"

if [[ ! -x "$BIN" ]]; then
  if [[ "$BACKEND" == "cpu" ]]; then
    "$ROOT_DIR/tools/compile_tool.sh" --mode "$MODE" --cuda off >/dev/null
  else
    "$ROOT_DIR/tools/compile_tool.sh" --mode "$MODE" --cuda on >/dev/null
  fi
fi

if [[ ! -x "$BIN" ]]; then
  echo "sim_run binary not found at: $BIN" >&2
  exit 2
fi

if [[ "$OUT_DIR" == "auto" ]]; then
  model_name="$(sanitize_component "$(basename "$MODEL_DIR")")"
  run_tag=""
  if [[ -n "$TAG" ]]; then
    run_tag="__$(sanitize_component "$TAG")"
  fi
  RUN_ID="$(date +%Y%m%d_%H%M%S_%3N)__${model_name}__${BACKEND}__n${STEPS}__oe${OUTPUT_EVERY}${run_tag}"
  OUT_DIR="$MODEL_DIR/outputs/$PURPOSE/$RUN_ID"
else
  case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$MODEL_DIR/$OUT_DIR" ;;
  esac
fi

echo "Running sim_run"
echo "  mode=$MODE backend=$BACKEND steps=$STEPS output_every=$OUTPUT_EVERY"
echo "  purpose=$PURPOSE tag=${TAG:-none}"
echo "  case=$MODEL_DIR/model.yaml"
echo "  out=$OUT_DIR"

set +e
attempt=0
max_attempts=1
if [[ "$BACKEND" == "gpu" ]]; then
  max_attempts=$((GPU_INIT_RETRIES + 1))
fi
while true; do
  attempt=$((attempt + 1))
  "$BIN" \
    --case "$MODEL_DIR/model.yaml" \
    --backend "$BACKEND" \
    --steps "$STEPS" \
    --output-every "$OUTPUT_EVERY" \
    --out "$OUT_DIR"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    break
  fi
  if [[ "$BACKEND" != "gpu" || $rc -ne 6 || $attempt -ge $max_attempts ]]; then
    break
  fi
  echo "GPU init retry $attempt/$GPU_INIT_RETRIES after rc=$rc ..."
  sleep 1
done
set -e
if [[ $rc -eq 0 ]]; then
  echo "Run completed successfully."
  echo "Output directory: $OUT_DIR"
else
  if [[ -d "$OUT_DIR" ]] && [[ -z "$(ls -A "$OUT_DIR" 2>/dev/null)" ]]; then
    rmdir "$OUT_DIR" || true
  fi
  echo "Run failed with exit code: $rc" >&2
  echo "Output directory: $OUT_DIR" >&2
fi
exit $rc
