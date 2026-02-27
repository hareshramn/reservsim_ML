#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$MODEL_DIR/model.yaml" ]]; then
  echo "model.yaml not found in $MODEL_DIR" >&2
  exit 2
fi

MODE="release"
BACKEND="cpu"
STEPS="10"
SEED="7"
OUT_DIR="auto"

if [[ -f "$MODEL_DIR/run.env" ]]; then
  # shellcheck disable=SC1091
  source "$MODEL_DIR/run.env"
  MODE="${SIM_MODE:-$MODE}"
  BACKEND="${SIM_BACKEND:-$BACKEND}"
  STEPS="${SIM_STEPS:-$STEPS}"
  SEED="${SIM_SEED:-$SEED}"
  OUT_DIR="${SIM_OUT_DIR:-$OUT_DIR}"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --backend) BACKEND="${2:-}"; shift 2 ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --out) OUT_DIR="${2:-}"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage:
  ./run [--mode debug|release] [--backend cpu|gpu] [--steps N] [--seed N] [--out path]

Defaults come from run.env in this model folder.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "debug" && "$MODE" != "release" ]]; then
  echo "Invalid mode: $MODE (debug|release)" >&2
  exit 2
fi
if [[ "$BACKEND" != "cpu" && "$BACKEND" != "gpu" ]]; then
  echo "Invalid backend: $BACKEND (cpu|gpu)" >&2
  exit 2
fi

BIN_FLAVOR="cpu"
if [[ "$BACKEND" == "gpu" ]]; then
  BIN_FLAVOR="cuda"
fi
BIN="$ROOT_DIR/core-cpp/build/${MODE}-${BIN_FLAVOR}/sim_run"

if [[ ! -x "$BIN" ]]; then
  if [[ "$BACKEND" == "cpu" ]]; then
    "$ROOT_DIR/build" "$MODE" >/dev/null
  else
    "$ROOT_DIR/tools/mcp_tools.sh" --mode "$MODE" --cuda on >/dev/null
  fi
fi

if [[ ! -x "$BIN" ]]; then
  echo "sim_run binary not found at: $BIN" >&2
  exit 2
fi

if [[ "$OUT_DIR" == "auto" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)_${BACKEND}_${SEED}"
  OUT_DIR="$MODEL_DIR/outputs/$RUN_ID"
else
  case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$MODEL_DIR/$OUT_DIR" ;;
  esac
fi
mkdir -p "$OUT_DIR"

echo "Running sim_run"
echo "  mode=$MODE backend=$BACKEND steps=$STEPS seed=$SEED"
echo "  case=$MODEL_DIR/model.yaml"
echo "  out=$OUT_DIR"

"$BIN" \
  --case "$MODEL_DIR/model.yaml" \
  --backend "$BACKEND" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --out "$OUT_DIR"

rc=$?
if [[ $rc -eq 0 ]]; then
  echo "Run completed successfully."
  echo "Output directory: $OUT_DIR"
else
  echo "Run failed with exit code: $rc" >&2
  echo "Output directory: $OUT_DIR" >&2
fi
exit $rc
