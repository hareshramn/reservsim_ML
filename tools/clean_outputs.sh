#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASES_DIR="$ROOT_DIR/cases"

TARGET_MODEL=""
CLEAN_ALL=0
KEEP=5
APPLY=0

usage() {
  cat <<'EOF'
Usage:
  tools/clean_outputs.sh [--model <name> | --all] [--keep <N>] [--apply]

Description:
  Cleans old run folders under cases/*/outputs/.
  Default mode is dry-run (no deletion). Use --apply to delete.

Options:
  --model <name>   Clean one model (for example: model1)
  --all            Clean all models under cases/
  --keep <N>       Keep newest N output folders per model (default: 5)
  --apply          Perform deletion (without this, script only prints plan)
  -h, --help       Show help

Examples:
  tools/clean_outputs.sh --model model1
  tools/clean_outputs.sh --model model1 --keep 3 --apply
  tools/clean_outputs.sh --all --keep 2 --apply
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      TARGET_MODEL="${2:-}"; shift 2 ;;
    --all)
      CLEAN_ALL=1; shift ;;
    --keep)
      KEEP="${2:-}"; shift 2 ;;
    --apply)
      APPLY=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -n "$TARGET_MODEL" && "$CLEAN_ALL" -eq 1 ]]; then
  echo "Use only one of --model or --all." >&2
  exit 2
fi

if [[ -z "$TARGET_MODEL" && "$CLEAN_ALL" -eq 0 ]]; then
  echo "Provide --model <name> or --all." >&2
  exit 2
fi

if ! [[ "$KEEP" =~ ^[0-9]+$ ]]; then
  echo "--keep must be a non-negative integer." >&2
  exit 2
fi

collect_models() {
  if [[ "$CLEAN_ALL" -eq 1 ]]; then
    find "$CASES_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
  else
    printf '%s\n' "$TARGET_MODEL"
  fi
}

clean_model_outputs() {
  local model_name="$1"
  local model_dir="$CASES_DIR/$model_name"
  local outputs_dir="$model_dir/outputs"
  local removed=0

  if [[ ! -d "$model_dir" ]]; then
    echo "[skip] model not found: $model_name"
    return 0
  fi
  if [[ ! -d "$outputs_dir" ]]; then
    echo "[skip] no outputs dir: $outputs_dir"
    return 0
  fi

  mapfile -t runs < <(find "$outputs_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | awk '{print $2}')
  local total="${#runs[@]}"

  if [[ "$total" -le "$KEEP" ]]; then
    echo "[ok] $model_name: total=$total keep=$KEEP remove=0"
    return 0
  fi

  echo "[plan] $model_name: total=$total keep=$KEEP remove=$((total - KEEP))"
  local idx=0
  for run_dir in "${runs[@]}"; do
    idx=$((idx + 1))
    if [[ "$idx" -le "$KEEP" ]]; then
      continue
    fi
    echo "  - $run_dir"
    if [[ "$APPLY" -eq 1 ]]; then
      rm -rf "$run_dir"
      removed=$((removed + 1))
    fi
  done

  if [[ "$APPLY" -eq 1 ]]; then
    echo "[done] $model_name: removed=$removed"
  fi
}

if [[ "$APPLY" -eq 0 ]]; then
  echo "Dry-run mode. No folders will be deleted. Use --apply to execute."
fi

while IFS= read -r model; do
  [[ -z "$model" ]] && continue
  clean_model_outputs "$model"
done < <(collect_models)
