#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASES_DIR="$ROOT_DIR/cases"

TARGET_MODEL=""
CLEAN_ALL=0
KEEP=5
APPLY=0
BUCKETS_CSV=""
declare -a TARGET_BUCKETS=()

usage() {
  cat <<'EOF'
Usage:
  tools/clean_outputs.sh [--model <name> | --all] [--bucket <name>[,<name>...]] [--keep <N>] [--apply]

Description:
  Cleans old run folders under cases/*/outputs/.
  Keeps the newest N runs per output bucket (for example: adhoc, benchmark, ml-data, legacy).
  Default mode is dry-run (no deletion). Use --apply to delete.

Options:
  --model <name>   Clean one model (for example: model1)
  --all            Clean all models under cases/
  --bucket <name>  Clean only selected buckets (adhoc, benchmark, ml-data, legacy).
                   Accepts comma-separated values and can be repeated.
  --keep <N>       Keep newest N output folders per model (default: 5)
  --apply          Perform deletion (without this, script only prints plan)
  -h, --help       Show help

Examples:
  tools/clean_outputs.sh --model model1
  tools/clean_outputs.sh --model model1 --bucket adhoc --keep 0 --apply
  tools/clean_outputs.sh --model model1 --bucket benchmark,ml-data --keep 3 --apply
  tools/clean_outputs.sh --model model1 --keep 3 --apply
  tools/clean_outputs.sh --all --keep 2 --apply
EOF
}

bucket_is_selected() {
  local bucket="$1"
  if [[ "${#TARGET_BUCKETS[@]}" -eq 0 ]]; then
    return 0
  fi
  local selected
  for selected in "${TARGET_BUCKETS[@]}"; do
    if [[ "$selected" == "$bucket" ]]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      TARGET_MODEL="${2:-}"; shift 2 ;;
    --all)
      CLEAN_ALL=1; shift ;;
    --bucket)
      BUCKETS_CSV="${BUCKETS_CSV},${2:-}"
      shift 2 ;;
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

if [[ -n "$BUCKETS_CSV" ]]; then
  BUCKETS_CSV="${BUCKETS_CSV#,}"
  IFS=',' read -r -a raw_buckets <<< "$BUCKETS_CSV"
  for b in "${raw_buckets[@]}"; do
    b="${b#"${b%%[![:space:]]*}"}"
    b="${b%"${b##*[![:space:]]}"}"
    [[ -z "$b" ]] && continue
    case "$b" in
      adhoc|benchmark|ml-data|legacy) TARGET_BUCKETS+=("$b") ;;
      *)
        echo "Unknown --bucket value: $b (expected adhoc|benchmark|ml-data|legacy)" >&2
        exit 2
        ;;
    esac
  done
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

  mapfile -t buckets < <(
    find "$outputs_dir" -mindepth 2 -maxdepth 3 -type f -name meta.json -printf '%h\n' |
      while IFS= read -r run_dir; do
        rel="${run_dir#$outputs_dir/}"
        bucket="${rel%%/*}"
        if [[ "$rel" == "$bucket" ]]; then
          bucket="legacy"
        fi
        printf '%s\n' "$bucket"
      done | sort -u
  )

  if [[ "${#buckets[@]}" -eq 0 ]]; then
    echo "[skip] no runs found: $outputs_dir"
    return 0
  fi

  for bucket in "${buckets[@]}"; do
    if ! bucket_is_selected "$bucket"; then
      continue
    fi
    if [[ "$bucket" == "legacy" ]]; then
      mapfile -t runs < <(
        find "$outputs_dir" -mindepth 2 -maxdepth 2 -type f -name meta.json -printf '%T@ %h\n' |
          while IFS= read -r line; do
            run_dir="${line#* }"
            rel="${run_dir#$outputs_dir/}"
            if [[ "$rel" != */* ]]; then
              printf '%s\n' "$line"
            fi
          done | sort -nr | awk '{print $2}'
      )
    else
      mapfile -t runs < <(find "$outputs_dir/$bucket" -mindepth 2 -maxdepth 2 -type f -name meta.json -printf '%T@ %h\n' | sort -nr | awk '{print $2}')
    fi
    local total="${#runs[@]}"
    if [[ "$total" -le "$KEEP" ]]; then
      echo "[ok] $model_name/$bucket: total=$total keep=$KEEP remove=0"
      continue
    fi

    echo "[plan] $model_name/$bucket: total=$total keep=$KEEP remove=$((total - KEEP))"
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
  done

  if [[ "${#TARGET_BUCKETS[@]}" -gt 0 ]]; then
    local requested_bucket
    for requested_bucket in "${TARGET_BUCKETS[@]}"; do
      local found_bucket=0
      local existing_bucket
      for existing_bucket in "${buckets[@]}"; do
        if [[ "$existing_bucket" == "$requested_bucket" ]]; then
          found_bucket=1
          break
        fi
      done
      if [[ "$found_bucket" -eq 1 ]]; then
        continue
      fi
      echo "[skip] $model_name/$requested_bucket: no runs found"
    done
  fi

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
