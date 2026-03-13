#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

MODEL_DIR=""
PLAN_FILE=""
MODE="release"
BACKEND="cpu"
STEPS="200"
OUTPUT_EVERY="1"
GPU_INIT_RETRIES="0"
KEEP_TEMP=0

usage() {
  cat <<'USAGE'
Usage:
  tools/ml_data_generate.sh --model-dir <path> [options]

Options:
  --plan <csv>               Candidate CSV file (default: <model-dir>/ml_scenarios.csv)
  --mode <debug|release>     Simulator mode for generated runs (default: release)
  --backend <cpu|gpu>        Backend for generated runs (default: cpu)
  --steps <N>                Steps per run (default: 200)
  --output-every <N>         Output frequency (default: 1)
  --gpu-init-retries <N>     GPU init retries (default: 0)
  --keep-temp                Keep temporary generated YAML files for inspection

Scenario CSV contract:
  - First non-comment line is header.
  - One required column: tag
  - Other columns are YAML keys to override.
  - Nested keys use dot notation, for example: rock.permeability_md
  - Blank values keep base model.yaml values.

Example:
  tag,injector_rate_stb_day,producer_bhp_psi,rock.permeability_md
  candidate-a,80.0,2800.0,100.0
  candidate-b,120.0,2600.0,150.0
USAGE
}

trim() {
  local v="$1"
  v="${v#"${v%%[![:space:]]*}"}"
  v="${v%"${v##*[![:space:]]}"}"
  printf '%s\n' "$v"
}

sanitize_component() {
  local raw="$1"
  local sanitized
  sanitized="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
  if [[ -z "$sanitized" ]]; then
    sanitized="scenario"
  fi
  printf '%s\n' "$sanitized"
}

replace_yaml_key() {
  local file="$1"
  local key_path="$2"
  local value="$3"
  local tmp
  tmp="$(mktemp)"

  if [[ "$key_path" == *.* ]]; then
    local section="${key_path%%.*}"
    local key="${key_path#*.}"
    if ! awk -v section="$section" -v key="$key" -v value="$value" '
      BEGIN {
        in_section = 0
        found = 0
      }
      /^[^[:space:]]/ {
        if ($0 ~ ("^" section ":[[:space:]]*$")) {
          in_section = 1
          print
          next
        }
        in_section = 0
      }
      {
        if (in_section && $0 ~ ("^[[:space:]]*" key ":[[:space:]]*")) {
          print "  " key ": " value
          found = 1
          next
        }
        print
      }
      END {
        if (!found) {
          exit 17
        }
      }
    ' "$file" > "$tmp"; then
      rm -f "$tmp"
      echo "Failed to apply override (missing key): $key_path in $file" >&2
      exit 2
    fi
  else
    local key="$key_path"
    if ! awk -v key="$key" -v value="$value" '
      BEGIN { found = 0 }
      {
        if ($0 ~ ("^" key ":[[:space:]]*")) {
          print key ": " value
          found = 1
          next
        }
        print
      }
      END {
        if (!found) {
          exit 17
        }
      }
    ' "$file" > "$tmp"; then
      rm -f "$tmp"
      echo "Failed to apply override (missing key): $key_path in $file" >&2
      exit 2
    fi
  fi

  mv "$tmp" "$file"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="${2:-}"; shift 2 ;;
    --plan) PLAN_FILE="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --backend) BACKEND="${2:-}"; shift 2 ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    --output-every) OUTPUT_EVERY="${2:-}"; shift 2 ;;
    --gpu-init-retries) GPU_INIT_RETRIES="${2:-}"; shift 2 ;;
    --keep-temp) KEEP_TEMP=1; shift ;;
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

if [[ -z "$MODEL_DIR" ]]; then
  echo "Missing required argument: --model-dir <path>" >&2
  exit 2
fi

if [[ "$MODEL_DIR" != /* ]]; then
  MODEL_DIR="$(pwd)/$MODEL_DIR"
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 2
fi

BASE_YAML="$MODEL_DIR/model.yaml"
HISTORY_CONTROLS="$MODEL_DIR/history_controls.csv"
HISTORY_OBSERVATIONS="$MODEL_DIR/history_observations.csv"
if [[ ! -f "$BASE_YAML" ]]; then
  echo "Base model file not found: $BASE_YAML" >&2
  exit 2
fi
if [[ ! -f "$HISTORY_CONTROLS" || ! -f "$HISTORY_OBSERVATIONS" ]]; then
  echo "History CSV files not found in model directory: $MODEL_DIR" >&2
  exit 2
fi

if [[ -z "$PLAN_FILE" ]]; then
  PLAN_FILE="$MODEL_DIR/ml_scenarios.csv"
elif [[ "$PLAN_FILE" != /* ]]; then
  PLAN_FILE="$(pwd)/$PLAN_FILE"
fi
if [[ ! -f "$PLAN_FILE" ]]; then
  echo "Scenario file not found: $PLAN_FILE" >&2
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

model_name="$(basename "$MODEL_DIR")"
tmp_root="$MODEL_DIR/.tmp_ml_cases_$$"
mkdir -p "$tmp_root"

cleanup() {
  if [[ "$KEEP_TEMP" -eq 0 && -d "$tmp_root" ]]; then
    rm -rf "$tmp_root"
  fi
}
trap cleanup EXIT

header_line=""
while IFS= read -r line || [[ -n "$line" ]]; do
  stripped="$(trim "$line")"
  if [[ -z "$stripped" || "${stripped:0:1}" == "#" ]]; then
    continue
  fi
  header_line="$line"
  break
done < "$PLAN_FILE"

if [[ -z "$header_line" ]]; then
  echo "Scenario file has no header: $PLAN_FILE" >&2
  exit 2
fi

IFS=',' read -r -a headers <<< "$header_line"
for i in "${!headers[@]}"; do
  headers[$i]="$(trim "${headers[$i]}")"
done

tag_idx=-1
case_name_idx=-1
for i in "${!headers[@]}"; do
  if [[ "${headers[$i]}" == "tag" ]]; then
    tag_idx=$i
  fi
  if [[ "${headers[$i]}" == "case_name" ]]; then
    case_name_idx=$i
  fi
done

if [[ "$tag_idx" -lt 0 ]]; then
  echo "Scenario header must include a 'tag' column: $PLAN_FILE" >&2
  exit 2
fi

base_case_name="$(sed -n 's/^case_name:[[:space:]]*//p' "$BASE_YAML" | head -n 1 | sed -E "s/^['\"]//; s/['\"]$//")"
if [[ -z "$base_case_name" ]]; then
  base_case_name="$model_name"
fi

row_num=0
ran_count=0
while IFS= read -r line || [[ -n "$line" ]]; do
  stripped="$(trim "$line")"
  if [[ -z "$stripped" || "${stripped:0:1}" == "#" ]]; then
    continue
  fi
  row_num=$((row_num + 1))
  if [[ "$row_num" -eq 1 ]]; then
    continue
  fi

  IFS=',' read -r -a values <<< "$line"
  tag_raw="${values[$tag_idx]:-}"
  tag="$(sanitize_component "$(trim "$tag_raw")")"
  if [[ -z "$tag" ]]; then
    echo "Skipping row $row_num with empty tag." >&2
    continue
  fi

  tmp_yaml="$tmp_root/model_${tag}.yaml"
  cp "$BASE_YAML" "$tmp_yaml"
  replace_yaml_key "$tmp_yaml" "history.controls_csv" "\"$HISTORY_CONTROLS\""
  replace_yaml_key "$tmp_yaml" "history.observations_csv" "\"$HISTORY_OBSERVATIONS\""

  if [[ "$case_name_idx" -lt 0 ]]; then
    replace_yaml_key "$tmp_yaml" "case_name" "\"${base_case_name}__${tag}\""
  fi

  for i in "${!headers[@]}"; do
    key="${headers[$i]}"
    if [[ "$key" == "tag" ]]; then
      continue
    fi
    value_raw="${values[$i]:-}"
    value="$(trim "$value_raw")"
    if [[ -z "$value" ]]; then
      continue
    fi
    replace_yaml_key "$tmp_yaml" "$key" "$value"
  done

  echo "[run] model=$model_name tag=$tag case=$tmp_yaml"
  set +e
  run_output="$(RESERV_OUTPUT_BUCKET="ml-data" "$ROOT_DIR/tools/history_run.sh" \
    --model-dir "$MODEL_DIR" \
    --mode "$MODE" \
    --backend "$BACKEND" \
    --steps "$STEPS" \
    --output-every "$OUTPUT_EVERY" \
    --gpu-init-retries "$GPU_INIT_RETRIES" \
    --tag "$tag" \
    --case-file "$tmp_yaml" 2>&1)"
  rc=$?
  set -e
  printf '%s\n' "$run_output"
  if [[ $rc -ne 0 ]]; then
    echo "Failed ML data run for tag=$tag (rc=$rc)" >&2
    exit $rc
  fi

  out_dir="$(printf '%s\n' "$run_output" | awk -F': ' '/^Output directory:/ {print $2}' | tail -n 1)"
  if [[ -z "$out_dir" || ! -d "$out_dir" ]]; then
    echo "Could not determine output directory for tag=$tag" >&2
    exit 2
  fi
  cp "$tmp_yaml" "$out_dir/case_input.yaml"
  ran_count=$((ran_count + 1))
done < "$PLAN_FILE"

"$PYTHON_BIN" "$ROOT_DIR/python/ml/build_history_match_dataset.py" \
  --runs-root "$MODEL_DIR/outputs/ml-data" \
  --plan "$PLAN_FILE" \
  --out "$MODEL_DIR/outputs/ml-data/history_ml_dataset.csv"

echo "[done] generated_ml_runs=$ran_count plan=$PLAN_FILE dataset=$MODEL_DIR/outputs/ml-data/history_ml_dataset.csv"
if [[ "$KEEP_TEMP" -eq 1 ]]; then
  echo "[keep-temp] $tmp_root"
fi
