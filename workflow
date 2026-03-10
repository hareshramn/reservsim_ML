#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  ./workflow <command> [options]

Commands:
  compile|build [compile options]
      Delegate to tools/compile_tool.sh.

  doctor
      Check local toolchain and Python deps needed by this repo.

  ui
      Launch a local GUI for selecting workflow mode and command arguments.

  web-ui [--host 127.0.0.1] [--port 8765]
      Launch a browser-based local UI for selecting mode and arguments.

  gpu-check --model <modelN> [gpu-check options]
      Probe CUDA readiness with a tiny GPU run.

  run --model <modelN> [run options]
      Run one model from repo root.
      Run options: --mode --backend --steps --output-every --out --gpu-init-retries --tag --case-file

  ml-data-gen --model <modelN> [ml-data-gen options]
      Generate ML-data runs from temporary case YAML variants.
      ml-data-gen options: --plan --mode --backend --steps --output-every --gpu-init-retries --keep-temp

  ml-check --model <modelN> [ml-check options]
      Run ml-data-gen, validate latest ml-data run, parity, and bench in sequence.
      ml-check options:
        --plan --mode --backend --steps --output-every --gpu-init-retries
        --bench-repeats --bench-steps --bench-output-every
        --skip-parity --skip-bench

  plot --model <modelN> [--run <run_id_or_path>] [--out <dir>] [--check-only]
      Plot a run; if --run is omitted, latest run under cases/<model>/outputs is used.

  validate --run <run_id_or_path>
      Validate required run artifacts and schema contracts.

  parity --model <modelN> [--cpu-run <id_or_path>] [--gpu-run <id_or_path>]
      Compare latest CPU/GPU artifacts and report parity metrics (latest runs by default).

  bench --model <modelN> [bench options]
      Execute CPU/GPU matrix and write benchmarks/benchmark_summary.csv.

  clean [--model <modelN> | --all] [--keep <N>] [--apply]
      Clean output directories.

  all --model <modelN> [options]
      Compile, run, then plot in one command.
      Options:
        --mode <debug|release>        (default: release)
        --backend <cpu|gpu>           (default: cpu)
        --steps <N>                   (default: 10)
        --output-every <N>            (default: 1)
        --gpu-init-retries <N>        (default: 0)
        --tag <name>                  (optional run label)
        --out <path|auto>             (default: auto)
        --plot-out <dir>              (default: figs)
        --cuda <auto|on|off>          (default: backend-derived)
        --clean-build                 (compile with --clean)
        --tests                       (compile with --tests)
        --check-only                  (plot validation only)

Examples:
  ./workflow compile --mode debug --cuda off
  ./workflow doctor
  ./workflow ui
  ./workflow web-ui
  ./workflow gpu-check --model model1 --mode release
  ./workflow run --model model1 --steps 10 --mode release
  ./workflow ml-data-gen --model model1 --plan cases/model1/ml_scenarios.csv --steps 200
  ./workflow ml-check --model model1
  ./workflow validate --run cases/model1/outputs/ml-data/<run_id>
  ./workflow parity --model model1
  ./workflow bench --model model1 --repeats 3 --steps 50
  ./workflow plot --model model1
  ./workflow clean --model model1 --keep 3 --apply
  ./workflow all --model model1 --steps 10
USAGE
}

latest_run_dir_for_model() {
  local model_name="$1"
  local purpose="${2:-}"
  local outputs_dir="$ROOT_DIR/cases/$model_name/outputs"
  if [[ ! -d "$outputs_dir" ]]; then
    echo ""
    return 0
  fi
  if [[ -n "$purpose" ]]; then
    local purpose_dir="$outputs_dir/$purpose"
    if [[ ! -d "$purpose_dir" ]]; then
      echo ""
      return 0
    fi
    find "$purpose_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | awk 'NR==1{print $2}'
    return 0
  fi
  find "$outputs_dir" -mindepth 1 -maxdepth 2 -type f -name meta.json -printf '%T@ %h\n' | sort -nr | awk 'NR==1{print $2}'
}

resolve_model_dir() {
  local model_name="$1"
  local model_dir="$ROOT_DIR/cases/$model_name"
  if [[ ! -d "$model_dir" ]]; then
    echo "Model directory not found: $model_dir" >&2
    exit 2
  fi
  echo "$model_dir"
}

resolve_out_dir() {
  local model_dir="$1"
  local out_arg="$2"
  if [[ "$out_arg" == "auto" ]]; then
    echo "auto"
    return 0
  fi
  case "$out_arg" in
    /*) echo "$out_arg" ;;
    *) echo "$model_dir/$out_arg" ;;
  esac
}

cmd="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$cmd" in
  help|-h|--help)
    usage
    ;;

  compile|build)
    exec "$ROOT_DIR/tools/compile_tool.sh" "$@"
    ;;

  doctor)
    exec "$ROOT_DIR/tools/doctor.sh" "$@"
    ;;

  ui)
    if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
      exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/tools/workflow_gui.py" "$@"
    fi
    exec python3 "$ROOT_DIR/tools/workflow_gui.py" "$@"
    ;;

  web-ui)
    if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
      exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/tools/workflow_web_ui.py" "$@"
    fi
    exec python3 "$ROOT_DIR/tools/workflow_web_ui.py" "$@"
    ;;

  gpu-check)
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
      exec "$ROOT_DIR/tools/gpu_check.sh" --help
    fi
    model=""
    args=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --model)
          model="${2:-}"
          shift 2
          ;;
        *)
          args+=("$1")
          shift
          ;;
      esac
    done
    if [[ -z "$model" ]]; then
      echo "Missing required argument: --model <name>" >&2
      exit 2
    fi
    model_dir="$(resolve_model_dir "$model")"
    exec "$ROOT_DIR/tools/gpu_check.sh" --model-dir "$model_dir" "${args[@]}"
    ;;

  run)
    model=""
    run_args=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --model)
          model="${2:-}"
          shift 2
          ;;
        --purpose)
          echo "workflow run is adhoc-only; use 'workflow bench' for benchmark runs or 'workflow ml-data-gen' for ML data runs." >&2
          exit 2
          ;;
        *)
          run_args+=("$1")
          shift
          ;;
      esac
    done
    if [[ -z "$model" ]]; then
      echo "Missing required argument: --model <name>" >&2
      exit 2
    fi
    model_dir="$(resolve_model_dir "$model")"
    exec "$ROOT_DIR/tools/model_run.sh" --model-dir "$model_dir" "${run_args[@]}"
    ;;

  ml-data-gen)
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
      exec "$ROOT_DIR/tools/ml_data_generate.sh" --help
    fi
    model=""
    args=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --model)
          model="${2:-}"
          shift 2
          ;;
        *)
          args+=("$1")
          shift
          ;;
      esac
    done
    if [[ -z "$model" ]]; then
      echo "Missing required argument: --model <name>" >&2
      exit 2
    fi
    model_dir="$(resolve_model_dir "$model")"
    exec "$ROOT_DIR/tools/ml_data_generate.sh" --model-dir "$model_dir" "${args[@]}"
    ;;

  ml-check)
    model=""
    plan=""
    mode="release"
    backend="cpu"
    steps="200"
    output_every="1"
    gpu_init_retries="2"
    bench_repeats="3"
    bench_steps="50"
    bench_output_every="10"
    skip_parity=0
    skip_bench=0

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --model) model="${2:-}"; shift 2 ;;
        --plan) plan="${2:-}"; shift 2 ;;
        --mode) mode="${2:-}"; shift 2 ;;
        --backend) backend="${2:-}"; shift 2 ;;
        --steps) steps="${2:-}"; shift 2 ;;
        --output-every) output_every="${2:-}"; shift 2 ;;
        --gpu-init-retries) gpu_init_retries="${2:-}"; shift 2 ;;
        --bench-repeats) bench_repeats="${2:-}"; shift 2 ;;
        --bench-steps) bench_steps="${2:-}"; shift 2 ;;
        --bench-output-every) bench_output_every="${2:-}"; shift 2 ;;
        --skip-parity) skip_parity=1; shift ;;
        --skip-bench) skip_bench=1; shift ;;
        *)
          echo "Unknown argument for ml-check: $1" >&2
          exit 2
          ;;
      esac
    done

    if [[ -z "$model" ]]; then
      echo "Missing required argument: --model <name>" >&2
      exit 2
    fi
    model_dir="$(resolve_model_dir "$model")"

    ml_data_cmd=("$ROOT_DIR/tools/ml_data_generate.sh"
      --model-dir "$model_dir"
      --mode "$mode"
      --backend "$backend"
      --steps "$steps"
      --output-every "$output_every"
      --gpu-init-retries "$gpu_init_retries")
    if [[ -n "$plan" ]]; then
      ml_data_cmd+=(--plan "$plan")
    fi
    "${ml_data_cmd[@]}"

    latest_ml="$(latest_run_dir_for_model "$model" "ml-data")"
    if [[ -z "$latest_ml" ]]; then
      echo "No ml-data runs found after ml-data-gen for model: $model" >&2
      exit 2
    fi
    "$ROOT_DIR/tools/validate_run.py" --run "$latest_ml"
    if [[ "$skip_parity" -eq 0 ]]; then
      "$ROOT_DIR/tools/parity_report.py" --model "$model"
    fi
    if [[ "$skip_bench" -eq 0 ]]; then
      "$ROOT_DIR/tools/benchmark_matrix.py" \
        --model "$model" \
        --repeats "$bench_repeats" \
        --steps "$bench_steps" \
        --output-every "$bench_output_every" \
        --gpu-init-retries "$gpu_init_retries"
    fi
    ;;

  validate)
    exec "$ROOT_DIR/tools/validate_run.py" "$@"
    ;;

  parity)
    exec "$ROOT_DIR/tools/parity_report.py" "$@"
    ;;

  bench)
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
      exec "$ROOT_DIR/tools/benchmark_matrix.py" --help
    fi
    model=""
    args=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --model)
          model="${2:-}"
          shift 2
          ;;
        *)
          args+=("$1")
          shift
          ;;
      esac
    done
    if [[ -z "$model" ]]; then
      echo "Missing required argument: --model <name>" >&2
      exit 2
    fi
    exec "$ROOT_DIR/tools/benchmark_matrix.py" --model "$model" "${args[@]}"
    ;;

  plot)
    model=""
    run_arg=""
    out_arg="figs"
    check_only=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --model)
          model="${2:-}"
          shift 2
          ;;
        --run)
          run_arg="${2:-}"
          shift 2
          ;;
        --out)
          out_arg="${2:-}"
          shift 2
          ;;
        --check-only)
          check_only=1
          shift
          ;;
        *)
          echo "Unknown argument for plot: $1" >&2
          exit 2
          ;;
      esac
    done
    if [[ -z "$model" ]]; then
      echo "Missing required argument: --model <name>" >&2
      exit 2
    fi
    model_dir="$(resolve_model_dir "$model")"
    if [[ -z "$run_arg" ]]; then
      run_arg="$(latest_run_dir_for_model "$model")"
      if [[ -z "$run_arg" ]]; then
        echo "No runs found under $model_dir/outputs" >&2
        exit 2
      fi
      echo "Using latest run: $run_arg"
    fi
    plot_cmd=("$ROOT_DIR/tools/plot_run.sh" --run "$run_arg" --out "$out_arg")
    if [[ "$check_only" -eq 1 ]]; then
      plot_cmd+=(--check-only)
    fi
    exec "${plot_cmd[@]}"
    ;;

  clean)
    exec "$ROOT_DIR/tools/clean_outputs.sh" "$@"
    ;;

  all)
    model=""
    mode="release"
    backend="cpu"
    steps="10"
    output_every="1"
    gpu_init_retries="0"
    tag=""
    out="auto"
    plot_out="figs"
    cuda=""
    clean_build=0
    tests=0
    check_only=0

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --model) model="${2:-}"; shift 2 ;;
        --mode) mode="${2:-}"; shift 2 ;;
        --backend) backend="${2:-}"; shift 2 ;;
        --steps) steps="${2:-}"; shift 2 ;;
        --output-every) output_every="${2:-}"; shift 2 ;;
        --gpu-init-retries) gpu_init_retries="${2:-}"; shift 2 ;;
        --tag) tag="${2:-}"; shift 2 ;;
        --out) out="${2:-}"; shift 2 ;;
        --plot-out) plot_out="${2:-}"; shift 2 ;;
        --cuda) cuda="${2:-}"; shift 2 ;;
        --clean-build) clean_build=1; shift ;;
        --tests) tests=1; shift ;;
        --check-only) check_only=1; shift ;;
        *)
          echo "Unknown argument for all: $1" >&2
          exit 2
          ;;
      esac
    done

    if [[ -z "$model" ]]; then
      echo "Missing required argument: --model <name>" >&2
      exit 2
    fi
    if [[ "$mode" != "debug" && "$mode" != "release" ]]; then
      echo "Invalid mode: $mode (debug|release)" >&2
      exit 2
    fi
    if [[ "$backend" != "cpu" && "$backend" != "gpu" ]]; then
      echo "Invalid backend: $backend (cpu|gpu)" >&2
      exit 2
    fi
    if [[ -z "$cuda" ]]; then
      if [[ "$backend" == "gpu" ]]; then
        cuda="on"
      else
        cuda="off"
      fi
    fi

    compile_cmd=("$ROOT_DIR/tools/compile_tool.sh" --mode "$mode" --cuda "$cuda")
    if [[ "$clean_build" -eq 1 ]]; then
      compile_cmd+=(--clean)
    fi
    if [[ "$tests" -eq 1 ]]; then
      compile_cmd+=(--tests)
    fi
    "${compile_cmd[@]}"

    model_dir="$(resolve_model_dir "$model")"
    "$ROOT_DIR/tools/model_run.sh" \
      --model-dir "$model_dir" \
      --mode "$mode" \
      --backend "$backend" \
      --steps "$steps" \
      --output-every "$output_every" \
      --gpu-init-retries "$gpu_init_retries" \
      --tag "$tag" \
      --out "$out"

    run_arg=""
    resolved_out="$(resolve_out_dir "$model_dir" "$out")"
    if [[ "$resolved_out" == "auto" ]]; then
      run_arg="$(latest_run_dir_for_model "$model" "adhoc")"
      if [[ -z "$run_arg" ]]; then
        echo "Run completed, but no output directory was found for plotting." >&2
        exit 2
      fi
    else
      run_arg="$resolved_out"
    fi

    plot_cmd=("$ROOT_DIR/tools/plot_run.sh" --run "$run_arg" --out "$plot_out")
    if [[ "$check_only" -eq 1 ]]; then
      plot_cmd+=(--check-only)
    fi
    exec "${plot_cmd[@]}"
    ;;

  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 2
    ;;
esac
