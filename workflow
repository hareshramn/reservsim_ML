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

  gpu-check --model <modelN> [gpu-check options]
      Probe CUDA readiness with a tiny GPU run.

  run --model <modelN> [run options]
      Run one model from repo root.
      Run options: --mode --backend --steps --output-every --seed --out --gpu-init-retries --purpose --tag

  plot --model <modelN> [--run <run_id_or_path>] [--out <dir>] [--check-only]
      Plot a run; if --run is omitted, latest run under cases/<model>/outputs is used.

  validate --run <run_id_or_path>
      Validate required run artifacts and schema contracts.

  parity --model <modelN> --seed <N> [--cpu-run <id_or_path>] [--gpu-run <id_or_path>]
      Compare latest CPU/GPU artifacts and report parity metrics.

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
        --seed <N>                    (default: 7)
        --gpu-init-retries <N>        (default: 0)
        --purpose <adhoc|benchmark|ml-data> (default: adhoc)
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
  ./workflow gpu-check --model model1 --mode release --seed 7
  ./workflow run --model model1 --steps 10 --mode release
  ./workflow validate --run cases/model1/outputs/ml-data/<run_id>
  ./workflow parity --model model1 --seed 7
  ./workflow bench --model model1 --seeds 1,2,3 --steps 50
  ./workflow plot --model model1
  ./workflow clean --model model1 --keep 3 --apply
  ./workflow all --model model1 --steps 10 --seed 7
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
    seed="7"
    gpu_init_retries="0"
    purpose="adhoc"
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
        --seed) seed="${2:-}"; shift 2 ;;
        --gpu-init-retries) gpu_init_retries="${2:-}"; shift 2 ;;
        --purpose) purpose="${2:-}"; shift 2 ;;
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
      --seed "$seed" \
      --gpu-init-retries "$gpu_init_retries" \
      --purpose "$purpose" \
      --tag "$tag" \
      --out "$out"

    run_arg=""
    resolved_out="$(resolve_out_dir "$model_dir" "$out")"
    if [[ "$resolved_out" == "auto" ]]; then
      run_arg="$(latest_run_dir_for_model "$model" "$purpose")"
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
