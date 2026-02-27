#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_DIR="$ROOT_DIR/core-cpp"

MODE="debug"
CUDA_MODE="auto"
CLEAN=0
RUN_TESTS=0
BUILD_DIR=""

usage() {
  cat <<'EOF'
Usage:
  tools/compile_tool.sh [options]

Options:
  --mode <debug|release>     Build profile (default: debug)
  --cuda <auto|on|off>       CUDA toggle (default: auto)
  --clean                    Remove build dir before configuring
  --tests                    Run tests after build
  --build-dir <path>         Override build directory
  -h, --help                 Show help

Examples:
  tools/compile_tool.sh --mode debug --cuda off
  tools/compile_tool.sh --mode release --cuda on --tests
  tools/compile_tool.sh --mode release --cuda auto --clean
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"; shift 2 ;;
    --cuda)
      CUDA_MODE="${2:-}"; shift 2 ;;
    --clean)
      CLEAN=1; shift ;;
    --tests)
      RUN_TESTS=1; shift ;;
    --build-dir)
      BUILD_DIR="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ "$MODE" != "debug" && "$MODE" != "release" ]]; then
  echo "Invalid --mode: $MODE (expected debug|release)" >&2
  exit 2
fi

if [[ "$CUDA_MODE" != "auto" && "$CUDA_MODE" != "on" && "$CUDA_MODE" != "off" ]]; then
  echo "Invalid --cuda: $CUDA_MODE (expected auto|on|off)" >&2
  exit 2
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required but not found in PATH." >&2
  exit 127
fi

BUILD_TYPE="Debug"
if [[ "$MODE" == "release" ]]; then
  BUILD_TYPE="Release"
fi

CUDA_ENABLED="OFF"
if [[ "$CUDA_MODE" == "on" ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "CUDA requested (--cuda on), but nvcc was not found." >&2
    exit 3
  fi
  CUDA_ENABLED="ON"
elif [[ "$CUDA_MODE" == "auto" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_ENABLED="ON"
  fi
fi

if [[ -z "$BUILD_DIR" ]]; then
  suffix="cpu"
  if [[ "$CUDA_ENABLED" == "ON" ]]; then
    suffix="cuda"
  fi
  BUILD_DIR="$CORE_DIR/build/${MODE}-${suffix}"
fi

if [[ "$CLEAN" -eq 1 ]]; then
  rm -rf "$BUILD_DIR"
fi

echo "Configuring:"
echo "  mode=$MODE"
echo "  build_type=$BUILD_TYPE"
echo "  cuda=$CUDA_ENABLED"
echo "  build_dir=$BUILD_DIR"

cmake -S "$CORE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DRESERV_ENABLE_CUDA="$CUDA_ENABLED"

cmake --build "$BUILD_DIR" -j

echo "Build complete."
echo "sim_run: $BUILD_DIR/sim_run"
echo "config_parser_tests: $BUILD_DIR/config_parser_tests"

if [[ "$RUN_TESTS" -eq 1 ]]; then
  echo "Running tests..."
  RESERV_ML_REPO_ROOT="$ROOT_DIR" ctest --test-dir "$BUILD_DIR" --output-on-failure
fi
