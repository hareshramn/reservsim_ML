#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

status_ok=0
status_warn=0
status_fail=0

check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    echo "[ok] command: $name ($(command -v "$name"))"
    status_ok=$((status_ok + 1))
  else
    echo "[fail] missing command: $name"
    status_fail=$((status_fail + 1))
  fi
}

echo "Doctor check: $ROOT_DIR"
check_cmd cmake
check_cmd bash

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "[ok] python: $ROOT_DIR/.venv/bin/python"
  status_ok=$((status_ok + 1))
  set +e
  "$ROOT_DIR/.venv/bin/python" - <<'PY'
mods = ["numpy", "matplotlib"]
missing = []
for mod in mods:
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
if missing:
    print("MISSING:" + ",".join(missing))
    raise SystemExit(2)
print("PY_DEPS_OK")
PY
  py_rc=$?
  set -e
  if [[ $py_rc -eq 0 ]]; then
    echo "[ok] .venv Python deps: numpy, matplotlib"
    status_ok=$((status_ok + 1))
  else
    echo "[warn] .venv missing one or more deps (numpy/matplotlib)"
    status_warn=$((status_warn + 1))
  fi
else
  echo "[warn] missing .venv Python: $ROOT_DIR/.venv/bin/python"
  status_warn=$((status_warn + 1))
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "[ok] nvcc: $(command -v nvcc)"
  status_ok=$((status_ok + 1))
else
  echo "[warn] nvcc not found (GPU compile path unavailable)"
  status_warn=$((status_warn + 1))
fi

if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  echo "[ok] nvidia-smi: /usr/lib/wsl/lib/nvidia-smi"
  status_ok=$((status_ok + 1))
elif command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ok] nvidia-smi: $(command -v nvidia-smi)"
  status_ok=$((status_ok + 1))
else
  echo "[warn] nvidia-smi not found (GPU runtime visibility unknown)"
  status_warn=$((status_warn + 1))
fi

echo "Summary: ok=$status_ok warn=$status_warn fail=$status_fail"
if [[ $status_fail -gt 0 ]]; then
  exit 1
fi
