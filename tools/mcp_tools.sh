#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cmd="${1:-compile}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$cmd" in
  help|-h|--help)
    cat <<'EOF'
MCP-style custom tool wrapper

Usage:
  tools/mcp_tools.sh [compile options]
  tools/mcp_tools.sh compile [compile options]

Examples:
  tools/mcp_tools.sh --mode debug --cuda off
  tools/mcp_tools.sh compile --mode release --cuda on --tests
EOF
    exit 0
    ;;
  compile)
    exec "$ROOT_DIR/tools/compile_tool.sh" "$@"
    ;;
  *)
    exec "$ROOT_DIR/tools/compile_tool.sh" "$cmd" "$@"
    ;;
esac
