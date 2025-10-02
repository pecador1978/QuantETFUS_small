#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "[ERR] Usage: run_with_caffeinate.sh /absolute/path/to/script.sh [args...]"
  exit 2
fi
SCRIPT="$1"; shift || true
# Keep the Mac awake while the child script runs
/usr/bin/caffeinate -dimsu bash -lc "$SCRIPT ${*:-}"
