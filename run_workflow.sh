#!/usr/bin/env bash
set -euo pipefail

# ---- basic paths ----
ROOT="/Users/Finance/QuantETFUS_small"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
SCRIPTS="$ROOT/scripts"
SHARED="/Users/Finance/QuantShared/scripts"           # optional, if you keep fetchers there
LOGDIR="$ROOT/logs"
SIGNALS_DIR="$ROOT/signals"

mkdir -p "$LOGDIR" "$SIGNALS_DIR"

# ---- env for the pipeline ----
export TARGET_BUCKET="${TARGET_BUCKET:-targeted_ETFs_US}"
export MARKET_TZ="${MARKET_TZ:-Europe/London}"

timestamp(){ date -u +"%Y-%m-%d %H:%M:%S UTC"; }
say(){ echo "[$(timestamp)] $*"; }

# nice error message if anything fails
trap 'say "[ERR] ${BASH_SOURCE[0]} failed at line $LINENO"; exit 1' ERR

# ---- venv check/activation ----
if [[ ! -x "$PY" ]]; then
  say "[ERR] Python venv not found at $PY"
  exit 1
fi
# Ensure correct env vars from venv (pip, libs, etc.)
source "$VENV/bin/activate"

# Prefer a script in $SCRIPTS; if not there, try $SHARED
run_any() {
  local label="$1"; shift
  local candidates=("$@")
  local found=""
  for f in "${candidates[@]}"; do
    if [[ -f "$f" ]]; then found="$f"; break; fi
  done
  if [[ -z "$found" ]]; then
    say "[SKIP] $label (no script found among: ${candidates[*]})"
    return 0
  fi
  say "$label ..."
  "$PYV" "$found" "$@"
}

# run a python file (single known path)
run_py() {
  local label="$1"; shift
  local file="$1"; shift || true
  if [[ -f "$file" ]]; then
    say "$label ..."
    "$PY" "$file" "$@"
  else
    say "[SKIP] $(basename "$file") not found."
  fi
}

say "=== RUN START ==="

# ---------------- FETCH (nightly full) ----------------
# If you want to skip all fetching from cron, set SKIP_FETCH=1 on the cron line.
if [[ "${SKIP_FETCH:-0}" != "1" ]]; then
  # daily (prefer local repo script, else shared)
  run_any "s10_fetch_daily.py"   "$SCRIPTS/s10_fetch_daily.py"   "$SHARED/s10_fetch_daily.py"
  # 30m
  run_any "s11_fetch_30min.py"   "$SCRIPTS/s11_fetch_30min.py"   "$SHARED/s11_fetch_30min.py"
  # weekly
  run_any "s12_fetch_weekly.py"  "$SCRIPTS/s12_fetch_weekly.py"  "$SHARED/s12_fetch_weekly.py"
else
  say "[INFO] SKIP_FETCH=1 → skipping all fetch steps (s10/s11/s12)."
fi

# ---------------- BUILD/DECIDE/RENDER ----------------
# s32 → enrich intraday features + attach daily context
run_py "s32_enrich_30m_intraday.py" "$SCRIPTS/s32_enrich_30m_intraday.py"

# s76 → Gate-1 rule signals
run_py "s76_rule_signals.py"        "$SCRIPTS/s76_rule_signals.py"

# s80 → HTML dashboard (also writes stable '1_signals_dashboard_latest.html')
run_py "s80_decision_board.py"      "$SCRIPTS/s80_decision_board.py"

DASH="$SIGNALS_DIR/1_signals_dashboard_latest.html"
[[ -f "$DASH" ]] && say "Dashboard → $DASH"

say "=== RUN END ==="
echo "[$(timestamp)] [DONE] Workflow finished successfully."