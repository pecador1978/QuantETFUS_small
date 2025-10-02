#!/usr/bin/env bash
set -euo pipefail

# -------- paths --------
ROOT="/Users/Finance/QuantETFUS_small"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
SCRIPTS="$ROOT/scripts"
SHARED="/Users/Finance/QuantShared/scripts"
LOGDIR="$ROOT/logs"
SIGNALS_DIR="$ROOT/signals"
CONFIG="$ROOT/config/gate1_v1_rules.json"

mkdir -p "$LOGDIR" "$SIGNALS_DIR"

# -------- env --------
export TARGET_BUCKET="${TARGET_BUCKET:-targeted_ETFs_US}"
export MARKET_TZ="${MARKET_TZ:-Europe/London}"

timestamp(){ date -u +"%Y-%m-%d %H:%M:%S UTC"; }
say(){ echo "[$(timestamp)] $*"; }
trap 'say "[ERR] ${BASH_SOURCE[0]} failed at line $LINENO"; exit 1' ERR

# -------- venv --------
if [[ ! -x "$PY" ]]; then
  say "[ERR] Python venv not found at $PY"
  exit 1
fi
source "$VENV/bin/activate"

# -------- helpers (robust) --------
run_any() {
  local label="$1"; shift
  local -a candidates=()
  while (($#)); do
    [[ "$1" == "--" ]] && shift && break
    candidates+=("$1"); shift
  done

  # collect the rest as args (may be empty)
  local -a args=()
  while (($#)); do args+=("$1"); shift; done

  local found=""
  for f in "${candidates[@]}"; do
    if [[ -f "$f" ]]; then found="$f"; break; fi
  done
  if [[ -z "$found" ]]; then
    say "[SKIP] $label (no script found among: ${candidates[*]})"
    return 0
  fi

  if ((${#args[@]})); then
    say "$label → $(basename "$found") ${args[*]}"
    "$PY" "$found" "${args[@]}"
  else
    say "$label → $(basename "$found")"
    "$PY" "$found"
  fi
}

run_py() {
  local label="$1"; shift
  local file="$1"; shift || true
  local -a args=()
  while (($#)); do args+=("$1"); shift; done

  if [[ -f "$file" ]]; then
    if ((${#args[@]})); then
      say "$label → $(basename "$file") ${args[*]}"
      "$PY" "$file" "${args[@]}"
    else
      say "$label → $(basename "$file")"
      "$PY" "$file"
    fi
  else
    say "[SKIP] $(basename "$file") not found."
  fi
}

# -------- run --------
LOGFILE="$LOGDIR/us_preclose_v1_$(date -u +%Y%m%d_%H%M).log"
{
  say "=== US-PRECLOSE REFRESH (v1) START ==="

  # Daily refresh to finalize *_d
  run_any "s10_ibkr_download_daily" "$SCRIPTS/s10_ibkr_download_daily.py" "$SHARED/s10_ibkr_download_daily.py" -- --client_id 88
  run_any "s12_ibkr_download_weekly" "$SCRIPTS/s12_ibkr_download_weekly.py" "$SHARED/s12_ibkr_download_weekly.py"
  run_any "s13_market_trends_update" "$SCRIPTS/s13_market_trends_update.py" "$SHARED/s13_market_trends_update.py"
  run_any "s14_forex_update"         "$SCRIPTS/s14_forex_update.py"         "$SHARED/s14_forex_update.py"

  run_py  "s20_make_master_daily"        "$SCRIPTS/s20_make_master_daily.py"
  run_py  "s30_enrich_to_parquet"        "$SCRIPTS/s30_enrich_to_parquet.py"
  run_py  "s31_enrich_macro_forex_daily" "$SCRIPTS/s31_enrich_macro_forex_daily.py"

  # Intraday 30m just before close + merge
  run_any "s11_ibkr_download_30min" "$SCRIPTS/s11_ibkr_download_30min.py" "$SHARED/s11_ibkr_download_30min.py" -- --client_id 88
  run_py  "s32_enrich_30m_intraday"  "$SCRIPTS/s32_enrich_30m_intraday.py"

  # Signals + dashboard
  run_py  "s77_rules_signals_v1" "$SCRIPTS/s77_rules_signals_v1.py" --rules-config "$CONFIG"

  if [[ -f "$SCRIPTS/s81_decision_board_rules_v1.py" ]]; then
    run_py "s81_decision_board_v1" "$SCRIPTS/s81_decision_board_rules_v1.py" --label "gate1_v1.0"
  elif [[ -f "$SCRIPTS/s81_decision_board_rules_v1.pys" ]]; then
    run_py "s81_decision_board_v1" "$SCRIPTS/s81_decision_board_rules_v1.pys" --label "gate1_v1.0"
  else
    say "[SKIP] s81_decision_board_rules_v1.(py|pys) not found."
  fi

  DASH="$SIGNALS_DIR/1_signals_dashboard_latest_gate1_v1.0.html"
  [[ -f "$DASH" ]] && say "Dashboard updated → $DASH"

  say "=== US-PRECLOSE REFRESH (v1) END ==="
} | tee -a "$LOGFILE"