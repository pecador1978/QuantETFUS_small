#!/usr/bin/env bash
set -euo pipefail

# ------------ paths ------------
ROOT="/Users/Finance/QuantETFUS_small"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
SCRIPTS="$ROOT/scripts"
LOGDIR="$ROOT/logs"
SIGNALS_DIR="$ROOT/signals"

mkdir -p "$LOGDIR" "$SIGNALS_DIR"

# ------------ env --------------
export MARKET_TZ="${MARKET_TZ:-Europe/London}"
export ETF_SHEET="${ETF_SHEET:-signalsUSD}"
export ETF_LIST_XLSX="${ETF_LIST_XLSX:-/Users/Finance/QuantShared/ETF_list.xlsx}"

# How "fast" you want it:
DAYS="${DAYS:-15}"           # recent daily/30m horizon
WEEKS="${WEEKS:-4}"         # recent weekly horizon
RUN_WEEKLY="${RUN_WEEKLY:-0}" # 0=skip weekly, 1=download weekly
SLEEP_MS="${SLEEP_MS:-200}"   # IB pacing (lower is faster, but riskier)

# s81 label to print in the page header
S81_LABEL="${S81_LABEL:-gate1_v1.0}"

timestamp(){ date -u +"%Y-%m-%d %H:%M:%S UTC"; }
say(){ echo "[$(timestamp)] $*"; }

LOGFILE="$LOGDIR/us_refresh_fast_$(date -u +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOGFILE") 2>&1
trap 'say "[ERR] ${BASH_SOURCE[0]} failed at line $LINENO"' ERR

# ------------ venv -------------
if [[ ! -x "$PY" ]]; then
  say "[ERR] Python venv not found at $PY"
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# ------------ helpers ----------
run_any() {
  local label="$1"; shift
  local -a candidates=()
  while (($#)); do
    [[ "$1" == "--" ]] && shift && break
    candidates+=("$1"); shift
  done
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

  say "$label → $(basename "$found") ${args[*]}"
  "$PY" "$found" "${args[@]}"
}

# ------------ FAST flow --------
say "=== US REFRESH (FAST) START ==="

# 1) IBKR downloads (recent only)
# Your s10 shows --duration; we map DAYS -> "NN D"
run_any "s10_ibkr_download_daily (fast)" \
  "$SCRIPTS/s10_ibkr_download_daily.py" "/Users/Finance/QuantShared/scripts/s10_ibkr_download_daily.py" -- \
  --client_id 88 --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX" \
  --duration "${DAYS} D" --sleep_ms "$SLEEP_MS"

# Optional weekly (small window or skip entirely)
if [[ "$RUN_WEEKLY" == "1" ]]; then
  run_any "s12_ibkr_download_weekly (fast)" \
    "$SCRIPTS/s12_ibkr_download_weekly.py" "/Users/Finance/QuantShared/scripts/s12_ibkr_download_weekly.py" -- \
    --sheet "$ETF_SHEET" --duration "${WEEKS} W" --sleep_ms "$SLEEP_MS"
else
  say "[INFO] Skipping weekly (RUN_WEEKLY=0)"
fi

# 30m intraday (recent window)
run_any "s11_ibkr_download_30min (fast)" \
  "$SCRIPTS/s11_ibkr_download_30min.py" "/Users/Finance/QuantShared/scripts/s11_ibkr_download_30min.py" -- \
  --client_id 88 --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX" \
  --duration "${DAYS} D" --sleep_ms "$SLEEP_MS"

# 2) Build masters/enrichment from latest pulls
"$PY" "$SCRIPTS/s20_make_master_daily.py"        --sheet "$ETF_SHEET"
"$PY" "$SCRIPTS/s30_enrich_to_parquet.py"
"$PY" "$SCRIPTS/s31_enrich_macro_forex_daily.py"
"$PY" "$SCRIPTS/s32_enrich_30m_intraday.py"

# 3) Signals + Gate-1.5
CONFIG="$ROOT/config/gate1_v1_rules.json"
"$PY" "$SCRIPTS/s77_rules_signals_v1.py" --rules-config "$CONFIG"
"$PY" "$SCRIPTS/s57_build_stretch_stats.py"
"$PY" "$SCRIPTS/s78_gate15_stats.py"
"$PY" "$SCRIPTS/s78a_gate15_calibrate_next1d.py"
"$PY" "$SCRIPTS/s78_gate15_stats_agg.py"

# ------- Dashboard (Gate-1 + Gate-1.5 + Daily context) -------
LAST_RULE_CSV=""
if compgen -G "$SIGNALS_DIR/rule_live_signals_*.csv" > /dev/null; then
  LAST_RULE_CSV="$(ls -t "$SIGNALS_DIR"/rule_live_signals_*.csv | head -n1)"
fi

if [[ -z "$LAST_RULE_CSV" ]]; then
  say "[ERR] No rule_live_signals_*.csv found after s77."
else
  say "s81 will render snapshot → $(basename "$LAST_RULE_CSV")"

  S81_FILE="$SCRIPTS/s81_decision_board_rules_v1_5.py"
  if [[ ! -f "$S81_FILE" ]]; then
    say "[ERR] s81_decision_board_rules_v1_5.py not found at $S81_FILE"
    exit 1
  fi

  : "${S81_LABEL:=gate1_v1.0}"   # keep header consistent; change via env if you want
  ETF_LIST_XLSX="${ETF_LIST_XLSX:-/Users/Finance/QuantShared/ETF_list.xlsx}" \
  "$PY" "$S81_FILE" \
    --label "$S81_LABEL" \
    --csv "$LAST_RULE_CSV" \
    --universe-sheet "${ETF_SHEET:-signalsUSD}"
fi

say "=== US REFRESH (FAST) DONE ==="
say "Log → $LOGFILE"