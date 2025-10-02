#!/usr/bin/env bash
set -euo pipefail

# ===================== paths =====================
ROOT="/Users/Finance/QuantETFUS_small"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
SCRIPTS="$ROOT/scripts"
SHARED="/Users/Finance/QuantShared/scripts"
LOGDIR="$ROOT/logs"
SIGNALS_DIR="$ROOT/signals"
ANALYTICS_DIR="$SIGNALS_DIR/analytics"
CONFIG="$ROOT/config/gate1_v1_rules.json"
MAP_CSV="$ROOT/config/ticker_mapping.csv"

mkdir -p "$LOGDIR" "$SIGNALS_DIR" "$ANALYTICS_DIR"

# ===================== env =====================
export TARGET_BUCKET="${TARGET_BUCKET:-targeted_ETFs_US}"
export MARKET_TZ="${MARKET_TZ:-Europe/London}"

# Single source of truth for universe location
export ETF_SHEET="${ETF_SHEET:-signalsUSD}"
export ETF_LIST_XLSX="${ETF_LIST_XLSX:-/Users/Finance/QuantShared/ETF_list.xlsx}"

timestamp(){ date -u +"%Y-%m-%d %H:%M:%S UTC"; }
say(){ echo "[$(timestamp)] $*"; }

# Log everything to file + console
LOGFILE="$LOGDIR/us_preclose_v1_$(date -u +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOGFILE") 2>&1
trap 'say "[ERR] ${BASH_SOURCE[0]} failed at line $LINENO"' ERR

# ===================== venv =====================
if [[ ! -x "$PY" ]]; then
  say "[ERR] Python venv not found at $PY"
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# ===================== helpers =====================
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

# Require ticker_mapping.csv (build if missing)
ensure_ticker_map() {
  if [[ -f "$MAP_CSV" ]]; then
    say "ticker_mapping.csv present → $MAP_CSV"
    return 0
  fi
  say "ticker_mapping.csv missing → building via s08"
  run_py "s08_build_ticker_mapping" "$SCRIPTS/s08_build_ticker_mapping.py" \
    --client-id 88 --port 7497 --sheet "$ETF_SHEET" --write-missing-only
  if [[ ! -f "$MAP_CSV" ]]; then
    say "[ERR] ticker_mapping.csv still missing after s08. Aborting."
    exit 1
  fi
}

# ===================== run =====================
say "=== US-PRECLOSE REFRESH (v1) START ==="

# --- Gate: ticker map must exist
ensure_ticker_map

# ------- One-time seeding for new tickers (creates files only if missing) -------
run_py  "s06_full_30min_historical" "$SCRIPTS/s06_full_30min_historical.py" \
  --client_id 57 --years 10 --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX"

run_py  "s05_full_daily_historical" "$SCRIPTS/s05_full_daily_historical.py" \
  --client_id 59 --years 10 --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX"

run_py  "s07_full_weekly_historical" "$SCRIPTS/s07_full_weekly_historical.py" \
  --client_id 58 --years 10 --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX"

# Pull seeded CSVs into parquet now (so brand-new tickers are available downstream)
run_py "s30_enrich_to_parquet (post-seed)" "$SCRIPTS/s30_enrich_to_parquet.py"

# ------- Daily refresh -------
run_any "s10_ibkr_download_daily"  "$SCRIPTS/s10_ibkr_download_daily.py" "$SHARED/s10_ibkr_download_daily.py" -- \
  --client_id 88 --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX"

run_any "s12_ibkr_download_weekly" "$SCRIPTS/s12_ibkr_download_weekly.py" "$SHARED/s12_ibkr_download_weekly.py" -- \
  --sheet "$ETF_SHEET"

run_any "s13_market_trends_update" "$SCRIPTS/s13_market_trends_update.py" "$SHARED/s13_market_trends_update.py"
run_any "s14_forex_update"         "$SCRIPTS/s14_forex_update.py"         "$SHARED/s14_forex_update.py"

run_py  "s20_make_master_daily"        "$SCRIPTS/s20_make_master_daily.py"        --sheet "$ETF_SHEET"
run_py  "s30_enrich_to_parquet"        "$SCRIPTS/s30_enrich_to_parquet.py"
run_py  "s31_enrich_macro_forex_daily" "$SCRIPTS/s31_enrich_macro_forex_daily.py"

# ------- Intraday 30m just before close + merge -------
run_any "s11_ibkr_download_30min" "$SCRIPTS/s11_ibkr_download_30min.py" "$SHARED/s11_ibkr_download_30min.py" -- \
  --client_id 88 --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX"

run_py  "s32_enrich_30m_intraday"  "$SCRIPTS/s32_enrich_30m_intraday.py"

# ------- Signals (Gate-1) -------
# NOTE: Do NOT pass --sheet; s77 reads ETF_SHEET/ETF_LIST_XLSX from env.
run_py  "s77_rules_signals_v1" "$SCRIPTS/s77_rules_signals_v1.py" --rules-config "$CONFIG"

# ------- Gate-1.5 analytics (stretch + calibration) -------
run_py  "s57_build_stretch_stats"      "$SCRIPTS/s57_build_stretch_stats.py"
run_py  "s78_gate15_stats"             "$SCRIPTS/s78_gate15_stats.py"
run_py  "s78a_gate15_calibrate_next1d" "$SCRIPTS/s78a_gate15_calibrate_next1d.py"
run_py  "s78_gate15_stats_agg"         "$SCRIPTS/s78_gate15_stats_agg.py"

# ------- Dashboard (Gate-1 + Gate-1.5 merged) -------
LAST_RULE_CSV=""
if compgen -G "$SIGNALS_DIR/rule_live_signals_*.csv" > /dev/null; then
  LAST_RULE_CSV="$(ls -t "$SIGNALS_DIR"/rule_live_signals_*.csv | head -n1)"
fi

if [[ -z "$LAST_RULE_CSV" ]]; then
  say "[ERR] No rule_live_signals_*.csv found after s77."
else
  say "s81 will render snapshot → $(basename "$LAST_RULE_CSV")"
  if [[ -f "$SCRIPTS/s81_decision_board_rules_v1_5.py" ]]; then
    # s81 doesn’t accept --universe-xlsx; it resolves Excel internally.
    # Give it the correct path via env so it won’t look in project/config.
    ETF_LIST_XLSX="${ETF_LIST_XLSX:-/Users/Finance/QuantShared/ETF_list.xlsx}" \
    "$PY" "$SCRIPTS/s81_decision_board_rules_v1_5.py" \
      --label "gate1_v1.5" \
      --csv "$LAST_RULE_CSV" \
      --universe-sheet "$ETF_SHEET"
  else
    say "[SKIP] s81_decision_board_rules_v1_5.py not found."
  fi
fi

say "=== US-PRECLOSE REFRESH (v1) DONE ==="
say "Log → $LOGFILE"