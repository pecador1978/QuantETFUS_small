#!/usr/bin/env bash
set -euo pipefail

# ----- timestamp + colored logging -----
timestamp(){ date -u +"%Y-%m-%d %H:%M:%S UTC"; }
BOLD='\033[1m'; RED='\033[1;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
say(){
  local msg="$*"
  msg="${msg//\[ERR\]/${RED}[ERR]${NC}}"
  msg="${msg//\[ERROR\]/${RED}[ERROR]${NC}}"
  msg="${msg//\[WARN\]/${YELLOW}[WARN]${NC}}"
  msg="${msg//\[WARNING\]/${YELLOW}[WARNING]${NC}}"
  msg="${msg//\[OK\]/${GREEN}[OK]${NC}}"
  msg="${msg//\[INFO\]/${BLUE}[INFO]${NC}}"
  msg="${msg//\[CHECK\]/${BOLD}[CHECK]${NC}}"
  echo -e "[$(timestamp)] $msg"
}

# ========== PROJECT ==========
ROOT="/Users/Finance/QuantETF_NY"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
SCRIPTS="$ROOT/scripts"
export PROJECT_ROOT="$ROOT"
export PYTHONPATH="$ROOT:$SCRIPTS:${PYTHONPATH:-}"

# ========== SHARED ==========
export QSHARED="$HOME/Library/Mobile Documents/com~apple~CloudDocs/QuantShared"
export PROJECT_SHARED="$QSHARED/ETF_NY"
export SHARED_BASE="$PROJECT_SHARED/data_raw"
mkdir -p "$SHARED_BASE"/{daily,weekly,30min} "$PROJECT_SHARED/mappings"

# ========== UNIVERSE / IB ==========
export MARKET_TZ="${MARKET_TZ:-America/New_York}"
export ETF_SHEET="${ETF_SHEET:-signalsNY}"
export ETF_LIST_XLSX="${ETF_LIST_XLSX:-$QSHARED/ETF_list.xlsx}"
export TICKER_MAPPING_CSV="${TICKER_MAPPING_CSV:-$PROJECT_SHARED/mappings/ticker_mapping.csv}"

# Preferred mapping hints (only used by scripts that support them)
export MAPPING_PRIMARY_EXCH_SEGMENTS="${MAPPING_PRIMARY_EXCH_SEGMENTS:-ARCA,NASDAQ,NYSE,ISLAND,BATS,IEX}"
export MAPPING_PREFERRED_CCY="${MAPPING_PREFERRED_CCY:-USD}"

# Separate client ids
export IB_CLIENT_ID_DAILY="${IB_CLIENT_ID_DAILY:-78}"
export IB_CLIENT_ID_WEEKLY="${IB_CLIENT_ID_WEEKLY:-79}"
export IB_CLIENT_ID_30M="${IB_CLIENT_ID_30M:-80}"

LOGDIR="$ROOT/logs"; mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/ny_seed_new_ticker_$(date -u +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOGFILE") 2>&1
trap 'say "[ERR] $(basename "$0") failed at line $LINENO"' ERR

# ========== venv ==========
if [[ ! -x "$PY" ]]; then say "[ERR] Python venv not found at $PY"; exit 1; fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"

say "[ENV] ROOT=$ROOT"
say "[ENV] PROJECT_SHARED=$PROJECT_SHARED"
say "[ENV] ETF_LIST_XLSX=$ETF_LIST_XLSX  ETF_SHEET=$ETF_SHEET"
say "[ENV] MARKET_TZ=$MARKET_TZ"

# ---- helpers ----
run_any() {
  local label="$1"; shift
  local -a candidates=()
  while (($#)); do
    [[ "$1" == "--" ]] && shift && break
    candidates+=("$1"); shift
  done
  local -a args=(); while (($#)); do args+=("$1"); shift; done
  local found=""; for f in "${candidates[@]}"; do [[ -f "$f" ]] && { found="$f"; break; }; done
  if [[ -z "$found" ]]; then say "[SKIP] $label (missing: ${candidates[*]-})"; return 0; fi
  say "$label → $(basename "$found") ${args[*]-}"
  "$PY" "$found" "${args[@]}"
}

# ---------- 0) Ensure ticker mapping exists ----------
if [[ -f "$TICKER_MAPPING_CSV" ]]; then
  say "[OK] ticker_mapping.csv present → $TICKER_MAPPING_CSV"
else
  say "[INFO] Building ticker_mapping.csv via s08"
  run_any "s08_build_ticker_mapping" \
    "$SCRIPTS/s08_build_ticker_mapping.py" "$QSHARED/scripts/s08_build_ticker_mapping.py" -- \
    --client_id "$IB_CLIENT_ID_DAILY" --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX" \
    --outdir "$(dirname "$TICKER_MAPPING_CSV")"
  [[ -f "$TICKER_MAPPING_CSV" ]] || { say "[ERR] ticker_mapping.csv missing after s08"; exit 1; }
fi

say "=== SEEDING NEW TICKER(S) — deep history only (NY) ==="

# ---------- 1) Daily — full history (prefer s05; fallback to s10 10Y) ----------
if [[ -f "$SCRIPTS/s05_full_daily_historical.py" || -f "$QSHARED/scripts/s05_full_daily_historical.py" ]]; then
  run_any "s05_full_daily_historical (10y)" \
    "$SCRIPTS/s05_full_daily_historical.py" "$QSHARED/scripts/s05_full_daily_historical.py" -- \
    --client_id "$IB_CLIENT_ID_DAILY" --years 10 --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX" \
    --outdir "$SHARED_BASE/daily"
else
  run_any "s10_ibkr_download_daily (10y fallback)" \
    "$SCRIPTS/s10_ibkr_download_daily.py" "$QSHARED/scripts/s10_ibkr_download_daily.py" -- \
    --client_id "$IB_CLIENT_ID_DAILY" --duration "10 Y" --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX" \
    --mapping "$TICKER_MAPPING_CSV" \
    --outdir "$SHARED_BASE/daily"
fi

# ---------- 2) Weekly — full history (prefer s07; fallback to s12 10Y) ----------
if [[ -f "$SCRIPTS/s07_full_weekly_historical.py" || -f "$QSHARED/scripts/s07_full_weekly_historical.py" ]]; then
  run_any "s07_full_weekly_historical (10y)" \
    "$SCRIPTS/s07_full_weekly_historical.py" "$QSHARED/scripts/s07_full_weekly_historical.py" -- \
    --client_id "$IB_CLIENT_ID_WEEKLY" --years 10 --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX" \
    --outdir "$SHARED_BASE/weekly"
else
  run_any "s12_ibkr_download_weekly (10y fallback)" \
    "$SCRIPTS/s12_ibkr_download_weekly.py" "$QSHARED/scripts/s12_ibkr_download_weekly.py" -- \
    --client_id "$IB_CLIENT_ID_WEEKLY" --duration "10 Y" --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX" \
    --mapping "$TICKER_MAPPING_CSV" \
    --segments "$MAPPING_PRIMARY_EXCH_SEGMENTS" --ccy "$MAPPING_PREFERRED_CCY" \
    --outdir "$SHARED_BASE/weekly"
fi

# ---------- 3) 30m — deep backfill (~5y) ----------
if [[ -f "$SCRIPTS/s06_full_30min_historical.py" || -f "$QSHARED/scripts/s06_full_30min_historical.py" ]]; then
  run_any "s06_full_30min_historical (~5y)" \
    "$SCRIPTS/s06_full_30min_historical.py" "$QSHARED/scripts/s06_full_30min_historical.py" -- \
    --client_id "$IB_CLIENT_ID_30M" --years 5 --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX" \
    --outdir "$SHARED_BASE/30min"
else
  export IB_30M_SEED=1 IB_30M_FORCE_SEED=1 IB_30M_SEED_DAYS="${IB_30M_SEED_DAYS:-1825}" IB_30M_SEED_WINDOW_DAYS="${IB_30M_SEED_WINDOW_DAYS:-60}"
  run_any "s11_ibkr_download_30min (seed ~5y)" \
    "$SCRIPTS/s11_ibkr_download_30min.py" "$QSHARED/scripts/s11_ibkr_download_30min.py" -- \
    --client_id "$IB_CLIENT_ID_30M" --sheet "$ETF_SHEET" --excel "$ETF_LIST_XLSX" \
    --mapping "$TICKER_MAPPING_CSV" \
    --segments "$MAPPING_PRIMARY_EXCH_SEGMENTS" --ccy "$MAPPING_PREFERRED_CCY" \
    --outdir "$SHARED_BASE/30min" --sleep_ms 200
fi

say "[OK] NY seeding complete. Next, run your NY FAST refresh to enrich & build signals."