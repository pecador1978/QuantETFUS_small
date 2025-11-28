#!/usr/bin/env bash
set -euo pipefail

# ----- timestamp + colored logging -----
timestamp(){ date -u +"%Y-%m-%d %H:%M:%S UTC"; }

# color codes
BOLD='\033[1m'
RED='\033[1;31m'     # bold red
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'         # No Color

say(){
  local msg="$*"
  msg="${msg//\[ERR\]/${RED}[ERR]${NC}}"
  msg="${msg//\[ERROR\]/${RED}[ERROR]${NC}}"
  msg="${msg//\[WARN\]/${YELLOW}[WARN]${NC}}"
  msg="${msg//\[WARNING\]/${YELLOW}[WARNING]${NC}}"
  msg="${msg//\[OK\]/${GREEN}[OK]${NC}}"
  msg="${msg//\[INFO\]/${BLUE}[INFO]${NC}}"
  msg="${msg//\[CHECK\]/${BOLD}[CHECK]${NC}}"
  msg="$(echo "$msg" | sed -E \
    -e "s/[Ee][Rr][Rr][Oo][Rr]/${RED}${BOLD}ERROR${NC}/g" \
    -e "s/\b[Ww][Aa][Rr][Nn](ING)?\b/${YELLOW}WARN\1${NC}/g" \
    -e "s/\b[Oo][Kk]\b/${GREEN}OK${NC}/g" \
    -e "s/\b[Ii][Nn][Ff][Oo]\b/${BLUE}INFO${NC}/g")"
  echo -e "[$(timestamp)] $msg"
}

# ---------- Project root (portable) ----------
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPTS="$PROJECT_ROOT/scripts"
PY="${PY:-$PROJECT_ROOT/venv/bin/python}"
export PROJECT_ROOT SCRIPTS_DIR="$SCRIPTS" PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ---------- Error muter (IBKR noise suppression) ----------
export PYTHONSTARTUP="/Users/Finance/QuantShared/common/ibkr_error_mute.py"
export PYTHONPATH="/Users/Finance/QuantShared/common:${PYTHONPATH:-}"

# ---------- iCloud shared + project bucket (NY defaults) ----------
export QSHARED="${QSHARED:-$HOME/Library/Mobile Documents/com~apple~CloudDocs/QuantShared}"
export PROJECT_SHARED="${PROJECT_SHARED:-$QSHARED/ETF_NY}"
export SHARED_RAW_BASE="${SHARED_RAW_BASE:-$PROJECT_SHARED/data_raw}"
export DATA_ENRICHED_BASE="${DATA_ENRICHED_BASE:-$PROJECT_SHARED/data_enriched}"
export SIGNALS_SHARED_DIR="${SIGNALS_SHARED_DIR:-$PROJECT_SHARED/signals}"
mkdir -p "$SHARED_RAW_BASE"/{daily,weekly,30min} "$DATA_ENRICHED_BASE" "$SIGNALS_SHARED_DIR"

# ---------- Universe + TZ (NY defaults) ----------
export ETF_LIST_XLSX="${ETF_LIST_XLSX:-$QSHARED/ETF_list.xlsx}"
export ETF_SHEET="${ETF_SHEET:-signalsNY}"
export MARKET_TZ="${MARKET_TZ:-US/Eastern}"

# ---------- Venue/currency prefs for contract discovery (US venues) ----------
MAPPING_PRIMARY_EXCH_SEGMENTS="${MAPPING_PRIMARY_EXCH_SEGMENTS:-ARCA,NASDAQ,NYSE,ISLAND,BATS,IEX}"
MAPPING_PREFERRED_CCY="${MAPPING_PREFERRED_CCY:-USD}"
DEFAULT_CCY="${DEFAULT_CCY:-USD}"
TICKER_MAPPING_CSV="${TICKER_MAPPING_CSV:-$PROJECT_SHARED/mappings/ticker_mapping.csv}"
mkdir -p "$(dirname "$TICKER_MAPPING_CSV")"

# ---------- IB + windows ----------
IB_CLIENT_ID="${IB_CLIENT_ID:-99}"
DAYS="${DAYS:-15}"
WEEKS="${WEEKS:-4}"
SLEEP_MS="${SLEEP_MS:-200}"
RUN_WEEKLY="${RUN_WEEKLY:-0}"

# ---------- Logging ----------
LOGDIR="$PROJECT_ROOT/logs"; mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/ny_refresh_fast_$(date -u +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOGFILE") 2>&1
trap 'say "[ERR] failed at line $LINENO"' ERR

# ---------- Banner ----------
say "[ENV] PROJECT_ROOT=$PROJECT_ROOT"
say "[ENV] PROJECT_SHARED=$PROJECT_SHARED"
say "[ENV] SHARED_RAW_BASE=$SHARED_RAW_BASE"
say "[ENV] ETF_LIST_XLSX=$ETF_LIST_XLSX  ETF_SHEET=$ETF_SHEET"
say "[ENV] MARKET_TZ=$MARKET_TZ  IB_CLIENT_ID=$IB_CLIENT_ID"
say "[ENV] Segments=$MAPPING_PRIMARY_EXCH_SEGMENTS  CCY=$MAPPING_PREFERRED_CCY"
say "[ENV] DATA_ENRICHED_BASE=$DATA_ENRICHED_BASE"

# ---------- 1) Downloads ----------
# Daily (append-only)
"$PY" "$SCRIPTS/s10_ibkr_download_daily.py" \
  --client_id "$IB_CLIENT_ID" \
  --sheet "$ETF_SHEET" \
  --excel "$ETF_LIST_XLSX" \
  --duration "${DAYS} D" \
  --mapping "$TICKER_MAPPING_CSV" \
  --outdir "$SHARED_RAW_BASE/daily"

# Weekly (append-only, optional)
if [[ "$RUN_WEEKLY" == "1" ]]; then
  "$PY" "$SCRIPTS/s12_ibkr_download_weekly.py" \
    --client_id "$IB_CLIENT_ID" \
    --sheet "$ETF_SHEET" \
    --excel "$ETF_LIST_XLSX" \
    --mapping "$TICKER_MAPPING_CSV" \
    --segments "$MAPPING_PRIMARY_EXCH_SEGMENTS" \
    --ccy "$MAPPING_PREFERRED_CCY" \
    --duration "${WEEKS} W" \
    --outdir "$SHARED_RAW_BASE/weekly" \
    --sleep_ms "$SLEEP_MS"
else
  say "[INFO] Skipping weekly (RUN_WEEKLY=0)"
fi

# 30m (append-only, small delta window)
"$PY" "$SCRIPTS/s11_ibkr_download_30min.py" \
  --client_id "$IB_CLIENT_ID" \
  --sheet "$ETF_SHEET" \
  --excel "$ETF_LIST_XLSX" \
  --mapping "$TICKER_MAPPING_CSV" \
  --segments "$MAPPING_PRIMARY_EXCH_SEGMENTS" \
  --ccy "$MAPPING_PREFERRED_CCY" \
  --duration "${DAYS} D" \
  --outdir "$SHARED_RAW_BASE/30min" \
  --sleep_ms "$SLEEP_MS"

# ---------- 2) Build / Enrich ----------
"$PY" "$SCRIPTS/s20_make_master_daily.py" --sheet "$ETF_SHEET" --input_base "$SHARED_RAW_BASE"
"$PY" "$SCRIPTS/s30_enrich_to_parquet.py"

# Preflight: ensure DAILY vs 30m universe parity before s32
"$PY" "$SCRIPTS/s00_preflight_universe_check.py" \
  --sheet "$ETF_SHEET" \
  --daily_parquet "$DATA_ENRICHED_BASE/prices_enriched.parquet" \
  --m30_dir "$DATA_ENRICHED_BASE/30min"

# quick sanity checks (fail early if paths are wrong)
[[ -f "$DATA_ENRICHED_BASE/prices_enriched.parquet" ]] || { 
  say "[ERR] prices_enriched.parquet missing in $DATA_ENRICHED_BASE"; exit 1; 
}

"$PY" "$SCRIPTS/s31_enrich_macro_forex_daily.py" || true  # ok if you don't maintain macro

# explicitly pass NY parquets + outdir to s32 so it can’t drift
"$PY" "$SCRIPTS/s32_enrich_30m_intraday.py" \
  --daily_context "$DATA_ENRICHED_BASE/prices_enriched.parquet" \
  --macro_context "$DATA_ENRICHED_BASE/macro_forex_enriched.parquet" \
  --out_dir "$DATA_ENRICHED_BASE/30min"

# ---------- 3) Signals ----------
RULES_JSON="$PROJECT_ROOT/config/gate1_v1_rules.json"
"$PY" "$SCRIPTS/s77_rules_signals_v1.py" --rules-config "$RULES_JSON"
"$PY" "$SCRIPTS/s57_build_stretch_stats.py"
"$PY" "$SCRIPTS/s78_gate15_stats.py"
"$PY" "$SCRIPTS/s78a_gate15_calibrate_next1d.py"
"$PY" "$SCRIPTS/s78_gate15_stats_agg.py"

# Market traffic light from DAILY master
"$PY" "$SCRIPTS/s84_market_traffic_light.py" \
  --bench SPY \
  --components "XLU,XLP,XLV,XLC,GLD,TLT" \
  --output "$SIGNALS_SHARED_DIR/market_status.json"

# --- Gate-2 calibration (bounds, confidence, snapshot, trust) ---
say "[INFO] Updating Gate-2 calibration (bounds, confidence, snapshot, trust)…"

"$PY" "$SCRIPTS/s79_calibrate_gate2_bounds.py" \
  --input_dir "$DATA_ENRICHED_BASE/30min" \
  --output    "$PROJECT_ROOT/config/gate2_norm_bounds.json"

"$PY" "$SCRIPTS/s80_confidence_gate2.py" \
  --input_dir "$DATA_ENRICHED_BASE/30min" \
  --output    "$DATA_ENRICHED_BASE/gate2_confidence_30m.csv" \
  --weights   "$PROJECT_ROOT/config/gate2_weights.json" \
  --norms     "$PROJECT_ROOT/config/gate2_norm_bounds.json"

"$PY" "$SCRIPTS/s80a_prepare_confidence_snapshot.py" \
  --input  "$DATA_ENRICHED_BASE/gate2_confidence_30m.csv" \
  --output "$DATA_ENRICHED_BASE/gate2_confidence_snapshot.csv"

# Build trust JSON (use s83 if metrics exist; else fallback)
if [ -f "$PROJECT_ROOT/reports/gate2_per_ticker_metrics.csv" ]; then
  "$PY" "$SCRIPTS/s83_model_quality_gate2.py" \
    --input  "$PROJECT_ROOT/reports/gate2_per_ticker_metrics.csv" \
    --output "$PROJECT_ROOT/config/gate2_ticker_quality.json"
else
  "$PY" - <<'PY'
import json, pandas as pd, os
from pathlib import Path

root = os.environ.get("PROJECT_ROOT")
base = os.environ.get("DATA_ENRICHED_BASE") or str(Path(os.environ.get("PROJECT_SHARED","")) / "data_enriched")
snap = str(Path(base) / "gate2_confidence_snapshot.csv")
out  = str(Path(root) / "config" / "gate2_ticker_quality.json")

df = pd.read_csv(snap)
df["trust_score"] = df["pct_hist"] / 100.0
def _qual(x):
    return "High" if x >= 0.66 else ("Medium" if x >= 0.33 else "Low")
df["quality"] = df["trust_score"].map(_qual)

out_json = {r.ticker: {"trust_score": round(float(r.trust_score), 3), "quality": r.quality}
            for _, r in df.iterrows()}
with open(out, "w") as f:
    json.dump(out_json, f, indent=2)
print(f"[OK] Wrote fallback trust JSON → {out}")
PY
fi

say "[OK] Gate-2 calibration updated successfully."

# ---------- 4) Dashboard ----------
LAST_RULE_CSV="$(ls -t "$PROJECT_ROOT"/signals/rule_live_signals_*.csv 2>/dev/null | head -n1 || true)"
if [[ -n "$LAST_RULE_CSV" ]]; then
  "$PY" "$SCRIPTS/s81_decision_board_rules_v1_5.py" \
    --label "gate1_v1.0" \
    --csv "$LAST_RULE_CSV" \
    --universe-sheet "$ETF_SHEET"
else
  say "[WARN] No rule_live_signals_*.csv found after s77"
fi

LATEST="$SIGNALS_SHARED_DIR/1_signals_dashboard_latest_gate1_v1.0.html"
NEWEST="$(ls -t "$SIGNALS_SHARED_DIR"/signals_dashboard_gate1_v1.0_*.html 2>/dev/null | head -n1 || true)"
[[ -n "$NEWEST" ]] && cp -f "$NEWEST" "$LATEST" || true

# ---------- 5) ATR snapshot (shared tool) ----------
if [[ -f "$QSHARED/scripts/export_atr_snapshot.py" ]]; then
  mkdir -p "$SIGNALS_SHARED_DIR/analytics"
  OUT_CSV="$SIGNALS_SHARED_DIR/analytics/atr_snapshot_${ETF_SHEET}_$(date -u +%Y%m%d_%H%M).csv"
  say "[INFO] Building ATR snapshot → $OUT_CSV"
  "$PY" "$QSHARED/scripts/export_atr_snapshot.py"
  SRC="$QSHARED/outputs/atr_snapshot.csv"
  [[ -f "$SRC" ]] && cp -f "$SRC" "$OUT_CSV" || say "[WARN] ATR source $SRC not found"
  ln -sf "$OUT_CSV" "$SIGNALS_SHARED_DIR/analytics/atr_snapshot_latest_${ETF_SHEET}.csv" || true
  say "[OK] ATR snapshot → $OUT_CSV"
else
  say "[WARN] $QSHARED/scripts/export_atr_snapshot.py not found; skipping ATR snapshot"
fi

say "[OK] NY FAST refresh complete — log: $LOGFILE"