#!/usr/bin/env bash
set -euo pipefail

# ---- paths ----
QUANT="/Users/Finance/QuantETFUS_small"
SHARED="/Users/Finance/QuantShared"
PY="$QUANT/venv/bin/python"

LOGDIR="$QUANT/signals/logs"
mkdir -p "$LOGDIR"

# market/session env (used by s32/s76/s80)
export MARKET_TZ="Europe/London"
export TARGET_BUCKET="targeted_ETFs_US"   # scripts look under data_raw/<bucket>/{30min,daily}

timestamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }
log() { echo "[$(timestamp)] $*"; }

RUN_TS="$(date -u +"%Y%m%d_%H%M%S")"
LOGFILE="$LOGDIR/run_$RUN_TS.log"

{
  log "=== START signals workflow ==="
  cd "$QUANT"

  # 0) Download fresh data into QuantShared
  #    - s10: daily    → /Users/Finance/QuantShared/data_raw_ETF_US/daily
  #    - s11: 30min    → /Users/Finance/QuantShared/data_raw_ETF_US/30min
  #    - s12: weekly   → /Users/Finance/QuantShared/data_raw_ETF_US/weekly (optional)
  log "[0/6] Shared data downloads (s10/s11/s12)"
  "$PY" "$SHARED/scripts/s10_download_daily.py"   || log "[WARN] s10_daily returned non-zero"
  "$PY" "$SHARED/scripts/s11_download_30min.py"   || log "[WARN] s11_30min returned non-zero"
  # "$PY" "$SHARED/scripts/s12_download_weekly.py" || log "[WARN] s12_weekly returned non-zero"

  # 1) Ensure symlink so QuantETFUS_small sees Shared raw data at expected path
  #    Scripts like s32 expect: $QUANT/data_raw/targeted_ETFs_US/{30min,daily}
  RAW_EXPECTED="$QUANT/data_raw/targeted_ETFs_US"
  RAW_SHARED="$SHARED/data_raw_ETF_US"

  if [ ! -e "$RAW_EXPECTED" ]; then
    log "[1/6] Creating symlink: $RAW_EXPECTED → $RAW_SHARED"
    mkdir -p "$(dirname "$RAW_EXPECTED")"
    ln -s "$RAW_SHARED" "$RAW_EXPECTED"
  else
    # If it exists but is NOT the right symlink, print a warning
    if [ ! -L "$RAW_EXPECTED" ] || [ "$(readlink "$RAW_EXPECTED")" != "$RAW_SHARED" ]; then
      log "[WARN] $RAW_EXPECTED already exists and is not a symlink to $RAW_SHARED"
      log "       Please reconcile manually if needed."
    fi
  fi

  # 2) Build/refresh daily enriched parquet (prices_enriched.parquet)
  #    If your pipeline uses a different script name, replace the line below.
  log "[2/6] Build daily enriched (s00 / or your daily build step)"
  "$PY" "$QUANT/scripts/s00_data_integrity_enrichment.py"

  # 3) Enrich 30m with daily+macro overlays → data_enriched/30min/{TICKER}.parquet
  log "[3/6] s32_enrich_30m_intraday.py"
  "$PY" "$QUANT/scripts/s32_enrich_30m_intraday.py"

  # 4) Rule-based Gate-1 signals (BUY / SELL / DO NOTHING) → signals/rule_live_signals_*.csv
  log "[4/6] s76_rule_signals.py"
  "$PY" "$QUANT/scripts/s76_rule_signals.py"

  # 5) Operator dashboard → signals/1_signals_dashboard_latest.html (+ timestamped copy)
  log "[5/6] s80_decision_board.py"
  "$PY" "$QUANT/scripts/s80_decision_board.py"

  log "=== DONE signals workflow ==="
} | tee -a "$LOGFILE"