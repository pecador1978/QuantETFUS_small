#!/bin/bash
# run_seeds.sh — sequential s60 runs for QuantETFUS

SEEDS=(120 130 140 150 160 170 180 190 200 210 220 230)

for SEED in "${SEEDS[@]}"; do
  echo "=== Starting seed $SEED ==="
  python /Users/Finance/QuantETFUS/scripts/s60_parametizer_all_tickers.py \
    --skip_tickers 0 \
    --limit_tickers 34 \
    --sample_scope per_ticker \
    --sample_combos 6000 \
    --min_trades 5 \
    --n_jobs_tickers 2 \
    --n_jobs_combos 12 \
    --start_date 2016-01-01 --end_date 2025-12-31 \
    --seed $SEED
  echo "=== Finished seed $SEED ==="
done
