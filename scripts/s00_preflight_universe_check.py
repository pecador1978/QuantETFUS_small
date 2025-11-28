#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s00_preflight_universe_check.py — sanity checks before running s32/s77
- Ensures tickers present in P.DATA_ENRICHED/30min/*.parquet match tickers in prices_enriched.parquet
- Prints neat diff and exits non-zero if mismatch (so you can wire it into a Makefile/runner)
"""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

# project-aware paths
from common.paths import P  # uses your centralized resolver

def main() -> int:
    m30_dir = P.DATA_ENRICHED / "30min"
    daily_pq = P.DATA_ENRICHED / "prices_enriched.parquet"

    if not m30_dir.exists():
        print(f"[ERR] Missing 30m dir: {m30_dir}")
        return 2
    if not daily_pq.exists():
        print(f"[ERR] Missing daily parquet: {daily_pq}")
        return 2

    # tickers in 30m
    m30 = sorted(p.stem.upper() for p in m30_dir.glob("*.parquet"))

    # tickers in daily parquet
    D = pd.read_parquet(daily_pq)
    D.columns = [str(c).strip().lower() for c in D.columns]
    if "ticker" not in D.columns:
        print("[ERR] 'ticker' column not found in daily parquet.")
        return 2
    daily = sorted(D["ticker"].astype(str).str.upper().unique())

    only_m30  = sorted(set(m30)  - set(daily))
    only_day  = sorted(set(daily) - set(m30))
    overlap   = sorted(set(m30).intersection(daily))

    print(f"[INFO] 30m tickers : {len(m30)}")
    print(f"[INFO] Daily tickers: {len(daily)}")
    print(f"[INFO] Overlap      : {len(overlap)}")

    if only_m30 or only_day:
        if only_m30:
            print(f"[WARN] In 30m only ({len(only_m30)}): {only_m30[:20]}{' ...' if len(only_m30)>20 else ''}")
        if only_day:
            print(f"[WARN] In daily only ({len(only_day)}): {only_day[:20]}{' ...' if len(only_day)>20 else ''}")
        # non-zero exit to fail pipelines if wired in
        return 1

    print("[OK] Universe parity: 30m and daily match perfectly.")
    return 0

if __name__ == "__main__":
    sys.exit(main())