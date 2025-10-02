#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s30_enrich_to_parquet.py — Run enrichment on the master daily CSV (UTC, stream s00 progress)

- Calls s00_data_integrity_enrichment.py on:
    P.DATA_RAW/etf_prices_daily_master.csv
- Streams s00's tqdm progress bar live to the terminal.
- Outputs (written by s00):
    P.ROOT/data_clean/prices_clean.parquet
    P.ROOT/data_enriched/prices_enriched.parquet
    P.ROOT/reports/integrity_report_*.csv
"""

from pathlib import Path
import subprocess
import sys
import os

# --- import centralized paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import P  # noqa: E402

# Try to get MARKET_TZ from settings; fall back to env or heuristic
try:
    from common.settings import MARKET_TZ  # noqa: E402
except Exception:
    proj_name = Path(P.ROOT).name if hasattr(P, "ROOT") else PROJECT_ROOT.name
    MARKET_TZ = os.environ.get(
        "MARKET_TZ",
        "US/Eastern" if "US" in proj_name.upper() else "Europe/London"
    )

MASTER = P.DATA_RAW / "etf_prices_daily_master.csv"   # built by s20
OUTBASE = P.ROOT                                       # where s00 will place parquets

def main():
    if not MASTER.exists():
        raise SystemExit(f"[ERR] Master not found: {MASTER}")

    s00 = P.SCRIPTS_DIR / "s00_data_integrity_enrichment.py"
    if not s00.exists():
        raise SystemExit(f"[ERR] Missing script: {s00}")

    cmd = [
        sys.executable,
        str(s00),
        "--input", str(MASTER),
        "--freq", "D",
        "--tz", MARKET_TZ,
        "--outbase", str(OUTBASE)
        # IBKR-only flow; don't pass --use_yf
    ]

    print("Running:", " ".join(cmd))

    # Stream stdout/stderr so tqdm stays visible
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        cwd=str(PROJECT_ROOT)  # keep relative paths stable inside s00
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
        ret = proc.wait()

    if ret != 0:
        raise SystemExit(f"[ERR] s00_data_integrity_enrichment exited with code {ret}")

    print("[OK] Enrichment complete. Parquets updated in:", OUTBASE)

if __name__ == "__main__":
    main()