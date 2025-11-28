#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s30_enrich_to_parquet.py — run s00 on the project master and write outputs to the project iCloud bucket.

Inputs
------
- Master CSV: <PROJECT_SHARED>/data_raw/etf_prices_daily_master.csv
  (falls back to P.SHARED_RAW_BASE/etf_prices_daily_master.csv)

Outputs (written by s00 via --outbase)
--------------------------------------
- <PROJECT_SHARED>/data_clean/prices_clean.parquet
- <PROJECT_SHARED>/data_enriched/prices_enriched.parquet
- <PROJECT_SHARED>/reports/integrity_report_*.csv

Env it honors (optional)
------------------------
- PROJECT_SHARED        : iCloud project bucket (preferred; used for --outbase)
- DATA_RAW_SHARED       : raw base (preferred; used to locate master)
- MARKET_TZ             : e.g., America/New_York or Europe/London
- ETF_SHEET             : used for tz heuristic if MARKET_TZ is unset
"""

from pathlib import Path
import subprocess
import sys
import os
import argparse

# --- import centralized paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import P  # noqa: E402

def _env_path(name: str) -> Path | None:
    v = os.environ.get(name, "").strip()
    return Path(v).expanduser().resolve() if v else None

def _default_market_tz() -> str:
    # 1) explicit env wins
    env_tz = os.environ.get("MARKET_TZ", "").strip()
    if env_tz:
        return env_tz

    # 2) infer from project name and/or ETF_SHEET
    proj_name = Path(getattr(P, "ROOT", PROJECT_ROOT)).name.upper()
    sheet = os.environ.get("ETF_SHEET", "").upper()

    # Treat NY/US sheets/projects as US market hours
    if any(k in proj_name for k in ("NY", "US")) or any(k in sheet for k in ("NY", "USD", "US")):
        return "America/New_York"

    # Default to London if not clearly US
    return "Europe/London"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tz", default="", help="Timezone override (e.g., 'America/New_York').")
    args, _ = ap.parse_known_args()

    # ---- resolve INPUT master (prefer the project-scoped raw base) ----
    raw_base = _env_path("DATA_RAW_SHARED") or P.SHARED_RAW_BASE
    master = (raw_base / "etf_prices_daily_master.csv").resolve()

    # ---- resolve OUTBASE (where s00 writes clean/enriched/reports) ----
    outbase = _env_path("PROJECT_SHARED") or (P.PROJECT_SHARED if getattr(P, "PROJECT_SHARED", None) else P.ROOT)

    # ---- MARKET_TZ ----
    market_tz = args.tz.strip() or _default_market_tz()

    # ---- sanity checks ----
    if not master.exists():
        raise SystemExit(f"[ERR] Master not found: {master}\n"
                         f"Hint: build it first with s20, e.g.\n"
                         f"  python scripts/s20_make_master_daily.py --input-base \"{raw_base}\"")

    # ---- locate s00 with sensible fallbacks ----
    candidates = [
        Path(getattr(P, "SCRIPTS_DIR", SCRIPT_DIR)) / "s00_data_integrity_enrichment.py",
        (Path(os.environ.get("QSHARED", "")) / "scripts" / "s00_data_integrity_enrichment.py") if os.environ.get("QSHARED") else None,
        SCRIPT_DIR / "s00_data_integrity_enrichment.py",  # same folder as s30 (last resort)
    ]
    candidates = [p.resolve() for p in candidates if p]
    s00 = next((p for p in candidates if p.exists()), None)
    if not s00:
        tried = "\n  - " + "\n  - ".join(map(str, candidates))
        raise SystemExit(f"[ERR] Missing s00_data_integrity_enrichment.py. Looked in:{tried}")

    # ---- build command ----
    cmd = [
        sys.executable,
        str(s00),
        "--input", str(master),
        "--freq", "D",
        "--tz", str(market_tz),
        "--outbase", str(outbase),
    ]

    print("Running:", " ".join(cmd))
    print(f"[s30] INPUT  master : {master}")
    print(f"[s30] OUTBASE       : {outbase}")
    print(f"[s30] MARKET_TZ     : {market_tz}")

    # ---- stream s00 output so tqdm renders properly ----
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        cwd=str(PROJECT_ROOT)
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
        ret = proc.wait()

    if ret != 0:
        raise SystemExit(f"[ERR] s00_data_integrity_enrichment exited with code {ret}")

    print(f"[OK] Enrichment complete. Parquets updated under: {outbase}")

if __name__ == "__main__":
    main()