#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s77a_replay_backfill.py — Fast backfill runner for s77 across past local days.

Key features
------------
- In-process execution (default): run many ASOF dates inside each worker process,
  avoiding per-day interpreter startup overhead.
- Bundling: split the calendar into chunks (e.g., 50 days) and assign one bundle
  per worker → far fewer process launches.
- Parallel via joblib (loky). Supports negative --jobs like sklearn: -1 = all cores.
- Passthrough of ETF sheet and rules config.

Usage examples
--------------
  python3 scripts/s77a_replay_backfill.py --start 2019-01-01 --end 2025-09-12 \
      --jobs -1 --bundle 60 --etf-sheet signalsUSD --rules-config config/gate1_v1_rules.json

  python3 scripts/s77a_replay_backfill.py --last-days 120 --limit 50 --jobs 6

Notes
-----
- Uses ASOF_DATE=YYYY-MM-DD for each date.
- Calls s77 with: --append-parquet --no-operator-exports --no-board-csv
- Weekends skipped unless --include-weekends.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, date
import argparse, os, sys, math

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
S77_PATH = PROJECT_ROOT / "scripts" / "s77_rules_signals_v1.py"

def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def _cpu_count() -> int:
    try:
        import multiprocessing as mp
        return max(1, mp.cpu_count())
    except Exception:
        return 1

def _normalize_jobs(jobs_opt: int | None) -> int:
    if not jobs_opt:
        return 1
    if jobs_opt > 0:
        return jobs_opt
    # negative → like sklearn: -1 = all cores, -2 = all but 1, etc.
    cores = _cpu_count()
    want = cores + jobs_opt + 1
    return max(1, want)

def _chunk(lst, n):
    """Yield successive chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _run_batch_inproc(dates: list[date], limit: int | None, rules_config: str | None, etf_sheet: str | None) -> tuple[list[date], list[date]]:
    """
    Run many ASOF dates sequentially in THIS (worker) process by invoking s77 via runpy.
    Returns (ok_dates, bad_dates).
    """
    import runpy
    ok, bad = [], []
    # Stable base argv for s77
    base_argv = [str(S77_PATH), "--append-parquet", "--no-operator-exports", "--no-board-csv"]
    if limit is not None:
        base_argv += ["--limit", str(limit)]
    if rules_config:
        base_argv += ["--rules-config", str(rules_config)]

    # Preserve original environment to avoid leaking ASOF_DATE
    orig_env = os.environ.copy()
    if etf_sheet:
        os.environ["ETF_SHEET"] = etf_sheet

    for d in dates:
        try:
            os.environ["ASOF_DATE"] = d.strftime("%Y-%m-%d")
            # Each run should parse its own sys.argv; reset it
            sys.argv = list(base_argv)
            # Fresh __main__ each time:
            runpy.run_path(str(S77_PATH), run_name="__main__")
            ok.append(d)
        except SystemExit as e:
            # s77 may call SystemExit with code; non-zero → treat as failure
            if int(getattr(e, "code", 1) or 0) == 0:
                ok.append(d)
            else:
                print(f"[WARN] s77 failed @ {d} (SystemExit code={e.code})")
                bad.append(d)
        except Exception as e:
            print(f"[WARN] s77 exception @ {d}: {e}")
            bad.append(d)
        finally:
            # keep ETF_SHEET, but ensure ASOF_DATE is cleared before next date set
            os.environ.pop("ASOF_DATE", None)

    # restore environment (keep ETF_SHEET if we had set it)
    os.environ.clear()
    os.environ.update(orig_env)
    if etf_sheet:
        os.environ["ETF_SHEET"] = etf_sheet
    return ok, bad

def _run_batch_subproc(dates: list[date], limit: int | None, rules_config: str | None, etf_sheet: str | None) -> tuple[list[date], list[date]]:
    """
    Fallback: spawn a new Python subprocess ONCE per date (slower).
    Returns (ok_dates, bad_dates).
    """
    import subprocess
    ok, bad = [], []
    for d in dates:
        env = os.environ.copy()
        env["ASOF_DATE"] = d.strftime("%Y-%m-%d")
        if etf_sheet:
            env["ETF_SHEET"] = etf_sheet

        cmd = [sys.executable, str(S77_PATH), "--append-parquet", "--no-operator-exports", "--no-board-csv"]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        if rules_config:
            cmd += ["--rules-config", str(rules_config)]

        rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode
        if rc == 0:
            ok.append(d)
        else:
            print(f"[WARN] s77 failed @ {d} (rc={rc})")
            bad.append(d)
    return ok, bad

def main():
    ap = argparse.ArgumentParser(description="Replay s77 across past days to build boards (fast bundling).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--last-days", type=int, help="Replay this many most-recent calendar days.")
    g.add_argument("--start", type=str, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", type=str, help="YYYY-MM-DD (inclusive). Defaults to today if not given.")
    ap.add_argument("--step", type=int, default=1, help="Calendar day step. Default 1.")
    ap.add_argument("--include-weekends", action="store_true", help="Run also on Sat/Sun (default: skip).")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N tickers per day.")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers. Use -1 for all cores, -2 for all but one, etc.")
    ap.add_argument("--bundle", type=int, default=50, help="How many days a worker processes sequentially (default 50).")
    ap.add_argument("--exec", dest="exec_mode", choices=["inproc","subproc"], default="inproc",
                    help="inproc (default) = reuse process per bundle; subproc = new process per date (slow).")
    ap.add_argument("--etf-sheet", type=str, default=None, help="Set ETF_SHEET for s77 (e.g., signalsUSD).")
    ap.add_argument("--rules-config", type=str, default=None, help="Path to rules JSON to pass through to s77.")
    args = ap.parse_args()

    # Build calendar
    if args.last_days:
        end_d = date.today()
        start_d = end_d - timedelta(days=args.last_days - 1)
    else:
        start_d = _d(args.start)
        end_d = _d(args.end) if args.end else date.today()

    if start_d > end_d:
        print(f"[ERR] start > end: {start_d} > {end_d}")
        sys.exit(2)

    days: list[date] = []
    cur = start_d
    while cur <= end_d:
        if args.include_weekends or cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=args.step)

    if not days:
        print("[WARN] No days to replay after filters.")
        sys.exit(0)

    jobs = _normalize_jobs(args.jobs)
    bundle = max(1, int(args.bundle))
    bundles = list(_chunk(days, bundle))

    print(f"[INFO] Mode={args.exec_mode} | jobs={jobs} | bundles={len(bundles)} (size≈{bundle}) | total_days={len(days)}")
    if args.etf_sheet:
        print(f"[INFO] ETF_SHEET={args.etf_sheet}")
        os.environ["ETF_SHEET"] = args.etf_sheet  # make visible to workers too

    # Choose runner
    runner = _run_batch_inproc if args.exec_mode == "inproc" else _run_batch_subproc

    total_ok = 0
    all_bad: list[date] = []

    if jobs > 1:
        try:
            from joblib import Parallel, delayed
            print(f"[INFO] Running in parallel with {jobs} workers...")
            results = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
                delayed(runner)(b, args.limit, args.rules_config, args.etf_sheet) for b in bundles
            )
        except ImportError:
            print("[WARN] joblib not installed → falling back to serial.")
            results = [runner(b, args.limit, args.rules_config, args.etf_sheet) for b in bundles]
    else:
        print("[INFO] Running serial...")
        results = [runner(b, args.limit, args.rules_config, args.etf_sheet) for b in bundles]

    # Collate results
    for ok, bad in results:
        total_ok += len(ok)
        all_bad.extend(bad)

    # Final log
    print(f"[DONE] Replayed {total_ok}/{len(days)} days. Failed: {len(all_bad)}")
    if all_bad:
        print("[FAIL LIST]", ", ".join(sorted(d.strftime("%Y-%m-%d") for d in all_bad)))

if __name__ == "__main__":
    main()