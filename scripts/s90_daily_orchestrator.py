#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s90_daily_orchestrator.py
End-to-end runner for daily updates, param sweeps, ML training, and final tables.

Typical runs
------------
# 1) Light data + final table
python scripts/s90_daily_orchestrator.py --update-daily --include-30m --finalize-signals

# 2) Full ML refresh (rebuild s67→s74) with whitelist + operator cuts
python scripts/s90_daily_orchestrator.py \
  --retrain-ml --build-ml-whitelist --finalize-signals --operator-views \
  --decision-threshold 0.10

# 3) Print plan only
python scripts/s90_daily_orchestrator.py --update-daily --include-weekly --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

# ---------- project-aware paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from common.paths import P  # ROOT, etc.
except Exception:
    # Fallback: assume project root is two levels up from this script
    class _P:
        ROOT = PROJECT_ROOT
        SCRIPTS = PROJECT_ROOT / "scripts"
        PARAM_RESULTS = PROJECT_ROOT / "param_results"
        MODELS = PROJECT_ROOT / "models"
    P = _P()  # type: ignore

PY = sys.executable  # use this environment's Python


# ---------- CLI ----------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="s90_daily_orchestrator.py",
        description="Daily pipeline runner for data updates, param sweeps, ML, and final signals."
    )

    # General
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan but do not execute steps.")
    ap.add_argument("--keep-going", action="store_true",
                    help="Do not stop on first failing step.")
    ap.add_argument("--verbose", action="store_true",
                    help="More logging.")
    ap.add_argument("--max-workers", type=int, default=0,
                    help="Optional process/thread cap for internal steps (0 = script defaults).")

    # (1) Data updates
    ap.add_argument("--update-daily", action="store_true",
                    help="Append DAILY bars (s10).")
    ap.add_argument("--include-30m", action="store_true",
                    help="Also append 30-minute bars (s11).")
    ap.add_argument("--include-weekly", action="store_true",
                    help="Also append weekly bars (s12).")

    # (2) Param sweeps
    ap.add_argument("--run-s60", action="store_true",
                    help="Run the multi-ticker param sweep (s60).")
    ap.add_argument("--run-s63", action="store_true",
                    help="Aggregate/Rank param runs (s63→s63b→s64→s65 chain).")

    # (3) ML block
    ap.add_argument("--retrain-ml", action="store_true",
                    help="Rebuild s67→s68→s69→s70 (+ s71/72/72b/73) and export live CSV.")
    ap.add_argument("--decision-threshold", type=float, default=0.50,
                    help="Global decision threshold for s70 (y_prob ≥ thr ⇒ 1).")

    # (4) Final tables
    ap.add_argument("--finalize-signals", action="store_true",
                    help="Produce final signals table (s74).")
    ap.add_argument("--operator-views", action="store_true",
                    help="Export operator cuts (s74a).")

    # Whitelist (optional)
    ap.add_argument("--build-ml-whitelist", action="store_true",
                    help="Recompute ML whitelist from per-ticker metrics (s70d).")

    return ap


# ---------- helpers ----------
def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_step(cmd: list[str], desc: str, dry: bool, verbose: bool) -> int:
    """Run a subprocess step; return exit code."""
    pretty = " ".join(cmd)
    print(f"\n[RUN] {desc}\n      $ {pretty}")
    if dry:
        print("      (dry-run: skipped)")
        return 0
    try:
        if verbose:
            return subprocess.call(cmd, cwd=str(P.ROOT))
        # non-verbose: capture and echo only on failure
        proc = subprocess.run(cmd, cwd=str(P.ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
        else:
            if proc.stdout.strip():
                print(proc.stdout.strip())
        return proc.returncode
    except FileNotFoundError:
        print(f"[ERR] Script not found: {cmd[1]}")
        return 127


def script_path(filename: str) -> str:
    return str((P.ROOT / "scripts" / filename).resolve())


# ---------- plan builder ----------
def build_plan(args: argparse.Namespace) -> list[dict]:
    plan: list[dict] = []

    # (1) Data updates
    if args.update_daily:
        plan.append(dict(key="s10", desc="Append DAILY bars (s10_ibkr_download_daily.py)"))
    if args.include_30m:
        plan.append(dict(key="s11", desc="Append 30-minute bars (s11_ibkr_download_30min.py)"))
    if args.include_weekly:
        plan.append(dict(key="s12", desc="Append WEEKLY bars (s12_ibkr_download_weekly.py)"))

    # (2) Param sweeps / aggregation
    if args.run_s60:
        plan.append(dict(key="s60", desc="Param sweep all tickers (s60_parametizer_all_tickers.py)"))
    if args.run_s63:
        plan.extend([
            dict(key="s63",  desc="Aggregate param runs (s63_aggregate_param_runs.py)"),
            dict(key="s63b", desc="Per-seed champions + stability (s63b_champions_per_seed.py)"),
            dict(key="s64",  desc="Export final champion combos (s64_select_and_export.py)"),
            dict(key="s65",  desc="Subperiod split for champions (s65_subperiod_split.py)"),
        ])

    # (3) ML block
    if args.retrain_ml:
        plan.extend([
            dict(key="s67",   desc="Index all trades (s67_index_all_trades.py)"),
            dict(key="s67b",  desc="Enrich trades index (s67b_enrich_trades_index.py)"),
            dict(key="s68",   desc="Robust split (s68_split_dataset.py)"),
            dict(key="s69",   desc="Per-ticker quality audit (s69_quality_check.py)"),
            dict(key="s70",   desc=f"Train classifier @ thr={args.decision_threshold:.2f} (s70_train_model_classifier_calibrated.py)"),
            dict(key="s70a",  desc="Split metrics report (s70a_metrics_split_report.py)"),
            dict(key="s73",   desc="Per-ticker metrics (s73_eval_per_ticker.py)"),
            dict(key="s71",   desc="Permutation importance (s71_permutation_importance.py)"),
            dict(key="s72",   desc="Reliability audit (s72_reliability_audit_classifier.py)"),
            dict(key="s72b",  desc="Feature guard (s72a_check_features.py)"),
        ])
        if args.build_ml_whitelist:
            plan.append(dict(key="s70d", desc="Build ML whitelist (s70d_build_ml_whitelist.py)"))
        plan.append(dict(key="s70e", desc="Quick live summary (s70e_quick_live_summary.py)"))

    # (4) Final tables
    if args.finalize_signals:
        plan.append(dict(key="s74",  desc="Finalize signals table (s74_finalize_signals_table.py)"))
    if args.operator_views:
        plan.append(dict(key="s74a", desc="Operator cuts (s74a_operator_views.py)"))

    return plan


# ---------- execution map ----------
def execute_step(step_key: str, args: argparse.Namespace, dry: bool, verbose: bool) -> int:
    k = step_key.lower()
    mw = str(args.max_workers) if args.max_workers and args.max_workers > 0 else None

    # Data updates
    if k == "s10":
        cmd = [PY, script_path("s10_ibkr_download_daily.py")]
        return run_step(cmd, "s10: Append DAILY bars", dry, verbose)
    if k == "s11":
        cmd = [PY, script_path("s11_ibkr_download_30min.py")]
        return run_step(cmd, "s11: Append 30-minute bars", dry, verbose)
    if k == "s12":
        cmd = [PY, script_path("s12_ibkr_download_weekly.py")]
        return run_step(cmd, "s12: Append WEEKLY bars", dry, verbose)

    # Param sweeps / aggregation
    if k == "s60":
        cmd = [PY, script_path("s60_parametizer_all_tickers.py")]
        if mw:  # only pass if user specified
            cmd += ["--n_jobs", mw]
        return run_step(cmd, "s60: Param sweep multi-ticker", dry, verbose)

    if k == "s63":
        cmd = [PY, script_path("s63_aggregate_param_runs.py")]
        return run_step(cmd, "s63: Aggregate param runs", dry, verbose)

    if k == "s63b":
        cmd = [PY, script_path("s63b_champions_per_seed.py")]
        if mw:
            cmd += ["--n_jobs", mw]
        return run_step(cmd, "s63b: Per-seed champions + stability", dry, verbose)

    if k == "s64":
        cmd = [PY, script_path("s64_select_and_export.py")]
        return run_step(cmd, "s64: Export final champion combos", dry, verbose)

    if k == "s65":
        cmd = [PY, script_path("s65_subperiod_split.py")]
        return run_step(cmd, "s65: Subperiod split for champions", dry, verbose)

    # ML block
    if k == "s67":
        cmd = [PY, script_path("s67_index_all_trades.py")]
        return run_step(cmd, "s67: Index all trades", dry, verbose)

    if k == "s67b":
        cmd = [PY, script_path("s67b_enrich_trades_index.py")]
        return run_step(cmd, "s67b: Enrich trades index", dry, verbose)

    if k == "s68":
        cmd = [PY, script_path("s68_split_dataset.py")]
        return run_step(cmd, "s68: Split dataset", dry, verbose)

    if k == "s69":
        cmd = [PY, script_path("s69_quality_check.py")]
        return run_step(cmd, "s69: Per-ticker quality audit", dry, verbose)

    if k == "s70":
        cmd = [PY, script_path("s70_train_model_classifier_calibrated.py"),
               "--decision_threshold", f"{args.decision_threshold:.4f}"]
        return run_step(cmd, "s70: Train classifier (calibrated) + live export", dry, verbose)

    if k == "s70a":
        cmd = [PY, script_path("s70a_metrics_split_report.py")]
        return run_step(cmd, "s70a: Split metrics summary", dry, verbose)

    if k == "s73":
        cmd = [PY, script_path("s73_eval_per_ticker.py")]
        return run_step(cmd, "s73: Per-ticker metrics", dry, verbose)

    if k == "s71":
        cmd = [PY, script_path("s71_permutation_importance.py")]
        return run_step(cmd, "s71: Permutation importance", dry, verbose)

    if k == "s72":
        cmd = [PY, script_path("s72_reliability_audit_classifier.py")]
        return run_step(cmd, "s72: Reliability audit", dry, verbose)

    if k == "s72b":
        cmd = [PY, script_path("s72a_check_features.py")]
        return run_step(cmd, "s72b: Feature guard", dry, verbose)

    if k == "s70d":
        cmd = [PY, script_path("s70d_build_ml_whitelist.py")]
        return run_step(cmd, "s70d: Build ML whitelist", dry, verbose)

    if k == "s70e":
        cmd = [PY, script_path("s70e_quick_live_summary.py")]
        return run_step(cmd, "s70e: Quick live summary", dry, verbose)

    # Final tables
    if k == "s74":
        cmd = [PY, script_path("s74_finalize_signals_table.py")]
        return run_step(cmd, "s74: Finalize signals table", dry, verbose)

    if k == "s74a":
        cmd = [PY, script_path("s74a_operator_views.py")]
        return run_step(cmd, "s74a: Operator cuts", dry, verbose)

    print(f"[WARN] Unknown plan key: {step_key}")
    return 0


# ---------- main ----------
def main():
    args = build_arg_parser().parse_args()
    plan = build_plan(args)

    if not plan:
        print("Nothing to do. Provide flags (e.g., --update-daily, --retrain-ml, --finalize-signals).")
        return

    print(f"[s90] {ts()}")
    print("[s90] Project root:", P.ROOT)
    print("\n[PLAN]")
    for i, step in enumerate(plan, 1):
        print(f"  {i:02d}. {step['desc']}")

    if args.dry_run:
        print("\n[dry-run] Plan printed; no steps executed.")
        return

    failures = 0
    for i, step in enumerate(plan, 1):
        rc = execute_step(step["key"], args, dry=False, verbose=args.verbose)
        if rc != 0:
            failures += 1
            print(f"[FAIL] Step {i:02d} returned code {rc}: {step['desc']}")
            if not args.keep_going:
                print("[s90] Aborting due to failure. Use --keep-going to continue on errors.")
                sys.exit(rc)

    if failures == 0:
        print("\n[s90] ✅ All steps completed successfully.")
    else:
        print(f"\n[s90] ⚠️  Completed with {failures} failure(s).")


if __name__ == "__main__":
    main()