#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s09_backfill_missing_history.py
⚠️ One-time migration script — only needed if you have old files without `_raw` suffix.
Not required in daily/weekly pipeline once files are normalized.

Rename existing historical ETF files to the new "*_raw.csv" convention.

Changes per subfolder under the shared raw repo:
  daily : *_daily.csv   -> *_daily_raw.csv
  30min : *_30min.csv   -> *_30min_raw.csv
  weekly: *_weekly.csv  -> *_weekly_raw.csv

Usage examples:
  python scripts/s09_backfill_missing_history.py --dry_run
  python scripts/s09_backfill_missing_history.py --only daily
  python scripts/s09_backfill_missing_history.py --base /Users/Finance/QuantShared/data_raw_ETF_EU
"""

from pathlib import Path
import argparse
import os

# ---------------- args ----------------
DEFAULT_BASE = "/Users/Finance/QuantShared/data_raw_ETF_US"

ap = argparse.ArgumentParser()
ap.add_argument(
    "--base",
    default=os.environ.get("SHARED_RAW_BASE", DEFAULT_BASE),
    help="Path to shared raw repo (default US). E.g. /Users/Finance/QuantShared/data_raw_ETF_EU",
)
ap.add_argument(
    "--dry_run",
    action="store_true",
    help="Preview changes without renaming.",
)
ap.add_argument(
    "--only",
    choices=["daily", "30min", "weekly"],
    nargs="*",
    default=None,
    help="Restrict to one or more buckets: daily 30min weekly",
)
args = ap.parse_args()

BASE_DIR = Path(args.base)

# ---------------- config ----------------
SUFFIX_MAP = {
    "daily":  ("_daily.csv",  "_daily_raw.csv"),
    "30min":  ("_30min.csv",  "_30min_raw.csv"),
    "weekly": ("_weekly.csv", "_weekly_raw.csv"),
}

# ---------------- impl ----------------
def rename_files(folder_name: str, old_suffix: str, new_suffix: str, dry_run: bool = False) -> int:
    folder = BASE_DIR / folder_name
    if not folder.exists():
        print(f"[WARN] Missing folder: {folder}")
        return 0
    count = 0
    for p in sorted(folder.glob(f"*{old_suffix}")):
        new_path = p.with_name(p.name.replace(old_suffix, new_suffix))
        if new_path == p:
            continue
        if dry_run:
            print(f"DRY-RUN: {p.name}  ->  {new_path.name}")
        else:
            p.rename(new_path)
            print(f"RENAMED: {p.name}  ->  {new_path.name}")
            count += 1
    print(f"[INFO] {'Would rename' if dry_run else 'Renamed'} {count} file(s) in '{folder_name}'.")
    return count

def main():
    targets = args.only or list(SUFFIX_MAP.keys())
    print(f"[INFO] Base dir: {BASE_DIR}")
    total = 0
    for bucket in targets:
        old_sfx, new_sfx = SUFFIX_MAP[bucket]
        total += rename_files(bucket, old_sfx, new_sfx, dry_run=args.dry_run)
    if args.dry_run:
        print(f"[INFO] DRY-RUN complete. Total files that would be renamed: {total}")
    else:
        print(f"[OK] Done. Total files renamed: {total}")

if __name__ == "__main__":
    main()