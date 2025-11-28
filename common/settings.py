#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os

# Import shared paths & helpers from the same package
# If your package name differs, keep the relative import (.)
from .paths import ETF_LIST as ETF_LIST_PATH, default_etf_sheet

# -------- Excel list & sheet ----------
# Env wins; otherwise use paths.py heuristic (project-aware).
DEFAULT_ETF_SHEET: str = os.environ.get("ETF_SHEET", default_etf_sheet())

# Public names used elsewhere
ETF_LIST_PATH_STR = str(ETF_LIST_PATH)

# -------- Target bucket (project-aware) ----------
def _default_bucket() -> str:
    # 1) Explicit env takes precedence
    env = os.environ.get("TARGET_BUCKET", "").strip()
    if env:
        return env

    # 2) Infer from project identity
    ident = (os.environ.get("PROJECT_NAME")
             or os.environ.get("PROJECT_ROOT", "")
             or "").lower()

    # ETF workflows
    if "ny" in ident:
        return "targeted_ETFs_NY"
    if "lse" in ident or "uk" in ident:
        return "targeted_ETFs_LSE"
    if "us" in ident and "etf" in ident:
        return "targeted_ETFs_US"

    # Stock workflows (adjust to your naming)
    if "stock" in ident and ("eu" in ident or "europe" in ident):
        return "targeted_Stocks_EU"
    if "stock" in ident and ("ny" in ident or "us" in ident):
        return "targeted_Stocks_NY"

    # Safe fallback
    return "targeted_ETFs_US"

TARGET_BUCKET: str = _default_bucket()

__all__ = [
    "ETF_LIST_PATH_STR",
    "DEFAULT_ETF_SHEET",
    "TARGET_BUCKET",
]