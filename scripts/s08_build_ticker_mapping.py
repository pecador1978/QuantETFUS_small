#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s08_build_ticker_mapping.py
Discover correct IBKR contract lines for LSE-traded ETFs/ETPs (prefer USD lines)
and write config/ticker_mapping.csv (semicolon-separated).

Discovery strategy:
- reqMatchingSymbols(ticker) to enumerate candidates
- for each candidate, fetch ContractDetails by conId (only if conId>0)
- score candidates preferring: currency USD, and primaryExchange in
  [LSEETF, LSEETP, LSE, LSEIOB]
- save as: SecType=STK, Exchange=SMART, Currency=USD/GBP, PrimaryExchange=<chosen LSE segment>

Optional manual overrides: config/ticker_overrides.csv with columns:
  Ticker;SecType;Exchange;Currency;PrimaryExchange;ConId;Symbol;LocalSymbol
Rows found here are used as-is and discovery is skipped for that ticker.

Notes:
- NEVER set exchange to LSEETF/LSEETP/LSE/LSEIOB. Those belong in primaryExchange.
- Contracts should be created as: exchange="SMART", primaryExchange="LSEETF"/... .
"""

from __future__ import annotations
from pathlib import Path
import sys, argparse, time
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from ib_insync import IB, Contract, ContractDetails

# ---------- project-aware paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # expects CONFIG_DIR, etc.

# Fallback shared (only if you point --excel there)
SHARED_BASE = Path("/Users/Finance/QuantShared")

# ---------- constants ----------
LSE_SEGMENTS = ("LSEETF", "LSEETP", "LSE", "LSEIOB")
PREFERRED_CCY = ("USD", "GBP")   # prefer USD, fallback GBP
OUTPUT_CSV = P.CONFIG_DIR / "ticker_mapping.csv"
OVERRIDES_CSV = P.CONFIG_DIR / "ticker_overrides.csv"

# ---------- helpers ----------
def _norm_ticker(s: str) -> str:
    return str(s).strip().upper().replace(" ", "").replace("\u00A0", "")

def _load_universe(xlsx: Path, sheet: str, ticker_col: str) -> List[str]:
    if not xlsx.exists():
        raise SystemExit(f"[ERR] Universe Excel not found: {xlsx}")
    df = pd.read_excel(xlsx, sheet_name=sheet)
    cols_map = {c.strip().lower(): c for c in df.columns}
    if ticker_col.lower() not in cols_map:
        raise SystemExit(f"[ERR] Column '{ticker_col}' not found in {xlsx}:{sheet}")
    col = cols_map[ticker_col.lower()]
    vals = (
        df[col].astype(str).str.strip().str.upper()
          .replace({"": None}).dropna().unique().tolist()
    )
    return [_norm_ticker(v) for v in vals if v and v not in {"NAN","NONE"}]

def _read_overrides(path: Path) -> Dict[str, Dict[str, Any]]:
    d: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return d
    df = pd.read_csv(path, sep=";")
    need = {"Ticker","SecType","Exchange","Currency","PrimaryExchange"}
    if not need.issubset(set(df.columns)):
        raise SystemExit(f"[ERR] {path} must contain columns: {sorted(need)}")
    for _, r in df.iterrows():
        t = _norm_ticker(r.get("Ticker",""))
        if not t: continue
        d[t] = {
            "Ticker": t,
            "SecType": str(r.get("SecType","STK")).upper() or "STK",
            "Exchange": str(r.get("Exchange","SMART")).upper() or "SMART",
            "Currency": str(r.get("Currency","USD")).upper() or "USD",
            "PrimaryExchange": str(r.get("PrimaryExchange","")).upper(),
            "ConId": int(r.get("ConId")) if pd.notna(r.get("ConId")) else None,
            "Symbol": str(r.get("Symbol")) if pd.notna(r.get("Symbol")) else t,
            "LocalSymbol": str(r.get("LocalSymbol")) if pd.notna(r.get("LocalSymbol")) else t,
        }
    print(f"[INFO] Overrides loaded: {len(d)}")
    return d

def _score_candidate(cd: ContractDetails) -> Tuple[int, int, int]:
    """
    Higher tuple is better when sorted reverse:
    (ccyScore, pxScore, conIdScore)
    """
    ccy = (cd.contract.currency or "").upper()
    px  = (cd.contract.primaryExchange or "").upper()
    ccyScore = 2 if ccy == "USD" else (1 if ccy == "GBP" else 0)
    pxScore  = 1 if px in LSE_SEGMENTS else 0
    conIdScore = int(cd.contract.conId or 0)
    return (ccyScore, pxScore, conIdScore)

def _fetch_cd_by_conid(ib: IB, conId: int) -> Optional[ContractDetails]:
    if not conId or conId <= 0:
        return None
    try:
        cds = ib.reqContractDetails(Contract(conId=conId))
        return cds[0] if cds else None
    except Exception as e:
        print(f"[WARN] reqContractDetails({conId}) failed: {e}")
        return None

def _is_lse_px(cd: ContractDetails) -> bool:
    return (cd.contract.primaryExchange or "").upper() in LSE_SEGMENTS

def _discover_one(ib: IB, ticker: str) -> Optional[Dict[str, Any]]:
    # 1) Enumerate by symbol matches (then expand via conId → details)
    details: List[ContractDetails] = []
    try:
        matches = ib.reqMatchingSymbols(ticker)
    except Exception as e:
        print(f"[WARN] reqMatchingSymbols({ticker}) failed: {e}")
        matches = []

    for m in matches:
        c = m.contract
        if (c.secType or "").upper() != "STK":
            continue
        conid = int(getattr(c, "conId", 0) or 0)
        if conid <= 0:
            continue
        cd = _fetch_cd_by_conid(ib, conid)
        if not cd:
            continue
        if _is_lse_px(cd):
            details.append(cd)
        time.sleep(0.03)

    # 2) Fallback probe: explicit SMART + primaryExchange + currency
    if not details:
        for px in LSE_SEGMENTS:
            for cur in PREFERRED_CCY:
                try:
                    probe = Contract(
                        symbol=ticker, secType="STK",
                        exchange="SMART", currency=cur,
                        primaryExchange=px
                    )
                    cds = ib.reqContractDetails(probe)
                    for cd in cds:
                        if (cd.contract.secType or "").upper() == "STK" and _is_lse_px(cd):
                            details.append(cd)
                    time.sleep(0.03)
                except Exception:
                    pass

    # 3) LocalSymbol probe: try <ticker>USD and <ticker>GBP explicitly
    if not details:
        for cur in PREFERRED_CCY:
            try_syms = [f"{ticker}{cur}", ticker]
            for sym in try_syms:
                for px in LSE_SEGMENTS:
                    probe = Contract(
                        symbol=sym, secType="STK",
                        exchange="SMART", currency=cur,
                        primaryExchange=px
                    )
                    try:
                        cds = ib.reqContractDetails(probe)
                        for cd in cds:
                            if (cd.contract.secType or "").upper() == "STK" and _is_lse_px(cd):
                                details.append(cd)
                        time.sleep(0.03)
                    except Exception:
                        continue

    if not details:
        print(f"[HINT] {ticker}: No LSE line found. If this UCITS trades under a different LSE symbol, add an override in {OVERRIDES_CSV}.")
        return None

    best = sorted(details, key=_score_candidate, reverse=True)[0]
    c = best.contract
    primary = (c.primaryExchange or "").upper()
    currency = (c.currency or "").upper() or "USD"
    symbol = c.symbol or ticker
    localsym = c.localSymbol or ticker
    conid = int(c.conId or 0)

    # Enforce SMART for exchange; segments go into primaryExchange
    return {
        "Ticker": ticker,
        "SecType": "STK",
        "Exchange": "SMART",
        "Currency": currency,
        "PrimaryExchange": primary if primary else "LSE",
        "ConId": conid if conid > 0 else None,
        "Symbol": symbol,
        "LocalSymbol": localsym,
    }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Build/refresh IBKR ticker mapping for LSE ETFs/ETPs.")
    ap.add_argument("--excel", type=str, default=str(SHARED_BASE / "ETF_list.xlsx"),
                    help="Path to ETF list Excel (default: QuantShared/ETF_list.xlsx)")
    ap.add_argument("--sheet", type=str, default="signalsUSD",
                    help="Worksheet name with the Ticker column (default: signalsUSD)")
    ap.add_argument("--ticker_col", type=str, default="Ticker")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496, help="TWS: 7496, IBGW paper: 4002, TWS live: 7496")
    ap.add_argument("--client-id", type=int, default=88)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", type=str, default="", help="Comma-separated tickers to map (skip universe)")
    ap.add_argument("--write-missing-only", action="store_true",
                    help="Skip discovery for tickers already in mapping (unless overridden).")
    args = ap.parse_args()

    # Universe
    if args.only:
        tickers = [_norm_ticker(x) for x in args.only.split(",") if x.strip()]
    else:
        tickers = _load_universe(Path(args.excel), args.sheet, args.ticker_col)
    if args.limit > 0:
        tickers = tickers[: args.limit]
    if not tickers:
        print("[ERR] No tickers to process.")
        sys.exit(1)

    # Existing mapping (keep rows for unknown sheets too)
    existing: Dict[str, Dict[str, Any]] = {}
    if OUTPUT_CSV.exists():
        dfm = pd.read_csv(OUTPUT_CSV, sep=";")
        for _, r in dfm.iterrows():
            t = _norm_ticker(r.get("Ticker",""))
            if not t: continue
            row = {k: r.get(k) for k in dfm.columns}
            row["Ticker"] = t
            existing[t] = row

    # ✅ FIX: pass the path
    overrides = _read_overrides(OVERRIDES_CSV)

    # Connect IB
    ib = IB()
    print(f"[IB] Connecting {args.host}:{args.port} (cid={args.client_id}) …")
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected.")

    try:
        out_rows: List[Dict[str, Any]] = []
        # start by carrying existing rows so we preserve other lists
        if existing:
            out_rows.extend(existing.values())

        for i, t in enumerate(tickers, 1):
            # override ⇒ use as-is
            if t in overrides:
                m = overrides[t]
                print(f"[{i}/{len(tickers)}] {t} (override) → {m.get('PrimaryExchange','')}/{m.get('Currency','')}")
                row = {
                    "Ticker": t,
                    "SecType": m.get("SecType","STK"),
                    "Exchange": "SMART",  # enforce SMART even if override forgot
                    "Currency": m.get("Currency","USD"),
                    "PrimaryExchange": m.get("PrimaryExchange","").upper(),
                    "ConId": m.get("ConId", None),
                    "Symbol": m.get("Symbol", t),
                    "LocalSymbol": m.get("LocalSymbol", t),
                }
            else:
                # skip if already mapped and write-missing-only
                if args.write_missing_only and t in existing:
                    print(f"[{i}/{len(tickers)}] {t} already mapped; skipping discovery")
                    # keep the existing one (already in out_rows)
                    continue
                m = _discover_one(ib, t)
                if not m:
                    print(f"[WARN] {t}: discovery failed; omitting from map")
                    # If it exists already, keep the existing mapping; else skip
                    if t in existing:
                        continue
                    else:
                        continue
                row = m

            # de-duplicate per Ticker (replace)
            out_rows = [r for r in out_rows if _norm_ticker(r.get("Ticker","")) != t]
            out_rows.append(row)

            ib.sleep(0.05)  # be gentle

    finally:
        ib.disconnect()

    if not out_rows:
        raise SystemExit("[ERR] No rows to write.")

    cols = ["Ticker","SecType","Exchange","Currency","PrimaryExchange","ConId","Symbol","LocalSymbol"]
    df_out = pd.DataFrame(out_rows, columns=cols).drop_duplicates(subset=["Ticker"])
    df_out = df_out.sort_values("Ticker").reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, sep=";", index=False)
    print(f"[OK] Mapping saved → {OUTPUT_CSV} (rows={len(df_out)})")
    if overrides:
        print(f"[INFO] Overrides file → {OVERRIDES_CSV}")

if __name__ == "__main__":
    main()