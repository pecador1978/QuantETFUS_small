#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s12_ibkr_download_weekly.py — Append-only WEEKLY updater (IBKR via ib_insync, UTC)

- Universe: ETF_list.xlsx (sheet from env ETF_SHEET or paths.default_etf_sheet()).
- Output: <OUTDIR>/{TICKER}_weekly_raw.csv (append-only). OUTDIR defaults to P.SHARED_WEEKLY_DIR.
- Mapping: prefers conId; robust CSV reader (comma/semicolon/BOM/case-insensitive).
- Venue & currency: driven by env/CLI (no LSE hard-coding):
    MAPPING_PRIMARY_EXCH_SEGMENTS="ARCA,NASDAQ,NYSE,ISLAND,BATS"  # example for US
    MAPPING_PREFERRED_CCY="USD,EUR"
- First-run vs delta:
    * If CSV missing → FULL_DURATION_W (default "30 Y")
    * If CSV exists → DELTA_DURATION_W (default "26 W")
- Optional --duration overrides both behaviors for all tickers.
"""

from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple, List
import os, argparse, sys, time, pandas as pd
from ib_insync import IB, Contract, Stock, ContractDetails, util

# ---------- project-aware imports ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import P                          # SHARED_WEEKLY_DIR, CONFIG_DIR, etc.
from common.settings import ETF_LIST_PATH_STR, DEFAULT_ETF_SHEET

# ---------- env-driven defaults (local vars; no globals) ----------
ENV_SEGMENTS = os.environ.get("MAPPING_PRIMARY_EXCH_SEGMENTS", "")
DEFAULT_EXCHANGE_SEGMENTS: Tuple[str, ...] = tuple(
    s.strip().upper() for s in ENV_SEGMENTS.split(",") if s.strip()
) or ("ARCA","NASDAQ","NYSE","ISLAND","BATS")

ENV_CCY = os.environ.get("MAPPING_PREFERRED_CCY", "")
DEFAULT_PREFERRED_CCY: Tuple[str, ...] = tuple(
    c.strip().upper() for c in ENV_CCY.split(",") if c.strip()
) or ("USD","GBP")

# First-run / delta windows (overridable via env)
FULL_DURATION_W  = os.environ.get("IB_FULL_DURATION_W",  "30 Y")
DELTA_DURATION_W = os.environ.get("IB_DELTA_DURATION_W", "26 W")

# ---------- paths & files ----------
OUTDIR_DEFAULT = P.SHARED_WEEKLY_DIR
ETF_XLSX  = Path(ETF_LIST_PATH_STR).resolve()
ETF_SHEET = DEFAULT_ETF_SHEET
ETF_COL   = os.environ.get("ETF_TICKER_COL", "Ticker")
MAPPING_CSV_DEFAULT = Path(
    os.environ.get("TICKER_MAPPING_CSV", str(P.CONFIG_DIR / "ticker_mapping.csv"))
).expanduser().resolve()

# ---------- IB noise muter ----------
@contextmanager
def mute_ib(ib: IB, suppress=(162, 200, 321)):
    def _h(reqId, code, msg, *a, **k):
        try:
            c = int(code)
        except Exception:
            c = None
        if c in suppress:
            return
        print(f"[IB {code}] {msg}")
    ib.errorEvent += _h
    try:
        yield
    finally:
        try:
            ib.errorEvent -= _h
        except Exception:
            pass

# ---------- Excel & mapping ----------
def load_tickers_from_excel(path_xlsx: str | Path, sheet: str, col_name: str = "Ticker") -> List[str]:
    x = Path(path_xlsx).expanduser().resolve()
    if not x.exists():
        raise SystemExit(f"[ERR] Universe Excel not found: {x}")
    df = pd.read_excel(x, sheet_name=sheet)
    colmap = {c.lower().strip(): c for c in df.columns}
    key = col_name.lower().strip()
    if key not in colmap:
        raise SystemExit(f"[ERR] Column '{col_name}' not found in {x.name} (sheet '{sheet}').")
    s = df[colmap[key]].astype(str).str.strip().str.upper()
    return [t for t in s.unique().tolist() if t and t not in {"NAN","NONE"}]

def _read_mapping_df(path_csv: Path) -> pd.DataFrame:
    if not path_csv.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path_csv, sep=None, engine="python")
    except Exception:
        for sep in (",",";"):
            try:
                df = pd.read_csv(path_csv, sep=sep)
                break
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            return df
    df.columns = [str(c).encode("utf-8").decode("utf-8-sig").strip() for c in df.columns]
    return df

def load_mapping(path_csv: str | Path) -> Dict[str, Dict[str, Any]]:
    df = _read_mapping_df(Path(path_csv))
    if df.empty:
        return {}
    cmap = {c.lower(): c for c in df.columns}
    if "ticker" not in cmap:
        raise ValueError(f"{path_csv} must include a 'Ticker' column.")
    def col_or_default(name: str, default: str) -> pd.Series:
        k = name.lower()
        if k in cmap:
            s = df[cmap[k]]
            return s if isinstance(s, pd.Series) else pd.Series([default] * len(df))
        return pd.Series([default] * len(df))
    norm = pd.DataFrame({
        "Ticker":          col_or_default("Ticker",          "").astype(str).str.strip().str.upper(),
        "conId":           col_or_default("ConId",           "").astype(str).str.strip(),
        "SecType":         col_or_default("SecType",         "STK").astype(str).str.strip(),
        "Exchange":        col_or_default("Exchange",        "SMART").astype(str).str.strip(),
        "Currency":        col_or_default("Currency",        (DEFAULT_PREFERRED_CCY[0] if DEFAULT_PREFERRED_CCY else "USD")).astype(str).str.strip(),
        "PrimaryExchange": col_or_default("PrimaryExchange", "").astype(str).str.strip(),
        "Symbol":          col_or_default("Symbol",          "").astype(str).str.strip(),
        "LocalSymbol":     col_or_default("LocalSymbol",     "").astype(str).str.strip(),
    }).fillna("")
    out: Dict[str, Dict[str, Any]] = {}
    for _, r in norm.iterrows():
        t = r["Ticker"]
        if not t:
            continue
        out[t] = {
            "conId":           r["conId"],
            "SecType":         r["SecType"] or "STK",
            "Exchange":        r["Exchange"] or "SMART",
            "Currency":        r["Currency"] or (DEFAULT_PREFERRED_CCY[0] if DEFAULT_PREFERRED_CCY else "USD"),
            "PrimaryExchange": r["PrimaryExchange"],
            "Symbol":          r["Symbol"] or t,
            "LocalSymbol":     r["LocalSymbol"],
        }
    return out

# ---------- Contract helpers ----------
def contract_from_map(ticker: str, mapping: Dict[str, Dict[str, Any]]) -> Optional[Contract]:
    m = mapping.get(ticker.upper())
    if not m:
        return None
    conId = m.get("conId","")
    if conId and str(conId).isdigit():
        c = Contract()
        c.conId = int(conId)
        return c
    c = Contract()
    c.symbol        = m.get("Symbol") or ticker
    c.localSymbol   = m.get("LocalSymbol") or c.symbol
    c.secType       = m.get("SecType") or "STK"
    c.exchange      = m.get("Exchange") or "SMART"
    c.currency      = m.get("Currency") or (DEFAULT_PREFERRED_CCY[0] if DEFAULT_PREFERRED_CCY else "USD")
    pe              = m.get("PrimaryExchange","")
    if pe:
        c.primaryExchange = pe
    return c

def _is_primary_px(cd: ContractDetails, segments: Tuple[str, ...]) -> bool:
    return (cd.contract.primaryExchange or "").upper() in segments

def discover_contract(ib: IB, ticker: str, preferred_ccy: Tuple[str, ...], segments: Tuple[str, ...]) -> Optional[Contract]:
    for cur in preferred_ccy:
        for seg in segments:
            probe = Stock(ticker, 'SMART', cur, primaryExchange=seg)
            try:
                cds = ib.reqContractDetails(probe)
                for cd in cds:
                    c = cd.contract
                    if (c.secType or "").upper() in {"STK","ETF"} and _is_primary_px(cd, segments):
                        return c
            except Exception:
                continue
    for cur in preferred_ccy:
        try:
            got = ib.qualifyContracts(Stock(ticker, 'SMART', cur))
            if got:
                return got[0]
        except Exception:
            continue
    return None

# ---------- IO ----------
def append_only(out_csv: Path, df_new: pd.DataFrame) -> int:
    if df_new is None or df_new.empty:
        return 0
    df_new = df_new.copy()
    df_new["date"] = pd.to_datetime(df_new["date"], utc=True, errors="coerce")
    df_new = df_new.dropna(subset=["date"])
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        if "date" not in old.columns:
            raise ValueError(f"{out_csv} missing 'date'")
        old["date"] = pd.to_datetime(old["date"], utc=True, errors="coerce")
        last = old["date"].max()
        add = df_new[df_new["date"] > last] if pd.notna(last) else df_new
        merged = pd.concat([old, add], ignore_index=True)
    else:
        add = df_new
        merged = df_new
    merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return len(add)

# ---------- Data fetch ----------
def fetch_weekly(ib: IB, con: Contract, duration: str, what: str, use_rth: bool) -> pd.DataFrame:
    bars = ib.reqHistoricalData(
        con,
        endDateTime="",
        durationStr=duration,
        barSizeSetting="1 week",
        whatToShow=what,
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
    )
    if not bars:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df = util.df(bars)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume","average","barCount"]:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date","open","high","low","close","volume","average","barCount"]]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)
    ap.add_argument("--client_id", type=int, default=62)

    ap.add_argument("--duration", default=None, help="Override for all tickers (e.g., '10 Y', '52 W').")
    ap.add_argument("--what", default="TRADES", choices=["TRADES","MIDPOINT","BID_ASK"])
    ap.add_argument("--use_rth", type=int, default=1)

    ap.add_argument("--excel", default=str(ETF_XLSX))
    ap.add_argument("--sheet", default=str(ETF_SHEET))
    ap.add_argument("--ticker_col", default=str(ETF_COL))
    ap.add_argument("--mapping", default=str(MAPPING_CSV_DEFAULT))
    ap.add_argument("--outdir", default=str(OUTDIR_DEFAULT))
    ap.add_argument("--sleep_ms", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--skip", nargs="*", default=[])

    # venue/currency preferences
    ap.add_argument("--segments", default=",".join(DEFAULT_EXCHANGE_SEGMENTS),
                    help="PrimaryExchange segments to prefer (env MAPPING_PRIMARY_EXCH_SEGMENTS)")
    ap.add_argument("--ccy", default=",".join(DEFAULT_PREFERRED_CCY),
                    help="Preferred currencies (env MAPPING_PREFERRED_CCY)")

    args = ap.parse_args()

    # local prefs (no globals)
    ex_segments: Tuple[str, ...] = tuple(s.strip().upper() for s in args.segments.split(",") if s.strip()) or DEFAULT_EXCHANGE_SEGMENTS
    pref_ccy: Tuple[str, ...]    = tuple(c.strip().upper() for c in args.ccy.split(",") if c.strip()) or DEFAULT_PREFERRED_CCY

    out_root = Path(args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Excel={ETF_XLSX} | sheet={ETF_SHEET}")
    print(f"[INFO] outdir={out_root}")
    print(f"[INFO] Mapping={Path(args.mapping).expanduser().resolve()}")
    print(f"[INFO] Segments={ex_segments} | CCY={pref_ccy}")
    print(f"[INFO] Duration(full={FULL_DURATION_W}, delta={DELTA_DURATION_W}) | RTH={args.use_rth}")

    tickers = load_tickers_from_excel(args.excel, args.sheet, args.ticker_col)
    if args.only:
        allow = {t.upper() for t in args.only}
        tickers = [t for t in tickers if t in allow]
    if args.skip:
        block = {t.upper() for t in args.skip}
        tickers = [t for t in tickers if t not in block]
    if args.limit and args.limit > 0:
        tickers = tickers[: args.limit]

    mapping = load_mapping(args.mapping)

    ib = IB()
    print(f"[IB] Connecting {args.host}:{args.port} (clientId={args.client_id}) …")
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected].")

    with mute_ib(ib):
        for i, t in enumerate(tickers, 1):
            try:
                con = contract_from_map(t, mapping)
                if con:
                    q = ib.qualifyContracts(con)
                    con = q[0] if q else None
                if not con:
                    con = discover_contract(ib, t, pref_ccy, ex_segments)
                if not con:
                    print(f"[{i}/{len(tickers)}] {t}: [SKIP] could not qualify")
                    continue

                out_csv = out_root / f"{t}_weekly_raw.csv"
                if args.duration:
                    duration = args.duration
                else:
                    duration = DELTA_DURATION_W if out_csv.exists() else FULL_DURATION_W
                    mode = "delta" if out_csv.exists() else "full"
                    print(f"[INFO] {t}: {mode} duration {duration}")

                df = fetch_weekly(ib, con, duration, args.what, bool(args.use_rth))
                if df.empty:
                    print(f"[{i}/{len(tickers)}] {t}: +0 → {out_csv}")
                    continue

                added = append_only(out_csv, df)
                print(f"[{i}/{len(tickers)}] {t}: +{added} → {out_csv}")
                ib.sleep(max(args.sleep_ms, 0) / 1000.0)

            except Exception as e:
                print(f"[{i}/{len(tickers)}] {t}: [ERR] {e}")

    ib.disconnect()
    print(f"[DONE] s12_ibkr_download_weekly (append-only). Dir → {out_root}")

if __name__ == "__main__":
    main()