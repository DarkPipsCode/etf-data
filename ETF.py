"""End-to-end UCITS ETF ranking report - single-script edition.

Pulls the Xetra ETF master sheet, classifies every product by strategy /
asset class, fetches adjusted-close history from Yahoo, computes performance
/ risk / technical metrics (incl. tracking error vs a benchmark proxy), and
writes a ranked Markdown report plus CSVs.

Usage:
    python etf_report.py                       # default 5y, full universe
    python etf_report.py --max-etfs 500        # stratified cap for speed
    python etf_report.py --refresh             # re-download master + repull prices
    python etf_report.py --period 3y --rf 0.025
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "output"

XETRA_MASTER_URL = (
    "https://www.cashmarket.deutsche-boerse.com/resource/blob/1553442/"
    "e9c6cbeee02c4be0f53d1d8e31d373b3/data/Master_DataSheet_Download.xls"
)
XETRA_CACHE = DATA_DIR / "xetra_master.xls"
PRICE_CACHE = CACHE_DIR / "prices.csv"
JUSTETF_CACHE = DATA_DIR / "justetf_universe.json"

JUSTETF_SCREENER_URL = "https://www.justetf.com/servlet/etfs-table"
JUSTETF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.justetf.com/en/find-etf.html",
    "X-Requested-With": "XMLHttpRequest",
}
JUSTETF_TTL_DAYS = 7

# ── OpenFIGI (free ISIN→ticker resolver) ────────────────────────────────────
OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
OPENFIGI_API_KEY = os.environ.get("OPENFIGI_API_KEY", "")  # optional, raises rate limit
OPENFIGI_CACHE = DATA_DIR / "openfigi_cache.json"
OPENFIGI_TTL_DAYS = 30
OPENFIGI_BATCH = 100

# OpenFIGI Bloomberg-style exchange code → Yahoo Finance ticker suffix.
_FIGI_TO_YF: dict[str, str] = {
    "LN": "L",   "GR": "DE",  "GY": "DE",  "GF": "F",
    "AS": "AS",  "IM": "MI",  "FP": "PA",  "SM": "MC",
    "SW": "SW",  "VX": "SW",  "SS": "ST",  "DC": "CO",
    "FH": "HE",  "NO": "OL",  "ID": "IR",  "BB": "BR",
    "AV": "VI",
}
# Preferred listing order when an ISIN has multiple matches (best yfinance coverage first).
_YF_PREF_ORDER = ["L", "DE", "AS", "MI", "PA", "SW", "MC", "ST", "CO", "HE", "OL", "IR", "BR", "VI", "F"]

# ── EODHD (paid: Morningstar ratings, fundamentals) ─────────────────────────
EODHD_API_KEY = os.environ.get("EODHD_API_KEY", "")
EODHD_BASE = "https://eodhistoricaldata.com/api/fundamentals"
EODHD_CACHE_DIR = DATA_DIR / "eodhd_cache"
EODHD_TTL_DAYS = 7

TRADING_DAYS = 252


# ============================================================================
# Universe loading
# ============================================================================

@dataclass
class UniverseStats:
    rows_raw: int
    rows_dedup: int
    ucits_count: int
    with_ter: int
    with_replication: int
    with_benchmark: int


def download_xetra_master(cache_path: Path = XETRA_CACHE, force: bool = False) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        return cache_path
    resp = requests.get(XETRA_MASTER_URL, timeout=60)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return cache_path


def _normalize_replication(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().lower()
    if not s:
        return ""
    if "synthetic" in s or "swap" in s:
        return "Synthetic"
    if "sampling" in s:
        return "Physical (sampling)"
    if "physical" in s or "full replication" in s or "full" in s:
        return "Physical (full)"
    return s.title()


def _normalize_distribution(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().lower()
    if "accum" in s:
        return "Accumulating"
    if "distrib" in s:
        return "Distributing"
    return s.title()


def _xetra_to_yahoo(symbol: object) -> str:
    if symbol is None or (isinstance(symbol, float) and pd.isna(symbol)):
        return ""
    s = str(symbol).strip()
    if not s:
        return ""
    return f"{s}.DE"


def parse_xetra(path: Path) -> tuple[pd.DataFrame, UniverseStats]:
    raw = pd.read_excel(path, sheet_name=0, header=8, engine="openpyxl")
    raw = raw.rename(columns=lambda c: str(c).strip())

    df = pd.DataFrame({
        "isin": raw["ISIN"].astype(str).str.strip(),
        "name": raw["PRODUCT NAME"].astype(str).str.strip(),
        "issuer": raw["PRODUCT FAMILY"].astype(str).str.strip(),
        "xetra_symbol": raw["XETRA SYMBOL"].astype(str).str.strip(),
        "bloomberg_ticker": raw["BLOOMBERG TICKER"].astype(str).str.strip(),
        "ter": pd.to_numeric(raw["ONGOING CHARGES"], errors="coerce"),
        "distribution": raw["USE OF PROFITS"].map(_normalize_distribution),
        "replication": raw["REPLICATION METHOD"].map(_normalize_replication),
        "fund_currency": raw["FUND CURRENCY"].astype(str).str.strip(),
        "trading_currency": raw["TRADING CURRENCY"].astype(str).str.strip(),
        "benchmark": raw["BENCHMARK"].astype(str).str.strip().replace({"nan": ""}),
        "product_type": raw["PRODUCT TYPE"].astype(str).str.strip(),
    })
    df["yahoo_ticker"] = raw["XETRA SYMBOL"].map(_xetra_to_yahoo)
    rows_raw = len(df)

    df = df[df["isin"].str.match(r"^[A-Z]{2}[A-Z0-9]{9}\d$", na=False)]

    df.loc[df["product_type"].str.contains("Active", case=False, na=False),
           "replication"] = df["replication"].where(df["replication"] != "", "Active")

    df["__pref"] = df["trading_currency"].map({"EUR": 0, "USD": 1}).fillna(2)
    df = df.sort_values(["isin", "__pref"]).drop_duplicates("isin", keep="first")
    df = df.drop(columns="__pref").reset_index(drop=True)

    stats = UniverseStats(
        rows_raw=rows_raw,
        rows_dedup=len(df),
        ucits_count=int(df["name"].str.contains("UCITS", case=False, na=False).sum()),
        with_ter=int(df["ter"].notna().sum()),
        with_replication=int((df["replication"] != "").sum()),
        with_benchmark=int((df["benchmark"] != "").sum()),
    )
    return df, stats


def load_universe(
    ucits_only: bool = True,
    refresh: bool = False,
    cache_path: Path = XETRA_CACHE,
    include_justetf: bool = False,
) -> tuple[pd.DataFrame, UniverseStats]:
    path = download_xetra_master(cache_path, force=refresh)
    df, stats = parse_xetra(path)
    if ucits_only:
        df = df[df["name"].str.contains("UCITS", case=False, na=False)].reset_index(drop=True)
    if include_justetf:
        je = fetch_justetf_universe(force=refresh)
        df, stats = merge_justetf_into_xetra(df, je, stats)
    return df, stats


# ============================================================================
# justETF supplemental scraper
# ============================================================================
#
# Xetra master data covers everything listed on Xetra (~3,400 products) but is
# missing UCITS ETFs that aren't cross-listed in Frankfurt, and its benchmark
# / strategy fields are sparser than justETF's. This module fills both gaps:
#   - extends the universe with non-Xetra UCITS ISINs from justETF
#   - fills missing TER, benchmark, replication, strategy fields where Xetra
#     has nothing useful
#
# The endpoint behind justETF's screener is a DataTables-style POST that
# returns ~3,000 UCITS ETFs in pages of 100. Cached on disk for a week.

def _fetch_justetf_page(start: int, length: int = 100, country: str = "DE") -> dict:
    form = {
        "draw": "1",
        "start": str(start),
        "length": str(length),
        "lang": "en",
        "country": country,
        "universeType": "private",
        "etfsParams": "",
    }
    resp = requests.post(
        JUSTETF_SCREENER_URL,
        data=form,
        headers=JUSTETF_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        m = re.search(r"\{.*\}", resp.text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _normalize_justetf_row(row: dict) -> dict:
    def g(*keys: str, default=None):
        for k in keys:
            v = row.get(k)
            if v not in (None, "", "-"):
                return v
        return default

    ter_raw = g("ter", "totalExpenseRatio")
    if isinstance(ter_raw, str):
        m = re.search(r"([0-9]+(?:[.,][0-9]+)?)", ter_raw)
        ter = float(m.group(1).replace(",", ".")) / 100 if m else None
    elif isinstance(ter_raw, (int, float)):
        ter = float(ter_raw) / 100 if ter_raw > 1 else float(ter_raw)
    else:
        ter = None

    return {
        "isin": g("isin"),
        "wkn": g("wkn"),
        "ticker": g("ticker", "symbol"),
        "name": g("name", "fundName"),
        "asset_class_je": g("assetClass"),
        "instrument": g("instrument"),
        "region": g("region"),
        "strategy": g("strategy"),
        "currency_je": g("currency", "fundCurrency"),
        "replication_je": g("replication"),
        "distribution_je": g("distributionPolicy", "useOfProfits", "dividends"),
        "inception": g("inceptionDate"),
        "ter_je": ter,
        "fund_size_je": g("fundSize"),
        "index_name": g("indexName", "index", "underlying"),
        "industry": g("industry", "sector"),
    }


def fetch_justetf_universe(
    force: bool = False, cache_path: Path = JUSTETF_CACHE
) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        age = time.time() - cache_path.stat().st_mtime
        if age < JUSTETF_TTL_DAYS * 86400:
            try:
                return pd.DataFrame(json.loads(cache_path.read_text(encoding="utf-8")))
            except Exception as exc:
                print(f"  ! justETF cache read failed ({exc}); refetching")

    print("  Scraping justETF screener…")
    rows: list[dict] = []
    try:
        first = _fetch_justetf_page(0, 100)
    except Exception as exc:
        print(f"  ! justETF first page failed: {exc}; skipping enrichment")
        return pd.DataFrame()
    total = int(first.get("recordsTotal") or first.get("recordsFiltered") or 0)
    rows.extend(_normalize_justetf_row(r) for r in (first.get("data") or []))

    for offset in range(100, max(total, 100), 100):
        try:
            page = _fetch_justetf_page(offset, 100)
        except Exception as exc:
            print(f"  ! justETF page offset={offset} failed: {exc}")
            continue
        rows.extend(_normalize_justetf_row(r) for r in (page.get("data") or []))
        time.sleep(0.15)  # be polite

    rows = [r for r in rows if r.get("isin")]
    by_isin: dict[str, dict] = {}
    for r in rows:
        cur = by_isin.get(r["isin"])
        if cur is None or sum(v is not None for v in r.values()) > sum(v is not None for v in cur.values()):
            by_isin[r["isin"]] = r
    rows = list(by_isin.values())

    cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"  justETF: {len(rows)} unique UCITS ETFs (cached)")
    return pd.DataFrame(rows)


def merge_justetf_into_xetra(
    xetra: pd.DataFrame, je: pd.DataFrame, stats: UniverseStats
) -> tuple[pd.DataFrame, UniverseStats]:
    """Union by ISIN; fill TER/benchmark/strategy/etc. gaps from justETF."""
    if je.empty:
        return xetra, stats

    xetra = xetra.copy()
    je = je.copy()

    # 1. Fill missing fields on Xetra rows from justETF.
    je_idx = je.set_index("isin")
    for col_xetra, col_je in [
        ("ter", "ter_je"),
        ("benchmark", "index_name"),
        ("replication", "replication_je"),
        ("distribution", "distribution_je"),
        ("fund_currency", "currency_je"),
    ]:
        if col_je not in je_idx.columns:
            continue
        mapper = je_idx[col_je]
        if col_xetra not in xetra.columns:
            xetra[col_xetra] = pd.NA
        if pd.api.types.is_numeric_dtype(xetra[col_xetra]):
            mask = xetra["isin"].isin(mapper.index) & xetra[col_xetra].isna()
        else:
            mask = xetra["isin"].isin(mapper.index) & (
                xetra[col_xetra].isna() | (xetra[col_xetra].astype(str).str.strip() == "")
            )
        if mask.any():
            xetra.loc[mask, col_xetra] = xetra.loc[mask, "isin"].map(mapper)

    # 2. Bring across justETF strategy/industry as new columns for the classifier.
    for col in ("strategy", "industry", "asset_class_je", "region"):
        if col in je_idx.columns and col not in xetra.columns:
            xetra[col] = xetra["isin"].map(je_idx[col])

    # 3. Add justETF-only ISINs (ETFs not listed on Xetra).
    extra = je[~je["isin"].isin(xetra["isin"])].copy()
    if not extra.empty:
        extra = extra.rename(columns={
            "ter_je": "ter",
            "index_name": "benchmark",
            "replication_je": "replication",
            "distribution_je": "distribution",
            "currency_je": "fund_currency",
        })
        # Pad columns to match Xetra schema
        for c in xetra.columns:
            if c not in extra.columns:
                extra[c] = pd.NA
        # Best-effort yahoo ticker from primary listing if available
        extra["yahoo_ticker"] = extra.get("ticker", pd.Series([""] * len(extra))).fillna("").astype(str).str.upper()
        # Tag origin so downstream knows it came from justETF
        extra["product_type"] = extra.get("product_type", pd.Series([""] * len(extra))).fillna("UCITS ETF (justETF)")
        merged = pd.concat([xetra, extra[xetra.columns]], ignore_index=True)
    else:
        merged = xetra

    # Refresh stats
    new_stats = UniverseStats(
        rows_raw=stats.rows_raw,
        rows_dedup=len(merged),
        ucits_count=int(merged["name"].str.contains("UCITS", case=False, na=False).sum()),
        with_ter=int(pd.to_numeric(merged["ter"], errors="coerce").notna().sum()),
        with_replication=int((merged["replication"].fillna("").astype(str) != "").sum()),
        with_benchmark=int((merged["benchmark"].fillna("").astype(str) != "").sum()),
    )
    print(f"  After justETF merge: {len(merged)} rows ({len(extra) if not extra.empty else 0} new from justETF)")
    return merged, new_stats


# ============================================================================
# OpenFIGI — ISIN → exchange ticker resolver (free, no key required)
# ============================================================================
#
# Many UCITS ETFs added via justETF have no yfinance-compatible ticker, and
# yfinance often lacks data for the .DE listing even when one exists. OpenFIGI
# returns every known listing for an ISIN; we pick the most-tradeable one and
# fall back through a preference order (London > XETRA > Amsterdam > …).

def _openfigi_request(batch: list[dict]) -> list[dict]:
    headers = {"Content-Type": "application/json"}
    if OPENFIGI_API_KEY:
        headers["X-OPENFIGI-APIKEY"] = OPENFIGI_API_KEY
    r = requests.post(OPENFIGI_URL, json=batch, headers=headers, timeout=30)
    if r.status_code == 429:
        # Backoff and retry once.
        time.sleep(8)
        r = requests.post(OPENFIGI_URL, json=batch, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_openfigi(isins: list[str], force: bool = False,
                   cache_path: Path = OPENFIGI_CACHE) -> dict[str, list[dict]]:
    """Map ISINs → list of {ticker, exchCode, name, securityType} candidates.
    Cached on disk for OPENFIGI_TTL_DAYS days."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached: dict[str, list[dict]] = {}
    fresh = (
        cache_path.exists()
        and not force
        and (time.time() - cache_path.stat().st_mtime) < OPENFIGI_TTL_DAYS * 86400
    )
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  ! OpenFIGI cache read failed ({exc}); refetching")
            cached = {}

    todo = sorted({i for i in isins if i and (force or i not in cached)})
    if not todo and fresh:
        print(f"  OpenFIGI: cache hit ({len(cached)} ISINs)")
        return cached
    if not todo:
        return cached

    print(f"  OpenFIGI: resolving {len(todo)} ISINs (have {len(cached)} cached)")
    sleep_s = 0.3 if OPENFIGI_API_KEY else 0.6
    for start in range(0, len(todo), OPENFIGI_BATCH):
        batch_isins = todo[start:start + OPENFIGI_BATCH]
        body = [{"idType": "ID_ISIN", "idValue": i} for i in batch_isins]
        try:
            results = _openfigi_request(body)
        except Exception as exc:
            print(f"  ! OpenFIGI batch start={start} failed: {exc}")
            time.sleep(2)
            continue
        for isin, result in zip(batch_isins, results):
            data = result.get("data") if isinstance(result, dict) else None
            cached[isin] = [
                {
                    "ticker": d.get("ticker"),
                    "exchCode": d.get("exchCode"),
                    "name": d.get("name"),
                    "securityType": d.get("securityType"),
                }
                for d in (data or [])
                if d.get("ticker")
            ]
        time.sleep(sleep_s)

    cache_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    return cached


def best_yf_ticker_from_figi(figi_results: list[dict]) -> str:
    """Pick the most likely-tradeable yfinance ticker from FIGI matches."""
    candidates: list[tuple[str, str]] = []  # (suffix, ticker)
    for d in figi_results:
        exch = (d.get("exchCode") or "").strip()
        sec = (d.get("securityType") or "").strip()
        tkr = (d.get("ticker") or "").strip()
        if not exch or not tkr:
            continue
        if sec and sec not in {"ETP", "Equity", "Equity Index", "Mutual Fund", "Open-End Fund"}:
            continue
        suffix = _FIGI_TO_YF.get(exch)
        if not suffix:
            continue
        candidates.append((suffix, tkr))
    for pref in _YF_PREF_ORDER:
        for suf, tkr in candidates:
            if suf == pref:
                return f"{tkr}.{suf}"
    return ""


def enrich_tickers_via_openfigi(universe: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """For ETFs with no yahoo_ticker, ask OpenFIGI for one. Returns a new DF."""
    out = universe.copy()
    if "yahoo_ticker" not in out.columns:
        out["yahoo_ticker"] = ""
    out["yahoo_ticker"] = out["yahoo_ticker"].fillna("").astype(str)
    needs = out[out["yahoo_ticker"].str.strip() == ""]
    if needs.empty:
        print("  OpenFIGI: every ETF already has a ticker; skipping")
        return out

    figi = fetch_openfigi(needs["isin"].tolist(), force=force)
    filled = 0
    for isin in needs["isin"]:
        cand = best_yf_ticker_from_figi(figi.get(isin, []))
        if cand:
            out.loc[out["isin"] == isin, "yahoo_ticker"] = cand
            filled += 1
    print(f"  OpenFIGI: filled {filled} previously-missing tickers")
    return out


# ============================================================================
# EODHD — Morningstar rating + fundamentals (paid, opt-in)
# ============================================================================
#
# Requires EODHD_API_KEY env var. Returns NaN columns gracefully if unset.
# Fields surfaced: Morningstar star rating, sustainability rating, holdings
# count, AUM ($M), distribution yield, top-10 concentration.

def _calc_top10_concentration(top_holdings) -> float:
    if not top_holdings or not isinstance(top_holdings, dict):
        return float("nan")
    total = 0.0
    for v in top_holdings.values():
        if isinstance(v, dict):
            w = v.get("Assets_%") or v.get("Weight") or 0
            try:
                total += float(w)
            except (ValueError, TypeError):
                pass
    return total / 100 if total > 0 else float("nan")


def fetch_eodhd_one(ticker: str) -> dict:
    """Fetch ETF fundamentals from EODHD. Cached per ticker on disk."""
    if not EODHD_API_KEY or not ticker:
        return {}
    EODHD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = EODHD_CACHE_DIR / f"{ticker.replace('.', '_').replace('/', '_')}.json"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < EODHD_TTL_DAYS * 86400:
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    url = f"{EODHD_BASE}/{ticker}?api_token={EODHD_API_KEY}"
    data: dict = {}
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            raw = r.json() or {}
            etf = raw.get("ETF_Data") or {}
            ms = etf.get("Morningstar") or {}
            data = {
                "morningstar_rating": ms.get("Ratio"),
                "sustainability_rating": ms.get("Sustainability_Ratio"),
                "category_benchmark": ms.get("Category_Benchmark"),
                "holdings_count": etf.get("Holdings_Count"),
                "aum_mil": etf.get("Market_Capitalisation_Mil"),
                "yield": etf.get("Yield"),
                "top10_concentration": _calc_top10_concentration(etf.get("Top_10_Holdings")),
            }
        elif r.status_code != 404:
            print(f"  ! EODHD {ticker}: HTTP {r.status_code}")
    except Exception as exc:
        print(f"  ! EODHD {ticker}: {exc}")

    cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def enrich_with_eodhd(merged: pd.DataFrame) -> pd.DataFrame:
    """Add Morningstar/EODHD columns to the merged DataFrame."""
    new_cols = ["morningstar_rating", "sustainability_rating", "holdings_count",
                "aum_mil", "yield", "top10_concentration", "category_benchmark"]
    out = merged.copy()
    for c in new_cols:
        if c not in out.columns:
            out[c] = pd.NA
    if not EODHD_API_KEY:
        print("  ! EODHD_API_KEY env var not set — skipping EODHD overlay")
        return out
    if "yahoo_ticker" not in out.columns:
        return out

    tickers = sorted({t for t in out["yahoo_ticker"].dropna().astype(str) if t})
    print(f"  EODHD: fetching fundamentals for {len(tickers)} tickers (TTL {EODHD_TTL_DAYS}d)")
    fetched = 0
    for tkr in tickers:
        d = fetch_eodhd_one(tkr)
        if not d:
            continue
        mask = out["yahoo_ticker"] == tkr
        for c in new_cols:
            if c in d and d[c] is not None:
                out.loc[mask, c] = d[c]
        fetched += 1
        if fetched % 50 == 0:
            print(f"    EODHD: {fetched}/{len(tickers)} done")
    print(f"  EODHD: fetched {fetched} (rest from cache or skipped)")
    return out


# ============================================================================
# Categorisation
# ============================================================================

LEVERAGED_RE = re.compile(
    r"(?:\b|[-\s])(?:2x|3x|-1x|-2x|-3x|leveraged|inverse|daily\s+leveraged)\b|\bshort\s+(?:dax|stoxx|s&p|ftse|msci)",
    re.IGNORECASE,
)

ESG_RE = re.compile(
    r"\b(esg|sri|socially\s+responsible|sustainab|paris[-\s]?aligned|"
    r"\bpab\b|\bctb\b|\bcta\b|climate\s+(transition|aligned|action)|"
    r"low\s+carbon|net[-\s]?zero|screened|ex[-\s]?fossil)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Rule:
    category: str
    asset_class: str
    pattern: re.Pattern[str]


def _r(pat: str) -> re.Pattern[str]:
    return re.compile(pat, re.IGNORECASE)


RULES: list[Rule] = [
    Rule("Crypto - Bitcoin",       "Crypto",   _r(r"\bbitcoin|btc\b")),
    Rule("Crypto - Ethereum",      "Crypto",   _r(r"\bethereum|ether\b|\beth\b")),
    Rule("Crypto - Other",         "Crypto",   _r(r"\b(solana|sol|xrp|cardano|polkadot|avalanche|crypto)\b")),

    Rule("Commodity - Gold",       "Commodity", _r(r"\bgold\b")),
    Rule("Commodity - Silver",     "Commodity", _r(r"\bsilver\b")),
    Rule("Commodity - Platinum",   "Commodity", _r(r"\bplatinum\b")),
    Rule("Commodity - Palladium",  "Commodity", _r(r"\bpalladium\b")),
    Rule("Commodity - Oil & Gas",  "Commodity", _r(r"\b(oil|brent|wti|natural\s+gas)\b")),
    Rule("Commodity - Industrial Metals", "Commodity", _r(r"\b(copper|nickel|zinc|aluminium|aluminum|industrial\s+metals)\b")),
    Rule("Commodity - Agriculture","Commodity", _r(r"\b(agriculture|wheat|corn|soybean|coffee|sugar|livestock)\b")),
    Rule("Commodity - Broad",      "Commodity", _r(r"\b(commodit|brc|broad\s+commodities|bloomberg\s+commodity)\b")),

    Rule("Bond - EUR Govt",        "Fixed Income", _r(r"(eur|euro)\s+(gov|government|sovereign|treasury)|\bbund\b|\bbtp\b|\boat\b")),
    Rule("Bond - US Treasury",     "Fixed Income", _r(r"(us|usd|u\.s\.|treasury|t-bond|t-bill).*?(treasur|gov)|treasury\s+bond")),
    Rule("Bond - UK Gilts",        "Fixed Income", _r(r"\bgilt|uk\s+gov|gbp\s+gov")),
    Rule("Bond - Inflation-Linked","Fixed Income", _r(r"inflation|tips|linker|index-linked|inflation-linked")),
    Rule("Bond - High Yield",      "Fixed Income", _r(r"\bhigh\s+yield|hy\b|fallen\s+angels")),
    Rule("Bond - EM Debt",         "Fixed Income", _r(r"(emerging|em)\s+(market.?\s+)?(debt|bond|sovereign)|\bembi\b")),
    Rule("Bond - IG Corporate",    "Fixed Income", _r(r"corporate\s+bond|ig\s+corp|investment\s+grade")),
    Rule("Bond - Aggregate",       "Fixed Income", _r(r"aggregate|global\s+(bond|aggregate)|multiverse|euro\s+aggregate")),
    Rule("Bond - Money Market",    "Fixed Income", _r(r"money\s+market|t-bill|ultra[-\s]?short|overnight|eonia|estr|sofr|sonia")),
    Rule("Bond - Convertible",     "Fixed Income", _r(r"convertible")),
    Rule("Bond - Other",           "Fixed Income", _r(r"\bbond|fixed\s+income|debt\b")),

    Rule("Sector - Real Estate",   "Real Estate", _r(r"\b(real\s+estate|reit|property)\b")),

    Rule("Factor - Momentum",      "Equity Factor", _r(r"\bmomentum\b")),
    Rule("Factor - Quality",       "Equity Factor", _r(r"\bquality\b")),
    Rule("Factor - Value",         "Equity Factor", _r(r"\bvalue\b(?!.*growth)")),
    Rule("Factor - Min Vol",       "Equity Factor", _r(r"min(?:imum)?\s+volatility|low\s+vol(?:atility)?|min\s*vol")),
    Rule("Factor - Multifactor",   "Equity Factor", _r(r"multi[-\s]?factor|multifactor")),
    Rule("Factor - Equal Weight",  "Equity Factor", _r(r"equal[-\s]?weight")),
    Rule("Factor - Dividend",      "Equity Factor", _r(r"dividend|yield(?!.*bond)|income(?!.*bond)|aristocrat|dividend\s+leaders")),
    Rule("Factor - Buyback",       "Equity Factor", _r(r"buyback|share\s+repurch")),
    Rule("Factor - Growth",        "Equity Factor", _r(r"\bgrowth\b")),

    Rule("Thematic - Clean Energy",    "Thematic", _r(r"clean\s+energy|renewable|solar|wind\s+energy")),
    Rule("Thematic - Battery / EV",    "Thematic", _r(r"battery|electric\s+vehicle|\bev\b|lithium|future\s+mobility|smart\s+mobility")),
    Rule("Thematic - Robotics & AI",   "Thematic", _r(r"robotic|artificial\s+intelligence|\bai\b|automation")),
    Rule("Thematic - Cybersecurity",   "Thematic", _r(r"cyber|security\b")),
    Rule("Thematic - Semiconductors",  "Thematic", _r(r"semiconductor|chip\b")),
    Rule("Thematic - Cloud / Software","Thematic", _r(r"cloud|software|saas|digital\s+economy|digitali[sz]ation")),
    Rule("Thematic - Biotech",         "Thematic", _r(r"biotech|genomic|genome|immuno")),
    Rule("Thematic - Water",           "Thematic", _r(r"\bwater\b")),
    Rule("Thematic - Infrastructure",  "Thematic", _r(r"infrastructure")),
    Rule("Thematic - Defence",         "Thematic", _r(r"defen[cs]e|aerospace\s*&?\s*defen|military")),
    Rule("Thematic - Blockchain",      "Thematic", _r(r"blockchain|metaverse|web3|nft")),
    Rule("Thematic - Healthcare Innovation", "Thematic", _r(r"digital\s+health|healthcare\s+innovation|aging|longevity")),
    Rule("Thematic - Travel & Leisure","Thematic", _r(r"travel|leisure|gaming|esports")),
    Rule("Thematic - Fintech",         "Thematic", _r(r"\bfintech\b|digital\s+payment|payment\s+(innov|tech)|digital\s+bank|neobank")),
    Rule("Thematic - Smart Cities",    "Thematic", _r(r"smart\s+cit|smart\s+infrastructure|resilient\s+future|future\s+cit|urbani[sz]")),
    Rule("Thematic - Innovation",      "Thematic", _r(r"\binnovat")),
    Rule("Thematic - Other",           "Thematic", _r(r"thematic|disruptive|next\s+gen|future\s+of")),

    Rule("Sector - Technology",        "Sector", _r(r"\btechnology|info\s*tech|technology\s+sector|\bit\s+sector")),
    Rule("Sector - Financials",        "Sector", _r(r"financials?|banks?\b|insurance")),
    Rule("Sector - Healthcare",        "Sector", _r(r"health\s*care|healthcare|pharma")),
    Rule("Sector - Energy",            "Sector", _r(r"\benergy\b")),
    Rule("Sector - Consumer Staples",  "Sector", _r(r"consumer\s+staples|food\s*&\s*bever")),
    Rule("Sector - Consumer Discretionary","Sector", _r(r"consumer\s+discretionary|consumer\s+services|automobile")),
    Rule("Sector - Industrials",       "Sector", _r(r"\bindustrials?\b")),
    Rule("Sector - Utilities",         "Sector", _r(r"utilit")),
    Rule("Sector - Materials",         "Sector", _r(r"basic\s+materials|materials\b|mining|metals\s+&\s+mining")),
    Rule("Sector - Communication",     "Sector", _r(r"communication\s+services|telecom|media\b")),

    Rule("Equity - World",             "Equity Broad", _r(r"\bmsci\s+world\b(?!\s+small)|ftse\s+(?:developed\s+world|all[-\s]?world)|world\s+(?:equity|index|all\s*cap)|\bacwi\b|developed\s+markets?\s+(?:equity|index|all\s*cap)|global\s+(?:equity|developed)")),
    Rule("Equity - World Small Cap",   "Equity Broad", _r(r"world\s+small|developed\s+small")),
    Rule("Equity - US Large Cap",      "Equity Broad", _r(r"\bs&?p\s*500|russell\s*1000|msci\s+usa(?!\s+small)|us\s+large")),
    Rule("Equity - US Total",          "Equity Broad", _r(r"crsp\s+us|russell\s+3000|us\s+total\s+market")),
    Rule("Equity - Nasdaq / Tech",     "Equity Broad", _r(r"nasdaq|nyse\s+fang")),
    Rule("Equity - US Small Cap",      "Equity Broad", _r(r"russell\s*2000|us\s+small|usa\s+small")),
    Rule("Equity - Europe Broad",      "Equity Broad", _r(r"stoxx\s*600|euro\s*stoxx\s*600|stoxx\s+europe(?!\s+small)(?:\s*50|\s*600|\s*total)?|msci\s+europe(?!\s+small)|ftse\s+europe|ftse\s+developed\s+europe|europe\s+equity|stoxx\s+europe\s+select")),
    Rule("Equity - Eurozone",          "Equity Broad", _r(r"euro\s*stoxx\s*50|emu\b|eurozone")),
    Rule("Equity - Europe Small Cap",  "Equity Broad", _r(r"europe\s+small|eu\s+small")),
    Rule("Equity - UK",                "Equity Broad", _r(r"ftse\s*100|ftse\s*250|\bukx\b|united\s+kingdom")),
    Rule("Equity - Germany",           "Equity Broad", _r(r"\bdax\b|germany")),
    Rule("Equity - France",            "Equity Broad", _r(r"cac\s*40|france\b")),
    Rule("Equity - Switzerland",       "Equity Broad", _r(r"\bsmi\b|swiss|switzerland")),
    Rule("Equity - Japan",             "Equity Broad", _r(r"\btopix|nikkei|msci\s+japan|japan\b")),
    Rule("Equity - China",             "Equity Broad", _r(r"csi\s*300|msci\s+china|ftse\s+china|china\s+a|hsi\b|hang\s+seng|\bchina\b")),
    Rule("Equity - India",             "Equity Broad", _r(r"\bindia\b|nifty|sensex")),
    Rule("Equity - Korea",             "Equity Broad", _r(r"\bkorea\b|kospi")),
    Rule("Equity - Taiwan",            "Equity Broad", _r(r"\btaiwan\b")),
    Rule("Equity - Brazil / LatAm",    "Equity Broad", _r(r"\bbrazil\b|\blatam\b|latin\s+america|mexico\b")),
    Rule("Equity - Canada",            "Equity Broad", _r(r"\bcanada\b|s&p\s+tsx")),
    Rule("Equity - Australia",         "Equity Broad", _r(r"australia|asx\s*200")),
    Rule("Equity - Asia ex-Japan",     "Equity Broad", _r(r"asia\s+ex[-\s]?japan|asia\s+pacific")),
    Rule("Equity - Emerging Markets",  "Equity Broad", _r(r"emerging\s+market|\bem\b|msci\s+em|ftse\s+em")),
    Rule("Equity - Frontier",          "Equity Broad", _r(r"frontier")),

    # ── Europe — single country (additions) ─────────────────────────────
    Rule("Equity - Italy",             "Equity Broad", _r(r"\bitaly\b|\bitalia\b|ftse\s*mib|\bftsemib\b|msci\s+italy")),
    Rule("Equity - Spain",             "Equity Broad", _r(r"\bspain\b|ibex\s*35|msci\s+spain")),
    Rule("Equity - Netherlands",       "Equity Broad", _r(r"netherlands|\baex\b|msci\s+netherlands")),
    Rule("Equity - Sweden",            "Equity Broad", _r(r"\bsweden\b|omx\s*stockholm|omxs\s*30|msci\s+sweden")),
    Rule("Equity - Norway",            "Equity Broad", _r(r"\bnorway\b|\bobx\b|msci\s+norway")),
    Rule("Equity - Denmark",           "Equity Broad", _r(r"\bdenmark\b|omx\s*copenhagen|omxc\s*25|msci\s+denmark")),
    Rule("Equity - Finland",           "Equity Broad", _r(r"\bfinland\b|omx\s*helsinki|omxh\s*25|msci\s+finland")),
    Rule("Equity - Belgium",           "Equity Broad", _r(r"\bbelgium\b|bel\s*20|msci\s+belgium")),
    Rule("Equity - Austria",           "Equity Broad", _r(r"\baustria\b|\batx\b|msci\s+austria")),
    Rule("Equity - Ireland",           "Equity Broad", _r(r"\bireland\b|\biseq\b|msci\s+ireland")),
    Rule("Equity - Greece",            "Equity Broad", _r(r"\bgreece\b|ase\s+composite|msci\s+greece")),
    Rule("Equity - Portugal",          "Equity Broad", _r(r"\bportugal\b|psi\s*20|msci\s+portugal")),
    Rule("Equity - Poland",            "Equity Broad", _r(r"\bpoland\b|\bwig\b|msci\s+poland")),
    Rule("Equity - Russia",            "Equity Broad", _r(r"\brussia\b|\bmoex\b|\brts\b|msci\s+russia")),
    Rule("Equity - Turkey",            "Equity Broad", _r(r"\bturkey\b|\bturkiye\b|bist\s*30|msci\s+turkey")),
    Rule("Equity - Hungary",           "Equity Broad", _r(r"\bhungary\b|\bbux\b|msci\s+hungary")),
    Rule("Equity - Czech Republic",    "Equity Broad", _r(r"\bczech\b|\bpx\s+index|msci\s+czech")),
    Rule("Equity - Nordic",            "Equity Broad", _r(r"\bnordic\b|vinx\s*30")),
    Rule("Equity - Eastern Europe",    "Equity Broad", _r(r"eastern\s+europe|\bcee\b|emerging\s+europe")),

    # ── Asia — single country (additions) ───────────────────────────────
    Rule("Equity - Indonesia",         "Equity Broad", _r(r"indonesi|jakarta|\bjci\b|msci\s+indonesia")),
    Rule("Equity - Vietnam",           "Equity Broad", _r(r"vietnam|vn\s*30|msci\s+vietnam")),
    Rule("Equity - Thailand",          "Equity Broad", _r(r"thailand|set\s*50|msci\s+thailand")),
    Rule("Equity - Malaysia",          "Equity Broad", _r(r"malaysia|\bklci\b|msci\s+malaysia")),
    Rule("Equity - Singapore",         "Equity Broad", _r(r"singapore|sti\s+index|msci\s+singapore")),
    Rule("Equity - Philippines",       "Equity Broad", _r(r"philippines|\bpsei\b|msci\s+philippines")),
    Rule("Equity - Pakistan",          "Equity Broad", _r(r"pakistan|kse\s*100|msci\s+pakistan")),
    Rule("Equity - New Zealand",       "Equity Broad", _r(r"new\s+zealand|nzx\s*50|msci\s+new\s+zealand")),
    Rule("Equity - ASEAN",             "Equity Broad", _r(r"\basean\b")),
    Rule("Equity - Asia Pacific",      "Equity Broad", _r(r"asia[-\s]?pacific|msci\s+ac\s+asia|ftse\s+asia")),

    # ── Americas — single country (additions) ───────────────────────────
    Rule("Equity - Mexico",            "Equity Broad", _r(r"\bmexico\b|ipc\s*mexico|msci\s+mexico")),
    Rule("Equity - Argentina",         "Equity Broad", _r(r"argentin|merval|msci\s+argentina")),
    Rule("Equity - Chile",             "Equity Broad", _r(r"\bchile\b|\bipsa\b|msci\s+chile")),
    Rule("Equity - Colombia",          "Equity Broad", _r(r"colombia|colcap|msci\s+colombia")),
    Rule("Equity - Peru",              "Equity Broad", _r(r"\bperu\b|msci\s+peru")),

    # ── MENA / Africa (additions) ───────────────────────────────────────
    Rule("Equity - Israel",            "Equity Broad", _r(r"\bisrael\b|ta-?(35|125)|msci\s+israel")),
    Rule("Equity - Saudi Arabia",      "Equity Broad", _r(r"saudi|\btasi\b|msci\s+saudi")),
    Rule("Equity - UAE",               "Equity Broad", _r(r"\buae\b|united\s+arab|\badx\b|\bdfm\b|msci\s+uae")),
    Rule("Equity - Qatar",             "Equity Broad", _r(r"\bqatar\b|msci\s+qatar")),
    Rule("Equity - Egypt",             "Equity Broad", _r(r"\begypt\b|egx\s*30|msci\s+egypt")),
    Rule("Equity - South Africa",      "Equity Broad", _r(r"south\s+africa|jse\s*top|msci\s+south\s+africa")),
    Rule("Equity - Nigeria",           "Equity Broad", _r(r"nigeria|nse\s+30|msci\s+nigeria")),
    Rule("Equity - Kenya",             "Equity Broad", _r(r"\bkenya\b|nse\s+kenya|msci\s+kenya")),
    Rule("Equity - Morocco",           "Equity Broad", _r(r"morocc|\bmasi\b|msci\s+morocco")),
    Rule("Equity - GCC",               "Equity Broad", _r(r"\bgcc\b|gulf\s+coop|gulf\s+states")),
    Rule("Equity - MENA",              "Equity Broad", _r(r"\bmena\b|middle\s+east")),
    Rule("Equity - Africa",            "Equity Broad", _r(r"\bafrica\b(?!\s+south)|msci\s+africa|ftse\s+africa")),

    # ── Cross-region aggregations (additions) ───────────────────────────
    Rule("Equity - BRIC",              "Equity Broad", _r(r"\bbric\b|\bbrics\b")),
    Rule("Equity - World ex US",       "Equity Broad", _r(r"world\s+ex[-\s]?us|kokusai|world\s+excluding\s+us")),
    Rule("Equity - EAFE",              "Equity Broad", _r(r"\beafe\b|msci\s+eafe")),
    Rule("Equity - North America",     "Equity Broad", _r(r"north\s+america|msci\s+north\s+america")),
    Rule("Equity - Latin America",     "Equity Broad", _r(r"latin\s+america|msci\s+latam(?!\s+ex)|ftse\s+latin")),

    Rule("Active - Equity",            "Active",      _r(r"active.*equity|equity.*active")),
    Rule("Active - Fixed Income",      "Active",      _r(r"active.*bond|bond.*active|active.*income")),
]


def classify_one(
    name: str,
    benchmark: str = "",
    product_type: str = "",
    strategy: str = "",
    industry: str = "",
) -> tuple[str, str]:
    text = " ".join(s for s in (name, benchmark, strategy, industry) if s).strip()
    for rule in RULES:
        if rule.pattern.search(text):
            return rule.category, rule.asset_class

    pt = (product_type or "").lower()
    if ESG_RE.search(text):
        return "Thematic - ESG / Climate", "Thematic"
    if "etc" in pt or "etn" in pt:
        return "Other - ETC/ETN", "Commodity"
    if "active" in pt:
        return "Active - Other", "Active"
    return "Uncategorized", "Other"


def add_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("strategy", "industry"):
        if col not in out.columns:
            out[col] = ""
    pairs = [
        classify_one(n, b, pt, s, ind)
        for n, b, pt, s, ind in zip(
            out["name"].fillna(""),
            out["benchmark"].fillna(""),
            out["product_type"].fillna(""),
            out["strategy"].fillna(""),
            out["industry"].fillna(""),
        )
    ]
    out["category"] = [p[0] for p in pairs]
    out["asset_class"] = [p[1] for p in pairs]
    out["esg"] = out["name"].apply(lambda n: bool(ESG_RE.search(str(n))))
    out["leveraged"] = out["name"].apply(lambda n: bool(LEVERAGED_RE.search(str(n))))
    return out


# ============================================================================
# Pricing (yfinance + on-disk cache)
# ============================================================================

def _chunk(seq: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def fetch_prices(
    tickers: list[str],
    period: str = "5y",
    batch_size: int = 40,
    sleep_between: float = 0.4,
) -> pd.DataFrame:
    tickers = sorted({t for t in tickers if t})
    frames: list[pd.DataFrame] = []
    for i, batch in enumerate(_chunk(tickers, batch_size)):
        try:
            data = yf.download(
                batch,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as exc:
            print(f"  ! batch {i} failed: {exc}")
            continue
        if data is None or data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            close = pd.DataFrame(
                {tkr: data[tkr]["Close"] for tkr in batch if tkr in data.columns.levels[0]}
            )
        else:
            close = data[["Close"]].rename(columns={"Close": batch[0]})
        frames.append(close.dropna(how="all"))
        if sleep_between:
            time.sleep(sleep_between)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out = out.dropna(how="all")
    return out


def load_or_fetch_prices(
    tickers: list[str],
    period: str = "5y",
    refresh: bool = False,
    cache_path: Path = PRICE_CACHE,
) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        cached = pd.DataFrame()

    needed = [t for t in tickers if t and t not in cached.columns]
    if needed:
        print(f"  Fetching prices for {len(needed)} new tickers (cached: {len(cached.columns)})")
        fresh = fetch_prices(needed, period=period)
        if not fresh.empty:
            cached = pd.concat([cached, fresh], axis=1)
            cached = cached.loc[:, ~cached.columns.duplicated()]
            cached.to_csv(cache_path)

    return cached.reindex(columns=[t for t in tickers if t in cached.columns])


# ============================================================================
# Performance / risk / technical metrics
# ============================================================================

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def total_return(prices: pd.Series, lookback_days: int) -> float:
    s = prices.dropna()
    if len(s) < lookback_days + 1:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-1 - lookback_days] - 1.0)


def cagr(prices: pd.Series, years: float) -> float:
    s = prices.dropna()
    if s.empty:
        return float("nan")
    days = int(years * TRADING_DAYS)
    if len(s) < days + 1:
        return float("nan")
    total = s.iloc[-1] / s.iloc[-1 - days]
    return float(total ** (1.0 / years) - 1.0)


def annualized_vol(returns: pd.Series, window: int = TRADING_DAYS) -> float:
    r = returns.dropna().tail(window)
    if len(r) < 30:
        return float("nan")
    return float(r.std() * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf_annual: float = 0.0, window: int = TRADING_DAYS) -> float:
    r = returns.dropna().tail(window)
    if len(r) < 30:
        return float("nan")
    excess = r - rf_annual / TRADING_DAYS
    sd = r.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf_annual: float = 0.0, window: int = TRADING_DAYS) -> float:
    r = returns.dropna().tail(window)
    if len(r) < 30:
        return float("nan")
    excess = r - rf_annual / TRADING_DAYS
    downside = r[r < 0]
    if downside.empty:
        return float("nan")
    dd = downside.std()
    if dd == 0:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(prices: pd.Series, window_days: int | None = None) -> float:
    s = prices.dropna()
    if window_days:
        s = s.tail(window_days)
    if s.empty:
        return float("nan")
    peak = s.cummax()
    dd = s / peak - 1.0
    return float(dd.min())


def rsi(prices: pd.Series, length: int = 14) -> float:
    s = prices.dropna()
    if len(s) < length + 2:
        return float("nan")
    delta = s.diff().dropna()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def ma_status(prices: pd.Series) -> tuple[float, float, str, float]:
    s = prices.dropna()
    if s.empty:
        return float("nan"), float("nan"), "n/a", float("nan")
    sma50 = s.tail(50).mean() if len(s) >= 50 else float("nan")
    sma200 = s.tail(200).mean() if len(s) >= 200 else float("nan")
    regime = "n/a"
    if not (np.isnan(sma50) or np.isnan(sma200)):
        regime = "Uptrend" if sma50 > sma200 else "Downtrend"
    last = s.iloc[-1]
    high_252 = s.tail(252).max() if len(s) >= 30 else float("nan")
    dist = float(last / high_252 - 1.0) if not np.isnan(high_252) else float("nan")
    return float(sma50), float(sma200), regime, dist


def momentum_12_1(prices: pd.Series) -> float:
    s = prices.dropna()
    if len(s) < TRADING_DAYS + 21:
        return float("nan")
    return float(s.iloc[-21] / s.iloc[-21 - TRADING_DAYS] - 1.0)


def compute_metrics(prices: pd.DataFrame, rf_annual: float = 0.02) -> pd.DataFrame:
    rets = daily_returns(prices)
    rows = []
    for tkr in prices.columns:
        s = prices[tkr].dropna()
        r = rets[tkr].dropna() if tkr in rets.columns else pd.Series(dtype=float)
        if s.empty:
            continue
        sma50, sma200, regime, dist_high = ma_status(s)
        rows.append({
            "ticker": tkr,
            "obs_days": len(s),
            "ret_1m":  total_return(s, 21),
            "ret_3m":  total_return(s, 63),
            "ret_6m":  total_return(s, 126),
            "ret_1y":  total_return(s, TRADING_DAYS),
            "ret_3y":  total_return(s, TRADING_DAYS * 3),
            "cagr_3y": cagr(s, 3),
            "cagr_5y": cagr(s, 5),
            "vol_1y":  annualized_vol(r, TRADING_DAYS),
            "vol_3y":  annualized_vol(r, TRADING_DAYS * 3),
            "sharpe_1y":  sharpe(r, rf_annual, TRADING_DAYS),
            "sharpe_3y":  sharpe(r, rf_annual, TRADING_DAYS * 3),
            "sortino_1y": sortino(r, rf_annual, TRADING_DAYS),
            "max_dd_3y":  max_drawdown(s, TRADING_DAYS * 3),
            "rsi_14":     rsi(s, 14),
            "sma_50":     sma50,
            "sma_200":    sma200,
            "trend":      regime,
            "off_52w_high": dist_high,
            "mom_12_1":   momentum_12_1(s),
        })
    return pd.DataFrame(rows)


# ============================================================================
# Benchmark proxy + tracking metrics
# ============================================================================

BENCHMARK_PROXIES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"s&?p\s*500", re.I),                    "^GSPC"),
    (re.compile(r"nasdaq[-\s]?100", re.I),               "^NDX"),
    (re.compile(r"nasdaq\s+composite", re.I),            "^IXIC"),
    (re.compile(r"russell\s*2000", re.I),                "^RUT"),
    (re.compile(r"russell\s*1000", re.I),                "^RUI"),
    (re.compile(r"msci\s+world\s+small", re.I),          "WSML.L"),
    (re.compile(r"msci\s+world(?!\s+small)", re.I),      "URTH"),
    (re.compile(r"msci\s+acwi|all\s+country\s+world", re.I), "ACWI"),
    (re.compile(r"ftse\s+all[-\s]?world", re.I),         "VWRL.L"),
    (re.compile(r"msci\s+europe(?!\s+small)", re.I),     "IMEU.L"),
    (re.compile(r"stoxx\s*600|euro\s*stoxx\s*600", re.I), "^STOXX"),
    (re.compile(r"euro\s*stoxx\s*50", re.I),             "^STOXX50E"),
    (re.compile(r"\bdax\b", re.I),                        "^GDAXI"),
    (re.compile(r"cac\s*40", re.I),                       "^FCHI"),
    (re.compile(r"\bsmi\b", re.I),                        "^SSMI"),
    (re.compile(r"ftse\s*100", re.I),                     "^FTSE"),
    (re.compile(r"ftse\s*250", re.I),                     "^FTMC"),
    (re.compile(r"\btopix\b", re.I),                      "^TPX"),
    (re.compile(r"nikkei", re.I),                         "^N225"),
    (re.compile(r"msci\s+japan", re.I),                   "EWJ"),
    (re.compile(r"msci\s+em|emerging\s+markets", re.I),   "EEM"),
    (re.compile(r"csi\s*300", re.I),                      "ASHR"),
    (re.compile(r"hang\s+seng|hsi\b", re.I),              "^HSI"),
    (re.compile(r"asx\s*200", re.I),                      "^AXJO"),
    (re.compile(r"s&p\s+tsx", re.I),                      "^GSPTSE"),
    (re.compile(r"\bgold\b", re.I),                       "GC=F"),
    (re.compile(r"\bsilver\b", re.I),                     "SI=F"),
    (re.compile(r"brent", re.I),                          "BZ=F"),
    (re.compile(r"\bwti\b|crude\s+oil", re.I),            "CL=F"),
]


def map_benchmark(text: str) -> str | None:
    if not text:
        return None
    for pat, proxy in BENCHMARK_PROXIES:
        if pat.search(text):
            return proxy
    return None


def tracking_metrics(etf_prices: pd.Series, bench_prices: pd.Series) -> dict[str, float]:
    df = pd.concat([etf_prices, bench_prices], axis=1, join="inner").dropna()
    if len(df) < 60:
        return {"tracking_error": float("nan"), "tracking_diff": float("nan"),
                "correlation": float("nan"), "r2": float("nan")}
    df.columns = ["etf", "bench"]
    rets = df.pct_change().dropna()
    diff = rets["etf"] - rets["bench"]
    te = float(diff.std() * np.sqrt(252))
    last_year = rets.tail(252)
    if len(last_year) < 60:
        td = float("nan")
    else:
        etf_ret = (1 + last_year["etf"]).prod() - 1
        bench_ret = (1 + last_year["bench"]).prod() - 1
        td = float(etf_ret - bench_ret)
    corr = float(rets["etf"].corr(rets["bench"]))
    r2 = float(corr ** 2) if not np.isnan(corr) else float("nan")
    return {"tracking_error": te, "tracking_diff": td, "correlation": corr, "r2": r2}


def add_benchmark_column(universe: pd.DataFrame) -> pd.DataFrame:
    out = universe.copy()
    src = (out["benchmark"].fillna("") + " " + out["name"].fillna("")).str.strip()
    out["benchmark_proxy"] = src.map(map_benchmark).fillna("")
    return out


# ============================================================================
# Report rendering
# ============================================================================

PCT_COLS = ["ter", "ret_1m", "ret_3m", "ret_6m", "ret_1y", "cagr_3y", "cagr_5y",
            "vol_1y", "vol_3y", "max_dd_3y", "off_52w_high", "mom_12_1",
            "tracking_error", "tracking_diff"]


def _fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x*100:.{digits}f}%"


def _fmt_num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"


def build_category_summary(merged: pd.DataFrame) -> pd.DataFrame:
    has_metrics = merged[merged["sharpe_1y"].notna()].copy()
    if has_metrics.empty:
        return pd.DataFrame()

    grouped = has_metrics.groupby(["asset_class", "category"])
    summary = grouped.agg(
        n_etfs=("isin", "size"),
        median_ter=("ter", "median"),
        median_ret_1y=("ret_1y", "median"),
        median_sharpe=("sharpe_1y", "median"),
        median_vol=("vol_1y", "median"),
        median_dd=("max_dd_3y", "median"),
        median_rsi=("rsi_14", "median"),
        median_te=("tracking_error", "median"),
    ).reset_index()

    top_idx = has_metrics.groupby(["asset_class", "category"])["sharpe_1y"].idxmax()
    top_etfs = has_metrics.loc[top_idx, ["asset_class", "category", "name", "isin",
                                          "ter", "ret_1y", "sharpe_1y"]]
    top_etfs = top_etfs.rename(columns={
        "name": "top_etf", "isin": "top_isin", "ter": "top_ter",
        "ret_1y": "top_ret_1y", "sharpe_1y": "top_sharpe",
    })
    summary = summary.merge(top_etfs, on=["asset_class", "category"], how="left")
    summary = summary.sort_values(
        ["median_sharpe", "median_ret_1y"], ascending=False
    ).reset_index(drop=True)
    summary.insert(0, "rank", summary.index + 1)
    return summary


def render_category_summary_md(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "_No ETFs with sufficient price history to rank._\n"
    rows = ["| # | Asset class | Category | # ETFs | Med TER | Med 1y | Med Sharpe | Med Vol | Med MaxDD | Med RSI | Top ETF (by Sharpe) | Top Sharpe |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for _, r in summary.iterrows():
        rows.append(
            f"| {int(r['rank'])} | {r['asset_class']} | {r['category']} | {int(r['n_etfs'])} | "
            f"{_fmt_pct(r['median_ter'], 2)} | {_fmt_pct(r['median_ret_1y'])} | "
            f"{_fmt_num(r['median_sharpe'])} | {_fmt_pct(r['median_vol'])} | "
            f"{_fmt_pct(r['median_dd'])} | {_fmt_num(r['median_rsi'], 1)} | "
            f"{r['top_etf']} | {_fmt_num(r['top_sharpe'])} |"
        )
    return "\n".join(rows)


def build_drilldown(merged: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    has_metrics = merged[merged["sharpe_1y"].notna()].copy()
    return (
        has_metrics.sort_values(["asset_class", "category", "sharpe_1y"], ascending=[True, True, False])
        .groupby(["asset_class", "category"], as_index=False)
        .head(top_n)
    )


def render_drilldown_md(drilldown: pd.DataFrame) -> str:
    if drilldown.empty:
        return "_No drilldown available._\n"
    blocks: list[str] = []
    for (asset_class, category), grp in drilldown.groupby(["asset_class", "category"]):
        blocks.append(f"\n### {asset_class} — {category}  ({len(grp)} shown)\n")
        blocks.append(
            "| Rank | ETF | ISIN | TER | Replication | 1y | 3y CAGR | Vol 1y | Sharpe | RSI(14) | Trend | MaxDD 3y | Tracking err |"
        )
        blocks.append("|---:|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|")
        for i, (_, r) in enumerate(grp.iterrows(), start=1):
            blocks.append(
                f"| {i} | {r['name']} | {r['isin']} | {_fmt_pct(r['ter'], 2)} | "
                f"{r.get('replication') or '—'} | {_fmt_pct(r['ret_1y'])} | "
                f"{_fmt_pct(r['cagr_3y'])} | {_fmt_pct(r['vol_1y'])} | "
                f"{_fmt_num(r['sharpe_1y'])} | {_fmt_num(r['rsi_14'], 1)} | "
                f"{r.get('trend') or '—'} | {_fmt_pct(r['max_dd_3y'])} | "
                f"{_fmt_pct(r.get('tracking_error', float('nan')))} |"
            )
    return "\n".join(blocks)


def render_top_overall_md(merged: pd.DataFrame, top_n: int = 25) -> str:
    has_metrics = merged[merged["sharpe_1y"].notna()].copy()
    top = has_metrics.sort_values("sharpe_1y", ascending=False).head(top_n)
    if top.empty:
        return ""
    rows = [f"\n## Top {len(top)} ETFs by 1y Sharpe ratio (any category)\n",
            "| # | ETF | Category | TER | 1y | Sharpe | Vol | RSI |",
            "|---:|---|---|---:|---:|---:|---:|---:|"]
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        rows.append(
            f"| {i} | {r['name']} | {r['category']} | {_fmt_pct(r['ter'])} | "
            f"{_fmt_pct(r['ret_1y'])} | {_fmt_num(r['sharpe_1y'])} | "
            f"{_fmt_pct(r['vol_1y'])} | {_fmt_num(r['rsi_14'], 1)} |"
        )
    return "\n".join(rows)


# ============================================================================
# HTML dashboard (optional — opt-in via --html)
# ============================================================================

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UCITS ETF Terminal — __GENERATED_AT__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg-0:#08080c; --bg-1:#0e0e15; --bg-2:#15151f; --bg-3:#1c1c28;
  --border:rgba(255,255,255,.08); --border-strong:rgba(255,255,255,.16);
  --text:#e4e4e7; --text-dim:#a1a1aa; --text-faint:#71717a;
  --accent:#22d3ee; --accent-glow:rgba(34,211,238,.18); --accent-2:#a78bfa;
  --pos:#34d399; --neg:#f87171;
}
html,body{background:var(--bg-0);color:var(--text);
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased;}
body::before{content:'';position:fixed;inset:0;z-index:-1;pointer-events:none;
  background:
    radial-gradient(ellipse 80% 50% at 20% 0%, rgba(34,211,238,.10), transparent 60%),
    radial-gradient(ellipse 60% 40% at 90% 10%, rgba(167,139,250,.07), transparent 60%),
    radial-gradient(ellipse 60% 40% at 50% 100%, rgba(34,211,238,.04), transparent 60%);
}
::selection{background:var(--accent-glow);color:#fff;}

/* ─── Top nav (sticky, glass) ─── */
.nav{position:sticky;top:0;z-index:100;height:56px;display:flex;align-items:center;
  padding:0 32px;gap:32px;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  background:rgba(8,8,12,.7);border-bottom:1px solid var(--border);}
.nav__brand{font-weight:600;font-size:13px;letter-spacing:.02em;display:flex;align-items:center;gap:10px;}
.nav__brand::before{content:'';width:8px;height:8px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 14px var(--accent),0 0 28px rgba(34,211,238,.4);
  animation:pulse 2.4s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.nav__links{display:flex;gap:24px;flex:1;}
.nav__links a{color:var(--text-dim);text-decoration:none;font-size:13px;transition:color .15s;}
.nav__links a:hover{color:var(--text);}
.nav__meta{color:var(--text-faint);font-size:11px;font-family:'JetBrains Mono',monospace;letter-spacing:.04em;}

/* ─── Container ─── */
.container{max-width:1400px;margin:0 auto;padding:56px 32px 96px;}

/* ─── Hero ─── */
.hero{margin-bottom:48px;}
.hero__eyebrow{color:var(--accent);font-size:11px;font-weight:500;letter-spacing:.12em;
  text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px;}
.hero__eyebrow::before{content:'';width:24px;height:1px;background:var(--accent);}
.hero__title{font-size:52px;font-weight:600;letter-spacing:-.035em;line-height:1.05;margin-bottom:14px;
  background:linear-gradient(180deg,#fff 0%,#a1a1aa 110%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero__sub{color:var(--text-dim);font-size:15px;max-width:680px;}
.hero__sub b{color:var(--text);font-weight:500;}

/* ─── KPI strip ─── */
.kpi-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border);
  border:1px solid var(--border);border-radius:14px;overflow:hidden;margin-top:36px;}
.kpi{background:var(--bg-1);padding:22px 26px;position:relative;transition:background .15s;}
.kpi:hover{background:var(--bg-2);}
.kpi__label{color:var(--text-faint);font-size:10px;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;margin-bottom:10px;}
.kpi__value{font-family:'JetBrains Mono',monospace;font-size:30px;font-weight:500;
  letter-spacing:-.025em;font-variant-numeric:tabular-nums;color:var(--text);}
.kpi__trend{font-size:11px;color:var(--text-faint);margin-top:6px;}

/* ─── Section heads ─── */
.section{margin-top:72px;}
.section__head{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:18px;gap:24px;}
.section__title{font-size:22px;font-weight:600;letter-spacing:-.02em;}
.section__sub{color:var(--text-dim);font-size:13px;flex:1;text-align:right;}

/* ─── Top picks ─── */
.pick-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}
.pick{background:linear-gradient(180deg,var(--bg-1) 0%,var(--bg-0) 100%);
  border:1px solid var(--border);border-radius:14px;padding:24px;position:relative;overflow:hidden;
  transition:border-color .2s,transform .2s;}
.pick:hover{border-color:var(--border-strong);transform:translateY(-1px);
  box-shadow:0 8px 32px rgba(34,211,238,.06);}
.pick::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent 0%,var(--accent) 50%,transparent 100%);opacity:.5;}
.pick__label{color:var(--accent);font-size:10px;font-weight:600;letter-spacing:.12em;
  text-transform:uppercase;margin-bottom:18px;}
.pick__name{font-size:17px;font-weight:500;letter-spacing:-.01em;line-height:1.3;margin-bottom:8px;}
.pick__isin{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text-faint);
  margin-bottom:18px;letter-spacing:.04em;}
.pick__metric{font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:500;
  color:var(--text);letter-spacing:-.02em;font-variant-numeric:tabular-nums;}

/* ─── Charts ─── */
.chart{background:var(--bg-1);border:1px solid var(--border);border-radius:14px;padding:18px;}

/* ─── Tables ─── */
.table-wrap{background:var(--bg-1);border:1px solid var(--border);border-radius:14px;overflow:hidden;}
.table-toolbar{display:flex;align-items:center;padding:14px 16px;border-bottom:1px solid var(--border);gap:12px;}
.table-toolbar input{flex:1;background:var(--bg-2);border:1px solid var(--border);border-radius:8px;
  padding:9px 14px;color:var(--text);font-family:inherit;font-size:13px;transition:border-color .15s;}
.table-toolbar input::placeholder{color:var(--text-faint);}
.table-toolbar input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow);}
.table-toolbar .count{color:var(--text-faint);font-size:12px;font-family:'JetBrains Mono',monospace;
  white-space:nowrap;}
.table-scroll{overflow:auto;max-height:640px;}
table.dt{width:100%;border-collapse:collapse;font-size:12px;}
table.dt thead th{position:sticky;top:0;background:var(--bg-2);color:var(--text-dim);font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;font-size:10px;padding:13px 14px;text-align:left;
  border-bottom:1px solid var(--border);cursor:pointer;user-select:none;white-space:nowrap;
  transition:color .15s,background .15s;}
table.dt thead th:hover{color:var(--text);background:var(--bg-3);}
table.dt thead th.sort-asc{color:var(--accent);}
table.dt thead th.sort-asc::after{content:' ↑';}
table.dt thead th.sort-desc{color:var(--accent);}
table.dt thead th.sort-desc::after{content:' ↓';}
table.dt thead th.num,table.dt tbody td.num{text-align:right;}
table.dt tbody td{padding:11px 14px;border-bottom:1px solid var(--border);font-size:12px;
  font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--text);}
table.dt tbody td.num,table.dt tbody td.mono{font-family:'JetBrains Mono',monospace;}
table.dt tbody tr{transition:background .1s;}
table.dt tbody tr:hover{background:var(--bg-2);}
table.dt tbody tr:last-child td{border-bottom:none;}
table.dt tbody td.pos{color:var(--pos);}
table.dt tbody td.neg{color:var(--neg);}
table.dt tbody td.dim{color:var(--text-faint);}
table.dt tbody td.spark{padding:4px 10px;vertical-align:middle;width:120px;}
table.dt tbody td.spark svg{display:block;opacity:.92;}
table.dt tbody tr:hover td.spark svg{opacity:1;}
table.dt tbody td.stars{font-family:'Inter';letter-spacing:1px;color:#fbbf24;}
table.dt tbody td.dots{font-family:'Inter';letter-spacing:1px;color:#34d399;}
table.dt tbody td.stars.dim,table.dt tbody td.dots.dim{color:var(--text-faint);}

/* ─── Footer ─── */
.footer{margin-top:96px;padding-top:32px;border-top:1px solid var(--border);
  color:var(--text-faint);font-size:11px;text-align:center;letter-spacing:.04em;
  font-family:'JetBrains Mono',monospace;}

@media (max-width:900px){
  .hero__title{font-size:36px;}
  .kpi-strip{grid-template-columns:repeat(2,1fr);}
  .pick-grid{grid-template-columns:1fr;}
  .container{padding:32px 16px 64px;}
  .nav{padding:0 16px;gap:16px;}
  .nav__links{gap:14px;overflow-x:auto;}
}
</style>
</head>
<body>

<nav class="nav">
  <div class="nav__brand">UCITS · TERMINAL</div>
  <div class="nav__links">
    <a href="#picks">Picks</a>
    <a href="#groups">Groups</a>
    <a href="#sharpe">Sharpe</a>
    <a href="#leaderboard">Leaderboard</a>
    <a href="#all">All ETFs</a>
  </div>
  <div class="nav__meta">__GENERATED_AT__</div>
</nav>

<main class="container">
  <header class="hero">
    <div class="hero__eyebrow">European UCITS · daily snapshot</div>
    <h1 class="hero__title">ETF Universe Report</h1>
    <p class="hero__sub">Performance, risk and benchmark-tracking metrics across <b>__N_TOTAL__</b> ETFs · <b>__N_PRICED__</b> with usable price history · sources: __JE_NOTE__.</p>

    <div class="kpi-strip">
      <div class="kpi"><div class="kpi__label">Universe</div><div class="kpi__value">__N_TOTAL__</div><div class="kpi__trend">UCITS-eligible</div></div>
      <div class="kpi"><div class="kpi__label">Priced</div><div class="kpi__value">__N_PRICED__</div><div class="kpi__trend">with metrics</div></div>
      <div class="kpi"><div class="kpi__label">Categories</div><div class="kpi__value">__N_CATS__</div><div class="kpi__trend">strategy / asset</div></div>
      <div class="kpi"><div class="kpi__label">Median Sharpe 1y</div><div class="kpi__value">__MED_SHARPE__</div><div class="kpi__trend">across all priced</div></div>
    </div>
  </header>

  <section id="picks" class="section">
    <div class="section__head"><h2 class="section__title">Top picks</h2>
      <p class="section__sub">Single-best ETF on each dimension across the priced universe.</p></div>
    <div class="pick-grid">
      <div class="pick">
        <div class="pick__label">Cheapest TER</div>
        <div class="pick__name">__CHEAPEST_NAME__</div>
        <div class="pick__isin">__CHEAPEST_ISIN__</div>
        <div class="pick__metric">__CHEAPEST_METRIC__</div>
      </div>
      <div class="pick">
        <div class="pick__label">Best Sharpe · 1y</div>
        <div class="pick__name">__BEST_SHARPE_NAME__</div>
        <div class="pick__isin">__BEST_SHARPE_ISIN__</div>
        <div class="pick__metric">__BEST_SHARPE_METRIC__</div>
      </div>
      <div class="pick">
        <div class="pick__label">Best benchmark tracker · 1y</div>
        <div class="pick__name">__BEST_TRACKER_NAME__</div>
        <div class="pick__isin">__BEST_TRACKER_ISIN__</div>
        <div class="pick__metric">__BEST_TRACKER_METRIC__</div>
      </div>
    </div>
  </section>

  <section id="groups" class="section">
    <div class="section__head"><h2 class="section__title">Group sizes</h2>
      <p class="section__sub">Number of priced ETFs per category — top 30 by count.</p></div>
    <div class="chart"><div id="group_bar" style="height:420px;"></div></div>
  </section>

  <section id="sharpe" class="section">
    <div class="section__head"><h2 class="section__title">Sharpe distribution by category</h2>
      <p class="section__sub">1-year Sharpe per ETF, grouped — boxes show within-category dispersion.</p></div>
    <div class="chart"><div id="sharpe_box" style="height:520px;"></div></div>
  </section>

  <section id="leaderboard" class="section">
    <div class="section__head"><h2 class="section__title">Category leaderboard</h2>
      <p class="section__sub">Median metrics per category, ranked by median 1y Sharpe.</p></div>
    <div class="table-wrap">
      <div class="table-toolbar">
        <input type="search" placeholder="Filter categories…" data-target="summary_dt">
        <span class="count" id="summary_count"></span>
      </div>
      <div class="table-scroll">__SUMMARY_TABLE__</div>
    </div>
  </section>

  <section id="uncat" class="section">
    <div class="section__head"><h2 class="section__title">Uncategorized <span style="color:var(--text-faint);font-weight:400;font-size:14px;">· __N_UNCAT__</span></h2>
      <p class="section__sub">ETFs whose name + benchmark didn't match any rule. Add patterns to <code style="color:var(--accent);font-family:'JetBrains Mono',monospace;">RULES</code> in ETF.py to absorb them.</p></div>
    <div class="table-wrap">
      <div class="table-toolbar">
        <input type="search" placeholder="Filter uncategorized…" data-target="uncat_dt">
        <span class="count" id="uncat_count"></span>
      </div>
      <div class="table-scroll">__UNCAT_TABLE__</div>
    </div>
  </section>

  <section id="all" class="section">
    <div class="section__head"><h2 class="section__title">All ETFs</h2>
      <p class="section__sub">Pre-sorted by Sharpe 1y descending. Click any column to sort (numeric columns flip to their preferred direction first).</p></div>
    <div class="table-wrap">
      <div class="table-toolbar">
        <input type="search" placeholder="Filter by name, ISIN, category…" data-target="all_dt">
        <span class="count" id="all_count"></span>
      </div>
      <div class="table-scroll">__ALL_TABLE__</div>
    </div>
  </section>

  <footer class="footer">XETRA · YAHOO FINANCE · JUSTETF · GENERATED BY ETF.PY</footer>
</main>

<script>
// ─── Sortable tables ──────────────────────────────────────────────────────
(function(){
  const parseNum = v => {
    if (v === '' || v === '—' || v == null) return null;
    const cleaned = String(v).replace(/[%,]/g, '').replace(/\s/g, '');
    const n = parseFloat(cleaned);
    return isNaN(n) ? null : n;
  };
  document.querySelectorAll('table.dt').forEach(table => {
    const heads = table.querySelectorAll('thead th');
    heads.forEach((th, i) => {
      th.addEventListener('click', () => {
        let dir;
        if (th.classList.contains('sort-asc')) dir = 'desc';
        else if (th.classList.contains('sort-desc')) dir = 'asc';
        else dir = th.dataset.prefer || 'asc';   // first click respects column's preferred direction
        heads.forEach(h => h.classList.remove('sort-asc','sort-desc'));
        th.classList.add('sort-' + dir);
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const cellVal = cell => {
          if (cell.dataset && cell.dataset.sort != null) {
            const n = parseFloat(cell.dataset.sort);
            return isNaN(n) ? null : n;
          }
          return parseNum(cell.textContent);
        };
        rows.sort((a, b) => {
          const av = cellVal(a.cells[i]);
          const bv = cellVal(b.cells[i]);
          let cmp;
          if (av !== null && bv !== null) cmp = av - bv;
          else if (av === null && bv !== null) cmp = 1;
          else if (av !== null && bv === null) cmp = -1;
          else cmp = a.cells[i].textContent.localeCompare(b.cells[i].textContent);
          return dir === 'asc' ? cmp : -cmp;
        });
        rows.forEach(r => tbody.appendChild(r));
      });
    });
  });
})();

// ─── Filter inputs ────────────────────────────────────────────────────────
document.querySelectorAll('.table-toolbar input').forEach(inp => {
  const cls = inp.dataset.target;
  const table = document.querySelector('table.' + cls);
  const countEl = document.getElementById(cls.replace('_dt','_count'));
  const updateCount = () => {
    if (!table || !countEl) return;
    const rows = table.querySelectorAll('tbody tr');
    const visible = Array.from(rows).filter(r => r.style.display !== 'none').length;
    countEl.textContent = visible + ' / ' + rows.length;
  };
  if (table) {
    inp.addEventListener('input', () => {
      const q = inp.value.toLowerCase();
      table.querySelectorAll('tbody tr').forEach(tr => {
        tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
      });
      updateCount();
    });
    updateCount();
  }
});

// ─── Plotly charts (dark theme baked in) ──────────────────────────────────
const _baseLayout = {
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{family:"Inter, -apple-system, sans-serif", size:12, color:'#a1a1aa'},
  xaxis:{gridcolor:'rgba(255,255,255,.05)', zerolinecolor:'rgba(255,255,255,.1)',
         linecolor:'rgba(255,255,255,.1)', tickfont:{color:'#a1a1aa'}},
  yaxis:{gridcolor:'rgba(255,255,255,.05)', zerolinecolor:'rgba(255,255,255,.1)',
         linecolor:'rgba(255,255,255,.1)', tickfont:{color:'#a1a1aa'}},
  margin:{t:24, r:24, b:100, l:60},
  colorway:['#22d3ee','#a78bfa','#34d399','#fbbf24','#f87171','#60a5fa','#f472b6','#facc15']
};
const _config = {displayModeBar:false, responsive:true};
function _draw(divId, fig, layoutOverrides){
  const merged = Object.assign({}, _baseLayout, fig.layout || {}, layoutOverrides || {});
  merged.xaxis = Object.assign({}, _baseLayout.xaxis, (fig.layout||{}).xaxis||{}, (layoutOverrides||{}).xaxis||{});
  merged.yaxis = Object.assign({}, _baseLayout.yaxis, (fig.layout||{}).yaxis||{}, (layoutOverrides||{}).yaxis||{});
  Plotly.newPlot(divId, fig.data, merged, _config);
}
_draw('group_bar', __GROUP_BAR_FIG__, {xaxis:{tickangle:-30}, margin:{b:140}});
_draw('sharpe_box', __SHARPE_BOX_FIG__, {showlegend:false, yaxis:{title:'Sharpe 1y'}, margin:{b:140}});
</script>
</body>
</html>
"""


# Column-type map: governs how the all-ETFs and summary tables get formatted.
_PCT_COLS = {
    "ter", "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_3y",
    "cagr_3y", "cagr_5y", "vol_1y", "vol_3y", "max_dd_3y",
    "off_52w_high", "mom_12_1", "tracking_error", "tracking_diff",
    "median_ter", "median_ret_1y", "median_vol", "median_dd",
    "median_te", "top_ter", "top_ret_1y",
    "top10_concentration",  # already a fraction 0..1
}
_RATIO_COLS = {
    "sharpe_1y", "sharpe_3y", "sortino_1y", "rsi_14",
    "median_sharpe", "median_rsi", "top_sharpe",
    "correlation", "r2",
}
_INT_COLS = {"obs_days", "n_etfs", "rank", "holdings_count"}
# Large counts rendered with thousands-separator (no percent/no decimals).
_LARGE_NUM_COLS = {"aum_mil"}
_TEXT_COLS = {
    "isin", "name", "ticker", "category", "asset_class", "replication",
    "trend", "issuer", "top_etf", "top_isin", "benchmark", "sparkline",
    "category_benchmark",
}
# Cells that contain pre-rendered HTML (don't escape).
_RAW_HTML_COLS = {"sparkline"}
# Pretty column headers (overrides raw DataFrame column names).
_COL_LABEL = {
    "isin": "ISIN", "name": "Name", "ticker": "Ticker",
    "category": "Category", "asset_class": "Asset class",
    "ter": "TER", "replication": "Replication",
    "ret_1m": "1m", "ret_3m": "3m", "ret_6m": "6m", "ret_1y": "1y", "ret_3y": "3y",
    "cagr_3y": "CAGR 3y", "cagr_5y": "CAGR 5y",
    "sharpe_1y": "Sharpe 1y", "sharpe_3y": "Sharpe 3y", "sortino_1y": "Sortino 1y",
    "vol_1y": "Vol 1y", "vol_3y": "Vol 3y",
    "max_dd_3y": "Max DD 3y", "rsi_14": "RSI 14", "mom_12_1": "Mom 12-1",
    "tracking_error": "Tracking err", "tracking_diff": "Tracking diff",
    "off_52w_high": "Off 52w high",
    "sparkline": "1y trend",
    "rank": "#", "n_etfs": "# ETFs",
    "median_ter": "Med TER", "median_ret_1y": "Med 1y", "median_sharpe": "Med Sharpe",
    "median_vol": "Med Vol", "median_dd": "Med DD", "median_rsi": "Med RSI",
    "median_te": "Med TE",
    "top_etf": "Top ETF (Sharpe)", "top_isin": "Top ISIN", "top_ter": "Top TER",
    "top_ret_1y": "Top 1y", "top_sharpe": "Top Sharpe",
    "benchmark": "Benchmark", "issuer": "Issuer",
    "morningstar_rating": "M★", "sustainability_rating": "Sustain",
    "holdings_count": "Holdings", "aum_mil": "AUM (M)", "yield": "Yield",
    "top10_concentration": "Top 10 %", "category_benchmark": "M* benchmark",
}
# Columns where first-click sort should go descending (higher = better).
# Everything else numeric defaults to ascending (lower = better, e.g. TER, vol).
_PREFER_DESC = {
    "sharpe_1y", "sharpe_3y", "sortino_1y",
    "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_3y",
    "cagr_3y", "cagr_5y", "mom_12_1", "rsi_14",
    "median_sharpe", "median_ret_1y", "median_rsi",
    "top_sharpe", "top_ret_1y", "n_etfs", "obs_days",
    # EODHD / qualitative overlays
    "morningstar_rating", "sustainability_rating",
    "aum_mil", "holdings_count", "yield",
}
# Rating columns get rendered as glyphs (stars/dots) — emit data-sort so
# the JS sorter ranks them numerically rather than by Unicode code-point.
_RATING_COLS = {"morningstar_rating", "sustainability_rating"}

def _render_stars(n, max_n: int = 5) -> str:
    """Render a 1..max_n rating as filled-then-empty star glyphs."""
    if n is None or (isinstance(n, float) and (np.isnan(n) or np.isinf(n))):
        return "—"
    try:
        k = int(round(float(n)))
    except (ValueError, TypeError):
        return "—"
    k = max(0, min(max_n, k))
    return "★" * k + "☆" * (max_n - k)


def _render_dots(n, max_n: int = 5) -> str:
    """Render a 1..max_n rating as filled-then-empty dot glyphs (for ESG / sustainability)."""
    if n is None or (isinstance(n, float) and (np.isnan(n) or np.isinf(n))):
        return "—"
    try:
        k = int(round(float(n)))
    except (ValueError, TypeError):
        return "—"
    k = max(0, min(max_n, k))
    return "●" * k + "○" * (max_n - k)


def _sparkline_svg(
    s: pd.Series, w: int = 100, h: int = 20, days: int = 252, max_pts: int = 40
) -> str:
    """Build an inline SVG polyline of the last `days` of a price series.
    Green if up over the window, red if down. Empty string if not enough data."""
    if s is None or s.empty:
        return ""
    s = s.dropna().tail(days)
    if len(s) < 2:
        return ""
    if len(s) > max_pts:
        step = max(1, len(s) // max_pts)
        s = s.iloc[::step]
    vals = s.values
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo if hi > lo else 1.0
    n = len(vals)
    pts = " ".join(
        f"{i/(n-1)*w:.1f},{h - (v-lo)/span*h:.1f}" for i, v in enumerate(vals)
    )
    last_x = w
    last_y = h - (float(vals[-1]) - lo) / span * h
    color = "#34d399" if vals[-1] >= vals[0] else "#f87171"
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'preserveAspectRatio="none" style="display:block">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.2" '
        f'stroke-linecap="round" stroke-linejoin="round" points="{pts}"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="1.6" fill="{color}"/>'
        f'</svg>'
    )


def _fmt_cell(col: str, val) -> tuple[str, str]:
    """Return (text, css_class) for one cell."""
    is_na = val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val)))
    if col == "sparkline":
        return (str(val) if (val and not is_na) else "—", "spark")
    if col == "morningstar_rating":
        return (_render_stars(val), "stars" if not is_na else "stars dim")
    if col == "sustainability_rating":
        return (_render_dots(val), "dots" if not is_na else "dots dim")
    if col == "aum_mil":
        if is_na:
            return ("—", "num dim")
        try:
            return (f"{float(val):,.0f}", "num")
        except (ValueError, TypeError):
            return ("—", "num dim")
    if col == "yield":
        if is_na:
            return ("—", "num dim")
        try:
            return (f"{float(val):.2f}%", "num")
        except (ValueError, TypeError):
            return ("—", "num dim")
    if col in _TEXT_COLS:
        return (str(val) if not is_na else "—", "")
    if is_na:
        return ("—", "num dim")
    if col in _PCT_COLS:
        sign_class = ""
        if col in {"ter", "vol_1y", "vol_3y", "tracking_error"}:
            cls = "num"
        elif col == "max_dd_3y" or col == "median_dd":
            cls = "num neg" if val < 0 else "num"
        else:
            cls = "num " + ("pos" if val > 0 else "neg" if val < 0 else "")
        return (f"{val*100:+.2f}%" if col in {"tracking_diff"} else f"{val*100:.2f}%", cls.strip())
    if col in _RATIO_COLS:
        if col == "rsi_14" or col == "median_rsi":
            return (f"{val:.1f}", "num")
        if col == "correlation" or col == "r2":
            return (f"{val:.3f}", "num")
        cls = "num " + ("pos" if val > 0 else "neg" if val < 0 else "")
        return (f"{val:.2f}", cls.strip())
    if col in _INT_COLS:
        try:
            return (f"{int(val):,}", "num")
        except (ValueError, TypeError):
            return (str(val), "num")
    if isinstance(val, float):
        return (f"{val:.4f}", "num")
    return (str(val), "")


def _df_to_html_dt(
    df: pd.DataFrame,
    table_id: str,
    max_rows: int = 5000,
    default_sort: tuple[str, str] | None = None,
) -> str:
    """Render a DataFrame as a styled <table>. `default_sort=(col, 'asc'|'desc')`
    pre-marks one column as already sorted (the data should already be sorted
    that way before being passed in — this just sets the visual indicator)."""
    if df is None or df.empty:
        return "<p style='padding:24px;color:var(--text-faint)'>(no data)</p>"
    df = df.head(max_rows)
    cols = list(df.columns)
    sort_col, sort_dir = (default_sort or (None, None))
    # Build header
    head_cells = []
    for c in cols:
        is_num = c not in _TEXT_COLS
        prefer = "desc" if c in _PREFER_DESC else "asc"
        classes = []
        if is_num:
            classes.append("num")
        if c == sort_col and sort_dir in ("asc", "desc"):
            classes.append(f"sort-{sort_dir}")
        cls = " ".join(classes)
        label = _COL_LABEL.get(c, c)
        head_cells.append(
            f'<th class="{cls}" data-col="{c}" data-prefer="{prefer}">{label}</th>'
        )
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            raw_val = row[c]
            txt, cls = _fmt_cell(c, raw_val)
            # Escape HTML in text columns *unless* the column is pre-rendered HTML.
            if c in _TEXT_COLS and c not in _RAW_HTML_COLS:
                txt = (str(txt).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            attrs = ""
            # Rating cells render as glyphs; emit numeric data-sort so the JS
            # sorter ranks them by value, not by Unicode code-point order.
            if c in _RATING_COLS:
                try:
                    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                        attrs = ' data-sort="-1"'
                    else:
                        attrs = f' data-sort="{int(round(float(raw_val)))}"'
                except (ValueError, TypeError):
                    attrs = ' data-sort="-1"'
            cells.append(f'<td class="{cls}"{attrs}>{txt}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<table class="dt {table_id}">'
        f'<thead><tr>{"".join(head_cells)}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        f'</table>'
    )


def _plotly_to_json(fig) -> str:
    """Plotly fig -> JSON string for embedding as a JS object literal."""
    try:
        import plotly.io as pio
        return pio.to_json(fig)
    except Exception as exc:
        print(f"  ! plotly serialization failed: {exc}")
        return '{"data":[],"layout":{}}'


def write_html_report(
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    universe_stats: dict[str, int] | None = None,
    used_justetf: bool = False,
    prices: pd.DataFrame | None = None,
) -> Path:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  ! plotly not installed — skip HTML (pip install plotly)")
        return Path()

    output_dir.mkdir(parents=True, exist_ok=True)
    has_metrics = (
        merged[merged["sharpe_1y"].notna()].copy()
        if "sharpe_1y" in merged.columns else pd.DataFrame()
    )

    # ── Sparkline column (1y price history, inline SVG per row) ──────────
    spark_map: dict[str, str] = {}
    if prices is not None and not prices.empty:
        for tkr in prices.columns:
            spark_map[tkr] = _sparkline_svg(prices[tkr])
    # Match by yahoo_ticker (the column we used to fetch prices).
    if "yahoo_ticker" in merged.columns:
        merged = merged.copy()
        merged["sparkline"] = merged["yahoo_ticker"].map(spark_map).fillna("")
    elif "ticker" in merged.columns:
        merged = merged.copy()
        merged["sparkline"] = merged["ticker"].map(spark_map).fillna("")

    # ── Top-pick rows ─────────────────────────────────────────────────────
    def _row_or_none(df: pd.DataFrame, sort_col: str, ascending: bool = True):
        if df.empty or sort_col not in df.columns:
            return None
        sub = df.dropna(subset=[sort_col])
        if sub.empty:
            return None
        return sub.sort_values(sort_col, ascending=ascending).iloc[0]

    cheapest_row = _row_or_none(has_metrics, "ter", ascending=True)
    best_sh_row = _row_or_none(has_metrics, "sharpe_1y", ascending=False)
    if "tracking_diff" in has_metrics.columns:
        td = has_metrics.dropna(subset=["tracking_diff"]).assign(_a=lambda d: d["tracking_diff"].abs())
        best_tr_row = td.sort_values("_a").iloc[0] if not td.empty else None
    else:
        best_tr_row = None

    def _name(r):  return ("—" if r is None else str(r.get("name", "?")))
    def _isin(r):  return ("" if r is None else str(r.get("isin", "")))
    cheapest_metric = (f"{cheapest_row['ter']*100:.2f}% TER" if cheapest_row is not None else "—")
    best_sh_metric  = (f"{best_sh_row['sharpe_1y']:.2f}" if best_sh_row is not None else "—")
    best_tr_metric  = (f"{best_tr_row['tracking_diff']*100:+.2f}% diff" if best_tr_row is not None else "—")

    # ── Plotly figs (theme applied in JS so colors stay aligned with CSS) ─
    if not has_metrics.empty:
        cat_n = has_metrics.groupby("category").size().sort_values(ascending=False).head(30)
        group_bar = go.Figure(go.Bar(
            x=cat_n.index.tolist(), y=cat_n.values.tolist(),
            marker=dict(color="#22d3ee", line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>%{y} ETFs<extra></extra>",
        ))
        group_bar.update_layout(yaxis_title="# ETFs (priced)")

        traces = []
        for cat in cat_n.index.tolist():
            vals = has_metrics.loc[has_metrics["category"] == cat, "sharpe_1y"].dropna().values
            if len(vals):
                traces.append(go.Box(
                    y=list(vals), name=cat, boxmean=True,
                    marker=dict(size=3, opacity=0.6),
                    line=dict(width=1.2),
                    fillcolor="rgba(34,211,238,0.10)",
                ))
        sharpe_box = go.Figure(traces)
    else:
        group_bar = go.Figure()
        sharpe_box = go.Figure()

    # ── Tables ───────────────────────────────────────────────────────────
    summary_show = summary.copy() if summary is not None and not summary.empty else pd.DataFrame()
    summary_table = _df_to_html_dt(
        summary_show, "summary_dt",
        default_sort=("median_sharpe", "desc") if "median_sharpe" in summary_show.columns else None,
    )

    show_cols = [c for c in [
        "isin", "name", "sparkline", "category", "asset_class", "ter", "replication",
        "morningstar_rating", "sustainability_rating",
        "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_3y",
        "sharpe_1y", "sharpe_3y", "vol_1y", "max_dd_3y", "rsi_14",
        "tracking_error", "tracking_diff",
        "aum_mil", "holdings_count", "top10_concentration", "yield",
    ] if c in merged.columns]
    merged_for_table = merged[show_cols] if show_cols else merged
    if "sharpe_1y" in merged_for_table.columns:
        merged_for_table = merged_for_table.sort_values(
            "sharpe_1y", ascending=False, na_position="last"
        )
    all_table = _df_to_html_dt(
        merged_for_table, "all_dt",
        default_sort=("sharpe_1y", "desc") if "sharpe_1y" in merged_for_table.columns else None,
    )

    # ── Uncategorized panel ──────────────────────────────────────────────
    if "category" in merged.columns:
        uncat = merged[merged["category"] == "Uncategorized"].copy()
    else:
        uncat = pd.DataFrame()
    uncat_show_cols = [c for c in ["isin", "name", "benchmark", "asset_class", "ter", "issuer"]
                       if c in uncat.columns]
    uncat_table = _df_to_html_dt(
        uncat[uncat_show_cols] if uncat_show_cols and not uncat.empty else pd.DataFrame(),
        "uncat_dt",
    )

    # ── Header / KPI values ──────────────────────────────────────────────
    je_note = "Xetra master + justETF screener" if used_justetf else "Xetra master sheet"
    n_total = len(merged)
    n_priced = int(has_metrics.shape[0]) if not has_metrics.empty else 0
    n_cats = int(has_metrics["category"].nunique()) if "category" in has_metrics.columns and not has_metrics.empty else 0
    n_uncat = int(len(uncat))
    med_sharpe = (f"{has_metrics['sharpe_1y'].median():.2f}"
                  if not has_metrics.empty and has_metrics["sharpe_1y"].notna().any()
                  else "—")

    html = (
        HTML_TEMPLATE
        .replace("__GENERATED_AT__", pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
        .replace("__JE_NOTE__", je_note)
        .replace("__N_TOTAL__", f"{n_total:,}")
        .replace("__N_PRICED__", f"{n_priced:,}")
        .replace("__N_CATS__", f"{n_cats}")
        .replace("__MED_SHARPE__", med_sharpe)
        .replace("__CHEAPEST_NAME__", _name(cheapest_row))
        .replace("__CHEAPEST_ISIN__", _isin(cheapest_row))
        .replace("__CHEAPEST_METRIC__", cheapest_metric)
        .replace("__BEST_SHARPE_NAME__", _name(best_sh_row))
        .replace("__BEST_SHARPE_ISIN__", _isin(best_sh_row))
        .replace("__BEST_SHARPE_METRIC__", best_sh_metric)
        .replace("__BEST_TRACKER_NAME__", _name(best_tr_row))
        .replace("__BEST_TRACKER_ISIN__", _isin(best_tr_row))
        .replace("__BEST_TRACKER_METRIC__", best_tr_metric)
        .replace("__SUMMARY_TABLE__", summary_table)
        .replace("__UNCAT_TABLE__", uncat_table)
        .replace("__N_UNCAT__", f"{n_uncat:,}")
        .replace("__ALL_TABLE__", all_table)
        .replace("__GROUP_BAR_FIG__", _plotly_to_json(group_bar))
        .replace("__SHARPE_BOX_FIG__", _plotly_to_json(sharpe_box))
    )
    out = output_dir / "ranked_report.html"
    out.write_text(html, encoding="utf-8")
    return out


def write_report(
    merged: pd.DataFrame,
    output_dir: Path,
    universe_stats: dict[str, int] | None = None,
    top_n_per_cat: int = 5,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_category_summary(merged)
    drilldown = build_drilldown(merged, top_n=top_n_per_cat)

    md_lines = ["# UCITS ETF ranking report",
                "",
                f"_Generated: {pd.Timestamp.utcnow():%Y-%m-%d %H:%M UTC}_",
                ""]
    if universe_stats:
        md_lines += [
            "## Universe coverage",
            "",
            f"- **Source:** Xetra ETF master sheet (cashmarket.deutsche-boerse.com)",
            f"- **Raw rows:** {universe_stats.get('rows_raw', 0):,}",
            f"- **Distinct ISINs:** {universe_stats.get('rows_dedup', 0):,}",
            f"- **UCITS ETFs:** {universe_stats.get('ucits_count', 0):,}",
            f"- **In analysis (price history available):** {merged['sharpe_1y'].notna().sum():,}",
            "",
        ]

    md_lines += [
        "## Methodology",
        "",
        "- **Universe:** Xetra master data sheet (the most comprehensive openly downloadable",
        "  European ETF reference dataset, 3,400+ products). Filtered to UCITS, deduplicated by ISIN.",
        "- **Categorisation:** rule-based on the ETF name + benchmark; ESG/SRI screening is",
        "  reported as a flag rather than a strategy bucket.",
        "- **Prices:** Yahoo Finance auto-adjusted close on the Xetra (.DE) listing. Adjusted",
        "  close incorporates dividends and splits, so returns are total returns.",
        "- **Risk-free rate:** 2.0% p.a. (EUR). Sharpe and Sortino are annualised from daily",
        "  excess returns.",
        "- **Tracking error:** market-price proxy using a Yahoo benchmark series",
        "  (annualised stdev of daily return differences). True NAV tracking requires the",
        "  issuer's NAV file and the index level - both paywalled - so this is the standard",
        "  open-source approximation.",
        "- **Group ranking:** categories are ordered by median 1y Sharpe of constituent ETFs.",
        "",
        "## Category summary (one line per strategy / asset class)",
        "",
        render_category_summary_md(summary),
        "",
    ]
    md_lines.append(render_top_overall_md(merged, top_n=25))
    md_lines += ["",
                 "## Top picks within each category",
                 "",
                 render_drilldown_md(drilldown)]

    md_path = output_dir / "ranked_report.md"
    md_path.write_text("\n".join(md_lines))

    summary.to_csv(output_dir / "category_summary.csv", index=False)
    drilldown.to_csv(output_dir / "category_drilldown.csv", index=False)
    merged.to_csv(output_dir / "etf_metrics_full.csv", index=False)
    return md_path


# ============================================================================
# CLI / orchestrator
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="Re-download Xetra master + price cache")
    p.add_argument("--max-etfs", type=int, default=0,
                   help="Cap number of ETFs sent for pricing (0 = no cap)")
    p.add_argument("--period", default="5y", help="Yahoo lookback (e.g. 1y, 3y, 5y, 10y)")
    p.add_argument("--rf", type=float, default=0.02, help="Annual risk-free rate (decimal)")
    p.add_argument("--top-per-cat", type=int, default=5, help="ETFs to show per category drilldown")
    p.add_argument("--exclude-leveraged", action="store_true",
                   help="Drop leveraged / inverse products from the analysis")
    p.add_argument("--ucits-only", action="store_true", default=True)
    p.add_argument("--include-non-ucits", dest="ucits_only", action="store_false")
    p.add_argument("--justetf", action="store_true",
                   help="Also pull justETF screener: fills TER/benchmark/strategy gaps and adds non-Xetra UCITS")
    p.add_argument("--openfigi", action="store_true",
                   help="Use OpenFIGI to fill missing tickers (improves yfinance hit-rate)")
    p.add_argument("--refresh-figi", action="store_true",
                   help="Bypass OpenFIGI cache and re-resolve all ISINs")
    p.add_argument("--eodhd", action="store_true",
                   help="Enrich with Morningstar rating + fundamentals via EODHD (requires EODHD_API_KEY env var)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("== Loading UCITS universe from Xetra master sheet ==")
    universe, stats = load_universe(
        ucits_only=args.ucits_only,
        refresh=args.refresh,
        include_justetf=args.justetf,
    )
    print(f"  rows_raw={stats.rows_raw}  distinct_isins={stats.rows_dedup}  "
          f"ucits={stats.ucits_count}  with_TER={stats.with_ter}  "
          f"with_replication={stats.with_replication}  with_benchmark={stats.with_benchmark}")

    print("\n== Categorising ==")
    universe = add_categories(universe)
    universe = add_benchmark_column(universe)
    cat_counts = universe.groupby("asset_class").size().sort_values(ascending=False)
    print(cat_counts.to_string())
    n_uncat = int((universe["category"] == "Uncategorized").sum())
    if n_uncat:
        sample = universe.loc[universe["category"] == "Uncategorized", "name"].head(5).tolist()
        print(f"  ! {n_uncat} uncategorized — see #uncat section in HTML. Examples:")
        for s in sample:
            print(f"      {s}")

    if args.exclude_leveraged:
        before = len(universe)
        universe = universe[~universe["leveraged"]].reset_index(drop=True)
        print(f"  Dropped {before - len(universe)} leveraged/inverse products")

    if args.openfigi:
        print("\n== Resolving tickers via OpenFIGI ==")
        universe = enrich_tickers_via_openfigi(universe, force=args.refresh_figi)

    pricing_pool = universe[universe["yahoo_ticker"] != ""].copy()
    if args.max_etfs and len(pricing_pool) > args.max_etfs:
        per_cat_quota = (
            pricing_pool.groupby("category").size()
            .mul(args.max_etfs / len(pricing_pool))
            .apply(lambda x: max(1, int(round(x))))
            .to_dict()
        )
        kept_idx = []
        for cat, grp in pricing_pool.groupby("category"):
            kept_idx.extend(grp.head(per_cat_quota.get(cat, 1)).index.tolist())
        pricing_pool = pricing_pool.loc[kept_idx].reset_index(drop=True)
        print(f"  Sampled {len(pricing_pool)} ETFs across {pricing_pool['category'].nunique()} categories")

    print(f"\n== Pulling prices ({len(pricing_pool)} tickers, period={args.period}) ==")
    prices = load_or_fetch_prices(
        pricing_pool["yahoo_ticker"].tolist(),
        period=args.period,
        refresh=args.refresh,
    )
    print(f"  Got prices for {prices.shape[1]} tickers, {prices.shape[0]} days")

    print("\n== Computing metrics ==")
    metrics = compute_metrics(prices, rf_annual=args.rf)
    print(f"  Metrics rows: {len(metrics)}")

    proxies = sorted({p for p in pricing_pool["benchmark_proxy"].unique() if p})
    if proxies:
        print(f"\n== Pulling {len(proxies)} benchmark proxies ==")
        bench_prices = load_or_fetch_prices(proxies, period=args.period, refresh=args.refresh)
        tracking_rows = []
        ticker_to_proxy = dict(zip(pricing_pool["yahoo_ticker"], pricing_pool["benchmark_proxy"]))
        for tkr in prices.columns:
            proxy = ticker_to_proxy.get(tkr, "")
            if not proxy or proxy not in bench_prices.columns:
                continue
            tm = tracking_metrics(prices[tkr], bench_prices[proxy])
            tm["ticker"] = tkr
            tm["benchmark_proxy"] = proxy
            tracking_rows.append(tm)
        if tracking_rows:
            tracking_df = pd.DataFrame(tracking_rows)
            metrics = metrics.merge(tracking_df, on="ticker", how="left")
            print(f"  Tracking metrics computed for {len(tracking_df)} ETFs")

    merged = pricing_pool.merge(metrics, left_on="yahoo_ticker", right_on="ticker", how="left")

    if args.eodhd:
        print("\n== Enriching with EODHD (Morningstar / fundamentals) ==")
        merged = enrich_with_eodhd(merged)

    print(f"\n== Writing report to {OUTPUT_DIR} ==")
    summary = build_category_summary(merged)
    html_path = write_html_report(
        merged, summary, OUTPUT_DIR,
        universe_stats=asdict(stats),
        used_justetf=args.justetf,
        prices=prices,
    )
    if html_path:
        print(f"  HTML report: {html_path}")
    else:
        print("  ! HTML not written (plotly missing? pip install plotly)")


if __name__ == "__main__":
    main()