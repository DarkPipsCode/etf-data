"""Export the UCITS ETF universe and a broadsheet-sourced news feed as JSON,
for consumption by the companion iPhone app.

Reuses the pipeline in ETF.py end-to-end, then writes two files:

    output/app/bundle.json   # universe + per-ETF metrics + 1y sparkline
    output/app/news.json     # broadsheet headlines: global top stories + per-category

News sources:
    - Direct RSS feeds from FT, WSJ, NYT, The Economist, The Guardian, CNBC,
      MarketWatch (general business/markets sections)
    - Google News, filtered with site: operators, for per-category stories
      from Reuters and Bloomberg (no public RSS) plus the rest of the broadsheets

Usage:
    python ETF_iPhone_App.py --max-etfs 50          # quick smoke test
    python ETF_iPhone_App.py --justetf --openfigi   # full nightly run
    python ETF_iPhone_App.py --no-news              # skip news fetch
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests

import ETF as etf_pipeline


OUTPUT_DIR = etf_pipeline.OUTPUT_DIR / "app"

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
}
NEWS_TIMEOUT = 15
TOP_STORIES_PER_SOURCE = 8
CATEGORY_STORIES = 6


# ============================================================================
# Bundle export
# ============================================================================

METRIC_COLS = [
    "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_3y",
    "cagr_3y", "cagr_5y",
    "vol_1y", "vol_3y",
    "sharpe_1y", "sharpe_3y", "sortino_1y",
    "max_dd_3y",
    "rsi_14", "sma_50", "sma_200", "trend", "off_52w_high", "mom_12_1",
    "obs_days",
]
TRACKING_COLS = ["tracking_error", "tracking_diff", "correlation", "r2"]
META_COLS = [
    "isin", "name", "issuer", "asset_class", "category", "strategy", "region",
    "leveraged", "esg", "ter", "replication", "distribution",
    "fund_currency", "trading_currency", "benchmark", "benchmark_proxy",
    "yahoo_ticker", "xetra_symbol", "bloomberg_ticker", "product_type",
]


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _round(value, digits=4):
    v = _clean(value)
    if isinstance(v, (int, float)):
        return round(float(v), digits)
    return v


def _sparkline(series: pd.Series, points: int = 252, digits: int = 2) -> list[float]:
    s = series.dropna().tail(points)
    if s.empty:
        return []
    base = s.iloc[0]
    if not base or math.isnan(base):
        return []
    rebased = (s / base) * 100.0
    return [round(float(x), digits) for x in rebased.tolist()]


def build_categories_summary(merged: pd.DataFrame) -> list[dict]:
    grouped = merged.groupby(["asset_class", "category"], dropna=False)
    rows = []
    for (asset_class, category), grp in grouped:
        rows.append({
            "asset_class": asset_class or "Other",
            "category": category or "Uncategorized",
            "count": int(len(grp)),
            "median_ter": _round(grp["ter"].median(), 5),
            "median_ret_1y": _round(grp["ret_1y"].median(), 4) if "ret_1y" in grp else None,
            "median_sharpe_1y": _round(grp["sharpe_1y"].median(), 3) if "sharpe_1y" in grp else None,
            "median_vol_1y": _round(grp["vol_1y"].median(), 4) if "vol_1y" in grp else None,
        })
    rows.sort(key=lambda r: (r["asset_class"], -r["count"]))
    return rows


def build_etf_records(
    merged: pd.DataFrame,
    prices: pd.DataFrame,
    holdings_by_ticker: dict[str, list[dict]] | None = None,
    aum_by_ticker: dict[str, float | None] | None = None,
    ff_returns: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    records = []
    for row in merged.itertuples(index=False):
        d = row._asdict()
        meta = {col: _clean(d.get(col)) for col in META_COLS if col in d}
        metrics = {col: _round(d.get(col), 5) for col in METRIC_COLS if col in d}
        tracking = {col: _round(d.get(col), 5) for col in TRACKING_COLS if col in d}
        ticker = d.get("yahoo_ticker") or ""
        spark = _sparkline(prices[ticker]) if ticker and ticker in prices.columns else []
        holdings = (holdings_by_ticker or {}).get(ticker, [])
        aum = (aum_by_ticker or {}).get(ticker)
        factor_exposure = None
        drawdowns: list[dict] = []
        if ticker and ticker in prices.columns:
            series = prices[ticker].dropna()
            if not series.empty:
                drawdowns = top_drawdowns_for_series(series, top_n=5)
                if ff_returns:
                    factor_exposure = compute_factor_exposure(series, ff_returns)
        records.append({
            **meta,
            "metrics": metrics,
            "tracking": tracking,
            "sparkline": spark,
            "top_holdings": holdings,
            "aum": aum,
            "factor_exposure": factor_exposure,
            "drawdowns": drawdowns,
        })
    return records


def write_bundle(merged: pd.DataFrame, prices: pd.DataFrame, stats, out_dir: Path,
                 also_gzip: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n== Top holdings (yfinance funds_data, cached) ==")
    holdings_by_ticker = fetch_top_holdings(merged)
    holdings_index = build_holdings_index(merged, holdings_by_ticker)
    print("\n== AUM (yfinance .info, cached) ==")
    aum_by_ticker = fetch_aum(merged)
    print("\n== Fama-French (for per-ETF factor regression) ==")
    home_static = build_homepage_static(merged, prices)
    ff_returns = _ff_factor_returns(home_static.get("factors"))
    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe_stats": asdict(stats),
        "categories": build_categories_summary(merged),
        "etfs": build_etf_records(merged, prices, holdings_by_ticker, aum_by_ticker, ff_returns),
        "holdings_index": holdings_index,
        "home": home_static,
    }
    path = out_dir / "bundle.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))
    if also_gzip:
        with path.open("rb") as src, gzip.open(path.with_suffix(".json.gz"), "wb") as dst:
            dst.writelines(src)
    return path


# ============================================================================
# Homepage: weekly movers, cross-asset correlation, Fama-French factors
# ============================================================================

# Asset-class proxies for the correlation heatmap. Tuple: (yahoo_ticker, category_substring).
HOMEPAGE_PROXIES: list[tuple[str, str, str | None]] = [
    ("US Equity",     "^GSPC",   "Equity - US Large Cap"),
    ("World Equity",  "URTH",    "Equity - World"),
    ("US Treasuries", "IEF",     "Bond - US Treasury"),
    ("High Yield",    "HYG",     "Bond - High Yield"),
    ("Gold",          "GC=F",    "Commodity - Gold"),
    ("Oil",           "CL=F",    "Commodity - Oil & Gas"),
    ("Bitcoin",       "BTC-USD", "Crypto - Bitcoin"),
]


MOVER_MIN_OBS = 60               # need ≥ 3 months of daily prices
MOVER_MAX_1D_RETURN = 0.25       # drop any series with a single-day jump > 25% (split/data glitch)
MOVER_MAX_5D_RETURN = 0.50       # drop reported 5d returns beyond ±50% as implausible


def _is_clean_series(s: pd.Series) -> bool:
    """Reject a price series that looks like a corporate-action / data glitch."""
    if len(s) < MOVER_MIN_OBS:
        return False
    rets = s.pct_change().dropna()
    if rets.empty:
        return False
    if rets.abs().max() > MOVER_MAX_1D_RETURN:
        return False
    return True


def compute_weekly_movers(merged: pd.DataFrame, prices: pd.DataFrame,
                          top_n: int = 5) -> dict:
    rows = []
    skipped = {"no_ticker": 0, "no_history": 0, "leveraged": 0,
               "dirty_series": 0, "implausible_return": 0}
    for r in merged.itertuples(index=False):
        d = r._asdict()
        if d.get("leveraged"):
            skipped["leveraged"] += 1
            continue
        tkr = d.get("yahoo_ticker") or ""
        if not tkr:
            skipped["no_ticker"] += 1
            continue
        if tkr not in prices.columns:
            skipped["no_history"] += 1
            continue
        s = prices[tkr].dropna()
        if not _is_clean_series(s):
            skipped["dirty_series"] += 1
            continue
        last = float(s.iloc[-1])
        prev = float(s.iloc[-6])
        if prev <= 0 or math.isnan(prev) or math.isnan(last):
            skipped["dirty_series"] += 1
            continue
        ret_5d = last / prev - 1
        if abs(ret_5d) > MOVER_MAX_5D_RETURN:
            skipped["implausible_return"] += 1
            continue
        rows.append({
            "isin": d.get("isin"),
            "name": d.get("name"),
            "category": d.get("category"),
            "asset_class": d.get("asset_class"),
            "ret_5d": round(float(ret_5d), 5),
            "ret_1y": _round(d.get("ret_1y"), 4),
        })
    print(f"  weekly_movers: kept {len(rows)}, skipped {skipped}")
    if not rows:
        return {"best": [], "worst": []}
    rows.sort(key=lambda r: r["ret_5d"], reverse=True)
    return {"best": rows[:top_n], "worst": rows[-top_n:][::-1]}


def _top_etfs_for_label(merged: pd.DataFrame, category_substring: str | None,
                        limit: int = 2) -> list[dict]:
    if not category_substring:
        return []
    cat_col = merged["category"].fillna("")
    matches = merged[cat_col.str.contains(re.escape(category_substring),
                                          case=False, na=False)].copy()
    if matches.empty:
        return []
    if "sharpe_1y" in matches.columns:
        matches["__s"] = pd.to_numeric(matches["sharpe_1y"], errors="coerce").fillna(-1e9)
    else:
        matches["__s"] = 0
    matches = matches.sort_values("__s", ascending=False).head(limit)
    return [{"isin": r["isin"], "name": r["name"]} for _, r in matches.iterrows()]


def compute_correlations(merged: pd.DataFrame, window: int = 60) -> dict:
    tickers = [tkr for _label, tkr, _cat in HOMEPAGE_PROXIES]
    proxy_prices = etf_pipeline.load_or_fetch_prices(tickers, period="1y")
    if proxy_prices.empty:
        return {"labels": [], "matrix": [], "linked_etfs": {}, "window_days": window}

    available = [(label, tkr, cat) for label, tkr, cat in HOMEPAGE_PROXIES
                 if tkr in proxy_prices.columns]
    if not available:
        return {"labels": [], "matrix": [], "linked_etfs": {}, "window_days": window}

    cols = [tkr for _l, tkr, _c in available]
    labels = [label for label, _t, _c in available]
    rets = proxy_prices[cols].pct_change().dropna(how="all").tail(window)
    if len(rets) < 10:
        return {"labels": labels, "matrix": [], "linked_etfs": {}, "window_days": window}

    rets.columns = labels
    corr = rets.corr()
    matrix = [
        [None if pd.isna(corr.iloc[i, j]) else round(float(corr.iloc[i, j]), 3)
         for j in range(len(labels))]
        for i in range(len(labels))
    ]
    linked = {label: _top_etfs_for_label(merged, cat) for label, _t, cat in available}
    return {
        "labels": labels,
        "matrix": matrix,
        "linked_etfs": linked,
        "window_days": int(len(rets)),
    }


FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF_5_URL = f"{FF_BASE}/F-F_Research_Data_5_Factors_2x3_CSV.zip"
FF_MOM_URL = f"{FF_BASE}/F-F_Momentum_Factor_CSV.zip"


def _parse_ff_csv(text: str, cols: list[str], start_year: int = 2000) -> list[dict]:
    rows = []
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d{6})\s*,(.*)$", line)
        if not m:
            if in_data:
                break  # end of monthly section
            continue
        in_data = True
        date_str = m.group(1)
        vals = [v.strip() for v in m.group(2).split(",")]
        year = int(date_str[:4])
        mo = int(date_str[4:])
        if year < start_year:
            continue
        row: dict = {"date": f"{year:04d}-{mo:02d}"}
        for i, c in enumerate(cols):
            try:
                row[c] = float(vals[i]) / 100.0
            except (ValueError, IndexError):
                row[c] = None
        rows.append(row)
    return rows


def _fetch_ff_zip_csv(url: str) -> str:
    import io
    import zipfile
    raw = _http_get_bytes(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                return z.read(name).decode("latin-1")
    return ""


def fetch_fama_french(start_year: int = 2000) -> dict | None:
    try:
        five_text = _fetch_ff_zip_csv(FF_5_URL)
        mom_text = _fetch_ff_zip_csv(FF_MOM_URL)
    except Exception as exc:
        print(f"  ! Fama-French fetch failed: {exc}")
        return None

    five = _parse_ff_csv(five_text, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"], start_year)
    mom = _parse_ff_csv(mom_text, ["Mom"], start_year)
    if not five:
        return None

    mom_by_date = {r["date"]: r.get("Mom") for r in mom}
    factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    cumulative: dict[str, list] = {f: [] for f in factor_names}
    base: dict[str, float] = {f: 100.0 for f in factor_names}
    dates: list[str] = []

    for r in five:
        dates.append(r["date"])
        mom_val = mom_by_date.get(r["date"])
        values = {**r, "Mom": mom_val}
        for f in factor_names:
            v = values.get(f)
            if v is None:
                cumulative[f].append(cumulative[f][-1] if cumulative[f] else 100.0)
            else:
                base[f] *= (1.0 + v)
                cumulative[f].append(round(base[f], 2))

    return {
        "source": "Ken French Data Library",
        "frequency": "monthly",
        "from": dates[0] if dates else None,
        "to": dates[-1] if dates else None,
        "dates": dates,
        "cumulative": cumulative,
    }


# ============================================================================
# Top holdings (per-ETF + reverse index by stock symbol)
# ============================================================================

HOLDINGS_CACHE = etf_pipeline.CACHE_DIR / "holdings.json"
HOLDINGS_TTL_DAYS = 7


def _load_holdings_cache() -> dict[str, dict]:
    if HOLDINGS_CACHE.exists():
        try:
            return json.loads(HOLDINGS_CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_holdings_cache(cache: dict[str, dict]) -> None:
    HOLDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    HOLDINGS_CACHE.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")


def _fetch_one_top_holdings(ticker: str) -> list[dict]:
    """Pull top holdings from yfinance.funds_data. Returns [] if unavailable."""
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        fd = t.funds_data
    except Exception:
        return []
    if fd is None:
        return []
    try:
        df = fd.top_holdings
    except Exception:
        return []
    if df is None or df.empty:
        return []

    out: list[dict] = []
    df = df.reset_index() if df.index.name else df
    sym_col = next((c for c in df.columns if str(c).strip().lower() in ("symbol", "ticker")), None)
    name_col = next((c for c in df.columns
                     if str(c).strip().lower() in ("holding name", "name", "holdingname")), None)
    pct_col = next((c for c in df.columns
                    if "percent" in str(c).strip().lower() or "weight" in str(c).strip().lower()),
                   None)

    for _, row in df.iterrows():
        sym = str(row.get(sym_col, "")).strip() if sym_col else ""
        nm = str(row.get(name_col, "")).strip() if name_col else ""
        pct_raw = row.get(pct_col) if pct_col else None
        try:
            pct = float(pct_raw)
        except (TypeError, ValueError):
            continue
        if pd.isna(pct):
            continue
        if not sym and not nm:
            continue
        out.append({
            "symbol": sym or nm[:10].upper(),
            "name": nm,
            "weight": round(pct, 5),
        })
    return out


def fetch_top_holdings(merged: pd.DataFrame, force: bool = False,
                       ttl_days: int = HOLDINGS_TTL_DAYS,
                       sleep_s: float = 0.05) -> dict[str, list[dict]]:
    """Return {ticker: [{symbol, name, weight}]} for every yahoo_ticker in merged.

    Uses an on-disk JSON cache (TTL = ttl_days) so the daily workflow only
    re-fetches stale or missing entries.
    """
    from datetime import timedelta

    cache = _load_holdings_cache()
    now = datetime.now(timezone.utc)
    cutoff_iso = (now - timedelta(days=ttl_days)).isoformat()
    now_iso = now.isoformat(timespec="seconds")

    tickers = [t for t in merged["yahoo_ticker"].fillna("").tolist() if t]
    tickers = sorted(set(tickers))
    needed = []
    for t in tickers:
        entry = cache.get(t)
        if not entry or force or (entry.get("fetched_at", "") < cutoff_iso):
            needed.append(t)

    print(f"  holdings: cache hit {len(tickers) - len(needed)} / {len(tickers)}, "
          f"fetching {len(needed)} fresh")

    if needed:
        empty = 0
        for i, tkr in enumerate(needed, 1):
            try:
                holdings = _fetch_one_top_holdings(tkr)
            except Exception:
                holdings = []
            if not holdings:
                empty += 1
            cache[tkr] = {"fetched_at": now_iso, "holdings": holdings}
            if i % 100 == 0:
                _save_holdings_cache(cache)
                print(f"    progress: {i}/{len(needed)}  ({empty} empty so far)")
            time.sleep(sleep_s)
        _save_holdings_cache(cache)
        print(f"  holdings: completed, {empty}/{len(needed)} returned no data")

    return {t: cache.get(t, {}).get("holdings", []) for t in tickers}


def build_holdings_index(merged: pd.DataFrame,
                         holdings_by_ticker: dict[str, list[dict]],
                         min_etfs_per_symbol: int = 1) -> dict:
    """Build a reverse index { SYMBOL: [{isin, name, weight, holding_name}, ...] }
    sorted by weight desc, so a stock-symbol search resolves in O(1) on the
    client.
    """
    by_symbol: dict[str, list[dict]] = {}
    name_by_symbol: dict[str, str] = {}

    for r in merged.itertuples(index=False):
        d = r._asdict()
        tkr = d.get("yahoo_ticker") or ""
        if not tkr:
            continue
        hl = holdings_by_ticker.get(tkr, [])
        if not hl:
            continue
        isin = d.get("isin")
        etf_name = d.get("name")
        category = d.get("category")
        asset_class = d.get("asset_class")
        for h in hl:
            sym = (h.get("symbol") or "").upper()
            if not sym:
                continue
            if sym not in name_by_symbol and h.get("name"):
                name_by_symbol[sym] = h["name"]
            by_symbol.setdefault(sym, []).append({
                "isin": isin,
                "etf_name": etf_name,
                "category": category,
                "asset_class": asset_class,
                "weight": h["weight"],
            })

    pruned: dict[str, dict] = {}
    for sym, etfs in by_symbol.items():
        if len(etfs) < min_etfs_per_symbol:
            continue
        etfs.sort(key=lambda x: x["weight"], reverse=True)
        pruned[sym] = {
            "name": name_by_symbol.get(sym, ""),
            "etfs": etfs,
        }
    print(f"  holdings_index: {len(pruned)} unique stock symbols across "
          f"{sum(len(v['etfs']) for v in pruned.values())} ETF-holdings rows")
    return pruned


# ============================================================================
# AUM (assets under management) — yfinance .info["totalAssets"]
# ============================================================================

AUM_CACHE = etf_pipeline.CACHE_DIR / "aum.json"
AUM_TTL_DAYS = 14


def _load_aum_cache() -> dict[str, dict]:
    if AUM_CACHE.exists():
        try:
            return json.loads(AUM_CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_aum_cache(c: dict[str, dict]) -> None:
    AUM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    AUM_CACHE.write_text(json.dumps(c, separators=(",", ":")), encoding="utf-8")


def fetch_aum(merged: pd.DataFrame, ttl_days: int = AUM_TTL_DAYS,
              sleep_s: float = 0.05) -> dict[str, float | None]:
    """{ticker: aum_in_fund_currency_units}. Cached with TTL since AUM drifts slowly."""
    from datetime import timedelta
    import yfinance as yf

    cache = _load_aum_cache()
    now = datetime.now(timezone.utc)
    cutoff_iso = (now - timedelta(days=ttl_days)).isoformat()
    now_iso = now.isoformat(timespec="seconds")

    tickers = sorted({t for t in merged["yahoo_ticker"].fillna("").tolist() if t})
    needed = [t for t in tickers
              if (not cache.get(t)) or cache[t].get("fetched_at", "") < cutoff_iso]
    print(f"  aum: cache hit {len(tickers) - len(needed)} / {len(tickers)}, "
          f"fetching {len(needed)} fresh")
    if needed:
        for i, t in enumerate(needed, 1):
            aum = None
            try:
                info = yf.Ticker(t).info or {}
                for key in ("totalAssets", "netAssets", "fundFamilyAssets"):
                    v = info.get(key)
                    if isinstance(v, (int, float)) and v > 0:
                        aum = float(v)
                        break
            except Exception:
                pass
            cache[t] = {"fetched_at": now_iso, "aum": aum}
            if i % 200 == 0:
                _save_aum_cache(cache)
                print(f"    aum progress: {i}/{len(needed)}")
            time.sleep(sleep_s)
        _save_aum_cache(cache)

    return {t: cache.get(t, {}).get("aum") for t in tickers}


# ============================================================================
# Top drawdowns per ETF (start, trough, recovery duration)
# ============================================================================

def top_drawdowns_for_series(s: pd.Series, top_n: int = 5,
                              min_pct: float = 0.05) -> list[dict]:
    """Find the top_n peak-to-trough drawdowns in `s` (sorted by depth desc).
    Returns [{start, trough, recovered, depth, duration_days, recovery_days}]."""
    s = s.dropna()
    if len(s) < 30:
        return []
    rolling_max = s.cummax()
    dd = s / rolling_max - 1.0
    out: list[dict] = []
    i = 0
    while i < len(dd):
        if dd.iloc[i] >= -min_pct:
            i += 1
            continue
        # Walk back to find peak (where rolling_max changed)
        start_idx = i
        while start_idx > 0 and rolling_max.iloc[start_idx] == rolling_max.iloc[start_idx - 1]:
            start_idx -= 1
        peak_val = rolling_max.iloc[i]
        # Walk forward to find trough + recovery
        j = i
        trough_idx = i
        trough_val = s.iloc[i]
        recovered_idx = None
        while j < len(s) and s.iloc[j] < peak_val:
            if s.iloc[j] < trough_val:
                trough_val = s.iloc[j]
                trough_idx = j
            j += 1
        if j < len(s):
            recovered_idx = j
        depth = trough_val / peak_val - 1.0
        if depth <= -min_pct:
            dur = (trough_idx - start_idx)
            rec = ((recovered_idx - trough_idx) if recovered_idx is not None else None)
            out.append({
                "start": str(s.index[start_idx].date()) if hasattr(s.index[start_idx], "date") else str(s.index[start_idx]),
                "trough": str(s.index[trough_idx].date()) if hasattr(s.index[trough_idx], "date") else str(s.index[trough_idx]),
                "recovered": (str(s.index[recovered_idx].date()) if (recovered_idx is not None and hasattr(s.index[recovered_idx], "date")) else None),
                "depth": round(float(depth), 5),
                "duration_days": int(dur),
                "recovery_days": int(rec) if rec is not None else None,
            })
        i = j if recovered_idx is not None else j + 1
    out.sort(key=lambda d: d["depth"])
    return out[:top_n]


# ============================================================================
# Per-ETF Fama-French factor regression
# ============================================================================

def _ff_factor_returns(factors: dict | None) -> dict[str, dict[str, float]] | None:
    """Convert the cumulative FF series into monthly returns indexed by 'YYYY-MM'."""
    if not factors:
        return None
    dates = factors.get("dates") or []
    cum = factors.get("cumulative") or {}
    if not dates or not cum:
        return None
    out: dict[str, dict[str, float]] = {}
    for f, series in cum.items():
        for i in range(1, len(series)):
            prev = series[i - 1]
            cur = series[i]
            if prev is None or cur is None or prev <= 0:
                continue
            ret = cur / prev - 1.0
            out.setdefault(dates[i], {})[f] = ret
    return out


def compute_factor_exposure(prices: pd.Series, ff_returns: dict[str, dict[str, float]]
                            ) -> dict | None:
    """OLS regression of ETF monthly returns on FF factors. Returns betas + r2."""
    if prices is None or len(prices.dropna()) < 36:
        return None
    monthly = prices.resample("M").last().dropna()
    if len(monthly) < 24:
        return None
    rets = monthly.pct_change().dropna()
    rows = []
    for ts, r in rets.items():
        ym = ts.strftime("%Y-%m")
        ff = ff_returns.get(ym)
        if not ff or "Mkt-RF" not in ff or "RF" not in ff:
            continue
        rows.append({
            "y": r - ff["RF"],  # excess return
            "Mkt-RF": ff["Mkt-RF"],
            "SMB": ff.get("SMB", 0.0),
            "HML": ff.get("HML", 0.0),
            "RMW": ff.get("RMW", 0.0),
            "CMA": ff.get("CMA", 0.0),
            "Mom": ff.get("Mom", 0.0),
        })
    if len(rows) < 24:
        return None
    df = pd.DataFrame(rows).dropna()
    if len(df) < 24:
        return None
    X = df[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]].values
    y = df["y"].values
    # Append intercept
    X_ = np.column_stack([np.ones(len(X)), X])
    try:
        coef, *_ = np.linalg.lstsq(X_, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    y_hat = X_ @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {
        "alpha_monthly": round(float(coef[0]), 6),
        "betas": {
            "Mkt-RF": round(float(coef[1]), 4),
            "SMB":    round(float(coef[2]), 4),
            "HML":    round(float(coef[3]), 4),
            "RMW":    round(float(coef[4]), 4),
            "CMA":    round(float(coef[5]), 4),
            "Mom":    round(float(coef[6]), 4),
        },
        "r2": round(r2, 4) if r2 is not None else None,
        "n_months": int(len(df)),
    }


def build_homepage_static(merged: pd.DataFrame, prices: pd.DataFrame) -> dict:
    print("\n== Homepage static data ==")
    movers = compute_weekly_movers(merged, prices)
    print(f"  weekly movers: {len(movers['best'])} best, {len(movers['worst'])} worst")

    correlations = compute_correlations(merged)
    print(f"  correlations: {len(correlations.get('labels', []))} assets, "
          f"window={correlations.get('window_days')}")

    factors = fetch_fama_french(start_year=2000)
    if factors:
        print(f"  factors: {factors['from']} → {factors['to']}, "
              f"{len(factors['dates'])} months")
    else:
        print("  factors: unavailable")

    return {
        "weekly_movers": movers,
        "correlations": correlations,
        "factors": factors,
    }


# ============================================================================
# Broadsheet news
# ============================================================================

# Direct RSS feeds for general markets / business sections.
BROADSHEET_FEEDS: list[tuple[str, str]] = [
    ("Financial Times",      "https://www.ft.com/markets?format=rss"),
    ("Financial Times",      "https://www.ft.com/companies?format=rss"),
    ("Wall Street Journal",  "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("Wall Street Journal",  "https://feeds.a.dj.com/rss/RSSWorldNews.xml"),
    ("New York Times",       "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
    ("New York Times",       "https://rss.nytimes.com/services/xml/rss/nyt/YourMoney.xml"),
    ("The Economist",        "https://www.economist.com/finance-and-economics/rss.xml"),
    ("The Guardian",         "https://www.theguardian.com/uk/business/rss"),
    ("CNBC",                 "https://www.cnbc.com/id/15839069/device/rss/rss.html"),    # Markets
    ("CNBC",                 "https://www.cnbc.com/id/10000664/device/rss/rss.html"),    # Investing
    ("MarketWatch",          "http://feeds.marketwatch.com/marketwatch/marketpulse/"),
    ("MarketWatch",          "http://feeds.marketwatch.com/marketwatch/topstories/"),
]

# For per-category news, query Google News and restrict to these domains.
# Includes Reuters and Bloomberg, neither of which publish open RSS any more.
CATEGORY_NEWS_SITES = [
    "ft.com", "wsj.com", "nytimes.com", "economist.com", "theguardian.com",
    "reuters.com", "bloomberg.com", "cnbc.com", "marketwatch.com",
    "barrons.com",
]
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


# ============================================================================
# News impact analysis (sentiment + ETF mapping)
# ============================================================================

# Each rule: (regex, [(category_substring_to_match, flip_sentiment)])
# Categories use substring matching against ETF.py's classification labels.
IMPACT_RULES: list[tuple[re.Pattern[str], list[tuple[str, bool]]]] = [
    # Geography — equities
    (re.compile(r"\b(s&?p\s*500|wall\s*street|us\s+stocks?|new york stock|nyse\b|the\s+fed|federal\s+reserve|fomc)\b", re.I),
        [("Equity - US Large Cap", False)]),
    (re.compile(r"\bnasdaq\b", re.I), [("Equity - Nasdaq / Tech", False)]),
    (re.compile(r"\b(russell\s*2000|us\s+small[-\s]?cap)\b", re.I), [("Equity - US Small Cap", False)]),
    (re.compile(r"\b(ftse\s*100|ftse\s*250|uk\s+stocks?|british\s+stocks?|london stock|\bboe\b|bank of england)\b", re.I),
        [("Equity - UK", False)]),
    (re.compile(r"\b(euro\s+stoxx|stoxx\s*600|eurozone|european\s+stocks?|\becb\b|european central bank)\b", re.I),
        [("Equity - Europe Broad", False)]),
    (re.compile(r"\bdax\b|\bgermany\b|german\s+(stocks?|economy|exports?)", re.I),
        [("Equity - Germany", False)]),
    (re.compile(r"\b(nikkei|topix|japan(ese)?\s+(stocks?|equities|economy)|bank of japan|\bboj\b)\b", re.I),
        [("Equity - Japan", False)]),
    (re.compile(r"\b(csi\s*300|china(?:'s)?\s+(stocks?|economy|market)|hang\s*seng|\bhsi\b|beijing|shanghai)\b", re.I),
        [("Equity - China", False)]),
    (re.compile(r"\b(india(?:'s)?\s+(stocks?|economy|markets?)|\bnifty\b|\bsensex\b|\brbi\b)\b", re.I),
        [("Equity - India", False)]),
    (re.compile(r"\b(emerging\s+markets?|em\s+stocks?|em\s+equities)\b", re.I),
        [("Equity - Emerging Markets", False)]),

    # Sectors
    (re.compile(r"\b(big\s+tech|tech\s+(stocks?|sector)|silicon\s+valley|apple|microsoft|google|alphabet|amazon|meta\b)\b", re.I),
        [("Sector - Technology", False)]),
    (re.compile(r"\b(bank(s|ing)?|wells\s+fargo|jpmorgan|jp\s+morgan|goldman|morgan\s+stanley|citi)\b", re.I),
        [("Sector - Financials", False)]),
    (re.compile(r"\b(pharma(ceutical)?|drug\s+(approval|trial)|fda\b|biotech)\b", re.I),
        [("Sector - Healthcare", False), ("Thematic - Biotech", False)]),
    (re.compile(r"\b(real\s+estate|housing\s+market|home\s+prices?|\breit\b)\b", re.I),
        [("Sector - Real Estate", False)]),
    (re.compile(r"\b(utilit(y|ies))\b", re.I), [("Sector - Utilities", False)]),

    # Themes
    (re.compile(r"\b(ai|artificial\s+intelligence|chatgpt|openai|nvidia|generative\s+ai)\b", re.I),
        [("Thematic - Robotics & AI", False), ("Thematic - Semiconductors", False)]),
    (re.compile(r"\b(semiconductor|chip\s+(maker|shortage|sales)|tsmc|asml)\b", re.I),
        [("Thematic - Semiconductors", False)]),
    (re.compile(r"\b(electric\s+vehicle|\bev\b|tesla|byd\b|battery|lithium)\b", re.I),
        [("Thematic - Battery / EV", False)]),
    (re.compile(r"\b(clean\s+energy|renewable|solar|wind\s+(power|farm))\b", re.I),
        [("Thematic - Clean Energy", False)]),
    (re.compile(r"\bcyber(security|attack)\b|data\s+breach", re.I),
        [("Thematic - Cybersecurity", False)]),
    (re.compile(r"\b(defen[cs]e|military|aerospace|nato)\b", re.I),
        [("Thematic - Defence", False)]),

    # Commodities
    (re.compile(r"\b(oil|crude|wti|brent|opec|saudi\s+aramco)\b", re.I),
        [("Commodity - Oil & Gas", False)]),
    (re.compile(r"\bnatural\s+gas|lng\b", re.I), [("Commodity - Oil & Gas", False)]),
    (re.compile(r"\bgold\b", re.I), [("Commodity - Gold", False)]),
    (re.compile(r"\bsilver\b", re.I), [("Commodity - Silver", False)]),
    (re.compile(r"\bcopper\b", re.I), [("Commodity - Industrial Metals", False)]),
    (re.compile(r"\b(wheat|corn|soybean|coffee|sugar|agricultur)", re.I),
        [("Commodity - Agriculture", False)]),

    # Crypto
    (re.compile(r"\bbitcoin|\bbtc\b", re.I), [("Crypto - Bitcoin", False)]),
    (re.compile(r"\bethereum|\beth\b\s+(price|rally)", re.I), [("Crypto - Ethereum", False)]),
    (re.compile(r"\b(crypto|digital\s+asset|blockchain|web3|defi\b)\b", re.I),
        [("Crypto", False)]),

    # Rates / bonds
    (re.compile(r"\btreasur(y|ies)|10-?year\s+yield|bond\s+(market|yield)\b", re.I),
        [("Bond - US Treasury", False), ("Bond - Aggregate", False)]),
    (re.compile(r"\b(gilt|uk\s+bond)\b", re.I), [("Bond - UK Gilts", False)]),
    (re.compile(r"\bhigh[-\s]?yield|junk\s+bond", re.I), [("Bond - High Yield", False)]),
    (re.compile(r"\binflation|\bcpi\b|consumer\s+price", re.I),
        [("Bond - Inflation-Linked", False)]),
]

POSITIVE_RE = re.compile(
    r"\b(surge|rally|jump|gain|rise|rises|risen|soar|boom|outperform|beat|beats|"
    r"breakthrough|approve|approved|deal|merger|upgrade|record\s+high|all[-\s]?time\s+high|"
    r"rebound|recover|recover(s|y|ed)|strong\s+(earnings|results|growth)|profit\s+(rise|jump|surge))\b",
    re.I,
)
NEGATIVE_RE = re.compile(
    r"\b(plunge|crash|fall|falls|fell|drop|slump|tumble|tumbles|"
    r"miss(es|ed)?\s+(earnings|estimates|forecast)?|recall|fraud|ban\b|sanction|fine|insider\s+trading|"
    r"downgrade|lawsuit|investigat|warn(s|ing|ed)?|profit\s+warning|cut\s+(forecast|outlook|jobs|workforce)|"
    r"layoff|bankrupt|default|sell[-\s]?off)\b",
    re.I,
)


def _classify_sentiment(title: str) -> str:
    pos = bool(POSITIVE_RE.search(title))
    neg = bool(NEGATIVE_RE.search(title))
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return "neutral"


def _flatten_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """If df came from bundle.json, metrics is a dict column — lift it."""
    if "metrics" in df.columns and df["metrics"].apply(lambda x: isinstance(x, dict)).any():
        for col in ("sharpe_1y", "ret_1y", "obs_days", "vol_1y"):
            if col not in df.columns:
                df[col] = df["metrics"].apply(
                    lambda m, c=col: (m or {}).get(c) if isinstance(m, dict) else None
                )
    return df


def _top_etfs_for_category(df: pd.DataFrame, category_substring: str,
                           limit: int = 5) -> list[dict]:
    cat_col = df["category"].fillna("")
    matches = df[cat_col.str.contains(re.escape(category_substring), case=False, na=False)].copy()
    if matches.empty:
        return []
    if "sharpe_1y" in matches.columns:
        matches["__sort"] = pd.to_numeric(matches["sharpe_1y"], errors="coerce").fillna(-1e9)
    else:
        matches["__sort"] = 0
    matches = matches.sort_values("__sort", ascending=False).head(limit)
    out = []
    for _, row in matches.iterrows():
        out.append({
            "isin": row.get("isin"),
            "name": row.get("name"),
            "category": row.get("category"),
            "ret_1y": _round(row.get("ret_1y"), 4),
            "sharpe_1y": _round(row.get("sharpe_1y"), 3),
        })
    return out


def _enrich_story(story: dict, df: pd.DataFrame, default_category: str = "") -> dict:
    title = story.get("title") or ""
    sentiment = _classify_sentiment(title)

    matched: list[tuple[str, bool]] = []
    for rx, targets in IMPACT_RULES:
        if rx.search(title):
            matched.extend(targets)

    # If we know the news bucket's category explicitly (by_category mode), seed it.
    if default_category:
        matched.append((default_category, False))

    seen_categories: set[str] = set()
    impacted: list[dict] = []
    for cat_sub, flip in matched:
        for etf in _top_etfs_for_category(df, cat_sub, limit=5):
            key = etf["isin"]
            if not key or key in {e["isin"] for e in impacted}:
                continue
            direction = sentiment
            if flip and sentiment in {"positive", "negative"}:
                direction = "negative" if sentiment == "positive" else "positive"
            etf["direction"] = direction
            impacted.append(etf)
        seen_categories.add(cat_sub)

    enriched = dict(story)
    enriched["sentiment"] = sentiment
    enriched["impacted_etfs"] = impacted[:12]
    enriched["matched_categories"] = sorted(seen_categories)
    return enriched


def _parse_pub_date(raw: str) -> str:
    if not raw:
        return ""
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
        except ValueError:
            continue
    return raw


_RSS_NS = {
    "media": "http://search.yahoo.com/mrss/",
    "content": "http://purl.org/rss/1.0/modules/content/",
}
_IMG_SRC_RE = re.compile(r"""<img[^>]+src=["']([^"']+)["']""", re.I)


def _extract_image(item) -> str:
    # media:content / media:thumbnail (FT, NYT, Guardian, Economist)
    for tag in ("media:content", "media:thumbnail"):
        el = item.find(tag, _RSS_NS)
        if el is not None:
            url = (el.get("url") or "").strip()
            if url:
                return url
    # enclosure (CNBC, MarketWatch, WSJ)
    enc = item.find("enclosure")
    if enc is not None:
        mime = (enc.get("type") or "").lower()
        url = (enc.get("url") or "").strip()
        if url and (mime.startswith("image/") or url.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))):
            return url
    # First <img> inside <description> or <content:encoded>
    for tag in ("description", "content:encoded"):
        el = item.find(tag, _RSS_NS) if ":" in tag else item.find(tag)
        if el is not None and el.text:
            m = _IMG_SRC_RE.search(el.text)
            if m:
                return m.group(1).strip()
    return ""


def _parse_rss(xml_text: str, default_source: str = "") -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source = ""
        if source_el is not None and source_el.text:
            source = source_el.text.strip()
        if not source:
            source = default_source
        if not title or not link:
            continue
        items.append({
            "title": title,
            "url": link,
            "source": source,
            "published_at": _parse_pub_date(pub),
            "image_url": _extract_image(item),
        })
    return items


_USE_POWERSHELL_HTTP = sys.platform == "win32"


def _http_get_urllib(url: str) -> str:
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=NEWS_TIMEOUT) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _http_get_powershell(url: str) -> str:
    # Windows fallback: this venv's Python 3.14 OpenSSL build is broken
    # (OPENSSL_Uplink / no OPENSSL_Applink), so we shell out to PowerShell's
    # Invoke-WebRequest, which uses .NET / Schannel and works reliably.
    # URL is passed via an env var to avoid PowerShell re-parsing query
    # strings that contain `&`.
    ps = (
        "$ProgressPreference='SilentlyContinue';"
        "$ErrorActionPreference='Stop';"
        "$r = Invoke-WebRequest -UseBasicParsing "
        "  -UserAgent $env:HTTP_UA "
        f"  -TimeoutSec {NEWS_TIMEOUT} "
        "  -Uri $env:HTTP_URL;"
        "[Console]::OpenStandardOutput().Write($r.RawContentStream.ToArray(), 0, "
        "  [int]$r.RawContentStream.Length)"
    )
    env = {**os.environ,
           "HTTP_URL": url,
           "HTTP_UA": HTTP_HEADERS["User-Agent"],
           "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
        capture_output=True, timeout=NEWS_TIMEOUT + 10, env=env,
    )
    if result.returncode != 0:
        raise OSError(f"powershell http get failed ({result.returncode}): "
                      f"{result.stderr.decode('utf-8', errors='replace').strip()}")
    return result.stdout.decode("utf-8", errors="replace")


def _http_get(url: str) -> str:
    if _USE_POWERSHELL_HTTP:
        return _http_get_powershell(url)
    return _http_get_urllib(url)


def _http_get_bytes_urllib(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=NEWS_TIMEOUT * 2) as resp:
        return resp.read()


def _http_get_bytes_powershell(url: str) -> bytes:
    ps = (
        "$ProgressPreference='SilentlyContinue';"
        "$ErrorActionPreference='Stop';"
        "$r = Invoke-WebRequest -UseBasicParsing "
        "  -UserAgent $env:HTTP_UA "
        f"  -TimeoutSec {NEWS_TIMEOUT * 2} "
        "  -Uri $env:HTTP_URL;"
        "[Console]::OpenStandardOutput().Write($r.RawContentStream.ToArray(), 0, "
        "  [int]$r.RawContentStream.Length)"
    )
    env = {**os.environ, "HTTP_URL": url, "HTTP_UA": HTTP_HEADERS["User-Agent"],
           "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
        capture_output=True, timeout=NEWS_TIMEOUT * 2 + 10, env=env,
    )
    if result.returncode != 0:
        raise OSError(f"powershell http bytes failed ({result.returncode}): "
                      f"{result.stderr.decode('utf-8', errors='replace').strip()}")
    return result.stdout


def _http_get_bytes(url: str) -> bytes:
    if _USE_POWERSHELL_HTTP:
        return _http_get_bytes_powershell(url)
    return _http_get_bytes_urllib(url)


def fetch_top_stories(merged: pd.DataFrame,
                      per_source: int = TOP_STORIES_PER_SOURCE,
                      sleep_s: float = 0.4) -> dict[str, list[dict]]:
    by_source: dict[str, list[dict]] = {}
    for source, url in BROADSHEET_FEEDS:
        try:
            xml = _http_get(url)
            stories = _parse_rss(xml, default_source=source)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  ! {source:22s} {url}  ->  {exc}")
            stories = []

        bucket = by_source.setdefault(source, [])
        seen = {s["url"] for s in bucket}
        for s in stories:
            if s["url"] in seen:
                continue
            bucket.append(_enrich_story(s, merged))
            seen.add(s["url"])
            if len(bucket) >= per_source:
                break
        print(f"  {source:22s} +{len(stories):3d} items  (cumulative: {len(bucket)})")
        time.sleep(sleep_s)
    return by_source


def _category_query(asset_class: str, category: str) -> str:
    cat = (category or "").strip()
    ac = (asset_class or "").strip()
    if cat and cat.lower() != "uncategorized":
        base = cat.split(" - ")[0]
    elif ac:
        base = ac
    else:
        return ""
    sites = " OR ".join(f"site:{d}" for d in CATEGORY_NEWS_SITES)
    return f'"{base}" ETF ({sites})'


def fetch_category_news(merged: pd.DataFrame,
                        per_category: int = CATEGORY_STORIES,
                        sleep_s: float = 0.5) -> dict[str, list[dict]]:
    """Pull per-(asset_class::category) stories. `merged` already flattened."""
    pairs = (
        merged[["asset_class", "category"]]
        .dropna(how="all")
        .drop_duplicates()
        .sort_values(["asset_class", "category"])
        .itertuples(index=False, name=None)
    )

    by_category: dict[str, list[dict]] = {}
    seen_q: dict[str, str] = {}

    for asset_class, category in pairs:
        ac = asset_class or "Other"
        cat = category or "Uncategorized"
        key = f"{ac}::{cat}"
        query = _category_query(ac, cat)
        if not query:
            continue
        if query in seen_q:
            by_category[key] = list(by_category[seen_q[query]])
            continue
        seen_q[query] = key

        url = f"{GOOGLE_NEWS_RSS}?q={urllib.parse.quote(query)}&hl=en-GB&gl=GB&ceid=GB:en"
        try:
            xml = _http_get(url)
            stories = _parse_rss(xml)[:per_category]
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  ! {key}  ->  {exc}")
            stories = []
        by_category[key] = [_enrich_story(s, merged, default_category=cat) for s in stories]
        print(f"  {key:48s}  stories={len(stories)}")
        time.sleep(sleep_s)
    return by_category


FORWARD_LOOKING_RE = re.compile(
    r"\b(ahead\s+of|set\s+to|preview|coming\s+(week|days|month)|"
    r"will\s+(release|decide|announce|publish|cut|raise|hike|hold|meet|address)|"
    r"next\s+(week|month|meeting|wednesday|thursday|friday)|"
    r"due\s+(out|to)|expected\s+to|outlook\s+for|"
    r"fed\s+(meeting|decision|minutes)|ecb\s+(meeting|decision|to\s+meet)|"
    r"fomc|cpi\s+(report|release|print|data)|jobs?\s+report|payrolls?\b|"
    r"\bnfp\b|earnings\s+(preview|expected|due|season)|to\s+report|"
    r"\b(monday|tuesday|wednesday|thursday|friday)\b)\b",
    re.I,
)


def scan_upcoming_events(top_by_source: dict[str, list[dict]],
                         max_items: int = 10) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for stories in top_by_source.values():
        for s in stories:
            url = s.get("url") or ""
            if url in seen:
                continue
            title = s.get("title") or ""
            if not FORWARD_LOOKING_RE.search(title):
                continue
            seen.add(url)
            out.append({
                "title": title,
                "url": url,
                "source": s.get("source"),
                "published_at": s.get("published_at"),
                "sentiment": s.get("sentiment"),
                "impacted_etfs": (s.get("impacted_etfs") or [])[:3],
                "matched_categories": s.get("matched_categories") or [],
            })
    out.sort(key=lambda e: e.get("published_at") or "", reverse=True)
    return out[:max_items]


def pick_market_movers(top_by_source: dict[str, list[dict]],
                       max_items: int = 5) -> list[dict]:
    """Top stories ranked by breadth of likely impact (impacted_etfs count) and sentiment != neutral."""
    pool: list[dict] = []
    seen: set[str] = set()
    for stories in top_by_source.values():
        for s in stories:
            url = s.get("url") or ""
            if url in seen:
                continue
            seen.add(url)
            impact = len(s.get("impacted_etfs") or [])
            sentiment = s.get("sentiment") or "neutral"
            if impact == 0 or sentiment == "neutral":
                continue
            pool.append({
                "title": s.get("title"),
                "url": url,
                "source": s.get("source"),
                "published_at": s.get("published_at"),
                "sentiment": sentiment,
                "impacted_etfs": (s.get("impacted_etfs") or [])[:4],
                "matched_categories": s.get("matched_categories") or [],
                "_score": impact,
            })
    pool.sort(key=lambda e: (e["_score"], e.get("published_at") or ""), reverse=True)
    for e in pool:
        e.pop("_score", None)
    return pool[:max_items]


# X / Twitter mentions via Nitter (unofficial mirror). Often flaky — we try
# several instances and silently fall back to empty if all fail.
NITTER_INSTANCES = [
    "nitter.privacydev.net",
    "nitter.poast.org",
    "nitter.net",
    "nitter.tiekoetter.com",
    "nitter.holo-mix.com",
]
NITTER_ACCOUNTS = [
    "LizAnnSonders", "lisaabramowicz1", "SoberLook", "charliebilello",
    "MichaelKantro", "biancoresearch", "M_McDonough", "JeffWeniger",
    "FT", "WSJmarkets", "EconomistFinance",
]


def fetch_nitter_mentions(accounts: list[str] = NITTER_ACCOUNTS,
                          per_account: int = 3,
                          max_age_h: int = 36) -> list[dict]:
    out: list[dict] = []
    now = datetime.now(timezone.utc)
    for handle in accounts:
        for inst in NITTER_INSTANCES:
            try:
                xml = _http_get(f"https://{inst}/{handle}/rss")
            except Exception:
                continue
            items = _parse_rss(xml, default_source=f"@{handle}")
            if not items:
                continue
            kept = 0
            for it in items:
                pub = it.get("published_at", "")
                try:
                    t = datetime.fromisoformat(pub) if pub else now
                    age_h = (now - t).total_seconds() / 3600
                    if age_h > max_age_h:
                        continue
                except ValueError:
                    pass
                out.append({**it, "handle": handle})
                kept += 1
                if kept >= per_account:
                    break
            break  # successful instance, stop trying others for this handle
    out.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    return out


def fetch_news(merged: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged = _flatten_metrics(merged.copy())

    print("\n  -- Top stories from broadsheet RSS --")
    top = fetch_top_stories(merged)

    print("\n  -- Per-category stories (Google News, site-filtered to broadsheets) --")
    by_cat = fetch_category_news(merged)

    print("\n  -- Homepage news sections --")
    upcoming = scan_upcoming_events(top)
    print(f"  upcoming events: {len(upcoming)}")
    movers = pick_market_movers(top)
    print(f"  market movers: {len(movers)}")
    x_posts = fetch_nitter_mentions()
    print(f"  X mentions: {len(x_posts)} (Nitter is flaky; 0 is normal)")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [s for s, _ in BROADSHEET_FEEDS],
        "top_stories_by_source": top,
        "by_category": by_cat,
        "home": {
            "upcoming_events": upcoming,
            "market_movers": movers,
            "x_mentions": x_posts,
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return out_path


# ============================================================================
# Pipeline orchestration
# ============================================================================

def build_merged(args: argparse.Namespace):
    print("== Loading UCITS universe from Xetra master sheet ==")
    universe, stats = etf_pipeline.load_universe(
        ucits_only=True,
        refresh=args.refresh,
        include_justetf=args.justetf,
    )
    print(f"  distinct_isins={stats.rows_dedup}  ucits={stats.ucits_count}  "
          f"with_TER={stats.with_ter}")

    print("\n== Categorising ==")
    universe = etf_pipeline.add_categories(universe)
    universe = etf_pipeline.add_benchmark_column(universe)

    if args.exclude_leveraged:
        before = len(universe)
        universe = universe[~universe["leveraged"]].reset_index(drop=True)
        print(f"  Dropped {before - len(universe)} leveraged/inverse products")

    if args.openfigi:
        print("\n== Resolving tickers via OpenFIGI ==")
        universe = etf_pipeline.enrich_tickers_via_openfigi(universe, force=args.refresh_figi)

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
        print(f"  Sampled {len(pricing_pool)} ETFs across "
              f"{pricing_pool['category'].nunique()} categories")

    print(f"\n== Pulling prices ({len(pricing_pool)} tickers, period={args.period}) ==")
    prices = etf_pipeline.load_or_fetch_prices(
        pricing_pool["yahoo_ticker"].tolist(),
        period=args.period,
        refresh=args.refresh,
    )
    print(f"  Got prices for {prices.shape[1]} tickers, {prices.shape[0]} days")

    print("\n== Computing metrics ==")
    metrics = etf_pipeline.compute_metrics(prices, rf_annual=args.rf)

    proxies = sorted({p for p in pricing_pool["benchmark_proxy"].unique() if p})
    if proxies:
        print(f"\n== Pulling {len(proxies)} benchmark proxies ==")
        bench_prices = etf_pipeline.load_or_fetch_prices(proxies, period=args.period,
                                                         refresh=args.refresh)
        rows = []
        ticker_to_proxy = dict(zip(pricing_pool["yahoo_ticker"],
                                   pricing_pool["benchmark_proxy"]))
        for tkr in prices.columns:
            proxy = ticker_to_proxy.get(tkr, "")
            if not proxy or proxy not in bench_prices.columns:
                continue
            tm = etf_pipeline.tracking_metrics(prices[tkr], bench_prices[proxy])
            tm["ticker"] = tkr
            rows.append(tm)
        if rows:
            metrics = metrics.merge(pd.DataFrame(rows), on="ticker", how="left")
            print(f"  Tracking metrics computed for {len(rows)} ETFs")

    merged = pricing_pool.merge(metrics, left_on="yahoo_ticker", right_on="ticker",
                                how="left")

    if args.eodhd:
        print("\n== Enriching with EODHD ==")
        merged = etf_pipeline.enrich_with_eodhd(merged)

    return merged, prices, stats


# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true",
                   help="Re-download Xetra master + price cache")
    p.add_argument("--max-etfs", type=int, default=0,
                   help="Cap ETFs (stratified by category, 0 = no cap)")
    p.add_argument("--period", default="5y", help="Yahoo lookback")
    p.add_argument("--rf", type=float, default=0.02, help="Annual risk-free rate")
    p.add_argument("--exclude-leveraged", action="store_true")
    p.add_argument("--justetf", action="store_true",
                   help="Augment with justETF screener data")
    p.add_argument("--openfigi", action="store_true",
                   help="Resolve missing tickers via OpenFIGI")
    p.add_argument("--refresh-figi", action="store_true")
    p.add_argument("--eodhd", action="store_true",
                   help="Enrich with EODHD (requires EODHD_API_KEY)")
    p.add_argument("--no-news", dest="news", action="store_false",
                   help="Skip news fetch")
    p.add_argument("--news-only", action="store_true",
                   help="Skip bundle; just refresh news.json (needs an existing bundle)")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.news_only:
        bundle_path = args.out_dir / "bundle.json"
        if not bundle_path.exists():
            raise SystemExit(f"--news-only requires {bundle_path} to exist")
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        merged = pd.DataFrame(bundle["etfs"])
        print(f"\n== Fetching news for {len(merged)} ETFs (news-only mode) ==")
        news_path = fetch_news(merged, args.out_dir / "news.json")
        print(f"  news.json: {news_path.stat().st_size / 1024:.1f} KB")
        return

    merged, prices, stats = build_merged(args)

    print(f"\n== Writing bundle to {args.out_dir} ==")
    bundle_path = write_bundle(merged, prices, stats, args.out_dir)
    size_mb = bundle_path.stat().st_size / 1024 / 1024
    print(f"  bundle.json: {size_mb:.2f} MB  ({len(merged)} ETFs)")

    if args.news:
        print(f"\n== Fetching broadsheet news ==")
        news_path = fetch_news(merged, args.out_dir / "news.json")
        print(f"  news.json: {news_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
