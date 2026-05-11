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


def build_etf_records(merged: pd.DataFrame, prices: pd.DataFrame) -> list[dict]:
    records = []
    for row in merged.itertuples(index=False):
        d = row._asdict()
        meta = {col: _clean(d.get(col)) for col in META_COLS if col in d}
        metrics = {col: _round(d.get(col), 5) for col in METRIC_COLS if col in d}
        tracking = {col: _round(d.get(col), 5) for col in TRACKING_COLS if col in d}
        ticker = d.get("yahoo_ticker") or ""
        spark = _sparkline(prices[ticker]) if ticker and ticker in prices.columns else []
        records.append({**meta, "metrics": metrics, "tracking": tracking, "sparkline": spark})
    return records


def write_bundle(merged: pd.DataFrame, prices: pd.DataFrame, stats, out_dir: Path,
                 also_gzip: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "universe_stats": asdict(stats),
        "categories": build_categories_summary(merged),
        "etfs": build_etf_records(merged, prices),
    }
    path = out_dir / "bundle.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))
    if also_gzip:
        with path.open("rb") as src, gzip.open(path.with_suffix(".json.gz"), "wb") as dst:
            dst.writelines(src)
    return path


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


def fetch_top_stories(per_source: int = TOP_STORIES_PER_SOURCE,
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
            bucket.append(s)
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
        by_category[key] = stories
        print(f"  {key:48s}  stories={len(stories)}")
        time.sleep(sleep_s)
    return by_category


def fetch_news(merged: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n  -- Top stories from broadsheet RSS --")
    top = fetch_top_stories()

    print("\n  -- Per-category stories (Google News, site-filtered to broadsheets) --")
    by_cat = fetch_category_news(merged)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [s for s, _ in BROADSHEET_FEEDS],
        "top_stories_by_source": top,
        "by_category": by_cat,
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
