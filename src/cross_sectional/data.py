"""
Cross-Sectional Data Pipeline

Chunked yfinance download of the S&P 500 universe with a local CSV cache.

Why the download uses auto_adjust=False and keeps three fields per ticker:
Yahoo's 'Adj Close' folds dividends into past prices while 'Volume' is
split-adjusted only, so adjusted-close dollar volume understates history for
dividend payers by the cumulative dividend factor (checked on this machine:
AAPL 2010-01-04 raw close 7.64 vs adjusted 6.41, identical volume, so the
adjusted-close dollar volume understates the true 2010 figure by about 16%).
The same adjustment pulls payers' past prices down and makes them look closer
to their 52-week high. Both distortions are cross-sectional and correlated with
dividend yield, which is a dimension the ranking sorts on. The split of duties
is therefore:

- ALL returns (features and the forward-return target) come from 'Adj Close'
  ratios (total return, correct).
- Dollar volume and the 52-week-high feature come from 'Close' x 'Volume'
  (split factors cancel, no dividend contamination; George and Hwang used
  split-adjusted prices).

Cache: three wide CSVs (date x ticker) because no parquet engine is installed
in this environment. data/xsec_meta.json records the download date, the
requested universe, failed tickers with reasons, and a complete flag written
only after every chunk and retry finished. A partial or mismatched cache
refuses to run and demands --refresh.

Usage:
    python -m src.cross_sectional.data            # build or validate the cache
    python -m src.cross_sectional.data --refresh  # wipe and redownload
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import shared as yf_shared

from src.cross_sectional.universe import load_universe

DATA_DIR = Path("data")
CACHE_FILES = {
    "adj_close": DATA_DIR / "xsec_adjclose.csv",
    "close": DATA_DIR / "xsec_close.csv",
    "volume": DATA_DIR / "xsec_volume.csv",
}
META_FILE = DATA_DIR / "xsec_meta.json"
FAILED_FILE = DATA_DIR / "failed_tickers.txt"

DEFAULT_START = "2010-01-01"
CHUNK_SIZE = 50
CHUNK_SLEEP_S = 1.5  # rate-limit hygiene between chunks
FIELDS = ["Adj Close", "Close", "Volume"]


def _extract_fields(raw: pd.DataFrame, tickers: list[str]) -> dict[str, dict[str, pd.Series]]:
    """
    Pull Adj Close / Close / Volume per REQUESTED ticker out of a yf.download
    result. Selection is by requested ticker with a KeyError guard; the field
    set is never inferred from the column union, because failed symbols come
    back as all-NaN columns and can carry a different field set than
    successful ones.
    """
    out: dict[str, dict[str, pd.Series]] = {f: {} for f in FIELDS}
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                sub = raw[t]
            except KeyError:
                continue
        else:
            # Single-ticker download can come back with flat columns.
            sub = raw
        for field in FIELDS:
            if field in sub.columns:
                out[field][t] = sub[field]
    return out


def _failure_reason(ticker: str) -> str:
    """Best-effort failure reason from yfinance's shared error registry."""
    err = yf_shared._ERRORS.get(ticker)
    if err is None:
        return "no data returned"
    return str(err)


def download_prices(
    tickers: list[str], start: str, end: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """
    Chunked batch download with one retry per failed ticker.

    A ticker counts as failed if its Adj Close column is missing or entirely
    NaN after the chunked pass. Each failure is retried once with a
    single-ticker download before being recorded, because a rate-limited name
    is otherwise indistinguishable from a dead one and would be baked into the
    cache forever. Partial history is fine.

    Args:
        tickers: Yahoo-format symbols (BRK-B, not BRK.B).
        start: Start date 'YYYY-MM-DD'.
        end: End date, defaults to today.

    Returns:
        Tuple of (adj_close, close, volume) wide DataFrames (date x ticker,
        all-NaN columns dropped) and {ticker: reason} for failures.
    """
    collected: dict[str, dict[str, pd.Series]] = {f: {} for f in FIELDS}
    chunks = [tickers[i : i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    for k, chunk in enumerate(chunks):
        print(f"  chunk {k + 1}/{len(chunks)} ({len(chunk)} tickers)...")
        raw = yf.download(
            tickers=chunk,
            start=start,
            end=end,
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        got = _extract_fields(raw, chunk)
        for field in FIELDS:
            collected[field].update(got[field])
        if k < len(chunks) - 1:
            time.sleep(CHUNK_SLEEP_S)

    def _failed(store: dict[str, dict[str, pd.Series]]) -> list[str]:
        return [
            t
            for t in tickers
            if t not in store["Adj Close"] or store["Adj Close"][t].isna().all()
        ]

    # One retry per failed ticker, single-ticker download.
    failed_reasons: dict[str, str] = {}
    for t in _failed(collected):
        print(f"  retrying {t}...")
        time.sleep(CHUNK_SLEEP_S)
        raw = yf.download(
            tickers=[t],
            start=start,
            end=end,
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=False,
        )
        got = _extract_fields(raw, [t])
        for field in FIELDS:
            collected[field].update(got[field])

    for t in _failed(collected):
        failed_reasons[t] = _failure_reason(t)
        for field in FIELDS:
            collected[field].pop(t, None)

    adj = pd.DataFrame(collected["Adj Close"]).sort_index()
    close = pd.DataFrame(collected["Close"]).sort_index()
    volume = pd.DataFrame(collected["Volume"]).sort_index()

    # Drop all-NaN columns so the cache contains only usable names.
    adj = adj.dropna(axis=1, how="all")
    close = close.reindex(columns=adj.columns)
    volume = volume.reindex(columns=adj.columns)
    return adj, close, volume, failed_reasons


def write_cache(
    adj: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    requested: list[str],
    failed: dict[str, str],
    start: str,
    end: str,
) -> None:
    """Write the three CSVs, the failed-ticker log, and the meta file (last)."""
    DATA_DIR.mkdir(exist_ok=True)
    adj.to_csv(CACHE_FILES["adj_close"], index_label="Date")
    close.to_csv(CACHE_FILES["close"], index_label="Date")
    volume.to_csv(CACHE_FILES["volume"], index_label="Date")

    with open(FAILED_FILE, "w", encoding="utf-8") as f:
        for t, reason in sorted(failed.items()):
            f.write(f"{t}: {reason}\n")

    meta = {
        "download_date": date.today().isoformat(),
        "start": start,
        "end": end,
        "requested": requested,
        "failed": failed,
        "complete": True,  # written only after every chunk and retry finished
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_cache(requested: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Load and validate the cache.

    Raises:
        RuntimeError: if the meta file is missing, the complete flag is not
            set, the start date differs, or the cache columns do not cover the
            requested universe minus recorded failures. The message demands
            --refresh; a partial cache is never silently used.
    """
    if not META_FILE.exists() or not all(p.exists() for p in CACHE_FILES.values()):
        raise RuntimeError("Price cache missing or incomplete. Run with --refresh.")
    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f)
    if not meta.get("complete"):
        raise RuntimeError("Price cache is marked incomplete. Run with --refresh.")
    if meta.get("start") != start:
        raise RuntimeError(
            f"Cache start {meta.get('start')} != requested start {start}. Run with --refresh."
        )

    adj = pd.read_csv(CACHE_FILES["adj_close"], index_col=0, parse_dates=True)
    close = pd.read_csv(CACHE_FILES["close"], index_col=0, parse_dates=True)
    volume = pd.read_csv(CACHE_FILES["volume"], index_col=0, parse_dates=True)

    expected = set(requested) - set(meta.get("failed", {}))
    missing = expected - set(adj.columns)
    if missing:
        raise RuntimeError(
            f"Cache does not cover the requested universe (missing {sorted(missing)[:5]}...). "
            "Run with --refresh."
        )
    return adj, close, volume, meta


def wipe_cache() -> None:
    """Delete all three cache files and the meta file."""
    for p in list(CACHE_FILES.values()) + [META_FILE]:
        if p.exists():
            p.unlink()


def get_prices(
    tickers: list[str], start: str = DEFAULT_START, refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Return (adj_close, close, volume, meta), from cache when valid, otherwise
    downloading and caching. refresh=True wipes the cache first.
    """
    if refresh:
        wipe_cache()
    if not refresh:
        try:
            return load_cache(tickers, start)
        except RuntimeError as exc:
            if META_FILE.exists() or any(p.exists() for p in CACHE_FILES.values()):
                # A partial cache is an error, not a trigger for a silent redownload.
                raise
            print(f"  no cache ({exc}); downloading...")

    end = date.today().isoformat()
    print(f"Downloading {len(tickers)} tickers, {start} to {end}...")
    adj, close, volume, failed = download_prices(tickers, start, end)
    write_cache(adj, close, volume, tickers, failed, start, end)
    for t, reason in sorted(failed.items()):
        print(f"  FAILED {t}: {reason}")
    print(f"  cached {adj.shape[1]} tickers x {adj.shape[0]} days")
    return load_cache(tickers, start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or validate the price cache.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--refresh", action="store_true", help="wipe cache and redownload")
    args = parser.parse_args()

    universe = load_universe()
    tickers = [row["yf_ticker"] for row in universe]
    adj, close, volume, meta = get_prices(tickers, start=args.start, refresh=args.refresh)
    print(
        f"Cache OK: {adj.shape[1]} tickers, {adj.shape[0]} days, "
        f"{len(meta['failed'])} failed, downloaded {meta['download_date']}"
    )


if __name__ == "__main__":
    main()
