"""
S&P 500 Universe Fetch

Fetches the current constituent list ONCE from Wikipedia and writes it to
data/universe_sp500.csv, which is committed to the repo so every rerun of the
study uses the identical universe.

This is a survivorship-biased universe by construction: today's members applied
retroactively. The bias direction and consequences are documented in the README.
The `date_added` column is kept because it makes the disclosure concrete (it
shows how many names joined the index after the sample start).

Parsing notes:
- Uses requests + BeautifulSoup with the stdlib "html.parser". pandas.read_html
  is NOT used because neither of its parser backends (lxml, html5lib) is
  installed in this environment and it raises ImportError.
- The constituents table has id="constituents" and a header starting
  Symbol, Security, GICS Sector. If the table id or the required headers are
  missing, this script fails loudly rather than guessing.
- yf_ticker is the Yahoo Finance symbol: '.' replaced with '-' (BRK.B -> BRK-B,
  BF.B -> BF-B). Yahoo rejects the dotted form.

Usage:
    python -m src.cross_sectional.universe          # refuses to overwrite
    python -m src.cross_sectional.universe --force  # re-fetch and overwrite
"""

import argparse
import csv
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UNIVERSE_CSV = Path("data") / "universe_sp500.csv"
REQUIRED_HEADERS = ["Symbol", "Security", "GICS Sector", "Date added"]


def fetch_constituents() -> list[dict]:
    """
    Fetch the current S&P 500 constituent table from Wikipedia.

    Returns:
        List of dicts with keys: ticker, yf_ticker, name, sector, date_added,
        fetched_on.

    Raises:
        RuntimeError: if the constituents table or its expected headers are
            not found. No fallback parsing is attempted.
    """
    resp = requests.get(WIKI_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id="constituents")
    if table is None:
        raise RuntimeError(
            "Wikipedia page has no table with id='constituents'. "
            "The page layout changed; update universe.py rather than guessing."
        )

    def _text(cell) -> str:
        # Header cells can contain line breaks ("GICS<br>Sector"); collapse
        # all internal whitespace so matching is layout-independent.
        return " ".join(cell.get_text().split())

    header_row = table.find("tr")
    headers = [_text(th) for th in header_row.find_all("th")]
    col_idx = {}
    for name in REQUIRED_HEADERS:
        if name not in headers:
            raise RuntimeError(
                f"Constituents table header is missing '{name}'. "
                f"Got headers: {headers}. Update universe.py rather than guessing."
            )
        col_idx[name] = headers.index(name)

    fetched_on = date.today().isoformat()
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < len(headers):
            continue
        ticker = _text(cells[col_idx["Symbol"]])
        if not ticker:
            continue
        rows.append(
            {
                "ticker": ticker,
                "yf_ticker": ticker.replace(".", "-"),
                "name": _text(cells[col_idx["Security"]]),
                "sector": _text(cells[col_idx["GICS Sector"]]),
                "date_added": _text(cells[col_idx["Date added"]]),
                "fetched_on": fetched_on,
            }
        )

    if len(rows) < 400:
        raise RuntimeError(
            f"Parsed only {len(rows)} constituents; expected roughly 500. "
            "The table layout likely changed; refusing to write a broken universe."
        )
    return rows


def write_universe(rows: list[dict], path: Path = UNIVERSE_CSV, force: bool = False) -> None:
    """
    Write the universe CSV. Idempotent: refuses to overwrite an existing file
    unless force=True, so reruns of the study never silently change the universe.
    """
    if path.exists() and not force:
        raise SystemExit(
            f"{path} already exists. The committed universe is intentionally "
            "frozen; pass --force to re-fetch and overwrite."
        )
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ticker", "yf_ticker", "name", "sector", "date_added", "fetched_on"]
        )
        writer.writeheader()
        writer.writerows(rows)


def load_universe(path: Path = UNIVERSE_CSV) -> list[dict]:
    """Load the committed universe CSV as a list of dicts."""
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run 'python -m src.cross_sectional.universe' first."
        )
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the S&P 500 universe from Wikipedia.")
    parser.add_argument("--force", action="store_true", help="overwrite the committed CSV")
    args = parser.parse_args()

    rows = fetch_constituents()
    write_universe(rows, force=args.force)
    print(f"Wrote {len(rows)} constituents to {UNIVERSE_CSV}")


if __name__ == "__main__":
    main()
