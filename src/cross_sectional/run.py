"""
Phase 2 Entry Point

Runs the full cross-sectional study: universe -> data -> panel -> baselines
-> walk-forward models -> evaluation -> results JSON and chart. The default
invocation reproduces every phase 2 number in the README.

Pre-registered headline test (fixed before any result is seen): the
Newey-West t-stat on the mean monthly GROSS XGBoost D10-minus-D1 long-short
return over the common evaluation window. The net-of-cost table is the
economic qualifier on that result; every other number in the study is
descriptive.

Usage:
    python -m src.cross_sectional.run
    python -m src.cross_sectional.run --refresh --models lin,xgb,rf --costs 0,5,10,20

Console output is ASCII-only: under redirection Windows falls back to the
locale codepage and non-cp1252-safe output raises UnicodeEncodeError.
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

from src.cross_sectional.baselines import baseline_scores
from src.cross_sectional.data import DEFAULT_START, get_prices
from src.cross_sectional.evaluate import equal_weight_benchmark, evaluate_scores
from src.cross_sectional.features import build_panel
from src.cross_sectional.models import BURN_IN_MONTHS, walk_forward_scores
from src.cross_sectional.plots import plot_decile_returns
from src.cross_sectional.universe import load_universe

RESULTS_JSON = Path("outputs") / "cross_sectional_results.json"

METHOD_LABELS = {
    "b1_mom": "B1 momentum",
    "b2_rev": "B2 reversal",
    "b3_combo": "B3 combo",
    "lin": "Linear",
    "xgb": "XGBoost",
    "rf": "Random Forest",
}


def _print_method(name: str, res: dict) -> None:
    print(
        f"  {METHOD_LABELS[name]:<14} IC {res['mean_ic']:+.4f} (NW t {res['ic_tstat_nw']:+.2f}, "
        f"{res['pct_positive_ic']:.0%} pos) | LS ann {res['ls_ann_return']:+.2%} "
        f"vol {res['ls_ann_vol']:.2%} Sharpe {res['ls_sharpe']:+.2f} "
        f"maxDD {res['ls_max_dd']:.2%} NW t {res['ls_tstat_nw']:+.2f} | "
        f"TO {res['turnover_long']:.2f}/{res['turnover_short']:.2f} | "
        f"D10-EW {res['d10_minus_ew_mean_monthly']:+.4%}/mo (t {res['d10_minus_ew_tstat_nw']:+.2f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phase 2 cross-sectional study.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--refresh", action="store_true", help="wipe price cache and redownload")
    parser.add_argument("--models", default="lin,xgb,rf")
    parser.add_argument("--costs", default="0,5,10,20", help="per-side cost levels in bps")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    cost_bps = [float(c) for c in args.costs.split(",")]
    t0 = time.time()

    # Universe
    universe = load_universe()
    tickers = [row["yf_ticker"] for row in universe]
    print(f"UNIVERSE: {len(tickers)} current S&P 500 members "
          f"(fetched {universe[0]['fetched_on']}, survivorship-biased by construction)")

    # Data
    adj, close, volume, meta = get_prices(tickers, start=args.start, refresh=args.refresh)
    print(f"DATA: {adj.shape[1]} tickers x {adj.shape[0]} days, "
          f"{meta['start']} to {meta['end']}, downloaded {meta['download_date']}, "
          f"{len(meta['failed'])} failed")
    for t, reason in sorted(meta["failed"].items()):
        print(f"  dropped {t}: {reason}")

    # Panel
    panel = build_panel(adj, close, volume)
    months = panel.index.get_level_values("date").unique().sort_values()
    names_per_month = panel.groupby(level="date").size()
    print(f"PANEL: {len(months)} months, {months[0].date()} to {months[-1].date()}, "
          f"{len(panel)} rows, names/month mean {names_per_month.mean():.0f} "
          f"min {names_per_month.min()} max {names_per_month.max()}")

    # Common evaluation window: month BURN_IN_MONTHS+1 of the panel through the
    # last month with realizable forward returns. Identical for every method
    # and the benchmark.
    realizable = panel["fwd_ret"].notna().groupby(level="date").any()
    eval_months = pd.DatetimeIndex([m for m in months[BURN_IN_MONTHS:] if realizable[m]])
    print(f"EVALUATION WINDOW: {eval_months[0].date()} to {eval_months[-1].date()} "
          f"({len(eval_months)} months, common to all methods)")

    # Baselines first
    print("BASELINES (no fitting):")
    results: dict[str, dict] = {}
    scores_by_method = dict(baseline_scores(panel))
    for name in ["b1_mom", "b2_rev", "b3_combo"]:
        results[name] = evaluate_scores(scores_by_method[name], panel, eval_months, cost_bps)
        _print_method(name, results[name])

    # Models
    print("MODELS (walk-forward, expanding window, embargoed):")
    for name in model_names:
        fit_t0 = time.time()
        scores, _ = walk_forward_scores(panel, name)
        scores_by_method[name] = scores
        results[name] = evaluate_scores(scores, panel, eval_months, cost_bps)
        _print_method(name, results[name])
        print(f"    ({METHOD_LABELS[name]} walk-forward took {time.time() - fit_t0:.1f}s)")

    # Cost table
    print("NET LONG-SHORT (per-side costs on traded notional):")
    header = "  method          " + "".join(f"{int(b):>4d} bps ann ret / Sharpe   " for b in cost_bps)
    print(header)
    for name, res in results.items():
        row = f"  {METHOD_LABELS[name]:<14}"
        for b in cost_bps:
            n = res["net"][str(int(b))]
            row += f"  {n['ann_return']:+8.2%} / {n['sharpe']:+5.2f}     "
        print(row)

    # Benchmark
    bench = equal_weight_benchmark(panel, eval_months)
    print(f"BENCHMARK equal-weight universe (gross, long-only): "
          f"ann {bench['ew_ann_return']:+.2%} vol {bench['ew_ann_vol']:.2%} "
          f"Sharpe {bench['ew_sharpe']:+.2f}")

    # Headline
    headline_method = "xgb" if "xgb" in results else model_names[-1]
    headline = {
        "method": headline_method,
        "metric": "ls_tstat_nw_gross",
        "value": results[headline_method]["ls_tstat_nw"],
        "nw_lag": results[headline_method]["nw_lag"],
    }
    print(f"HEADLINE (pre-registered): {headline_method} gross long-short NW t = "
          f"{headline['value']:+.2f} (lag {headline['nw_lag']})")

    # Dropped-at-evaluation counts are method-independent (same eval sets).
    dropped = results["b1_mom"]["dropped_fwd_ret"]

    out = {
        "run_date": date.today().isoformat(),
        "data": {
            "download_date": meta["download_date"],
            "start": meta["start"],
            "end": meta["end"],
            "n_tickers_requested": len(tickers),
            "n_tickers_used": int(adj.shape[1]),
            "failed_tickers": meta["failed"],
        },
        "panel": {
            "n_months": int(len(months)),
            "first_rebalance": str(months[0].date()),
            "last_rebalance": str(months[-1].date()),
            "n_rows": int(len(panel)),
            "mean_names_per_month": float(names_per_month.mean()),
            "min_names_per_month": int(names_per_month.min()),
        },
        "evaluation_window": {
            "first_month": str(eval_months[0].date()),
            "last_month": str(eval_months[-1].date()),
            "n_months": int(len(eval_months)),
        },
        "dropped_fwd_ret": dropped,
        "headline": headline,
        "methods": {
            name: {k: v for k, v in res.items() if k != "dropped_fwd_ret"}
            for name, res in results.items()
        },
        "benchmark": bench,
    }
    RESULTS_JSON.parent.mkdir(exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {RESULTS_JSON}")

    chart_methods = [m for m in ["b1_mom", "b3_combo", "xgb"] if m in results]
    plot_decile_returns(
        {METHOD_LABELS[m]: results[m]["decile_mean_returns"] for m in chart_methods}
    )
    print("Wrote outputs/xsec_decile_returns.png")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
