# ML Signal Generator

Do daily OHLC technical features predict next-day SPY direction?

**No — and this repo is built to show that honestly.**

Out-of-sample AUC is indistinguishable from a coin flip, and a long-only signal that is
long ~57% of days during a bull market **loses to simply buying and holding SPY**. That is
the expected result for this feature set on a liquid index, and the harness is constructed
so that it could not have concluded otherwise by accident.

All numbers below are the console output of `python main.py` on the default configuration
(SPY, 2020-01-01 → 2024-01-01, threshold 0.55, 2 bps cost). Reproduce them with one command.

---

## The headline result

**Test window: 2023-05-31 → 2023-12-29 (148 rows, 147 backtest days).**

| | Total Return | Sharpe | Max DD | Days Long |
|---|---|---|---|---|
| Random Forest (selected) | 7.32% | 1.56 | -4.97% | 57.1% |
| XGBoost | 3.04% | 0.62 | -8.47% | 62.6% |
| **Buy & Hold SPY** | **13.96%** | **2.20** | -9.97% | 100.0% |

**Excess return vs buy & hold: Random Forest −6.64pp, XGBoost −10.91pp.**

The strategy's Sharpe of 1.56 looks respectable in isolation. It is not. Over the same
window, doing nothing but holding SPY returned 13.96% at Sharpe 2.20. The strategy is long
most days of a melt-up, so it captures a fraction of the market's beta and reports it as
performance. **A long-only signal in a rising market earns beta, not alpha.** Without the
benchmark column, that distinction is invisible — which is exactly why the benchmark is
computed with the same metric code and plotted on the same equity curve.

## The model has no measurable edge

```
Bootstrap 95% CI on test AUC (2000 resamples):
  Random Forest   AUC 0.5255   95% CI [0.4269, 0.6235]   contains 0.5 - cannot reject coin flip
  XGBoost         AUC 0.5462   95% CI [0.4516, 0.6451]   contains 0.5 - cannot reject coin flip
```

Point AUCs of 0.5255 and 0.5462 sit above 0.5, and it would be easy to stop reading there
and call it edge. On 148 daily observations the error bar is roughly ±0.10 — **both
intervals comfortably contain 0.5**, so a coin flip cannot be ruled out. Reporting the point
estimate without the interval would be the single easiest way to fool yourself here.

The walk-forward estimate, which averages 5 expanding-window folds over 838 days and is far
more robust than one 148-day window, agrees:

```
WALK-FORWARD VALIDATION (expanding window, 5 folds)
  Random Forest  - mean AUC: 0.4997
  XGBoost        - mean AUC: 0.4903
```

Both are at or below 0.5. **The honest read is: no edge detected.**

## Overfitting: caught, not hidden

```
OVERFITTING DIAGNOSTIC (single 70/15 train/val cut)
  Random Forest  - train acc: 0.9855   val acc: 0.4730   val AUC: 0.4382
  XGBoost        - train acc: 0.9913   val acc: 0.4797   val AUC: 0.4637
```

The models classify the training set almost perfectly (98.6% / 99.1%) and then perform at
or below chance out of sample (47.3% / 48.0%). Tree ensembles will memorise noise in
financial time series, and this is what it looks like. The point of printing both columns is
that the walk-forward split *catches* this rather than letting it leak into a headline
number. A single in-sample fit here would have produced a spectacular and completely fake
result.

## Non-stationary features (fixed)

The model originally received `ma_5d` and `ma_20d` as **absolute dollar prices**. This is a
subtle and serious bug for tree models:

- Trees cannot extrapolate beyond the range of values seen in training.
- Training range was `ma_20d ∈ [229.52, 442.07]`; **11 of 148 test rows (7.4%)** fell
  outside it. Every one of those lands in the same terminal leaf regardless of market state.
- `ma_20d` was XGBoost's single most important feature (importance 0.159854) — so the
  top-ranked signal was effectively a **proxy for calendar time**, not for market state.

Fixed by feeding the same information in stationary form, `price / ma − 1` (percent distance
from the moving average). Effect on AUC, reported in both directions:

| | Walk-forward AUC (5 folds, 838 days) | Test AUC (148 days) |
|---|---|---|
| RF — raw price MAs | 0.5083 | 0.4945 |
| RF — stationary MAs | **0.4997** | **0.5255** |
| XGB — raw price MAs | 0.5027 | 0.4676 |
| XGB — stationary MAs | **0.4903** | **0.5462** |

The fix moved the single-window test AUC **up** and the robust walk-forward AUC **down**.
Neither move is evidence of anything: an AUC that swings by 0.08 on a change of feature
*representation alone* is telling you the metric is dominated by noise at this sample size.
The correct conclusion is not "the fix helped" — it is that **there is no stable signal here
to help**. The features are now stationary because that is correct practice, not because it
bought performance. It did not.

## Why you can trust the negative result

A negative result is only worth anything if the harness could have found a positive one. This
one could have:

- **No lookahead.** The target is `Close.shift(-1) / Close - 1`; features at time *t* use
  only data up to *t*. Signals are applied to the *next* day's return.
- **No leakage across the split.** Test data is the last 15%, held out entirely. Model
  selection happens via walk-forward on the first 85% and never touches the test set.
- **Costs charged on turnover, not on days held.** `Net_t = Gross_t − cost × |Signal_t − Signal_{t−1}|`.
  Charging per *day held* would have quietly penalised the strategy and made the benchmark
  comparison flattering by accident.
- **Benchmarked.** Buy & hold is computed on the identical return series with the identical
  Sharpe/annualisation code, so the columns are comparable.
- **Error bars on the headline metric.** Bootstrap CI on test AUC.

## Reproduce

```bash
pip install -r requirements.txt
python main.py                          # SPY, 2020-2024, defaults — produces every number above
python main.py --ticker AAPL --start 2021-01-01 --end 2024-01-01
python main.py --threshold 0.50 --transaction-cost 0.0005
```

Data is cached to `data/` on first download, so reruns are deterministic and offline.
Outputs (`outputs/`): `equity_curve.png` (strategy vs buy & hold), `roc_curve.png`,
`feature_importance_{rf,xgb}.png`.

## What the pipeline does

- **Data**: daily OHLC via yfinance (Alpha Vantage supported as a fallback; set `API_SOURCE`
  in `.env`). Cached locally after first fetch.
- **Features** (all stationary): `return_1d`, `return_5d`, `volatility_20d`,
  `price_to_ma_5d`, `price_to_ma_20d`, `ma_gap`, `z_score_20`.
- **Target**: `y = 1` if next-day return > 0, else `0`. (986 rows after NaN drop; 529 up days
  / 457 down days.)
- **Split**: last 15% held out as test. Walk-forward (expanding window, `TimeSeriesSplit`,
  5 folds) over the first 85% selects between Random Forest and XGBoost by mean fold AUC.
- **Signals**: long if `P(up) > threshold` (default 0.55), else flat. Long-only, so the
  strategy can never be short — a structural limitation, and the reason the buy & hold
  comparison matters so much.
- **Backtest**: total/annualised return, volatility, Sharpe (rf = 0), max drawdown, win rate,
  turnover-based transaction costs, and the buy & hold benchmark.

## Honest next steps

The result above says the feature set is exhausted, not that the pipeline is. What would
actually move the needle:

1. **Allow short positions.** Long-only cannot express a bearish view, and structurally
   guarantees beta contamination in any bull sample.
2. **Test on a bear or sideways regime** (e.g. 2022). A single 147-day bull window is not
   enough to conclude anything about the strategy; it is only enough to conclude the strategy
   did not beat holding.
3. **Features with a plausible economic reason to predict** — order-flow imbalance, options
   positioning, realised-vs-implied vol spread. Lagged returns and moving averages on a
   liquid index are the most heavily arbitraged signals in existence; finding no edge in them
   is the *prior*, not a surprise.
4. **Longer horizons.** Next-day direction on SPY is close to the hardest possible target.

Anything that reports a positive result here should be treated as a bug until proven otherwise.

## Project structure

```
ml-signal-generator/
├── main.py              # CLI entry point — runs the full pipeline
├── data/                # Cached OHLC downloads
├── notebooks/
│   └── 01_training.ipynb
├── src/
│   ├── features.py      # Feature engineering (stationary)
│   ├── model.py         # Training, walk-forward validation, bootstrap AUC CI
│   └── backtest.py      # Signals, backtest, buy & hold benchmark
├── outputs/             # Generated charts
└── requirements.txt
```

## Disclaimer

**For educational purposes only, not financial advice.** These are not trading
recommendations. Past performance does not guarantee future results.

## License

See LICENSE file for details.

## Author

GasparCoquet
