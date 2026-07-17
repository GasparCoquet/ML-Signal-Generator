"""
Monthly Panel and Cross-Sectional Features

Builds the (rebalance date, ticker) panel: six standard cross-sectional
features, the forward 1-month total return, and per-date rank transforms.

Rebalance grid: the last trading day of each calendar month in the price
index, then the final calendar month present is ALWAYS dropped (even if the
run date happens to be that month's last trading day). Run mid-month, the
naive grid would make today's partial month a rebalance date and label the
previous month-end's part-month forward return as monthly. Dropping the final
month guarantees every forward return spans exactly one complete calendar
month.

Leakage rules enforced here and tested in tests/test_leakage.py:
- Every feature at rebalance t uses only prices/volume with index <= t
  (rolling windows ending at t, sampled on t).
- The forward return is strictly from data after t:
  AdjClose(next grid date) / AdjClose(t) - 1.
- Eligibility at t uses t-information only (history length and feature
  availability at t). Membership does NOT condition on the forward return
  existing; requiring a valid t+1 price at formation time would be lookahead.
  Names whose forward return turns out missing are dropped at evaluation.
- Cross-sectional rank transforms use only that single date's cross-section,
  so no fitted scaler exists and nothing is fit on future data by
  construction.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
MOM_LONG = 252  # 12-1 momentum: total return from t-252 to t-21
MOM_SKIP = 21
REV_WINDOW = 21  # short-term reversal window
VOL_WINDOW = 63
HIGH_WINDOW = 252
LIQ_WINDOW = 63
MIN_HISTORY_DAYS = 253  # needed for mom_12_1 and dist_52w_high
MIN_NAMES_PER_MONTH = 100

FEATURE_COLS = [
    "mom_12_1",
    "rev_1m",
    "vol_63d",
    "dist_52w_high",
    "amihud_63d",
    "log_dollar_vol_63d",
]


def rebalance_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Last trading day of each calendar month in `index`, with the final
    calendar month present always dropped so every forward return spans one
    complete month.

    Raises:
        ValueError: if any two consecutive grid dates are not exactly one
            calendar month apart (a gap would silently mislabel forward
            returns).
    """
    s = index.to_series()
    grid = s.groupby(index.to_period("M")).max()
    grid = grid.iloc[:-1]  # drop the final calendar month present
    months = grid.index.year * 12 + grid.index.month
    gaps = np.diff(months)
    if len(gaps) and not (gaps == 1).all():
        raise ValueError("Rebalance grid has a calendar-month gap; refusing to continue.")
    return pd.DatetimeIndex(grid.values)


def compute_features(
    adj_close: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """
    Daily feature frames (date x ticker). All returns come from Adj Close
    ratios (total return); dollar volume and the 52-week high come from
    Close x Volume (split-adjusted only, no dividend contamination).

    Features:
    - mom_12_1: total return from t-252 to t-21 trading days, skipping the
      last month (Jegadeesh-Titman 12-1 momentum).
    - rev_1m: total return over the last 21 trading days (Jegadeesh 1990
      short-term reversal). Sign convention: raw return; the reversal signal
      is its negative.
    - vol_63d: std of daily returns over 63 days, annualized (low-volatility
      anomaly).
    - dist_52w_high: Close(t) / max Close over 252 days - 1 (George-Hwang
      52-week high). Uses the split-adjusted-only Close, NOT Adj Close:
      dividend adjustment would drag payers toward their high.
    - amihud_63d: log of mean(|daily return| / dollar volume) over 63 days
      (Amihud illiquidity). Zero-volume days are NaN, not clipped.
    - log_dollar_vol_63d: log of mean daily dollar volume over 63 days. Size
      and liquidity proxy; market cap is not available from the batch
      download without per-ticker calls, and this is the honest substitute.
    """
    ret = adj_close.pct_change(fill_method=None)
    dollar_vol = (close * volume).where(volume > 0)

    mom_12_1 = adj_close.shift(MOM_SKIP) / adj_close.shift(MOM_LONG) - 1
    rev_1m = adj_close / adj_close.shift(REV_WINDOW) - 1
    vol_63d = ret.rolling(VOL_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    dist_52w_high = close / close.rolling(HIGH_WINDOW).max() - 1

    with np.errstate(divide="ignore", invalid="ignore"):
        amihud_63d = np.log((ret.abs() / dollar_vol).rolling(LIQ_WINDOW).mean())
        log_dollar_vol_63d = np.log(dollar_vol.rolling(LIQ_WINDOW).mean())
    amihud_63d = amihud_63d.replace([np.inf, -np.inf], np.nan)
    log_dollar_vol_63d = log_dollar_vol_63d.replace([np.inf, -np.inf], np.nan)

    return {
        "mom_12_1": mom_12_1,
        "rev_1m": rev_1m,
        "vol_63d": vol_63d,
        "dist_52w_high": dist_52w_high,
        "amihud_63d": amihud_63d,
        "log_dollar_vol_63d": log_dollar_vol_63d,
    }


def _rank01(s: pd.Series) -> pd.Series:
    """Rank a cross-section to [0, 1]. NaN stays NaN."""
    n = s.notna().sum()
    if n <= 1:
        return s.where(s.isna(), 0.5)
    return (s.rank(method="average") - 1) / (n - 1)


def rank_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Per-date cross-sectional rank transforms.

    Each feature is ranked to [0, 1] WITHIN its rebalance date; fwd_ret_rank
    is the within-month percentile of the raw forward return (NaN where
    fwd_ret is NaN). The rank at date t uses only that date's cross-section,
    which tests/test_leakage.py verifies by deleting other dates.
    """
    out = panel.copy()
    grouped = out.groupby(level="date", group_keys=False)
    for col in FEATURE_COLS:
        out[col] = grouped[col].transform(_rank01)
    out["fwd_ret_rank"] = grouped["fwd_ret"].transform(_rank01)
    return out


def build_panel(
    adj_close: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    min_names: int = MIN_NAMES_PER_MONTH,
) -> pd.DataFrame:
    """
    Build the long-format monthly panel.

    A name enters the panel at rebalance t if it has >= MIN_HISTORY_DAYS
    non-NaN Adj Close observations up to t and all six features are non-NaN
    at t. A month enters the study only if it has >= min_names eligible
    names. Membership never looks at the forward return.

    Returns:
        DataFrame indexed by (date, ticker) with the six ranked features,
        fwd_ret (raw forward 1-month total return, NaN on the last grid date)
        and fwd_ret_rank (within-month percentile of fwd_ret, NaN where
        fwd_ret is NaN).
    """
    grid = rebalance_grid(adj_close.index)
    feats = compute_features(adj_close, close, volume)
    history_count = adj_close.notna().cumsum()

    frames = []
    for i, t in enumerate(grid):
        eligible = history_count.loc[t] >= MIN_HISTORY_DAYS
        for col in FEATURE_COLS:
            eligible &= feats[col].loc[t].notna()
        names = eligible.index[eligible]
        if len(names) < min_names:
            continue

        month = pd.DataFrame({col: feats[col].loc[t, names] for col in FEATURE_COLS})
        if i + 1 < len(grid):
            month["fwd_ret"] = adj_close.loc[grid[i + 1], names] / adj_close.loc[t, names] - 1
        else:
            month["fwd_ret"] = np.nan  # no next grid date; row stays, excluded at evaluation
        month.index = pd.MultiIndex.from_product([[t], month.index], names=["date", "ticker"])
        frames.append(month)

    if not frames:
        raise ValueError("No month has enough eligible names to build a panel.")
    panel = pd.concat(frames)
    # A mid-sample month dropped by the min_names filter would leave a gap:
    # fwd_ret would still span one grid step but the holding period between
    # consecutive panel months would be two, silently misstating turnover.
    kept = panel.index.get_level_values("date").unique().sort_values()
    positions = [grid.get_loc(t) for t in kept]
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise ValueError("Panel months are not contiguous on the rebalance grid; "
                         "a mid-sample month fell below min_names.")
    return rank_panel(panel)
