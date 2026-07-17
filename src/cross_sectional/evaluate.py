"""
Evaluation: IC, Deciles, Long-Short, Turnover, Costs

Every method (baselines and models) goes through this identical code, on the
identical evaluation window. All t-stats are Newey-West.

Accounting conventions (also stated in the README):
- Monthly arithmetic returns. Annualized return is x12 on the mean; annualized
  vol is x sqrt(12); Sharpe uses rf = 0 (phase 1 convention).
- The cumulative curve for max drawdown is the ARITHMETIC cumulative sum of
  monthly long-short returns, matching the x12 annualization (compounding a
  dollar-neutral series is a different convention and is not used). Max
  drawdown is the largest peak-to-trough decline of that cumsum, in return
  units, reported as a negative number.
- Newey-West lag, for every NW t-stat in the study:
  L = floor(4 * (T / 100) ** (2 / 9)) where T is the number of evaluated
  months (Newey-West 1994 rule of thumb), computed from the actual series
  length at run time, never hard-coded.
- Turnover is weight-based and captures both name replacement and the
  within-name trading needed to restore equal weights. Per leg at month t:
  target weights w_t(i) = 1/N_t for members else 0; drifted previous weights
  wd_{t-1}(i) = w_{t-1}(i) * (1 + r_i) / sum_j w_{t-1}(j) * (1 + r_j) with
  r_i the name's realized return over the month; traded notional
  TN_t = sum_i |w_t(i) - wd_{t-1}(i)| over the union of names; reported
  one-way turnover TO_t = 0.5 * TN_t. First evaluation month: the previous
  portfolio is empty, so TN_1 = 1 per leg (pure entry buys) and the entry
  cost IS charged.
- Costs: per-side cost c (in bps) multiplies every unit of notional bought or
  sold: net_ls_t = gross_ls_t - c * (TN_long_t + TN_short_t).
- Names whose forward return is missing are dropped HERE, not at formation,
  and the per-month dropped count is logged.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

N_DECILES = 10


def newey_west_lag(n_months: int) -> int:
    """Newey-West (1994) rule-of-thumb lag: floor(4 * (T / 100) ** (2 / 9))."""
    return int(np.floor(4.0 * (n_months / 100.0) ** (2.0 / 9.0)))


def nw_tstat(series: np.ndarray | pd.Series) -> tuple[float, float, int]:
    """
    Mean and Newey-West t-stat of a monthly series.

    OLS of the series on a constant with HAC covariance, maxlags from
    newey_west_lag on the actual series length.

    Returns:
        Tuple of (mean, t-stat, lag L used).
    """
    x = np.asarray(series, dtype=float)
    lag = newey_west_lag(len(x))
    res = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return float(res.params[0]), float(res.tvalues[0]), lag


def assign_deciles(scores: pd.Series) -> pd.Series:
    """
    Decile buckets 0..9 for one month's scores.

    Sorts by ticker (deterministic tie-break), ranks with method="first", and
    qcuts THAT rank into 10 equal-count buckets. Tree predictions tie heavily
    and qcut on raw scores can silently collapse bins, mislabeling "D10 minus
    D1"; ranking first guarantees 10 equal bins. Asserts exactly 10 buckets
    and max minus min bucket size <= 1.
    """
    s = scores.sort_index()
    ranks = s.rank(method="first")
    buckets = pd.qcut(ranks, N_DECILES, labels=False)
    counts = buckets.value_counts()
    if len(counts) != N_DECILES or counts.max() - counts.min() > 1:
        raise AssertionError(
            f"Decile assignment produced {len(counts)} buckets with sizes "
            f"{sorted(counts)}; expected {N_DECILES} buckets, sizes within 1."
        )
    return buckets


def evaluate_month(scores_t: pd.Series, fwd_t: pd.Series) -> dict:
    """
    One month's IC, decile mean returns, and portfolio memberships.

    Both inputs must be NaN-free: NaN forward returns are dropped by the
    caller, and a NaN score is a pipeline bug that must fail loudly here, not
    quietly deflate the mean IC.
    """
    if scores_t.isna().any():
        raise ValueError("NaN score reached the IC step; pipeline bug.")
    if fwd_t.isna().any():
        raise ValueError("NaN forward return reached the IC step; pipeline bug.")

    ic = float(spearmanr(scores_t.to_numpy(), fwd_t.to_numpy(), nan_policy="omit").statistic)
    buckets = assign_deciles(scores_t)
    fwd = fwd_t.reindex(buckets.index)
    decile_means = fwd.groupby(buckets).mean().to_numpy()
    long_names = buckets.index[buckets == N_DECILES - 1]
    short_names = buckets.index[buckets == 0]
    return {
        "ic": ic,
        "decile_means": decile_means,
        "d10_ret": float(fwd.loc[long_names].mean()),
        "d1_ret": float(fwd.loc[short_names].mean()),
        "ew_ret": float(fwd.mean()),
        "long_names": list(long_names),
        "short_names": list(short_names),
    }


def leg_turnover(
    prev_weights: pd.Series | None,
    prev_returns: pd.Series | None,
    new_names: list,
) -> tuple[float, pd.Series]:
    """
    Traded notional for one equal-weight leg at one rebalance.

    Args:
        prev_weights: previous target weights (ticker -> weight), or None on
            the entry month.
        prev_returns: realized returns of the previous members over the month
            just ended (their fwd_ret from the previous rebalance).
        new_names: this month's members (equal weight 1/N each).

    Returns:
        Tuple of (TN_t, new weights). Entry month: TN = 1 (pure buys).
    """
    w_new = pd.Series(1.0 / len(new_names), index=pd.Index(new_names))
    if prev_weights is None:
        return 1.0, w_new
    grown = prev_weights * (1.0 + prev_returns.reindex(prev_weights.index))
    drifted = grown / grown.sum()
    union = drifted.index.union(w_new.index)
    tn = float(
        (w_new.reindex(union, fill_value=0.0) - drifted.reindex(union, fill_value=0.0))
        .abs()
        .sum()
    )
    return tn, w_new


def net_series(
    gross_ls: np.ndarray, tn_long: np.ndarray, tn_short: np.ndarray, cost_bps: float
) -> np.ndarray:
    """Net long-short returns: gross_t - c * (TN_long_t + TN_short_t), c per side."""
    c = cost_bps / 1e4
    return np.asarray(gross_ls) - c * (np.asarray(tn_long) + np.asarray(tn_short))


def max_drawdown(returns: np.ndarray) -> float:
    """
    Largest peak-to-trough decline of the arithmetic cumsum of monthly
    returns, in return units, as a negative number.
    """
    curve = np.concatenate([[0.0], np.cumsum(np.asarray(returns, dtype=float))])
    peak = np.maximum.accumulate(curve)
    return float(-(peak - curve).max())


def _annualize(monthly: np.ndarray) -> tuple[float, float, float]:
    """(annualized return, annualized vol, Sharpe) from monthly arithmetic returns."""
    m = np.asarray(monthly, dtype=float)
    ann_ret = float(m.mean() * 12.0)
    ann_vol = float(m.std(ddof=1) * np.sqrt(12.0))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    return ann_ret, ann_vol, sharpe


def evaluate_scores(
    scores: pd.Series,
    panel: pd.DataFrame,
    eval_months: pd.DatetimeIndex,
    cost_bps: list[float],
) -> dict:
    """
    Full evaluation of one score series over the common window.

    Args:
        scores: (date, ticker) score Series covering every eval month.
        panel: the panel (for fwd_ret).
        eval_months: the common evaluation months (identical for every
            method; stated once in the results JSON).
        cost_bps: per-side cost levels in bps.

    Returns:
        Dict matching the per-method schema of the results JSON, plus
        'decile_mean_returns' and 'dropped_fwd_ret' {month: count}.
    """
    ics, decile_rows, ls_rows, d10_minus_ew_rows = [], [], [], []
    tn_long_rows, tn_short_rows = [], []
    memb_long_rows, memb_short_rows = [], []
    dropped: dict[str, int] = {}

    prev_w = {"long": None, "short": None}
    prev_names = {"long": None, "short": None}
    prev_fwd: pd.Series | None = None

    for t in eval_months:
        scores_t = scores.xs(t, level="date")
        fwd_t = panel["fwd_ret"].xs(t, level="date").reindex(scores_t.index)
        keep = fwd_t.notna()
        n_dropped = int((~keep).sum())
        if n_dropped:
            dropped[t.strftime("%Y-%m-%d")] = n_dropped
        scores_t, fwd_t = scores_t[keep], fwd_t[keep]

        month = evaluate_month(scores_t, fwd_t)
        ics.append(month["ic"])
        decile_rows.append(month["decile_means"])
        ls_rows.append(month["d10_ret"] - month["d1_ret"])
        d10_minus_ew_rows.append(month["d10_ret"] - month["ew_ret"])

        for leg, names in (("long", month["long_names"]), ("short", month["short_names"])):
            tn, w_new = leg_turnover(prev_w[leg], prev_fwd, names)
            (tn_long_rows if leg == "long" else tn_short_rows).append(tn)
            if prev_names[leg] is not None:
                new = len(set(names) - set(prev_names[leg]))
                frac = new / len(names)
                (memb_long_rows if leg == "long" else memb_short_rows).append(frac)
            prev_w[leg] = w_new
            prev_names[leg] = names
        prev_fwd = fwd_t

    ics = np.asarray(ics)
    ls = np.asarray(ls_rows)
    tn_long = np.asarray(tn_long_rows)
    tn_short = np.asarray(tn_short_rows)

    _, ic_t, _ = nw_tstat(ics)
    ls_mean, ls_t, nw_lag = nw_tstat(ls)
    d10ew_mean, d10ew_t, _ = nw_tstat(np.asarray(d10_minus_ew_rows))
    ann_ret, ann_vol, sharpe = _annualize(ls)

    net = {}
    for bps in cost_bps:
        n = net_series(ls, tn_long, tn_short, bps)
        n_ret, _, n_sharpe = _annualize(n)
        net[str(int(bps))] = {"ann_return": n_ret, "sharpe": n_sharpe}

    return {
        "mean_ic": float(ics.mean()),
        "ic_tstat_nw": ic_t,
        "pct_positive_ic": float((ics > 0).mean()),
        "decile_mean_returns": [float(x) for x in np.vstack(decile_rows).mean(axis=0)],
        "ls_ann_return": ann_ret,
        "ls_ann_vol": ann_vol,
        "ls_sharpe": sharpe,
        "ls_max_dd": max_drawdown(ls),
        "ls_tstat_nw": ls_t,
        "nw_lag": nw_lag,
        "turnover_long": float(0.5 * tn_long.mean()),
        "turnover_short": float(0.5 * tn_short.mean()),
        "membership_turnover_long": float(np.mean(memb_long_rows)),
        "membership_turnover_short": float(np.mean(memb_short_rows)),
        "d10_minus_ew_mean_monthly": d10ew_mean,
        "d10_minus_ew_tstat_nw": d10ew_t,
        "net": net,
        "dropped_fwd_ret": dropped,
    }


def equal_weight_benchmark(panel: pd.DataFrame, eval_months: pd.DatetimeIndex) -> dict:
    """
    Equal-weight average forward return of ALL eligible names each month over
    the common evaluation window. This is the long-only "own the whole
    universe" alternative; it is GROSS (no turnover charged) and is the
    comparator for the long leg, not for the roughly beta-neutral long-short.
    """
    ew = []
    for t in eval_months:
        fwd_t = panel["fwd_ret"].xs(t, level="date").dropna()
        ew.append(float(fwd_t.mean()))
    ann_ret, ann_vol, sharpe = _annualize(np.asarray(ew))
    return {
        "ew_ann_return": ann_ret,
        "ew_ann_vol": ann_vol,
        "ew_sharpe": sharpe,
        "gross_only": True,
    }
