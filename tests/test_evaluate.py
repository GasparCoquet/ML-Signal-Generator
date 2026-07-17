"""
IC, decile, turnover, and cost math on toy inputs. No network.
"""

import numpy as np
import pandas as pd

from src.cross_sectional.evaluate import (
    assign_deciles,
    evaluate_month,
    evaluate_scores,
    leg_turnover,
    max_drawdown,
    net_series,
    newey_west_lag,
    nw_tstat,
)


def test_newey_west_lag_rule_of_thumb():
    # floor(4 * (T/100)^(2/9))
    assert newey_west_lag(100) == 4
    assert newey_west_lag(150) == 4
    assert newey_west_lag(50) == 3
    assert newey_west_lag(400) == 5


def test_deciles_survive_heavy_ties():
    # 103 names, scores almost all tied: still exactly 10 buckets, sizes
    # within 1 of each other.
    names = [f"T{j:03d}" for j in range(103)]
    scores = pd.Series(1.0, index=pd.Index(names))
    scores.iloc[:3] = [0.5, 2.0, 1.5]
    buckets = assign_deciles(scores)
    counts = buckets.value_counts()
    assert len(counts) == 10
    assert counts.max() - counts.min() <= 1


def test_decile_means_and_ic_hand_computed():
    names = [f"T{j:02d}" for j in range(20)]
    scores = pd.Series(np.arange(20, dtype=float), index=pd.Index(names))
    fwd = scores / 100.0  # perfectly rank-correlated
    month = evaluate_month(scores, fwd)
    assert abs(month["ic"] - 1.0) < 1e-12
    # 20 names -> deciles of 2; D1 = {0, 0.01}, D10 = {0.18, 0.19}.
    assert abs(month["decile_means"][0] - 0.005) < 1e-12
    assert abs(month["decile_means"][9] - 0.185) < 1e-12
    assert abs(month["d10_ret"] - month["d1_ret"] - 0.18) < 1e-12
    assert abs(month["ew_ret"] - fwd.mean()) < 1e-12


def test_nan_score_fails_loudly():
    names = ["A", "B", "C"]
    scores = pd.Series([1.0, np.nan, 3.0], index=pd.Index(names))
    fwd = pd.Series([0.01, 0.02, 0.03], index=pd.Index(names))
    raised = False
    try:
        evaluate_month(scores, fwd)
    except ValueError:
        raised = True
    assert raised


def test_leg_turnover_hand_computed():
    # Entry month: pure buys, TN = 1.
    tn1, w1 = leg_turnover(None, None, ["A", "B"])
    assert tn1 == 1.0
    assert abs(w1["A"] - 0.5) < 1e-12
    # Month 2: A +10%, B -10%; drifted A 0.55, B 0.45. New leg {B, C} at 0.5.
    # TN = |0 - 0.55| + |0.5 - 0.45| + |0.5 - 0| = 1.10.
    prev_ret = pd.Series({"A": 0.10, "B": -0.10})
    tn2, w2 = leg_turnover(w1, prev_ret, ["B", "C"])
    assert abs(tn2 - 1.10) < 1e-12
    assert abs(w2["C"] - 0.5) < 1e-12


def test_net_series_and_max_drawdown_hand_computed():
    gross = np.array([0.02, 0.01])
    tn_long = np.array([1.0, 0.2])
    tn_short = np.array([1.0, 0.3])
    net = net_series(gross, tn_long, tn_short, cost_bps=10.0)
    # c = 0.001 per side per unit of traded notional.
    assert abs(net[0] - (0.02 - 0.001 * 2.0)) < 1e-12
    assert abs(net[1] - (0.01 - 0.001 * 0.5)) < 1e-12

    # Cumsum path 0, .05, .03, 0, .04 -> largest decline 0.05 (as a negative).
    dd = max_drawdown(np.array([0.05, -0.02, -0.03, 0.04]))
    assert abs(dd - (-0.05)) < 1e-12


def test_nw_tstat_zero_mean_series():
    # Alternating ICs of +-0.5: mean 0, t approximately 0.
    x = np.tile([0.5, -0.5], 30)
    mean, tstat, lag = nw_tstat(x)
    assert abs(mean) < 1e-12
    assert abs(tstat) < 1e-8
    assert lag == newey_west_lag(60)


def _toy_panel_and_scores(n_months=6, n_names=20, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-31", periods=n_months, freq="ME")
    names = [f"T{j:02d}" for j in range(n_names)]
    idx = pd.MultiIndex.from_product([dates, names], names=["date", "ticker"])
    panel = pd.DataFrame(index=idx)
    panel["fwd_ret"] = rng.normal(0.01, 0.05, size=len(idx))
    scores = pd.Series(rng.uniform(size=len(idx)), index=idx)
    return panel, scores, dates


def test_evaluate_scores_end_to_end_on_toy_panel():
    panel, scores, dates = _toy_panel_and_scores()
    res = evaluate_scores(scores, panel, pd.DatetimeIndex(dates), cost_bps=[0.0, 10.0])
    assert len(res["decile_mean_returns"]) == 10
    assert -1.0 <= res["mean_ic"] <= 1.0
    # Entry month TN = 1 per leg is included in the mean one-way turnover.
    assert 0.0 < res["turnover_long"] <= 1.0
    # Zero cost leaves the gross annualized return unchanged.
    assert abs(res["net"]["0"]["ann_return"] - res["ls_ann_return"]) < 1e-12
    # Positive costs strictly reduce it (turnover is positive).
    assert res["net"]["10"]["ann_return"] < res["ls_ann_return"]


def test_evaluate_scores_counts_dropped_forward_returns():
    panel, scores, dates = _toy_panel_and_scores()
    victim = (dates[2], "T05")
    panel.loc[victim, "fwd_ret"] = np.nan
    res = evaluate_scores(scores, panel, pd.DatetimeIndex(dates), cost_bps=[0.0])
    key = dates[2].strftime("%Y-%m-%d")
    assert res["dropped_fwd_ret"] == {key: 1}
