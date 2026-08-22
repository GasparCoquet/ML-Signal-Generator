"""
Feature math and panel construction on tiny hand-built frames. No network.
"""

import numpy as np
import pandas as pd

from src.cross_sectional.features import (
    MIN_HISTORY_DAYS,
    build_panel,
    compute_features,
    rank_panel,
    rebalance_grid,
)


def _constant_growth_frames(n_days=600, daily=0.001, n_names=1, volume_const=1000.0):
    """Price frames where every name grows by `daily` per day, constant volume."""
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    prices = 100.0 * (1.0 + daily) ** np.arange(n_days)
    cols = {f"T{j}": prices for j in range(n_names)}
    adj = pd.DataFrame(cols, index=idx)
    close = adj.copy()
    volume = pd.DataFrame(volume_const, index=idx, columns=adj.columns)
    return adj, close, volume


def test_rebalance_grid_complete_months_only():
    # Index ends mid-month: 2021-07 must never appear in the grid.
    idx = pd.bdate_range("2020-01-02", "2021-07-20")
    grid = rebalance_grid(idx)
    assert grid[-1].month == 6 and grid[-1].year == 2021
    last_month = idx[-1].to_period("M")
    assert last_month not in grid.to_period("M")
    # Every consecutive pair exactly one calendar month apart.
    months = grid.year * 12 + grid.month
    assert (np.diff(months) == 1).all()
    # Each grid date is the last trading day of its month within the index.
    for t in grid:
        same_month = idx[(idx.year == t.year) & (idx.month == t.month)]
        assert t == same_month.max()


def test_mom_and_rev_on_constant_growth():
    adj, close, volume = _constant_growth_frames()
    feats = compute_features(adj, close, volume)
    t = adj.index[300]
    # mom_12_1 = p(t-21) / p(t-252) - 1 = (1.001)^231 - 1
    expected_mom = 1.001**231 - 1
    assert abs(feats["mom_12_1"].loc[t, "T0"] - expected_mom) < 1e-12
    # rev_1m = p(t) / p(t-21) - 1 = (1.001)^21 - 1
    expected_rev = 1.001**21 - 1
    assert abs(feats["rev_1m"].loc[t, "T0"] - expected_rev) < 1e-12
    # Constant daily return: vol is zero.
    assert abs(feats["vol_63d"].loc[t, "T0"]) < 1e-12
    # Rising price: at its 252-day high, so dist_52w_high = 0.
    assert abs(feats["dist_52w_high"].loc[t, "T0"]) < 1e-12


def test_amihud_and_dollar_volume_hand_computed():
    adj, close, volume = _constant_growth_frames(volume_const=1000.0)
    feats = compute_features(adj, close, volume)
    i = 300
    t = adj.index[i]
    p = adj["T0"].to_numpy()
    dv = close["T0"].to_numpy() * 1000.0
    daily_ret = p[1:] / p[:-1] - 1
    # Windows ending at t (inclusive), 63 observations.
    illiq = np.abs(daily_ret[i - 63 : i]) / dv[i - 62 : i + 1]
    expected_amihud = np.log(illiq.mean())
    expected_ldv = np.log(dv[i - 62 : i + 1].mean())
    assert abs(feats["amihud_63d"].loc[t, "T0"] - expected_amihud) < 1e-10
    assert abs(feats["log_dollar_vol_63d"].loc[t, "T0"] - expected_ldv) < 1e-10


def test_zero_volume_is_nan_not_clipped():
    adj, close, volume = _constant_growth_frames()
    volume.iloc[280, 0] = 0.0
    feats = compute_features(adj, close, volume)
    # Any 63-day window containing the zero-volume day is NaN.
    t = adj.index[300]
    assert np.isnan(feats["amihud_63d"].loc[t, "T0"])
    assert np.isnan(feats["log_dollar_vol_63d"].loc[t, "T0"])
    # Windows entirely past the zero-volume day recover.
    t2 = adj.index[280 + 63]
    assert np.isfinite(feats["amihud_63d"].loc[t2, "T0"])


def test_forward_return_hand_computed_and_last_month_nan():
    adj, close, volume = _constant_growth_frames(n_days=700, n_names=3)
    panel = build_panel(adj, close, volume, min_names=1)
    dates = panel.index.get_level_values("date").unique().sort_values()
    grid = rebalance_grid(adj.index)
    t, t_next = dates[0], dates[1]
    expected = adj.loc[t_next, "T0"] / adj.loc[t, "T0"] - 1
    assert abs(panel.loc[(t, "T0"), "fwd_ret"] - expected) < 1e-12
    # Last grid date has no next grid date: fwd_ret NaN, row still present.
    assert dates[-1] == grid[-1]
    assert panel.loc[dates[-1]]["fwd_ret"].isna().all()
    assert panel.loc[dates[-1]]["fwd_ret_rank"].isna().all()


def test_eligibility_needs_min_history():
    adj, close, volume = _constant_growth_frames(n_days=800, n_names=3)
    # T2 starts trading late: NaN for the first 400 days.
    adj.iloc[:400, 2] = np.nan
    close.iloc[:400, 2] = np.nan
    volume.iloc[:400, 2] = np.nan
    panel = build_panel(adj, close, volume, min_names=1)
    sufficiency_checked = 0
    for t in panel.index.get_level_values("date").unique():
        names = panel.loc[t].index
        n_hist = adj.loc[:t, "T2"].notna().sum()
        if "T2" in names:
            assert n_hist >= MIN_HISTORY_DAYS
        elif n_hist >= MIN_HISTORY_DAYS + 63:
            # Well past warm-up for every rolling window: must be present.
            assert "T2" in names
        if n_hist >= MIN_HISTORY_DAYS + 63:
            sufficiency_checked += 1
    # Guard against this branch going dead: the sufficiency case must occur.
    assert sufficiency_checked > 0


def test_rank_is_per_date_no_pooling():
    # Ranking at date t must be invariant to deleting all other dates.
    rng = np.random.default_rng(7)
    dates = pd.DatetimeIndex(["2020-01-31", "2020-02-28", "2020-03-31"])
    names = [f"T{j}" for j in range(8)]
    idx = pd.MultiIndex.from_product([dates, names], names=["date", "ticker"])
    raw = pd.DataFrame(
        {col: rng.normal(size=len(idx)) for col in
         ["mom_12_1", "rev_1m", "vol_63d", "dist_52w_high", "amihud_63d", "log_dollar_vol_63d"]},
        index=idx,
    )
    raw["fwd_ret"] = rng.normal(size=len(idx))
    full = rank_panel(raw)
    t = dates[1]
    single = rank_panel(raw.loc[[t]])
    pd.testing.assert_frame_equal(full.loc[[t]], single)


def test_ranks_are_in_unit_interval():
    adj, close, volume = _constant_growth_frames(n_days=700, n_names=5)
    # Perturb prices so ranks are not degenerate.
    rng = np.random.default_rng(0)
    noise = rng.normal(1.0, 0.01, size=adj.shape)
    adj = adj * noise
    close = close * noise
    panel = build_panel(adj, close, volume, min_names=1)
    for col in ["mom_12_1", "rev_1m", "vol_63d", "dist_52w_high", "amihud_63d", "log_dollar_vol_63d"]:
        vals = panel[col]
        assert vals.notna().all()
        assert (vals >= 0).all() and (vals <= 1).all()
