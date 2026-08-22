"""
Anti-lookahead guards: feature timing, formation membership, forward-return
direction, and the walk-forward embargo. Synthetic data only, no network.
"""

import numpy as np
import pandas as pd

from src.cross_sectional.features import FEATURE_COLS, build_panel
from src.cross_sectional.models import BURN_IN_MONTHS, walk_forward_scores


def _random_walk_frames(n_days=800, n_names=6, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    rets = rng.normal(0.0004, 0.015, size=(n_days, n_names))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"T{j}" for j in range(n_names)]
    adj = pd.DataFrame(prices, index=idx, columns=cols)
    close = adj * (1.0 + rng.normal(0, 0.001, size=prices.shape))
    volume = pd.DataFrame(
        rng.integers(1e5, 1e6, size=prices.shape).astype(float), index=idx, columns=cols
    )
    return adj, close, volume


def test_features_and_membership_ignore_the_future():
    """
    Corrupt every price/volume strictly after a rebalance date t: features and
    membership at dates <= t must be identical, while fwd_ret at t must
    change (the target really looks forward).
    """
    adj, close, volume = _random_walk_frames()
    panel = build_panel(adj, close, volume, min_names=1)
    dates = panel.index.get_level_values("date").unique().sort_values()
    t = dates[-4]

    after = adj.index > t
    adj_c, close_c, volume_c = adj.copy(), close.copy(), volume.copy()
    adj_c.loc[after] *= 10.0
    close_c.loc[after] *= 10.0
    volume_c.loc[after] *= 10.0
    panel_c = build_panel(adj_c, close_c, volume_c, min_names=1)

    upto = dates[dates <= t]
    for d in upto:
        # Membership at d is invariant to the corruption.
        assert list(panel.loc[d].index) == list(panel_c.loc[d].index)
    pd.testing.assert_frame_equal(
        panel.loc[upto, FEATURE_COLS], panel_c.loc[upto, FEATURE_COLS]
    )
    # The forward return at t IS changed: it is built from post-t prices.
    fwd = panel.loc[t]["fwd_ret"]
    fwd_c = panel_c.loc[t]["fwd_ret"]
    assert not np.allclose(fwd.to_numpy(), fwd_c.to_numpy())


def test_truncating_the_future_leaves_past_rows_identical():
    """
    Rebuild the panel from inputs truncated at t: rows at common dates must be
    identical to the full build (rolling-window sampling == truncation).
    """
    adj, close, volume = _random_walk_frames()
    panel = build_panel(adj, close, volume, min_names=1)
    dates = panel.index.get_level_values("date").unique().sort_values()
    t = dates[-4]

    panel_tr = build_panel(adj.loc[:t], close.loc[:t], volume.loc[:t], min_names=1)
    dates_tr = panel_tr.index.get_level_values("date").unique().sort_values()
    common = dates_tr[dates_tr <= t].intersection(dates)
    assert len(common) > 0
    for d in common:
        assert list(panel.loc[d].index) == list(panel_tr.loc[d].index)
    pd.testing.assert_frame_equal(
        panel.loc[common, FEATURE_COLS], panel_tr.loc[common, FEATURE_COLS]
    )


def _synthetic_panel(n_months=48, n_names=10, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    names = [f"T{j}" for j in range(n_names)]
    idx = pd.MultiIndex.from_product([dates, names], names=["date", "ticker"])
    panel = pd.DataFrame(
        {col: rng.uniform(size=len(idx)) for col in FEATURE_COLS}, index=idx
    )
    panel["fwd_ret"] = rng.normal(0.01, 0.05, size=len(idx))
    # Last month has no realizable forward return, like the real panel.
    panel.loc[dates[-1], "fwd_ret"] = np.nan
    panel["fwd_ret_rank"] = panel.groupby(level="date")["fwd_ret"].rank(pct=True)
    return panel, dates


def test_walk_forward_embargo_and_expanding_window():
    panel, dates = _synthetic_panel()
    scores, fit_log = walk_forward_scores(panel, "lin")

    # Predictions exist for exactly the months after burn-in.
    pred_months = scores.index.get_level_values("date").unique().sort_values()
    assert list(pred_months) == list(dates[BURN_IN_MONTHS:])

    def month_number(ts):
        return ts.year * 12 + ts.month

    prev_train_end = None
    for train_end, pred_month in fit_log:
        # Embargo: train_end <= prediction month - 1 calendar month.
        assert month_number(train_end) <= month_number(pred_month) - 1
        # Expanding window: training cutoffs never move backward.
        if prev_train_end is not None:
            assert train_end >= prev_train_end
        prev_train_end = train_end
    assert len(fit_log) == len(dates) - BURN_IN_MONTHS


def test_walk_forward_scores_are_finite_everywhere():
    panel, dates = _synthetic_panel()
    scores, _ = walk_forward_scores(panel, "lin")
    assert np.isfinite(scores.to_numpy()).all()
    # Scores exist even for the final month, whose fwd_ret is NaN: scoring
    # does not require the target, evaluation excludes it later.
    assert dates[-1] in scores.index.get_level_values("date")
