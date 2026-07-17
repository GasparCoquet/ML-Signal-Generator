"""
Walk-Forward ML Models

Linear regression, XGBoost, and Random Forest trained walk-forward on the six
ranked features to predict the within-month percentile of the forward return
(fwd_ret_rank), so the models learn relative ordering, not market direction.

Protocol (identical for all models):
- Burn-in: the first BURN_IN_MONTHS panel months are training-only; the first
  prediction month is month BURN_IN_MONTHS + 1 of the panel.
- Embargo: when predicting at month t, the training set is all samples whose
  forward-return window has fully closed by t. Sample at month m has its
  target realized at m + 1, so training at prediction month t uses samples
  with m + 1 <= t, i.e. m <= t - 1. Rows with NaN fwd_ret_rank are excluded
  from training.
- Refit cadence: refit every REFIT_EVERY prediction months (expanding
  window); months in between are scored with the most recent fit.

The only fitted objects are the models themselves, refit inside the loop; no
scaler or other fitted state crosses the split (features are per-date rank
transforms, which involve no fitting at all).
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

from src.cross_sectional.features import FEATURE_COLS

BURN_IN_MONTHS = 36
REFIT_EVERY = 12  # refit cadence in prediction months

MODEL_NAMES = ["lin", "xgb", "rf"]


def make_model(name: str) -> Any:
    """Instantiate a fresh model by short name (lin, xgb, rf)."""
    if name == "lin":
        return LinearRegression()
    if name == "xgb":
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}.")


def walk_forward_scores(
    panel: pd.DataFrame, model_name: str
) -> tuple[pd.Series, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Run the expanding-window walk-forward loop for one model.

    Args:
        panel: (date, ticker) panel with ranked features, fwd_ret_rank.
        model_name: one of lin, xgb, rf.

    Returns:
        Tuple of (scores, fit_log). scores is a Series indexed by
        (date, ticker) covering every panel month after burn-in. fit_log has
        one (train_end_month, prediction_month) pair per prediction month,
        recording the training cutoff of the fit actually used; the leakage
        test asserts train_end <= prediction month - 1 for every entry.
    """
    months = panel.index.get_level_values("date").unique().sort_values()
    if len(months) <= BURN_IN_MONTHS:
        raise ValueError(
            f"Panel has {len(months)} months; need more than {BURN_IN_MONTHS} for burn-in."
        )

    dates = panel.index.get_level_values("date")
    model = None
    train_end: pd.Timestamp | None = None
    scores: list[pd.Series] = []
    fit_log: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for i in range(BURN_IN_MONTHS, len(months)):
        t = months[i]
        if (i - BURN_IN_MONTHS) % REFIT_EVERY == 0:
            # Embargo: samples at month m train only if m <= t - 1 month,
            # i.e. their forward-return window closed by t.
            train_end = months[i - 1]
            train = panel[(dates <= train_end) & panel["fwd_ret_rank"].notna()]
            model = make_model(model_name)
            model.fit(train[FEATURE_COLS].to_numpy(), train["fwd_ret_rank"].to_numpy())
        rows = panel.xs(t, level="date", drop_level=False)
        preds = model.predict(rows[FEATURE_COLS].to_numpy())
        scores.append(pd.Series(np.asarray(preds, dtype=float), index=rows.index))
        fit_log.append((train_end, t))

    return pd.concat(scores).rename(model_name), fit_log
