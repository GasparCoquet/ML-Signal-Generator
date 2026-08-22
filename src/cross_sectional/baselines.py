"""
No-Fit Baselines

Three baselines computed and evaluated BEFORE any model is trained. If the ML
models do not beat B3, the honest headline is "nonlinear ML adds nothing over
a naive combo", and the README must be able to say that sentence.

- B1 momentum: score = raw mom_12_1 rank. Decile 10 = past winners
  (Jegadeesh-Titman).
- B2 reversal: score = negative rev_1m rank. Decile 10 = past losers
  (Jegadeesh 1990).
- B3 naive multi-factor combo: equal-weight sum of the six feature ranks with
  signs fixed by the academic prior, no fitting: +mom_12_1, -rev_1m, -vol_63d,
  +dist_52w_high, +amihud_63d, -log_dollar_vol_63d. With each rank r in
  [0, 1], score = sum of r for + features plus (1 - r) for - features. This
  is the same-information baseline: it sees exactly the features the ML
  models see.
"""

import pandas as pd


def baseline_scores(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Compute the three baseline score series from the ranked panel.

    Args:
        panel: (date, ticker) panel with the six ranked features in [0, 1].

    Returns:
        Dict of {method: score Series} for b1_mom, b2_rev, b3_combo. Higher
        score = decile 10.
    """
    b1 = panel["mom_12_1"].rename("b1_mom")
    b2 = (1.0 - panel["rev_1m"]).rename("b2_rev")
    b3 = (
        panel["mom_12_1"]
        + (1.0 - panel["rev_1m"])
        + (1.0 - panel["vol_63d"])
        + panel["dist_52w_high"]
        + panel["amihud_63d"]
        + (1.0 - panel["log_dollar_vol_63d"])
    ).rename("b3_combo")
    return {"b1_mom": b1, "b2_rev": b2, "b3_combo": b3}
