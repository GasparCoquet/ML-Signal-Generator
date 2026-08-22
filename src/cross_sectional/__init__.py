"""
Phase 2: cross-sectional equity ranking study.

Ranks the S&P 500 cross-section each month on standard features (momentum,
reversal, volatility, 52-week high, liquidity) and tests whether the ranking
predicts relative next-month returns. Baselines first, then walk-forward ML,
Newey-West t-stats on every headline number, costs charged on turnover.

Entry point: python -m src.cross_sectional.run
"""
