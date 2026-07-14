"""
ML Signal Generator - Training Pipeline

Run the full pipeline from the command line:
    python main.py
    python main.py --ticker AAPL --start 2021-01-01 --end 2024-01-01
    python main.py --threshold 0.50
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plots save without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report, roc_curve

from src.features import download_data, engineer_features, prepare_features_for_training
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import xgboost as xgb

from src.model import (
    bootstrap_auc_ci,
    evaluate_model,
    get_feature_importance,
    train_random_forest,
    train_walk_forward,
    train_xgboost,
)
from src.backtest import (
    backtest_buy_and_hold,
    backtest_strategy,
    generate_signals,
    plot_equity_curve,
    plot_feature_importance,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_or_download(ticker, start, end, api_source, api_key, data_dir):
    """Return OHLC DataFrame, loading from cache or downloading."""
    csv_path = data_dir / f"{ticker}_{start}_{end}.csv"
    xlsx_path = data_dir / f"{ticker}_{start}_{end}.xlsx"

    if csv_path.exists():
        print(f"Loading saved data from {csv_path.name}...")
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if xlsx_path.exists():
        print(f"Loading saved data from {xlsx_path.name}...")
        return pd.read_excel(xlsx_path, index_col=0, parse_dates=True)

    print(f"Downloading data for {ticker}...")
    max_retries = 3
    for attempt in range(max_retries):
        if attempt > 0:
            wait = 30 * attempt
            print(f"  Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)
        try:
            data = download_data(ticker, start, end, api_source=api_source, api_key=api_key)
            if data is not None and not data.empty:
                data_dir.mkdir(exist_ok=True)
                data.to_csv(csv_path)
                print(f"  Data saved to {csv_path.name}")
                return data
        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ("rate limit", "too many requests", "ratelimit")):
                if attempt == max_retries - 1:
                    raise SystemExit(
                        "Rate-limited after all retries. Wait 15-20 min or set API_SOURCE=alpha_vantage."
                    ) from exc
                continue
            raise

    raise SystemExit(f"Failed to obtain data for {ticker}.")


def _plot_roc(y_test, proba_rf, proba_xgb, auc_rf, auc_xgb, save_path):
    fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, proba_xgb)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.4f})", linewidth=2, color="#2E86AB")
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.4f})", linewidth=2, color="#A23B72")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("ROC Curve Comparison: Random Forest vs XGBoost", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved to {save_path}")


# ── main pipeline ────────────────────────────────────────────────────────────

def run(ticker, start_date, end_date, signal_threshold, transaction_cost):
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    data_dir = project_root / "data"

    # Load .env
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    api_source = os.getenv("API_SOURCE", "yfinance")
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY") if api_source == "alpha_vantage" else None

    # ── 1. Data ──────────────────────────────────────────────────────────────
    data = _load_or_download(ticker, start_date, end_date, api_source, api_key, data_dir)
    print(f"Loaded {len(data)} days  ({data.index[0]}  to  {data.index[-1]})\n")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    print("Engineering features...")
    data_features = engineer_features(data, return_periods=[1, 5], volatility_window=20, ma_windows=[5, 20])
    X, y = prepare_features_for_training(data_features)
    print(f"  Features shape: {X.shape}")
    print(f"  Target distribution: {dict(y.value_counts())}\n")

    # ── 3. Train / test split ───────────────────────────────────────────────
    #   Hold out the last 15% as a pure test set.
    #   Walk-forward validation runs over the first 85% to select the model.
    n = len(X)
    test_start = int(n * 0.85)

    X_train_full, y_train_full = X.iloc[:test_start], y.iloc[:test_start]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

    print(f"Train+Val (walk-forward): {len(X_train_full)}  |  Test (held-out): {len(X_test)}")
    print(f"Test window: {X_test.index[0].date()}  to  {X_test.index[-1].date()}\n")

    # ── 3b. Overfitting diagnostic ───────────────────────────────────────────
    #   Single 70/15 train/validation cut, purely to expose the train-vs-validation
    #   gap. Tree models memorise noise in financial time series; printing both
    #   numbers side by side is how you catch it rather than discover it in production.
    print("=" * 70)
    print("OVERFITTING DIAGNOSTIC (single 70/15 train/val cut)")
    print("=" * 70)

    tr_end, val_end = int(n * 0.70), int(n * 0.85)
    X_a, y_a = X.iloc[:tr_end], y.iloc[:tr_end]
    X_v, y_v = X.iloc[tr_end:val_end], y.iloc[tr_end:val_end]

    _, diag_rf = train_random_forest(X_a, y_a, X_v, y_v)
    _, diag_xgb = train_xgboost(X_a, y_a, X_v, y_v)

    print(f"\n  Random Forest  —  train acc: {diag_rf['train_accuracy']:.4f}   "
          f"val acc: {diag_rf['val_accuracy']:.4f}   val AUC: {diag_rf['auc']:.4f}")
    print(f"  XGBoost        —  train acc: {diag_xgb['train_accuracy']:.4f}   "
          f"val acc: {diag_xgb['val_accuracy']:.4f}   val AUC: {diag_xgb['auc']:.4f}")
    print("\n  A large train/val accuracy gap means the model is memorising noise,")
    print("  not learning signal. Walk-forward validation below is what keeps this")
    print("  from leaking into the reported result.\n")

    # ── 4. Walk-forward model selection (expanding window, 5 folds) ─────────
    print("=" * 70)
    print("WALK-FORWARD VALIDATION (expanding window, 5 folds)")
    print("=" * 70)

    model_rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
    )
    model_xgb = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="auc",
    )

    wf_auc_rf = train_walk_forward(model_rf, X_train_full, y_train_full, n_splits=5)
    wf_auc_xgb = train_walk_forward(model_xgb, X_train_full, y_train_full, n_splits=5)

    print(f"\n  Random Forest  —  mean AUC: {wf_auc_rf:.4f}")
    print(f"  XGBoost        —  mean AUC: {wf_auc_xgb:.4f}")

    if wf_auc_rf >= wf_auc_xgb:
        print(f"\n  Walk-forward selects Random Forest (+{wf_auc_rf - wf_auc_xgb:.4f})")
    else:
        print(f"\n  Walk-forward selects XGBoost (+{wf_auc_xgb - wf_auc_rf:.4f})")

    # ── 5. Retrain both models on full training data ─────────────────────────
    print("\n" + "=" * 70)
    print("RETRAINING ON FULL TRAINING SET")
    print("=" * 70)

    print("\n  Training Random Forest...")
    model_rf = clone(model_rf)
    model_rf.fit(X_train_full, y_train_full)

    print("  Training XGBoost...")
    model_xgb = clone(model_xgb)
    model_xgb.fit(X_train_full, y_train_full)

    # ── 6. Feature importance ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    importance_rf = get_feature_importance(model_rf, list(X.columns))
    importance_xgb = get_feature_importance(model_xgb, list(X.columns))

    print("\nRandom Forest:")
    print(importance_rf.to_string(index=False))
    print("\nXGBoost:")
    print(importance_xgb.to_string(index=False))

    plot_feature_importance(importance_rf, str(output_dir / "feature_importance_rf.png"),
                            "Random Forest Feature Importance")
    plot_feature_importance(importance_xgb, str(output_dir / "feature_importance_xgb.png"),
                            "XGBoost Feature Importance")

    # ── 7. Evaluate on test set ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    test_results_rf = evaluate_model(model_rf, X_test, y_test)
    test_results_xgb = evaluate_model(model_xgb, X_test, y_test)

    print(f"\nRandom Forest  —  AUC: {test_results_rf['auc']:.4f}   Accuracy: {test_results_rf['accuracy']:.4f}")
    print(classification_report(y_test, test_results_rf["y_pred"]))

    print(f"XGBoost        —  AUC: {test_results_xgb['auc']:.4f}   Accuracy: {test_results_xgb['accuracy']:.4f}")
    print(classification_report(y_test, test_results_xgb["y_pred"]))

    # A point AUC on ~150 daily rows is a noisy statistic. Show the error bar and
    # state plainly whether a coin flip can be ruled out.
    print("Bootstrap 95% CI on test AUC (2000 resamples):")
    for name, res in (("Random Forest", test_results_rf), ("XGBoost", test_results_xgb)):
        pt, lo, hi = bootstrap_auc_ci(y_test, res["y_pred_proba"])
        verdict = "contains 0.5 — cannot reject coin flip" if lo <= 0.5 <= hi else "excludes 0.5"
        print(f"  {name:<15} AUC {pt:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   {verdict}")

    # ROC curve
    _plot_roc(y_test, test_results_rf["y_pred_proba"], test_results_xgb["y_pred_proba"],
              test_results_rf["auc"], test_results_xgb["auc"],
              str(output_dir / "roc_curve.png"))

    # ── 8. Generate trading signals ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SIGNAL GENERATION")
    print("=" * 70)

    test_returns = data_features.loc[X_test.index, "next_return"].dropna()

    signals_rf = pd.Series(
        generate_signals(test_results_rf["y_pred_proba"], threshold=signal_threshold),
        index=X_test.index,
    ).loc[test_returns.index]

    signals_xgb = pd.Series(
        generate_signals(test_results_xgb["y_pred_proba"], threshold=signal_threshold),
        index=X_test.index,
    ).loc[test_returns.index]

    print(f"\nRandom Forest — signals: {signals_rf.sum()}/{len(signals_rf)} days ({signals_rf.mean()*100:.1f}%)")
    print(f"XGBoost       — signals: {signals_xgb.sum()}/{len(signals_xgb)} days ({signals_xgb.mean()*100:.1f}%)")

    # ── 9. Backtest ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BACKTEST PERFORMANCE")
    print("=" * 70)

    def _backtest_one(name, signals, returns_series):
        aligned = pd.DataFrame({"signals": signals.values, "returns": returns_series.values},
                                index=signals.index)
        equity, metrics = backtest_strategy(
            aligned["signals"].values, aligned["returns"].values,
            initial_capital=10_000.0, transaction_cost=transaction_cost,
        )
        equity_series = pd.Series(equity, index=aligned.index) if not isinstance(equity, pd.Series) else equity
        print(f"\n  {name}:")
        print(f"    Total Return:      {metrics['total_return_pct']:>8.2f}%")
        print(f"    Annualized Return: {metrics['annualized_return_pct']:>8.2f}%")
        print(f"    Volatility:        {metrics['volatility_pct']:>8.2f}%")
        print(f"    Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
        print(f"    Max Drawdown:      {metrics['max_drawdown_pct']:>8.2f}%")
        print(f"    Win Rate:          {metrics['win_rate_pct']:>8.2f}%")
        print(f"    Total Trades:      {metrics['total_trades']:>8d}")
        return equity_series, metrics

    equity_rf, metrics_rf_bt = _backtest_one("Random Forest", signals_rf, test_returns)
    equity_xgb, metrics_xgb_bt = _backtest_one("XGBoost", signals_xgb, test_returns)

    # Buy-and-hold benchmark over the EXACT same window, same metric code.
    equity_bh, metrics_bh = backtest_buy_and_hold(test_returns, initial_capital=10_000.0)
    print(f"\n  Buy & Hold ({ticker}):")
    print(f"    Total Return:      {metrics_bh['total_return_pct']:>8.2f}%")
    print(f"    Annualized Return: {metrics_bh['annualized_return_pct']:>8.2f}%")
    print(f"    Volatility:        {metrics_bh['volatility_pct']:>8.2f}%")
    print(f"    Sharpe Ratio:      {metrics_bh['sharpe_ratio']:>8.2f}")
    print(f"    Max Drawdown:      {metrics_bh['max_drawdown_pct']:>8.2f}%")

    # Best model selected by walk-forward AUC
    if wf_auc_rf >= wf_auc_xgb:
        best_name, best_equity, best_metrics = "Random Forest", equity_rf, metrics_rf_bt
    else:
        best_name, best_equity, best_metrics = "XGBoost", equity_xgb, metrics_xgb_bt

    print(f"\n  Best model (by walk-forward AUC): {best_name}")

    # ── 9b. Strategy vs benchmark ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STRATEGY vs BUY & HOLD  ({X_test.index[0].date()} to {X_test.index[-1].date()})")
    print("=" * 70)
    print(f"\n  {'':<22}{'Total Ret':>12}{'Sharpe':>10}{'MaxDD':>10}{'Days Long':>12}")
    for name, m, sig in (
        ("Random Forest", metrics_rf_bt, signals_rf),
        ("XGBoost", metrics_xgb_bt, signals_xgb),
    ):
        print(f"  {name:<22}{m['total_return_pct']:>11.2f}%{m['sharpe_ratio']:>10.2f}"
              f"{m['max_drawdown_pct']:>9.2f}%{sig.mean()*100:>11.1f}%")
    print(f"  {'Buy & Hold ' + ticker:<22}{metrics_bh['total_return_pct']:>11.2f}%"
          f"{metrics_bh['sharpe_ratio']:>10.2f}{metrics_bh['max_drawdown_pct']:>9.2f}%"
          f"{100.0:>11.1f}%")

    excess_rf = metrics_rf_bt["total_return_pct"] - metrics_bh["total_return_pct"]
    excess_xgb = metrics_xgb_bt["total_return_pct"] - metrics_bh["total_return_pct"]
    print(f"\n  Excess return vs Buy & Hold  —  Random Forest: {excess_rf:+.2f}%")
    print(f"  Excess return vs Buy & Hold  —  XGBoost:       {excess_xgb:+.2f}%")

    best_excess = best_metrics["total_return_pct"] - metrics_bh["total_return_pct"]
    if best_excess <= 0:
        print(f"\n  The selected model ({best_name}) UNDERPERFORMS buy-and-hold by "
              f"{abs(best_excess):.2f}pp over this window.")
        print("  A long-only signal in a rising market earns beta, not alpha.")
    else:
        print(f"\n  The selected model ({best_name}) beats buy-and-hold by {best_excess:.2f}pp "
              f"over this window.")

    # ── 10. Plot results ─────────────────────────────────────────────────────
    plot_equity_curve(equity_rf, str(output_dir / "equity_curve_rf.png"),
                      "Random Forest Strategy vs Buy & Hold", benchmark=equity_bh,
                      benchmark_label=f"Buy & Hold {ticker}")
    plot_equity_curve(equity_xgb, str(output_dir / "equity_curve_xgb.png"),
                      "XGBoost Strategy vs Buy & Hold", benchmark=equity_bh,
                      benchmark_label=f"Buy & Hold {ticker}")
    plot_equity_curve(best_equity, str(output_dir / "equity_curve.png"),
                      f"{best_name} Strategy vs Buy & Hold (Best Model)", benchmark=equity_bh,
                      benchmark_label=f"Buy & Hold {ticker}")

    print(f"\nAll outputs saved to {output_dir}/")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ML Signal Generator — train models, generate signals, and backtest.",
    )
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD (default: 2020-01-01)")
    parser.add_argument("--end", default="2024-01-01", help="End date YYYY-MM-DD (default: 2024-01-01)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Signal probability threshold (default: 0.55)")
    parser.add_argument("--transaction-cost", type=float, default=0.0002,
                        help="Transaction cost in return space, e.g. 0.0002 = 2 bps (default: 0.0002)")
    args = parser.parse_args()

    run(args.ticker, args.start, args.end, args.threshold, args.transaction_cost)


if __name__ == "__main__":
    main()
