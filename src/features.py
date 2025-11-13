"""
Feature Engineering Module

This module contains functions to compute technical features from OHLC price data
for machine learning signal generation.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download OHLC data using yfinance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'SPY' for S&P 500)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    import yfinance as yf
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Flatten MultiIndex columns if present (yfinance returns MultiIndex for single ticker)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Ensure we have the required columns
    if 'Close' not in data.columns:
        raise ValueError(f"Failed to download data for {ticker}")
    
    return data


def compute_returns(data: pd.DataFrame, periods: list[int] = [1, 5]) -> pd.DataFrame:
    """
    Compute lagged returns for specified periods.
    
    Args:
        data: DataFrame with 'Close' column
        periods: List of periods for lagged returns (e.g., [1, 5] for 1d and 5d)
    
    Returns:
        DataFrame with additional return columns
    """
    df = data.copy()
    
    for period in periods:
        df[f'return_{period}d'] = df['Close'].pct_change(periods=period)
    
    return df


def compute_volatility(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute rolling volatility (standard deviation of returns).
    
    Args:
        data: DataFrame with 'Close' column
        window: Rolling window size in days
    
    Returns:
        DataFrame with additional volatility column
    """
    df = data.copy()
    
    # Compute daily returns first if not present
    if 'return_1d' not in df.columns:
        df['return_1d'] = df['Close'].pct_change()
    
    df[f'volatility_{window}d'] = df['return_1d'].rolling(window=window).std()
    
    return df


def compute_moving_averages(data: pd.DataFrame, windows: list[int] = [5, 20]) -> pd.DataFrame:
    """
    Compute moving averages for specified windows.
    
    Args:
        data: DataFrame with 'Close' column
        windows: List of window sizes (e.g., [5, 20])
    
    Returns:
        DataFrame with additional MA columns
    """
    df = data.copy()
    
    for window in windows:
        df[f'ma_{window}d'] = df['Close'].rolling(window=window).mean()
    
    return df


def compute_ma_gap(data: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    """
    Compute MA gap (ratio difference between short and long moving averages).
    
    Args:
        data: DataFrame with MA columns
        short_window: Short MA window (e.g., 5)
        long_window: Long MA window (e.g., 20)
    
    Returns:
        DataFrame with additional MA gap column
    """
    df = data.copy()
    
    # Ensure MAs are computed
    if f'ma_{short_window}d' not in df.columns:
        df = compute_moving_averages(df, [short_window, long_window])
    
    short_ma = df[f'ma_{short_window}d']
    long_ma = df[f'ma_{long_window}d']
    
    # Compute ratio difference: (short_ma - long_ma) / long_ma
    df['ma_gap'] = (short_ma - long_ma) / long_ma
    
    return df


def create_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target: 1 if next-day return > 0, else 0.
    
    Args:
        data: DataFrame with 'Close' column
    
    Returns:
        DataFrame with 'target' column
    """
    df = data.copy()
    
    # Compute next-day return
    df['next_return'] = df['Close'].shift(-1) / df['Close'] - 1
    
    # Binary target: 1 if positive return, 0 otherwise
    df['target'] = (df['next_return'] > 0).astype(int)
    
    return df


def engineer_features(
    data: pd.DataFrame,
    return_periods: list[int] = [1, 5],
    volatility_window: int = 20,
    ma_windows: list[int] = [5, 20]
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    This function orchestrates all feature engineering steps:
    1. Compute lagged returns
    2. Compute rolling volatility
    3. Compute moving averages
    4. Compute MA gap
    5. Create binary target
    
    Args:
        data: DataFrame with OHLC data
        return_periods: Periods for lagged returns
        volatility_window: Window for volatility calculation
        ma_windows: Windows for moving averages
    
    Returns:
        DataFrame with all features and target
    """
    df = data.copy()
    
    # Step 1: Compute lagged returns
    df = compute_returns(df, periods=return_periods)
    
    # Step 2: Compute rolling volatility
    df = compute_volatility(df, window=volatility_window)
    
    # Step 3: Compute moving averages
    df = compute_moving_averages(df, windows=ma_windows)
    
    # Step 4: Compute MA gap
    df = compute_ma_gap(df, short_window=ma_windows[0], long_window=ma_windows[1])
    
    # Step 5: Create target
    df = create_target(df)
    
    return df


def prepare_features_for_training(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for ML training.
    
    Args:
        data: DataFrame with engineered features and target
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Define feature columns
    feature_cols = [
        'return_1d', 'return_5d',
        'volatility_20d',
        'ma_5d', 'ma_20d',
        'ma_gap'
    ]
    
    # Select features and target
    X = data[feature_cols].copy()
    y = data['target'].copy()
    
    # Ensure column names are strings (handle MultiIndex if somehow still present)
    if isinstance(X.columns, pd.MultiIndex):
        X.columns = X.columns.get_level_values(0)
    X.columns = [str(col) for col in X.columns]
    
    # Remove rows with NaN values (from rolling calculations and target shift)
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y

