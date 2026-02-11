"""
Feature Engineering Module

This module contains functions to compute technical features from OHLC price data
for machine learning signal generation.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def download_data(
    ticker: str, 
    start_date: str, 
    end_date: str,
    api_source: str = 'yfinance',
    api_key: str = None
) -> pd.DataFrame:
    """
    Download OHLC data using various APIs.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'SPY' for S&P 500)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        api_source: API to use ('yfinance' or 'alpha_vantage')
        api_key: API key (required for alpha_vantage)
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    if api_source == 'yfinance':
        return _download_yfinance(ticker, start_date, end_date)
    elif api_source == 'alpha_vantage':
        if not api_key:
            raise ValueError("API key required for Alpha Vantage. Get a free key at https://www.alphavantage.co/support/#api-key")
        return _download_alpha_vantage(ticker, start_date, end_date, api_key)
    else:
        raise ValueError(f"Unknown API source: {api_source}. Choose 'yfinance' or 'alpha_vantage'")


def _download_yfinance(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download data using yfinance with improved error handling."""
    import yfinance as yf
    import time
    
    try:
        # Try using Ticker object first (more reliable)
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=start_date, end=end_date)
        
        if data.empty:
            # Fallback to download method
            time.sleep(1)  # Small delay to avoid rate limiting
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        # Fallback to standard download
        time.sleep(2)  # Delay before retry
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Ensure we have the required columns
    if data.empty:
        raise ValueError(
            f"yfinance returned empty data for {ticker}. "
            f"This usually means rate limiting or invalid ticker. "
            f"Try waiting 15-20 minutes or use an alternative API."
        )
    if 'Close' not in data.columns:
        raise ValueError(f"yfinance data missing 'Close' column for {ticker}")
    
    return data


def _download_alpha_vantage(ticker: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """Download data using Alpha Vantage API."""
    import requests
    from datetime import datetime
    
    # Alpha Vantage only provides daily data
    # Use TIME_SERIES_DAILY (free) instead of TIME_SERIES_DAILY_ADJUSTED (premium)
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',  # Free endpoint (not ADJUSTED which is premium)
        'symbol': ticker,
        'outputsize': 'full',
        'apikey': api_key,
        'datatype': 'json'
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data_json = response.json()
    
    # Check for API errors first
    if 'Error Message' in data_json:
        raise ValueError(f"Alpha Vantage API Error: {data_json['Error Message']}")
    if 'Note' in data_json:
        raise ValueError(f"Alpha Vantage Rate Limit: {data_json['Note']}")
    
    # Check for invalid API key or other info messages
    if 'Information' in data_json:
        info_msg = data_json['Information']
        if 'API key' in info_msg.lower() or 'invalid' in info_msg.lower():
            raise ValueError(f"Alpha Vantage API Key Error: {info_msg}")
        else:
            raise ValueError(f"Alpha Vantage API Info: {info_msg}")
    
    # Parse the data - TIME_SERIES_DAILY returns 'Time Series (Daily)'
    time_series = data_json.get('Time Series (Daily)', {})
    
    if not time_series:
        # Provide more diagnostic info
        available_keys = list(data_json.keys())
        error_details = f"Response keys: {available_keys}"
        
        # Check if there's a meta data section that might give clues
        if 'Meta Data' in data_json:
            meta = data_json['Meta Data']
            error_details += f"\nMeta Data: {meta}"
        
        raise ValueError(
            f"No data returned for {ticker} from Alpha Vantage.\n"
            f"{error_details}\n"
            f"Possible issues:\n"
            f"1. Invalid or expired API key\n"
            f"2. Invalid ticker symbol\n"
            f"3. API rate limit exceeded\n"
            f"4. Try using 'yfinance' instead: Set API_SOURCE=yfinance in .env"
        )
    
    # Convert to DataFrame
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    records = []
    for date_str, values in time_series.items():
        date = pd.to_datetime(date_str)
        if start_dt <= date <= end_dt:
            # TIME_SERIES_DAILY uses different keys than ADJUSTED
            # Keys: '1. open', '2. high', '3. low', '4. close', '5. volume'
            records.append({
                'Date': date,
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])  # Note: '5. volume' not '6. volume' for non-adjusted
            })
    
    if not records:
        raise ValueError(f"No data found for {ticker} in date range {start_date} to {end_date}")
    
    data = pd.DataFrame(records)
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    
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


def compute_z_score(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Z-Score: (Price - rolling mean) / rolling std.
    Stationarity-aware: oscillates around 0, better for ML than raw price or MA.
    """
    df = data.copy()
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    df['z_score_20'] = (df['Close'] - rolling_mean) / rolling_std
    # Avoid inf/nan when std=0
    df['z_score_20'] = df['z_score_20'].replace([np.inf, -np.inf], np.nan)
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
    5. Z-Score (stationarity-aware)
    6. Create binary target
    
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

    # Step 5: Z-Score (stationarity-aware: oscillates around 0, better for ML)
    df = compute_z_score(df, window=volatility_window)

    # Step 6: Create target
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
        'ma_gap',
        'z_score_20'
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

