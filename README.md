# ML Signal Generator

A machine learning project for predicting next-day return direction of financial assets and generating binary trading signals with backtesting capabilities.

## Overview

This project implements a complete ML pipeline to:
- Download historical OHLC (Open, High, Low, Close) market data
- Engineer technical features (returns, volatility, moving averages, etc.)
- Train machine learning models (Random Forest or XGBoost) to predict next-day return direction
- Generate probability-based trading signals
- Backtest strategy performance with comprehensive metrics

## Project Structure

```
ml-signal-generator/
├── data/                # Sample data or placeholder
├── notebooks/
│   └── 01_training.ipynb  # Complete training pipeline
├── src/
│   ├── __init__.py
│   ├── features.py      # Feature engineering functions
│   ├── model.py         # ML model training functions
│   └── backtest.py      # Backtesting logic
├── outputs/             # Generated charts and results
├── .env.example         # Environment variables template
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher

### Setup

1. Clone the repository:
```bash
git clone git@github.com:GasparCoquet/ml-signal-generator.git
cd ml-signal-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional, for alternative APIs):
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys if using Alpha Vantage
# Get free API keys:
# - Alpha Vantage: https://www.alphavantage.co/support/#api-key
```

## Usage

### Running the Training Pipeline

1. Open the Jupyter notebook:
```bash
jupyter notebook notebooks/01_training.ipynb
```

2. Execute all cells to:
   - Download market data (default: SPY ETF from 2020-2024)
   - Engineer features
   - Train a Random Forest or XGBoost model
   - Generate trading signals
   - Backtest the strategy
   - Generate performance plots

### Key Configuration

In the notebook, you can modify:
- `TICKER`: Stock ticker symbol (default: 'SPY')
- `START_DATE` / `END_DATE`: Date range for data
- `MODEL_TYPE`: 'random_forest' or 'xgboost'
- `SIGNAL_THRESHOLD`: Probability threshold for signal generation (default: 0.55)

### API Configuration

The project supports multiple data sources to avoid rate limiting:

1. **yfinance** (default): Free, no API key needed, but can be rate limited
2. **Alpha Vantage**: Free tier (5 calls/min, 500 calls/day) - requires API key

Configure via `.env` file:
- Copy `.env.example` to `.env`
- Set `API_SOURCE` to your preferred source
- Add API key if using Alpha Vantage

### Troubleshooting

**Rate Limiting (Too Many Requests):**
- Yahoo Finance (via yfinance) has rate limits to prevent abuse
- **Solution 1**: Use an alternative API (Alpha Vantage) - see API Configuration above
- **Solution 2**: If using yfinance, wait 15-20 minutes before trying again
- **Solution 3**: Set `USE_SAMPLE_DATA = True` in the notebook to use synthetic data for testing
- The download function includes automatic retry logic (3 attempts with increasing delays)
- Once data is downloaded, it's saved locally and reused automatically

## Features

### Feature Engineering

The pipeline computes the following features:
- **Lagged Returns**: 1-day and 5-day returns
- **Rolling Volatility**: 20-day rolling standard deviation of returns
- **Moving Averages**: 5-day and 20-day moving averages
- **MA Gap**: Ratio difference between short and long MAs

### Target Variable

Binary target: `y = 1` if next-day return > 0, else `y = 0`

### Model Training

- Time series aware train/validation/test split (70%/15%/15%)
- Supports Random Forest and XGBoost classifiers
- Feature importance analysis
- AUC and accuracy metrics

### Signal Generation

- Probability-based signals using model predictions
- Configurable threshold (default: 0.55)
- Binary signals: 1 (long) or 0 (flat)

### Backtesting

The backtest computes:
- **Total Return**: Overall strategy return
- **Annualized Return**: Return adjusted for time period
- **Volatility**: Annualized standard deviation
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

## Outputs

The pipeline generates two key visualizations saved to `outputs/`:
- `equity_curve.png`: Strategy equity curve over time
- `feature_importance.png`: Feature importance ranking

## Key Results

After running the notebook, you'll see:
- Model performance metrics (AUC, accuracy)
- Feature importance rankings
- Backtest performance metrics
- Equity curve visualization

*Note: Results will vary based on market conditions, data period, and model parameters.*

## Code Style

- Modern, modular Python with clear separation of concerns
- Type hints and comprehensive docstrings
- PEP8 compliant code
- Vectorized operations for performance
- Clear variable naming

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning models
- `xgboost`: Gradient boosting classifier
- `matplotlib`: Plotting
- `yfinance`: Market data download (default)
- `requests`: HTTP library for alternative APIs
- `python-dotenv`: Environment variable management
- `jupyter`: Notebook environment

## Disclaimer

**⚠️ IMPORTANT: For educational purposes only, not financial advice.**

This project is intended for educational and research purposes only. The trading signals and backtest results are not recommendations for actual trading. Past performance does not guarantee future results. Always conduct thorough research and consult with qualified financial advisors before making investment decisions.

## License

See LICENSE file for details.

## Author

GasparCoquet
