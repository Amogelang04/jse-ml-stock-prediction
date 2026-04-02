"""Feature engineering functions for stock price prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create predictive features from raw OHLCV stock data.

    The target is the next day's closing price.
    """
    data = df.copy()

    # Keep consistent names from downloaded data
    rename_map = {
        "Date": "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume",
    }
    data = data.rename(columns=rename_map)

    data["return_1d"] = data["Close"].pct_change()
    data["return_3d"] = data["Close"].pct_change(3)
    data["return_5d"] = data["Close"].pct_change(5)

    data["lag_close_1"] = data["Close"].shift(1)
    data["lag_close_2"] = data["Close"].shift(2)
    data["lag_close_3"] = data["Close"].shift(3)

    data["ma_5"] = data["Close"].rolling(window=5).mean()
    data["ma_10"] = data["Close"].rolling(window=10).mean()
    data["ma_20"] = data["Close"].rolling(window=20).mean()

    data["volatility_5"] = data["return_1d"].rolling(window=5).std()
    data["volatility_10"] = data["return_1d"].rolling(window=10).std()

    data["price_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["volume_change"] = data["Volume"].pct_change()
    data["momentum_5"] = data["Close"] - data["Close"].shift(5)

    # Ratios relative to moving averages
    data["close_ma5_ratio"] = data["Close"] / data["ma_5"]
    data["close_ma10_ratio"] = data["Close"] / data["ma_10"]

    # Target: next day's close
    data["target_next_close"] = data["Close"].shift(-1)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna().reset_index(drop=True)
    return data


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "return_1d",
    "return_3d",
    "return_5d",
    "lag_close_1",
    "lag_close_2",
    "lag_close_3",
    "ma_5",
    "ma_10",
    "ma_20",
    "volatility_5",
    "volatility_10",
    "price_range",
    "volume_change",
    "momentum_5",
    "close_ma5_ratio",
    "close_ma10_ratio",
]
