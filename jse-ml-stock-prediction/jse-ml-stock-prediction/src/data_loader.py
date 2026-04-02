"""Utilities for downloading and storing JSE stock data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_stock_data(ticker: str, start: str = "2018-01-01") -> pd.DataFrame:
    """Download daily stock data from Yahoo Finance."""
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError(
            f"No data was downloaded for ticker '{ticker}'. "
            "Check the ticker symbol and internet connection."
        )

    # Fix for newer yfinance versions that may return MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # If ticker is one level, keep only the price-field names
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)

    # Make sure column names are plain strings
    df.columns = [str(col) for col in df.columns]

    # Reorder/select expected columns if they exist
    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available_cols = [col for col in expected_cols if col in df.columns]
    df = df[available_cols]

    df = df.reset_index()

    output_path = DATA_DIR / f"{ticker.replace('.', '_')}.csv"
    df.to_csv(output_path, index=False)
    return df
