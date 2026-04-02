"""Utilities for downloading and storing JSE stock data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_stock_data(ticker: str, start: str = "2018-01-01") -> pd.DataFrame:
    """Download daily stock data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. "SBK.JO".
    start : str
        Start date for download in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Historical OHLCV data.
    """
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError(
            f"No data was downloaded for ticker '{ticker}'. "
            "Check the ticker symbol and internet connection."
        )

    df = df.reset_index()
    df.columns = [str(col) for col in df.columns]

    output_path = DATA_DIR / f"{ticker.replace('.', '_')}.csv"
    df.to_csv(output_path, index=False)
    return df
