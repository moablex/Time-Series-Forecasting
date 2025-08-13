# src/data_fetch.py
import yfinance as yf
from pathlib import Path
import pandas as pd
from datetime import datetime
from config import RAW_DIR, OHLC_COLS
from utils import logger, save_dataframe



def fetch_and_save(ticker: str, start: str, end: str) -> Path:
    """
    Fetch ticker data from Yahoo Finance and save raw CSV to data/raw/{ticker}.csv
    Returns the saved file path.
    """
    logger.info(f"Fetching {ticker} from {start} to {end}")
    data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        logger.warning(f"No data fetched for {ticker}")
        return None

    if "Adj Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    # Keep only standard columns
    data = data[OHLC_COLS]

    path = RAW_DIR / f"{ticker}.csv"
    save_dataframe(data, path)
    logger.info(f"Saved raw data for {ticker} to {path}")
    return path


def fetch_multiple(tickers, start, end):
    saved_paths = {}
    for t in tickers:
        path = fetch_and_save(t, start, end)
        if path:
            saved_paths[t] = path
    return saved_paths


if __name__ == "__main__":
    # Parameters
    tickers = ["TSLA", "BND", "SPY"]
    start_date = "2020-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    logger.info("Starting Yahoo Finance data fetch...")
    saved = fetch_multiple(tickers, start_date, end_date)
    logger.info(f"Data fetch complete. Saved files: {saved}")
