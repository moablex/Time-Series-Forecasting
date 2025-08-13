# preprocess.py
from pathlib import Path
import pandas as pd
import numpy as np
from .config import RAW_DIR, PROCESSED_DIR, OHLC_COLS, DEFAULT_TICKERS
from .utils import logger, save_dataframe, load_dataframe


def load_raw(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}.csv"
    if not path.exists():
        logger.error(f"Raw data for {ticker} not found in {RAW_DIR}")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def reindex_market_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


def fill_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(method=method).fillna(method="bfill")
    return df


def feature_engineer(df: pd.DataFrame, window_vol: int = 21) -> pd.DataFrame:
    df = df.copy()
    df["adj_close"] = df["Adj Close"]
    df["daily_return"] = df["adj_close"].pct_change()
    df["log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    df[f"vol_{window_vol}d"] = df["daily_return"].rolling(window=window_vol).std() * np.sqrt(252)
    df[f"ret_mean_{window_vol}d"] = df["daily_return"].rolling(window=window_vol).mean()
    df["sma_21"] = df["adj_close"].rolling(21).mean()
    df["sma_63"] = df["adj_close"].rolling(63).mean()
    return df


def compute_risk_metrics(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    returns = df["daily_return"].dropna()
    if returns.empty:
        return {}
    hist_var = returns.quantile(alpha)
    mu = returns.mean()
    sigma = returns.std()
    param_var = mu + sigma * np.percentile(returns, 100 * alpha)
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if sigma != 0 else np.nan
    return {
        "hist_var": hist_var,
        "param_var": param_var,
        "annual_return": returns.mean() * 252,
        "annual_vol": returns.std() * np.sqrt(252),
        "sharpe": sharpe,
    }


def process_and_save(ticker: str):
    logger.info(f"Preprocessing {ticker}")
    df = load_raw(ticker)
    if df.empty:
        logger.warning(f"No raw data for {ticker}, skipping.")
        return None, None
    df = reindex_market_days(df)
    df = fill_missing(df)
    df = feature_engineer(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{ticker}_processed.csv"
    save_dataframe(df, out_path)
    logger.info(f"Saved processed data to {out_path}")
    metrics = compute_risk_metrics(df)
    return df, metrics


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in DEFAULT_TICKERS:
        df, metrics = process_and_save(ticker)
        if metrics:
            logger.info(f"Metrics for {ticker}: {metrics}")
