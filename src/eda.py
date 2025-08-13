# eda.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from .utils import logger, load_dataframe
from .config import PROCESSED_DIR, DEFAULT_TICKERS

def plot_close(df: pd.DataFrame, ticker: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Adj Close"], label=f"{ticker} Adj Close")
    ax.set_title(f"{ticker} Adjusted Close Price")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.legend()
    fig.autofmt_xdate()
    path = outdir / f"{ticker}_adj_close.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved close plot to {path}")
    return path


def plot_returns_hist(df: pd.DataFrame, ticker: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    returns = df["daily_return"].dropna()
    ax.hist(returns, bins=80)
    ax.set_title(f"{ticker} Daily Returns Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    path = outdir / f"{ticker}_returns_hist.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved returns histogram to {path}")
    return path


def plot_rolling_vol(df: pd.DataFrame, ticker: str, window: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    col = f"vol_{window}d"
    if col not in df.columns:
        raise ValueError(f"{col} not in df")
    ax.plot(df.index, df[col])
    ax.set_title(f"Rolling Volatility ({window}d annualized) — {ticker}")
    ax.set_ylabel("Volatility")
    ax.set_xlabel("Date")
    path = outdir / f"{ticker}_vol_{window}d.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved rolling vol plot to {path}")
    return path


def adf_test_print(series: pd.Series, ticker: str):
    from statsmodels.tsa.stattools import adfuller
    s = series.dropna()
    if s.empty:
        logger.warning(f"Series empty for ADF test: {ticker}")
        return None
    res = adfuller(s)
    logger.info(f"ADF for {ticker}: statistic={res[0]:.4f}, pvalue={res[1]:.4f}")
    return res


if __name__ == "__main__":
    FIG_DIR = Path("reports") / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in DEFAULT_TICKERS:
        file_path = PROCESSED_DIR / f"{ticker}_processed.csv"
        if not file_path.exists():
            logger.warning(f"No processed data for {ticker}. Run preprocess first.")
            continue

        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        plot_close(df, ticker, FIG_DIR)
        plot_returns_hist(df, ticker, FIG_DIR)
        plot_rolling_vol(df, ticker, 21, FIG_DIR)

        adf_test_print(df["daily_return"], ticker)
