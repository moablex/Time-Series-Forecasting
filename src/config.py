# config.py
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Defaults
DEFAULT_TICKERS = ["TSLA", "BND", "SPY"]
DEFAULT_START = "2015-07-01"
DEFAULT_END = "2025-07-31"

# Data columns
OHLC_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
