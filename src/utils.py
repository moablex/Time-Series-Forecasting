import logging
from pathlib import Path
import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("ts_portfolio")


logger = setup_logging()


def save_dataframe(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)