"""
Data Loader
============
Handles reading MuMax3 table.txt files into pandas DataFrames.
All file-access and format-validation logic lives here.
"""

from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class LoadError(Exception):
    """Raised when a table.txt cannot be loaded or is malformed."""


def load_table(path: str | Path) -> pd.DataFrame:
    """
    Load a MuMax3 table.txt file.

    Parameters
    ----------
    path : path to the file

    Returns
    -------
    pd.DataFrame with column names taken from the file header.

    Raises
    ------
    LoadError  on any problem (file missing, empty, bad format …)
    """
    path = Path(path)

    if not path.exists():
        raise LoadError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise LoadError(f"File is empty: {path}")

    try:
        df = pd.read_csv(path, sep="\t", comment=None)
    except Exception as exc:
        raise LoadError(f"Could not parse '{path}': {exc}") from exc

    if df.empty:
        raise LoadError(f"No data rows in '{path}'.")

    # Strip leading/trailing whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    logger.info("Loaded %d rows × %d cols from '%s'", len(df), len(df.columns), path)
    return df


def get_columns(df: pd.DataFrame) -> list[str]:
    """Return column names, excluding the time column."""
    return list(df.columns)
