"""
Hysteresis Processing
======================
Pure data-processing helpers for hysteresis analysis.
Edit this file to customise data transformations without touching GUI code.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def extract_xy(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pull x and y arrays from a DataFrame.

    Returns (x, y) after dropping rows where either column is NaN.
    """
    sub = df[[x_col, y_col]].dropna()
    return sub[x_col].to_numpy(), sub[y_col].to_numpy()


def merge_datasets(
    datasets: list[tuple[np.ndarray, np.ndarray, str]],
    x_label: str,
) -> pd.DataFrame:
    """
    Merge multiple (x, y, label) pairs into one DataFrame.

    Uses an outer join on the x-axis so unequal lengths are handled via NaN.

    Parameters
    ----------
    datasets : list of (x_array, y_array, column_label)
    x_label  : name for the shared x column

    Returns
    -------
    pd.DataFrame  with columns [x_label, label_0, label_1, …]
    """
    frames = []
    for x, y, label in datasets:
        frames.append(pd.DataFrame({x_label: x, label: y}))

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for frame in frames[1:]:
        result = pd.merge(result, frame, on=x_label, how="outer")

    return result.sort_values(x_label).reset_index(drop=True)
