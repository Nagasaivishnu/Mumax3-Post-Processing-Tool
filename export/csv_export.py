"""
CSV Export
===========
All export-to-disk logic lives here.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd


def export_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_heatmap_dataframe(
    fields: "np.ndarray",
    f: "np.ndarray",
    mFFTs_abs: "np.ndarray",
    label: str,
) -> pd.DataFrame:
    """
    Flatten a 2-D heatmap array into a tidy CSV-friendly DataFrame.

    Columns: Field_T, Frequency_Hz, <label>_Absorption
    """
    import numpy as np
    rows = []
    for fi, freq in enumerate(f):
        for bi, field in enumerate(fields):
            rows.append({
                "Field_T":    field,
                "Frequency_Hz": freq,
                f"{label}_Absorption": mFFTs_abs[fi, bi],
            })
    return pd.DataFrame(rows)


def build_slice_dataframe(
    x: "np.ndarray",
    y_datasets: list[tuple["np.ndarray", str]],
    x_label: str,
) -> pd.DataFrame:
    """
    Build a DataFrame for a 1-D slice plot (multiple datasets, same x).

    Uses outer merge so unequal lengths become NaN.
    """
    import pandas as pd
    frames = [pd.DataFrame({x_label: x})]
    for y, label in y_datasets:
        frames.append(pd.DataFrame({x_label: x, label: y}))
    result = frames[0]
    for f in frames[1:]:
        result = pd.merge(result, f, on=x_label, how="outer")
    return result.sort_values(x_label).reset_index(drop=True)
