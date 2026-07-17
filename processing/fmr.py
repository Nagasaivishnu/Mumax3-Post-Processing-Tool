"""
FMR Post-Processing Module
===========================
Core signal-processing routines for Ferromagnetic Resonance analysis.
Ported and adapted from the original fmr_post_processing.py.

Edit this file to modify the physics/signal-processing logic without
touching any GUI code.
"""

import numpy as np
import pandas as pd
import scipy.interpolate as sp_interp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calc_susceptibility(
    df: pd.DataFrame,
    dt: float,
    interpolate: bool = False,
    BiasFieldDir: str = "y",
    MWFieldDir: str = "x",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the dynamic susceptibility from a MuMax3 table.txt DataFrame.

    Parameters
    ----------
    df            : loaded table.txt as a DataFrame
    dt            : saving interval of the simulation [seconds]
    interpolate   : if True, re-sample m(t) to a uniform grid before FFT
    BiasFieldDir  : direction of the static (swept) external field ('x','y','z')
    MWFieldDir    : direction of the microwave (dynamic) field ('x','y','z')

    Returns
    -------
    fields  : 1-D array of swept-field values [T]
    f       : 1-D array of positive frequencies [Hz]
    mFFTs   : 2-D complex array, shape (n_freqs, n_fields)
    """

    # ── input validation ──────────────────────────────────────────────────
    BiasFieldDir = _validate_dir(BiasFieldDir, "BiasFieldDir")
    MWFieldDir   = _validate_dir(MWFieldDir,   "MWFieldDir")

    # ── find the swept field values ───────────────────────────────────────
    use_b_stat = "B_stat (T)" in df.columns
    if use_b_stat:
        B_stats = np.sort(df["B_stat (T)"].unique())
    else:
        col = f"B_ext{BiasFieldDir} (T)"
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found. Available: {list(df.columns)}"
            )
        B_stats = np.sort(df[col].unique())

    # ── sweep through field values ────────────────────────────────────────
    mFFTs_list: list[np.ndarray] = []
    fields_list: list[float]     = []
    Ls: list[int]                = []

    t_col = "# t (s)"
    m_col = f"m{MWFieldDir} ()"
    if t_col not in df.columns:
        raise KeyError(f"Time column '{t_col}' not found.")
    if m_col not in df.columns:
        raise KeyError(f"Magnetisation column '{m_col}' not found.")

    for B in B_stats:
        sub = df[df["B_stat (T)"] == B].copy() if use_b_stat \
              else df[df[f"B_ext{BiasFieldDir} (T)"] == B].copy()

        # drop duplicate time-points (artefact of some MuMax3 runs)
        sub = sub.drop_duplicates(subset=t_col)
        t = sub[t_col].to_numpy()
        m = sub[m_col].to_numpy()
        m = m - m.mean()          # remove DC offset

        if interpolate and len(t) > 3:
            spl = sp_interp.CubicSpline(t, m)
            t   = np.linspace(0, t[-1], int(t[-1] / dt), endpoint=True)
            m   = spl(t)

        L     = len(t)
        mFFT  = np.fft.fft(m)[: L // 2]
        mFFTs_list.append(mFFT)
        fields_list.append(float(B))
        Ls.append(L)

    # ── remove incomplete runs (unequal length) ───────────────────────────
    Ls_arr = np.array(Ls)
    L_vals, L_counts = np.unique(Ls_arr, return_counts=True)
    L_vals  = L_vals[np.argsort(-L_counts)]  # most-common length first
    L_keep  = int(L_vals[0])

    if len(L_vals) > 1:
        # remove runs whose length is not the most common
        bad_idx = sorted(
            [i for i, l in enumerate(Ls) if l != L_keep], reverse=True
        )
        for idx in bad_idx:
            mFFTs_list.pop(idx)
            fields_list.pop(idx)

    # ── build output arrays ───────────────────────────────────────────────
    fields = np.array(fields_list)
    mFFTs  = np.array(mFFTs_list).T      # shape: (n_freqs, n_fields)
    f      = np.fft.fftfreq(L_keep, d=dt)[: L_keep // 2]

    return fields, f, mFFTs


def get_absorption_curve(
    fields: np.ndarray,
    mFFTs: np.ndarray,
    f: np.ndarray,
    Fmeas: float = 5e9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the absorption vs. field at a fixed frequency.

    Returns (fields, absorption) where absorption = |mFFT| at Fmeas.
    """
    idx        = int(np.argmin(np.abs(f - Fmeas)))
    absorption = np.abs(mFFTs[idx, :])
    return fields, absorption


def get_mfft_at_field(
    fields: np.ndarray,
    mFFTs: np.ndarray,
    f: np.ndarray,
    Bstat: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract FFT(m(t)) at a fixed applied field value.

    Returns (f, |mFFT|) at the field closest to Bstat.
    """
    idx  = int(np.argmin(np.abs(fields - Bstat)))
    mFFT = np.abs(mFFTs[:, idx])
    return f, mFFT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_dir(value: str, name: str) -> str:
    v = value.strip().lower()
    if v not in ("x", "y", "z"):
        raise ValueError(f"{name} must be 'x', 'y', or 'z'  (got '{value}')")
    return v
