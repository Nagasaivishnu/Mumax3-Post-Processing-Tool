"""
Mode Profile Processing
========================
All physics/signal-processing logic for spatial spin-wave mode profiling.
No GUI code lives here — edit freely without touching any UI.

Data convention
---------------
After loading, the raw dataset has shape:

    m_raw : (n_time, nz, ny, nx, 3)   float32

Axis mapping
    axis 0 → time
    axis 1 → z  (slowest spatial, matches OVF file ordering)
    axis 2 → y
    axis 3 → x  (fastest spatial)
    axis 4 → magnetisation component (0=Mx, 1=My, 2=Mz)

Averaging axis (applied after FFT on the spatial power array):
    avg_axis='Z'    → np.mean(..., axis=0) on (nz, ny, nx)  → (ny, nx)  → XY view
    avg_axis='Y'    → np.mean(..., axis=1)                   → (nz, nx)  → XZ view
    avg_axis='X'    → np.mean(..., axis=2)                   → (nz, ny)  → YZ view
    avg_axis='None' → central z slice                        → (ny, nx)  → XY view
"""

from __future__ import annotations
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import find_peaks as _scipy_find_peaks

from utils.ovf_reader import read_ovf

logger = logging.getLogger(__name__)

# Mapping: user-facing component name → array axis-4 index
COMPONENT_MAP = {"Mx": 0, "My": 1, "Mz": 2}

# Averaging direction → which spatial axis to reduce
AVG_AXIS_MAP = {"Z": 0, "Y": 1, "X": 2}   # axes of the (nz, ny, nx) sub-array

# Suggested view plane for each averaging direction
AVG_TO_VIEW = {"Z": "XY", "Y": "XZ", "X": "YZ", "None": "XY"}


# ---------------------------------------------------------------------------
# Dataset loading (OVF or NPY cache)
# ---------------------------------------------------------------------------

NPY_CACHE_FILENAME = "m_txyz.npy"


def load_dataset(
    sim_dir: str | Path,
    progress_cb: Callable[[int, int], None] | None = None,
    status_cb:   Callable[[str], None] | None = None,
) -> np.ndarray:
    """
    Load magnetisation snapshots from *sim_dir*.

    1. If ``m_txyz.npy`` exists in *sim_dir*: load it (fast path).
    2. Otherwise: read all ``m*.ovf`` files in parallel → save ``m_txyz.npy``.

    Parameters
    ----------
    sim_dir     : simulation output directory (contains table.txt + *.ovf)
    progress_cb : called as (n_done, n_total) while loading OVFs
    status_cb   : called with a human-readable status string

    Returns
    -------
    np.ndarray, shape (n_time, nz, ny, nx, 3), dtype float32
    """
    sim_dir  = Path(sim_dir).resolve()
    npy_path = sim_dir / NPY_CACHE_FILENAME

    def _status(msg: str) -> None:
        logger.info(msg)
        if status_cb:
            status_cb(msg)

    # ── fast path: npy cache ──────────────────────────────────────────
    if npy_path.exists():
        _status(f"Loading cache: {npy_path.name} …")
        data = np.load(npy_path)
        _status(f"Loaded from cache  –  shape {data.shape}")
        return data

    # ── slow path: read OVF files ─────────────────────────────────────
    ovf_files = sorted(glob(str(sim_dir / "m*.ovf")))
    if not ovf_files:
        raise FileNotFoundError(
            f"No m*.ovf files found in '{sim_dir}'.\n"
            "Make sure the path points to a MuMax3 output directory."
        )

    n = len(ovf_files)
    _status(f"Found {n} OVF files – loading in parallel …")

    results: list[np.ndarray | None] = [None] * n
    max_workers = min(4, os.cpu_count() or 1)

    def _load_one(args):
        idx, path = args
        return idx, read_ovf(path)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_load_one, (i, p)): i
            for i, p in enumerate(ovf_files)
        }
        for fut in as_completed(futures):
            idx, arr = fut.result()
            results[idx] = arr
            done += 1
            if progress_cb:
                progress_cb(done, n)
            _status(f"Loading OVF {done} / {n}")

    _status("Stacking arrays …")
    data = np.array(results, dtype=np.float32)   # (n_time, nz, ny, nx, 3)
    data = np.transpose(data, (0, 3, 2, 1, 4))   # ensure correct axis order
    del results

    _status(f"Saving cache → {npy_path.name} …")
    np.save(npy_path, data)
    _status(f"Done  –  shape {data.shape}")
    return data


# ---------------------------------------------------------------------------
# FFT processing
# ---------------------------------------------------------------------------

def compute_fft(
    m_raw: np.ndarray,
    component: int,
    dt: float,
    t_start: float,
    t_end: float,
) -> dict:
    """
    Extract one magnetisation component, apply a time window, and FFT.

    Parameters
    ----------
    m_raw     : (n_time, nz, ny, nx, 3)
    component : 0=Mx, 1=My, 2=Mz
    dt        : simulation saving interval [s]
    t_start   : start of analysis window [s]
    t_end     : end   of analysis window [s]

    Returns
    -------
    dict with keys:
        'f'     : positive frequency array [Hz]  shape (n_freq,)
        'P'     : spatial power             shape (n_freq, nz, ny, nx)
        'P_int' : spatially integrated power shape (n_freq,)
    """
    # Select component → (n_time, nz, ny, nx)
    m_t = m_raw[:, :, :, :, component].astype(np.float32)

    # Apply time window
    n_time = m_t.shape[0]
    t      = np.arange(n_time, dtype=np.float64) * dt
    mask   = (t >= t_start) & (t <= t_end)
    m_t    = m_t[mask]

    if m_t.shape[0] < 4:
        raise ValueError(
            f"Time window [{t_start:.2e}, {t_end:.2e}] s contains only "
            f"{m_t.shape[0]} samples — widen the window."
        )

    # Remove DC per spatial point
    m_t -= m_t.mean(axis=0, keepdims=True)

    # FFT along time axis
    m_f = np.fft.fft(m_t, axis=0)             # (n_time_win, nz, ny, nx)
    f   = np.fft.fftfreq(m_t.shape[0], d=dt)

    # Keep positive frequencies only
    pos = f >= 0
    f   = f[pos]
    m_f = m_f[pos]

    # Amplitude spectrum
    P     = np.abs(m_f)                       # (n_freq, nz, ny, nx)
    P_int = P.sum(axis=(1, 2, 3))             # (n_freq,)

    return {"f": f, "P": P, "P_int": P_int}


# ---------------------------------------------------------------------------
# FFT result caching
# ---------------------------------------------------------------------------
#
# After compute_fft, the result can be saved to an .npz file in the simulation
# directory.  The filename encodes the FFT identification parameters
# (component, dt, time window), so each distinct parameter set gets its own
# cache file:
#
#     fft_My_dt5e-12_ts3e-09_te2.5e-08.npz
#
# On the next "Compute FFT" with the same parameters, the cache is loaded
# instead of recomputing (unless the user forces a recompute).

FFT_CACHE_PREFIX = "fft"

# Reverse of COMPONENT_MAP: axis index → name
_COMPONENT_NAMES = {v: k for k, v in COMPONENT_MAP.items()}


def fft_cache_path(
    sim_dir: str | Path,
    component: int,
    dt: float,
    t_start: float,
    t_end: float,
) -> Path:
    """
    Build the cache-file path for one FFT parameter set.

    The name encodes component, dt, and the time window so that different
    parameter combinations never collide.
    """
    comp_name = _COMPONENT_NAMES.get(component, f"c{component}")
    fname = (
        f"{FFT_CACHE_PREFIX}_{comp_name}"
        f"_dt{dt:g}_ts{t_start:g}_te{t_end:g}.npz"
    )
    return Path(sim_dir) / fname


def save_fft_result(
    sim_dir: str | Path,
    component: int,
    dt: float,
    t_start: float,
    t_end: float,
    result: dict,
) -> Path:
    """
    Save a compute_fft result dict to the simulation directory.

    Returns the path of the written .npz file.
    """
    path = fft_cache_path(sim_dir, component, dt, t_start, t_end)
    np.savez(path, f=result["f"], P=result["P"], P_int=result["P_int"])
    logger.info("Saved FFT cache → %s", path)
    return path


def load_fft_result(path: str | Path) -> dict:
    """
    Load a previously saved FFT result.

    Returns the same dict structure as compute_fft:
    {'f', 'P', 'P_int'}.

    Raises
    ------
    ValueError  if the file does not contain the expected arrays.
    """
    path = Path(path)
    with np.load(path) as data:
        missing = [k for k in ("f", "P", "P_int") if k not in data.files]
        if missing:
            raise ValueError(
                f"FFT cache '{path.name}' is missing arrays: {missing}. "
                "Delete the file and recompute."
            )
        result = {"f": data["f"], "P": data["P"], "P_int": data["P_int"]}
    logger.info("Loaded FFT cache ← %s", path)
    return result


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def find_fmr_peaks(
    f: np.ndarray,
    P_int: np.ndarray,
    n_peaks: int,
    f_min_ghz: float,
    f_max_ghz: float,
) -> list[dict]:
    """
    Locate the strongest *n_peaks* peaks in the integrated FMR spectrum.

    Peaks are sorted strongest-first so Mode 1 = dominant resonance.

    Returns
    -------
    list of dicts, each with:
        'mode'   : 1-based mode index
        'f_peak' : resonance frequency [Hz]
        'pk_idx' : index into the full *f* / *P_int* arrays
    """
    # Restrict to user-specified frequency window
    mask  = (f >= f_min_ghz * 1e9) & (f <= f_max_ghz * 1e9)
    f_win = f[mask]
    P_win = P_int[mask]

    if len(P_win) < 3:
        raise ValueError(
            "Frequency window too narrow – fewer than 3 points. "
            "Increase the frequency range."
        )

    peak_locs, _ = _scipy_find_peaks(P_win)
    if len(peak_locs) == 0:
        raise ValueError(
            "No peaks found in the spectrum. "
            "Try widening the frequency range or adjusting the time window."
        )

    # Sort by amplitude (strongest last) then reverse for strongest-first
    peak_locs = peak_locs[np.argsort(P_win[peak_locs])][-n_peaks:][::-1]

    results = []
    for rank, local_idx in enumerate(peak_locs):
        # Map the windowed index back to the full array
        global_idx = int(np.where(f == f_win[local_idx])[0][0])
        results.append({
            "mode":   rank + 1,
            "f_peak": float(f[global_idx]),
            "pk_idx": global_idx,
        })
    return results


# ---------------------------------------------------------------------------
# Spatial profile extraction
# ---------------------------------------------------------------------------

def get_spatial_profile(
    P: np.ndarray,
    pk_idx: int,
    avg_axis: str,
) -> np.ndarray:
    """
    Extract the 2-D spatial power map for one resonant mode.

    Parameters
    ----------
    P        : (n_freq, nz, ny, nx)  from compute_fft
    pk_idx   : frequency index of the resonance peak
    avg_axis : 'Z' | 'Y' | 'X' | 'None'
               Direction to average over before displaying.
               'None' → central z-slice (no averaging).

    Returns
    -------
    2-D np.ndarray ready for imshow
    """
    sp = P[pk_idx]   # (nz, ny, nx)

    if avg_axis == "Z":
        return sp.mean(axis=0)      # (ny, nx)  → XY view
    elif avg_axis == "Y":
        return sp.mean(axis=1)      # (nz, nx)  → XZ view
    elif avg_axis == "X":
        return sp.mean(axis=2)      # (nz, ny)  → YZ view
    else:                           # "None" – central z slice
        return sp[sp.shape[0] // 2] # (ny, nx)  → XY view
