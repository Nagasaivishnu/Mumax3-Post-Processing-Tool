"""
Cluster / Low-Memory Loading and FFT
====================================
Drop-in, memory-efficient replacements for the two heavy routines in
``processing.mode_profile``:

    load_dataset      ->  load_dataset_lowmem
    compute_fft       ->  compute_fft_chunked

Why these exist
---------------
The GUI versions are fine on a laptop with a handful of OVF frames, but they
have two peak-memory problems that bite on large cluster runs:

1. ``load_dataset`` builds a Python ``list`` of N per-frame arrays, then calls
   ``np.array(results)`` (full copy #1), then ``np.transpose(...)`` which for a
   non-contiguous permutation forces another copy (copy #2).  Peak RSS is
   therefore roughly **2.5x** the final array size.  Here we preallocate the
   final array once and let each worker write its transposed frame straight
   into its slot, so peak is ~**1.0x**.

2. ``compute_fft`` runs ``np.fft.fft`` over the whole spatial grid at once.
   NumPy promotes float32 -> complex128, so the transient is
   ``16 * n_time_win * nz * ny * nx`` bytes -- 4x the float32 input, on top of
   the input.  Here the spatial grid is flattened and processed in chunks, and
   ``np.abs`` is taken per chunk into a preallocated float32 output.  Peak
   transient is bounded by ``chunk_points`` regardless of grid size.

Numerical results are identical to the GUI versions (same FFT, same window,
same DC removal, same axis ordering).  Only the allocation strategy differs.

.. note::
   **Axis order.**  ``utils.ovf_reader`` returns ``(nz, ny, nx, vdim)``, which
   is the native OVF layout (x fastest, z slowest).  Stacking frames therefore
   gives ``(T, nz, ny, nx, 3)`` -- exactly what ``mode_profile.load_dataset``
   promises in its docstring and what ``get_spatial_profile`` assumes.

   The now-deleted GUI additionally applied ``np.transpose(data, (0,3,2,1,4))``,
   producing ``(T, nx, ny, nz, 3)`` instead.  With that ordering ``avg_axis="Z"``
   reduces index 0, which holds **x**, so it averaged over the wrong axis and
   returned a ``(y, z)`` map where a ``(y, x)`` one was intended.  Invisible on
   a cubic grid; very visible on a thin film.

   ``axis_order="zyx"`` (the default) uses the native, correct order.
   ``axis_order="xyz"`` reproduces the old GUI transpose, and exists only so you
   can compare against results generated before this was fixed.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Callable

import numpy as np

from utils.ovf_reader import read_ovf
from processing.mode_profile import extract_component, NPY_CACHE_FILENAME

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory estimation (use this to size --mem in your SLURM script)
# ---------------------------------------------------------------------------

def probe_grid(sim_dir: str | Path) -> dict:
    """
    Read only the *header* of the first OVF file to learn the grid size.

    Cheap enough to run on the login node before submitting a job.

    Returns
    -------
    dict with keys: n_frames, nx, ny, nz, vdim
    """
    sim_dir = Path(sim_dir).resolve()
    ovf_files = sorted(glob(str(sim_dir / "m*.ovf")))
    if not ovf_files:
        raise FileNotFoundError(f"No m*.ovf files found in '{sim_dir}'.")

    header: dict[str, str] = {}
    with open(ovf_files[0], "rb") as fh:
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError(f"Unexpected EOF reading header of {ovf_files[0]}")
            line = raw.decode("latin-1", errors="replace").strip()
            if line.lower().startswith("# begin: data"):
                break
            if line.startswith("#") and ":" in line:
                key, _, val = line[1:].partition(":")
                header[key.strip().lower()] = val.strip()

    return {
        "n_frames": len(ovf_files),
        "nx":   int(header["xnodes"]),
        "ny":   int(header["ynodes"]),
        "nz":   int(header["znodes"]),
        "vdim": int(header.get("valuedim", 3)),
    }


def estimate_memory_gb(
    n_frames: int,
    nx: int,
    ny: int,
    nz: int,
    vdim: int = 3,
    chunk_points: int = 1 << 20,
    n_time_win: int | None = None,
) -> dict:
    """
    Estimate peak RAM for a mode-profile run, in GiB.

    Returns a dict of named contributions plus ``total`` and ``recommended``
    (total x1.35 headroom, rounded up), which is what you should hand to
    ``#SBATCH --mem``.
    """
    n_time_win = n_time_win or n_frames
    n_spatial  = nx * ny * nz
    GiB = 1024 ** 3

    raw     = n_frames * n_spatial * vdim * 4 / GiB          # m_raw float32
    comp    = n_time_win * n_spatial * 4 / GiB               # extracted component
    n_freq  = n_time_win // 2 + 1
    power   = n_freq * n_spatial * 4 / GiB                   # P float32 output
    # transient: one chunk promoted to complex128 + its float64 source
    transient = chunk_points * n_time_win * (16 + 8) / GiB

    total = raw + comp + power + transient
    return {
        "m_raw_gb":     round(raw, 3),
        "component_gb": round(comp, 3),
        "power_gb":     round(power, 3),
        "fft_chunk_gb": round(transient, 3),
        "total_gb":     round(total, 3),
        "recommended_gb": int(np.ceil(total * 1.35)) + 2,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset_lowmem(
    sim_dir: str | Path,
    workers: int | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
    mmap: bool = False,
    axis_order: str = "zyx",
    progress_cb: Callable[[int, int], None] | None = None,
    status_cb:   Callable[[str], None] | None = None,
) -> np.ndarray:
    """
    Load all ``m*.ovf`` frames from *sim_dir* into one preallocated array.

    Parameters
    ----------
    sim_dir     : MuMax3 output directory (contains table.txt + m*.ovf)
    workers     : parallel OVF readers.  Default ``min(16, cpu_count)``.
                  On a cluster node set this from ``$SLURM_CPUS_PER_TASK``.
    use_cache   : load ``m_txyz.npy`` if present instead of re-reading OVFs
    write_cache : save ``m_txyz.npy`` after a fresh OVF read
    mmap        : memory-map the .npy cache instead of reading it into RAM.
                  Lets you analyse arrays larger than RAM at the cost of
                  slower random access.  Only applies to the cache fast path.
    axis_order  : ``"zyx"`` (default) keeps native OVF order -> (T, nz, ny, nx, 3);
                  ``"xyz"`` reproduces the old GUI transpose -> (T, nx, ny, nz, 3),
                  for reproducing pre-fix results only. See the module docstring.

    Returns
    -------
    np.ndarray, float32
    """
    sim_dir  = Path(sim_dir).resolve()
    npy_path = sim_dir / NPY_CACHE_FILENAME

    def _status(msg: str) -> None:
        logger.info(msg)
        if status_cb:
            status_cb(msg)

    # ── fast path: existing .npy cache ────────────────────────────────
    if use_cache and npy_path.exists():
        _status(f"Loading cache: {npy_path.name} (mmap={mmap}) ...")
        data = np.load(npy_path, mmap_mode="r" if mmap else None)
        _status(f"Loaded from cache - shape {data.shape}, dtype {data.dtype}")
        return data

    # ── slow path: read OVF files ─────────────────────────────────────
    ovf_files = sorted(glob(str(sim_dir / "m*.ovf")))
    if not ovf_files:
        raise FileNotFoundError(
            f"No m*.ovf files found in '{sim_dir}'.\n"
            "Make sure the path points to a MuMax3 output directory."
        )

    n = len(ovf_files)
    if workers is None:
        workers = min(16, os.cpu_count() or 1)
    workers = max(1, int(workers))

    grid = probe_grid(sim_dir)
    nx, ny, nz, vdim = grid["nx"], grid["ny"], grid["nz"], grid["vdim"]

    if axis_order == "xyz":
        shape = (n, nx, ny, nz, vdim)
        perm  = (2, 1, 0, 3)      # (nz,ny,nx,v) -> (nx,ny,nz,v)
    elif axis_order == "zyx":
        shape = (n, nz, ny, nx, vdim)
        perm  = None
    else:
        raise ValueError(f"axis_order must be 'xyz' or 'zyx', got {axis_order!r}")

    nbytes_gb = np.prod(shape) * 4 / 1024 ** 3
    _status(
        f"Found {n} OVF files, grid {nx}x{ny}x{nz}x{vdim} -> "
        f"allocating {nbytes_gb:.2f} GiB, {workers} workers"
    )

    # Preallocate once.  Workers write disjoint slices, so no lock is needed.
    out = np.empty(shape, dtype=np.float32)

    def _load_into(args) -> int:
        idx, path = args
        arr = read_ovf(path)                       # (nz, ny, nx, vdim)
        if arr.shape != (nz, ny, nx, vdim):
            raise ValueError(
                f"Grid mismatch in '{path}': expected {(nz, ny, nx, vdim)}, "
                f"got {arr.shape}. All frames must share one grid."
            )
        out[idx] = np.transpose(arr, perm) if perm else arr
        return idx

    done = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_into, (i, p)): p
                   for i, p in enumerate(ovf_files)}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:                      # noqa: BLE001
                errors.append(f"{Path(futures[fut]).name}: {exc}")
            done += 1
            if progress_cb:
                progress_cb(done, n)
            if done % max(1, n // 20) == 0 or done == n:
                _status(f"Loading OVF {done} / {n}")

    if errors:
        raise IOError(
            f"{len(errors)} OVF file(s) failed to load:\n  " +
            "\n  ".join(errors[:10]) +
            ("\n  ..." if len(errors) > 10 else "")
        )

    if write_cache:
        _status(f"Saving cache -> {npy_path.name} ({nbytes_gb:.2f} GiB) ...")
        try:
            np.save(npy_path, out)
        except OSError as exc:
            logger.warning("Could not write cache (%s) - continuing without it", exc)

    _status(f"Done - shape {out.shape}")
    return out


# ---------------------------------------------------------------------------
# Chunked FFT
# ---------------------------------------------------------------------------

def compute_fft_chunked(
    m_raw: np.ndarray,
    component: str,
    dt: float,
    t_start: float,
    t_end: float,
    chunk_points: int = 1 << 20,
    status_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Bounded-memory equivalent of ``processing.mode_profile.compute_fft``.

    Identical maths (component extraction, time window, per-point DC removal,
    ``np.fft.fft`` along time, positive frequencies, ``np.abs``), but the
    spatial grid is flattened and transformed ``chunk_points`` columns at a
    time so the complex128 transient never exceeds
    ``chunk_points * n_time_win * 24`` bytes.

    Parameters
    ----------
    chunk_points : spatial points per FFT batch.  Larger = faster but more
                   transient RAM.  1<<20 (~1M points) costs ~2.4 GiB at
                   n_time_win = 100.

    Returns
    -------
    dict with 'f' (n_freq,), 'P' (n_freq, *spatial) float32, 'P_int' (n_freq,)
    """
    def _status(msg: str) -> None:
        logger.info(msg)
        if status_cb:
            status_cb(msg)

    spatial_shape = m_raw.shape[1:4]

    # ── component extraction, done in time-slabs to avoid a full copy ──
    n_time = m_raw.shape[0]
    t      = np.arange(n_time, dtype=np.float64) * dt
    mask   = (t >= t_start) & (t <= t_end)
    idx    = np.flatnonzero(mask)

    if idx.size < 4:
        raise ValueError(
            f"Time window [{t_start:.3e}, {t_end:.3e}] s contains only "
            f"{idx.size} samples - widen the window. "
            f"(dt={dt:.3e} s, {n_time} frames, "
            f"t_max={ (n_time - 1) * dt:.3e} s)"
        )

    n_win = int(idx.size)
    _status(f"Time window: {n_win} of {n_time} frames")

    n_spatial = int(np.prod(spatial_shape))
    m_t = np.empty((n_win, n_spatial), dtype=np.float32)
    for j, i in enumerate(idx):
        # extract_component on a single frame keeps the temporary tiny
        frame = extract_component(m_raw[i:i + 1], component)   # (1, nz, ny, nx)
        m_t[j] = frame.reshape(-1)

    # per-point DC removal (matches compute_fft's mean over the window)
    m_t -= m_t.mean(axis=0, keepdims=True)

    f_all = np.fft.fftfreq(n_win, d=dt)
    pos   = f_all >= 0
    f     = f_all[pos]
    n_freq = int(pos.sum())

    P = np.empty((n_freq, n_spatial), dtype=np.float32)

    n_chunks = int(np.ceil(n_spatial / chunk_points))
    _status(
        f"FFT: {n_spatial:,} spatial points in {n_chunks} chunk(s) "
        f"of {min(chunk_points, n_spatial):,}"
    )
    for c, start in enumerate(range(0, n_spatial, chunk_points), 1):
        stop = min(start + chunk_points, n_spatial)
        block = np.fft.fft(m_t[:, start:stop], axis=0)
        P[:, start:stop] = np.abs(block[pos]).astype(np.float32, copy=False)
        del block
        if c % max(1, n_chunks // 10) == 0 or c == n_chunks:
            _status(f"FFT chunk {c} / {n_chunks}")

    del m_t

    P = P.reshape((n_freq,) + tuple(spatial_shape))
    P_int = P.sum(axis=(1, 2, 3), dtype=np.float64)

    return {"f": f, "P": P, "P_int": P_int}
