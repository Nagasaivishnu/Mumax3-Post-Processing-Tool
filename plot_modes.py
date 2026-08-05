#!/usr/bin/env python3
"""
plot_modes.py -- plot a mode bundle on your own PC (Windows / macOS / Linux)
===========================================================================
This is the *other half* of the workflow. The cluster does the heavy lifting
and writes one file:

    <sim>_<component>_modes.npz

You download that file and run this script on it. Nothing from the cluster
project is needed here -- only NumPy and Matplotlib:

    pip install numpy matplotlib

Usage
-----
    python plot_modes.py sim.out_My_modes.npz
    python plot_modes.py sim.out_My_modes.npz --save
    python plot_modes.py sim.out_My_modes.npz --mode 2 --cmap viridis
    python plot_modes.py sim.out_My_modes.npz --info

On Windows, if `python` is not on your PATH, use the full path to python.exe,
or run it from the Anaconda Prompt.

What is in the .npz
-------------------
    f_hz            (n_freq,)            frequency axis, Hz
    f_ghz           (n_freq,)            same, GHz -- for convenience
    spectrum        (n_freq,)            spatially integrated FFT amplitude
    peak_freqs_hz   (n_peaks,)           resonance frequencies
    peak_freqs_ghz  (n_peaks,)           same, GHz
    peak_indices    (n_peaks,)           index into f_hz / spectrum
    mode_numbers    (n_peaks,)           1 = strongest resonance
    profiles        (n_peaks, ny, nx)    2-D spatial map per mode, oriented
    metadata_json   scalar string        every parameter used, as JSON
    power_full      (n_freq, ...)        ONLY if written with --npz-mode full

The bare minimum to get at your data, with no helper code at all:

    import numpy as np
    d = np.load("sim.out_My_modes.npz")
    f_ghz    = d["f_ghz"]
    spectrum = d["spectrum"]
    profiles = d["profiles"]          # profiles[0] is the strongest mode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required:  pip install matplotlib")


# ---------------------------------------------------------------------------

def load_bundle(path: str | Path) -> tuple[dict, dict]:
    """Return (arrays, metadata). Raises a clear error on the usual mistakes."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"file not found: {path}")
    if path.suffix != ".npz":
        sys.exit(f"expected a .npz bundle, got '{path.suffix}'. "
                 "Use the *_modes.npz written by `cli.py modes`.")

    d = np.load(path, allow_pickle=False)

    required = ("f_ghz", "spectrum", "profiles", "peak_freqs_ghz")
    missing = [k for k in required if k not in d.files]
    if missing:
        sys.exit(f"'{path.name}' is missing {missing}.\n"
                 f"It contains: {list(d.files)}\n"
                 "This may be a raw fft_*.npz cache rather than a mode bundle.")

    meta = {}
    if "metadata_json" in d.files:
        try:
            meta = json.loads(str(d["metadata_json"]))
        except (ValueError, TypeError):
            pass
    return {k: d[k] for k in d.files}, meta


def print_info(arrays: dict, meta: dict) -> None:
    print()
    print("  Parameters")
    for k, v in meta.items():
        print(f"    {k:<18} {v}")
    print()
    print("  Arrays")
    for k, v in arrays.items():
        if k == "metadata_json":
            continue
        print(f"    {k:<18} shape={str(v.shape):<22} dtype={v.dtype}")
    print()
    print("  Detected modes")
    for n, fq in zip(arrays["mode_numbers"], arrays["peak_freqs_ghz"]):
        print(f"    mode {int(n)}: {fq:.4f} GHz")
    print()
    if meta.get("axes_match") is False:
        print("  NOTE: --avg-axis "
              f"{meta.get('avg_axis_requested')} actually reduced the "
              f"'{meta.get('avg_axis_effective')}' axis, because axis_order="
              f"{meta.get('axis_order')} puts it at that index.")
        print(f"        Profile axes are (row={meta.get('profile_row_axis')}, "
              f"col={meta.get('profile_col_axis')}).")
        print("        This is the known transpose issue -- see the")
        print("        axis-order note in cluster/README_CLUSTER.md.")
        print()


# ---------------------------------------------------------------------------

def plot_all(arrays: dict, meta: dict, cmap: str, save: bool,
             outdir: Path, dpi: int) -> None:
    f_ghz    = arrays["f_ghz"]
    spectrum = arrays["spectrum"]
    profiles = arrays["profiles"]
    peaks    = arrays["peak_freqs_ghz"]
    modes    = arrays["mode_numbers"]
    name     = meta.get("sim_name", "simulation")
    comp     = meta.get("component", "M")
    view     = meta.get("view_plane", "XY")

    n = len(profiles)
    if n == 0:
        sys.exit("bundle contains no mode profiles")

    # Spectrum on top spanning the full width, mode maps in a grid below.
    ncols = min(3, n)
    nrows = 1 + int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(4.6 * ncols, 3.6 * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(f_ghz, spectrum, lw=1.4, color="#1f4e79")
    for m, fq in zip(modes, peaks):
        ax.axvline(fq, color="crimson", ls="--", lw=0.9, alpha=0.8)
        ax.annotate(f"{int(m)}", xy=(fq, ax.get_ylim()[1]),
                    xytext=(0, -12), textcoords="offset points",
                    ha="center", fontsize=9, color="crimson")
    fmin = meta.get("f_min_ghz", float(f_ghz.min()))
    fmax = meta.get("f_max_ghz", float(f_ghz.max()))
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Integrated FFT amplitude (arb. u.)")
    ax.set_title(f"{name} - {comp} spectrum")
    ax.tick_params(direction="in", top=True, right=True)

    # Axis names come from the bundle, not from an assumption. Which physical
    # axes survive the averaging depends on avg_axis AND axis_order, so
    # hardcoding "x" and "y" here would mislabel plots for most settings.
    row_ax = meta.get("profile_row_axis", "y")
    col_ax = meta.get("profile_col_axis", "x")

    for i, prof in enumerate(profiles):
        r, c = divmod(i, ncols)
        axm = fig.add_subplot(gs[1 + r, c])
        im = axm.imshow(prof, origin="lower", aspect="auto", cmap=cmap)
        fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04)
        axm.set_title(f"Mode {int(modes[i])} - {peaks[i]:.3f} GHz [{view}]",
                      fontsize=10)
        axm.set_xlabel(f"{col_ax} (cells)")
        axm.set_ylabel(f"{row_ax} (cells)")

    fig.suptitle(f"{name}  ({comp})", fontsize=13, y=0.995)

    if save:
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"{name}_{comp}_overview.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"wrote {p}")
    else:
        plt.show()


def plot_one(arrays: dict, meta: dict, mode: int, cmap: str, save: bool,
             outdir: Path, dpi: int) -> None:
    modes = list(arrays["mode_numbers"])
    if mode not in modes:
        sys.exit(f"mode {mode} not in this bundle. Available: {modes}")
    i = modes.index(mode)

    prof = arrays["profiles"][i]
    fq   = arrays["peak_freqs_ghz"][i]
    name = meta.get("sim_name", "simulation")
    comp = meta.get("component", "M")
    view = meta.get("view_plane", "XY")

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    im = ax.imshow(prof, origin="lower", aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax, label="FFT amplitude (arb. u.)")
    ax.set_title(f"{name} - mode {mode} - {fq:.4f} GHz [{view}]")
    ax.set_xlabel(f"{meta.get('profile_col_axis', 'x')} (cells)")
    ax.set_ylabel(f"{meta.get('profile_row_axis', 'y')} (cells)")
    ax.tick_params(direction="in", color="white")

    if save:
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"{name}_{comp}_mode{mode}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"wrote {p}")
    else:
        plt.show()


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot a MuMax3 mode bundle produced by cli.py modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("npz", help="the *_modes.npz file you downloaded")
    ap.add_argument("--mode", type=int, default=None,
                    help="plot only this mode number (1 = strongest)")
    ap.add_argument("--cmap", default="inferno", help="matplotlib colormap")
    ap.add_argument("--save", action="store_true",
                    help="write PNGs instead of opening a window")
    ap.add_argument("--outdir", default=".", help="where --save writes")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--info", action="store_true",
                    help="print the contents and parameters, then exit")
    args = ap.parse_args()

    arrays, meta = load_bundle(args.npz)

    if args.info:
        print_info(arrays, meta)
        return 0

    if args.mode is not None:
        plot_one(arrays, meta, args.mode, args.cmap, args.save,
                 Path(args.outdir), args.dpi)
    else:
        plot_all(arrays, meta, args.cmap, args.save,
                 Path(args.outdir), args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
