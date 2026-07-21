"""
Publication ("Origin") Plot Style
==================================
Centralised Matplotlib styling that mimics OriginLab's default
publication look, which most journals accept:

  * Arial / Helvetica sans-serif font
  * full black box (all four spines) with thicker lines
  * inward-pointing MAJOR and MINOR ticks on all four sides
  * no gridlines, white background
  * bold-ish axis labels

Usage
-----
    from gui.plot_style import set_origin_rcparams, style_axis

    set_origin_rcparams()          # once, at application startup
    ...
    ax.plot(x, y)
    style_axis(ax)                 # after plotting, per-axes refinement
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator


# Preferred publication fonts, in order — Matplotlib falls back gracefully
_FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def set_origin_rcparams() -> None:
    """Apply Origin-like defaults to Matplotlib's global rcParams."""
    mpl.rcParams.update({
        # ── fonts ──────────────────────────────────────────────────────
        "font.family":      "sans-serif",
        "font.sans-serif":  _FONT_STACK,
        "font.size":        12,
        "axes.titlesize":   13,
        "axes.labelsize":   13,
        "xtick.labelsize":  11,
        "ytick.labelsize":  11,
        "legend.fontsize":  11,
        "mathtext.fontset": "dejavusans",

        # ── frame / spines ─────────────────────────────────────────────
        "axes.linewidth":   1.4,
        "axes.edgecolor":   "black",
        "axes.grid":        False,

        # ── ticks: inward, on all four sides ───────────────────────────
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.top":        True,
        "ytick.right":      True,
        "xtick.major.size": 6.0,
        "ytick.major.size": 6.0,
        "xtick.minor.size": 3.0,
        "ytick.minor.size": 3.0,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        # ── lines / legend / figure ────────────────────────────────────
        "lines.linewidth":  1.6,
        "legend.frameon":   False,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor": "white",
        "savefig.dpi":      300,
    })


def style_axis(ax, minor: bool = True, box: bool = True) -> None:
    """
    Apply per-axes Origin styling after plotting.

    Parameters
    ----------
    ax    : Matplotlib Axes
    minor : add automatic minor tick locators (linear axes only)
    box   : ensure all four spines are visible with the frame linewidth
    """
    lw = mpl.rcParams["axes.linewidth"]

    if box:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(lw)

    # Minor ticks only make sense on linear scales; log axes place their own.
    if minor:
        if ax.get_xscale() == "linear":
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        if ax.get_yscale() == "linear":
            ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(which="both", direction="in",
                   top=True, right=True)
    ax.tick_params(which="major", length=6.0, width=lw)
    ax.tick_params(which="minor", length=3.0, width=1.0)
