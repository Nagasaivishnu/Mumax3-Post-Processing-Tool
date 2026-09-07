"""
Plot Bundle
===========
Data model + file format for the Plotter tab.

A *bundle* groups several plots together with common display settings so
they can be edited, saved, reopened, and exported to PowerPoint as a set.
No source table.txt is needed once a plot is in a bundle — the plot data
lives inside the bundle file.

File format
-----------
A bundle is stored as JSON with the extension ``.mmxbundle``::

    {
      "format": "mmxbundle", "version": 1,
      "description": "...",
      "common": {logx, logy, xmin, xmax, ymin, ymax, aspect},
      "plots": [ {title, x[], y[], x_label, y_label, peaks[]}, ... ]
    }

A registry file (``bundles/registry.json`` in the repo) lists every bundle
that has been saved, with its description, so the Plotter tab can suggest
existing bundles when loading.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

BUNDLE_EXT = ".mmxbundle"
FORMAT_TAG = "mmxbundle"


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bundles_dir() -> Path:
    d = _repo_root() / "bundles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def registry_path() -> Path:
    return bundles_dir() / "registry.json"


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def default_common() -> dict:
    return {
        "logx": False, "logy": False,
        "xmin": None, "xmax": None,
        "ymin": None, "ymax": None,
        "aspect": 1.4,          # width / height of each plot picture
        "xlabel": "",           # override X axis title (blank = per-plot)
        "ylabel": "",           # override Y axis title (blank = per-plot)
        "font":   "",           # font family (blank = default)
        "fontsize": 12.0,       # base font size (labels; title +1, ticks -1)
    }


def new_bundle(description: str = "") -> dict:
    return {
        "format": FORMAT_TAG, "version": 1,
        "description": description,
        "common": default_common(),
        "plots": [],
    }


def make_plot_item(title, x, y, x_label="", y_label="", peaks=None) -> dict:
    """Build one plot entry. *peaks* is a list of {'mode':int,'f':float(GHz)}."""
    return {
        "title":   title or "",
        "x":       [float(v) for v in x],
        "y":       [float(v) for v in y],
        "x_label": x_label,
        "y_label": y_label,
        "peaks":   list(peaks or []),
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_bundle(bundle: dict, path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() != BUNDLE_EXT:
        path = path.with_suffix(BUNDLE_EXT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    _update_registry(path, bundle)
    logger.info("Saved bundle → %s", path)
    return path


def load_bundle(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        b = json.load(f)
    if b.get("format") != FORMAT_TAG:
        raise ValueError(f"'{path.name}' is not a MuMax plot bundle.")
    common = default_common()
    common.update(b.get("common", {}))
    b["common"] = common
    b.setdefault("plots", [])
    b.setdefault("description", "")
    return b


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def read_registry() -> list[dict]:
    p = registry_path()
    if not p.exists():
        return []
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def write_registry(reg: list[dict]) -> None:
    with open(registry_path(), "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)


def _update_registry(path: Path, bundle: dict) -> None:
    reg = read_registry()
    rpath = str(Path(path).resolve())
    entry = {
        "name":        Path(path).stem,
        "path":        rpath,
        "description": bundle.get("description", ""),
        "n_plots":     len(bundle.get("plots", [])),
        "modified":    time.strftime("%Y-%m-%d %H:%M"),
    }
    reg = [e for e in reg if e.get("path") != rpath]
    reg.append(entry)
    write_registry(reg)
