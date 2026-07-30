"""
OVF → Video
===========
Build (and cache) an MP4 movie from a numbered sequence of OVF snapshots.

Pipeline
--------
1. Find frames named with a leading number (0000.ovf, 0001.ovf, …).
2. Convert each to PNG with the external ``mumax3-convert -png`` tool.
3. Assemble the PNGs into an MP4 (imageio + bundled ffmpeg).
4. Delete the intermediate PNGs.
5. Cache the MP4 in the simulation directory as ``ovf_movie.mp4`` so a
   second request just re-opens it.

Requires
--------
* ``mumax3-convert`` on PATH  (ships with MuMax3)
* ``imageio`` and ``imageio-ffmpeg``  (pip)
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

VIDEO_CACHE_NAME = "ovf_movie.mp4"
_LEADING_NUM = re.compile(r"^(\d+)")
_CONVERT_CHUNK = 150          # OVF files per mumax3-convert call (arg-length safe)


# ---------------------------------------------------------------------------
# Discovery / paths
# ---------------------------------------------------------------------------

def find_ovf_frames(sim_dir: str | Path) -> list[Path]:
    """
    Return numbered ``*.ovf`` frames (name starts with digits) in *sim_dir*,
    sorted by their leading number.
    """
    sim_dir = Path(sim_dir)
    frames = [p for p in sim_dir.glob("*.ovf") if _LEADING_NUM.match(p.name)]
    return sorted(frames, key=lambda p: int(_LEADING_NUM.match(p.name).group(1)))


def video_cache_path(sim_dir: str | Path) -> Path:
    """Path of the cached movie for this directory."""
    return Path(sim_dir) / VIDEO_CACHE_NAME


def find_converter() -> str | None:
    """Locate the mumax3-convert executable, or None if not on PATH."""
    return shutil.which("mumax3-convert") or shutil.which("mumax3-convert.exe")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def make_video(
    sim_dir: str | Path,
    fps: int = 10,
    progress_cb: Callable[[int, int], None] | None = None,
    status_cb:   Callable[[str], None] | None = None,
    convert_exe: str | None = None,
) -> Path:
    """
    Convert the numbered OVF frames in *sim_dir* to an MP4 and cache it.

    Returns the path of the written .mp4.

    Raises
    ------
    FileNotFoundError  no numbered OVF frames found
    RuntimeError       mumax3-convert missing, or a conversion/encode failure
    ImportError        imageio not installed
    """
    sim_dir = Path(sim_dir)

    def _status(msg: str) -> None:
        logger.info(msg)
        if status_cb:
            status_cb(msg)

    frames = find_ovf_frames(sim_dir)
    if not frames:
        raise FileNotFoundError(
            "No numbered OVF frames (0000.ovf, 0001.ovf, …) found in:\n"
            f"{sim_dir}"
        )

    exe = convert_exe or find_converter()
    if not exe:
        raise RuntimeError(
            "'mumax3-convert' was not found on your PATH.\n"
            "It ships with MuMax3 — add the MuMax3 folder to PATH and retry."
        )

    # ── 1) OVF → PNG (chunked to stay under command-line length limits) ──
    _status(f"Converting {len(frames)} OVF files to PNG …")
    done = 0
    for chunk in _chunks(frames, _CONVERT_CHUNK):
        try:
            subprocess.run(
                [exe, "-png", "-arrows", "10", *[str(p) for p in chunk]],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
            raise RuntimeError(f"mumax3-convert failed:\n{err}") from exc
        done += len(chunk)
        if progress_cb:
            progress_cb(done, len(frames))

    pngs = [p.with_suffix(".png") for p in frames]
    missing = [p.name for p in pngs if not p.exists()]
    if missing:
        raise RuntimeError(
            "Expected PNG output was not produced for: "
            + ", ".join(missing[:5])
            + (" …" if len(missing) > 5 else "")
        )

    # ── 2) PNG → MP4 ────────────────────────────────────────────────────
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "imageio is not installed. Install it with:\n"
            "    pip install imageio imageio-ffmpeg"
        ) from exc

    out = video_cache_path(sim_dir)
    _status("Assembling video …")
    try:
        writer = imageio.get_writer(str(out), fps=fps, macro_block_size=None)
        try:
            for i, png in enumerate(pngs):
                writer.append_data(imageio.imread(str(png)))
                if progress_cb:
                    progress_cb(i + 1, len(pngs))
        finally:
            writer.close()
    except Exception as exc:
        raise RuntimeError(f"Could not encode the video: {exc}") from exc

    # ── 3) delete the intermediate PNGs (keep only the cached video) ────
    _status("Cleaning up PNG frames …")
    for png in pngs:
        try:
            png.unlink()
        except OSError:
            pass

    _status(f"Video ready: {out.name}")
    return out
