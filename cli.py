#!/usr/bin/env python3
"""
MuMax3 Post-Processing Tool - Headless CLI
==========================================
Batch entry point for running spatial FMR / spin-wave mode analysis on an HPC
compute node, where you have far more RAM and cores than on a laptop.

This project is **headless**. There is no GUI: the PyQt6 interface was removed,
along with its OVF video export. Nothing here imports Qt, and nothing needs a
display. The intended workflow is:

    compute on the cluster  ->  download one .npz  ->  plot on your own PC

The primary output of ``modes`` is therefore not a picture but a **download
bundle**: ``<sim>_<component>_modes.npz``, holding the FFT spectrum, the
detected resonance frequencies, and one 2-D spatial profile per mode, plus all
the parameters that produced them. It loads with NumPy alone -- see
``plot_modes.py`` for a copy-paste example that runs on Windows.

Usage
-----
    # 1. Cheap header-only probe -- run on the LOGIN node to size your job
    python3 cli.py probe --sim-dir ~/runs/sim.out

    # 2. The main event -- spatial mode profiles, inside an sbatch job
    python3 cli.py modes --sim-dir $TMPDIR/sim.out --outdir $TMPDIR/results \
        --component My --dt 5e-12 --t-start 3e-9 --t-end 25e-9 --n-peaks 5

    # ...or, when you only want the data to plot at home:
    python3 cli.py modes --sim-dir ... --outdir ... --dt 5e-12 \
        --no-plots --no-csv

    # 3. Secondary: table.txt analyses (much lighter than OVF work)
    python3 cli.py fmr --tables run1/table.txt --dt 5e-12 --outdir results
    python3 cli.py hysteresis --tables run*/table.txt \
        --x-col "B_extz (T)" --y-col "mz ()" --outdir results

    # 4. Or drive everything from one JSON file
    python3 cli.py run --config jobs.json
"""

from __future__ import annotations

# Headless rendering MUST be selected before pyplot is imported anywhere.
import matplotlib
matplotlib.use("Agg")

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project modules (all Qt-free)
from processing import mode_profile as mp
from processing import fmr as fmr_mod
from processing import hysteresis as hyst_mod
from processing.cluster_loader import (
    compute_fft_chunked,
    estimate_memory_gb,
    load_dataset_lowmem,
    probe_grid,
)
from processing.data_loader import load_table, LoadError
from export.csv_export import (
    build_heatmap_dataframe,
    build_slice_dataframe,
    export_dataframe,
)
from export.plot_style import set_origin_rcparams, style_axis  # pure matplotlib

log = logging.getLogger("mumax.cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Set by _setup_logging so cmd_* handlers can report where the log went.
LOG_PATHS: dict[str, Path | None] = {"run": None, "history": None}


def _iso_now() -> str:
    """
    ISO-8601 local time with a colon in the offset: 2026-08-04T20:26:02+08:00.

    Must match `date +%Y-%m-%dT%H:%M:%S%:z` in cluster/logging.sh exactly.
    time.strftime('%z') emits '+0800' without the colon, which would make the
    Python and bash lines in run_history.log sort and diff inconsistently.
    """
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _log_dir(args) -> Path:
    """
    Resolve where logs go, in priority order:
        --log-dir  >  $LOG_DIR (exported by the SLURM scripts)  >  <project>/logs
    """
    raw = getattr(args, "log_dir", None) or os.environ.get("LOG_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parent / "logs"


def _setup_logging(args) -> None:
    """
    Console + per-run file logging, plus one line in the shared history file.

    Timestamps use ISO-8601 with the local UTC offset, matching what
    cluster/logging.sh writes, so the bash and Python records of the same job
    interleave correctly when you sort them.
    """
    verbose = getattr(args, "verbose", False)
    level = logging.DEBUG if verbose else logging.INFO
    class _IsoFormatter(logging.Formatter):
        """Same timestamp shape as cluster/logging.sh, colon included."""
        def formatTime(self, record, datefmt=None):     # noqa: N802, ARG002
            from datetime import datetime
            return (datetime.fromtimestamp(record.created)
                    .astimezone().isoformat(timespec="seconds"))

    fmt = _IsoFormatter("%(asctime)s  %(levelname)-7s  %(name)s : %(message)s")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    if getattr(args, "no_log_file", False):
        return

    log_dir = _log_dir(args)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        root.warning("cannot create log dir %s (%s) - console logging only",
                     log_dir, exc)
        return

    # One file per run. Tag with the SLURM job (and array task) when present so
    # the Python log sits next to the matching bash trace file; fall back to PID
    # plus a timestamp for local runs, which have no job ID to key on.
    job = os.environ.get("SLURM_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job:
        tag = f"{job}_{task}" if task else job
    else:
        tag = f"{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}"

    run_path = log_dir / f"cli_{args.command}_{tag}.log"
    try:
        handler = logging.FileHandler(run_path, encoding="utf-8")
    except OSError as exc:
        root.warning("cannot open %s (%s) - console logging only", run_path, exc)
        return

    handler.setFormatter(fmt)
    handler.setLevel(logging.DEBUG)          # file always gets full detail
    root.addHandler(handler)

    LOG_PATHS["run"] = run_path
    LOG_PATHS["history"] = log_dir / "run_history.log"

    log.info("command   : %s", " ".join(_shell_quote(a) for a in sys.argv))
    log.info("cwd       : %s", Path.cwd())
    log.info("log file  : %s", run_path)


def _shell_quote(arg: str) -> str:
    """Quote an argv element only when it needs it, so logs stay readable."""
    import shlex
    return shlex.quote(arg)


def _history_line(state: str, args, rc: int = 0, dur: float = 0.0) -> None:
    """
    Append one tab-separated line to run_history.log, in the exact format
    cluster/logging.sh uses, so both halves of a job land in one file.

    Uses an exclusive flock where available: parallel array tasks all append to
    this file, and without the lock their lines can interleave mid-write.
    """
    path = LOG_PATHS["history"]
    if path is None:
        return

    job = os.environ.get("SLURM_JOB_ID", "local")
    if os.environ.get("SLURM_ARRAY_TASK_ID"):
        job += "." + os.environ["SLURM_ARRAY_TASK_ID"]

    fields = [
        _iso_now(),
        state,
        f"job={job}",
        f"host={os.environ.get('HOSTNAME') or _hostname()}",
        f"user={os.environ.get('USER', 'unknown')}",
        f"ctx=cli.{args.command}",
    ]
    if state in ("OK", "FAIL", "END"):
        fields += [f"rc={rc}", f"dur={dur:.0f}s"]
    fields.append("cmd=" + " ".join(_shell_quote(a) for a in sys.argv))

    line = "\t".join(fields) + "\n"
    try:
        with open(path, "a", encoding="utf-8") as fh:
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass                      # non-POSIX or NFS without locking
            fh.write(line)
    except OSError as exc:
        log.debug("could not append to history log: %s", exc)


def _hostname() -> str:
    import socket
    try:
        return socket.gethostname().split(".")[0]
    except OSError:
        return "unknown"


def _outdir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


# MuMax3 names its output directory after the .mx3 file, so a sweep is usually
# sweepA/sim.out, sweepB/sim.out, ... -- the leaf name is identical everywhere.
# Bundles named from the leaf alone would all be "sim.out_My_modes.npz" and
# collide the moment you download several into one folder.
_GENERIC_SIM_NAMES = {"sim.out", "output.out", "out", "sim", "output", "run.out"}


def bundle_stem(sim_dir: Path, override: str | None = None) -> str:
    """
    A filename stem that stays unique across a parameter sweep.

    Uses the simulation directory name, unless that name is a generic MuMax3
    default -- in which case the parent directory (which is what actually
    distinguishes the runs) is prepended.

        ~/runs/sweepA/sim.out   ->  sweepA_sim.out
        ~/runs/py_thickness.out ->  py_thickness.out

    Mirrored by sim_label() in cluster/config.sh. The SLURM scripts must compute
    this from the REAL simulation path and pass it as --bundle-name: they stage
    data onto node-local scratch first, so cli.py only ever sees
    /scratch/mumax_<jobid>/sim.out and would otherwise encode the job ID.
    """
    if override:
        return override
    name = sim_dir.name
    if name.lower() in _GENERIC_SIM_NAMES and sim_dir.parent.name:
        return f"{sim_dir.parent.name}_{name}"
    return name


def _resolve_outdir(args, base: Path) -> Path:
    """
    Decide where this run's outputs go.

    Default is ``<base>/results`` -- a folder inside the simulation directory
    itself, so every simulation carries its own analysis. That matters most for
    job arrays: 40 simulations produce 40 self-contained result folders with no
    chance of one overwriting another, and no central tree to keep in sync.

    An explicit ``--outdir`` always wins, which is what the SLURM scripts use
    when they redirect output to node-local scratch.
    """
    if getattr(args, "outdir", None):
        return _outdir(args.outdir)
    return _outdir(Path(base).expanduser().resolve() / args.results_subdir)


def _cpus() -> int:
    """Cores available to this job -- SLURM-aware, falls back to the machine."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val and val.isdigit():
            return int(val)
    try:
        return len(os.sched_getaffinity(0))       # respects cgroup pinning
    except AttributeError:
        return os.cpu_count() or 1


def _banner() -> None:
    log.info("=" * 68)
    log.info("MuMax3 Post-Processing - headless CLI")
    log.info("host=%s  job=%s  cpus=%d",
             os.uname().nodename if hasattr(os, "uname") else "?",
             os.environ.get("SLURM_JOB_ID", "-"), _cpus())
    log.info("=" * 68)


def _save_fig(fig, path: Path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def _human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if nbytes < 1024 or unit == "TiB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TiB"


# ---------------------------------------------------------------------------
# The download bundle
# ---------------------------------------------------------------------------
#
# This is the file you actually carry off the cluster. Design constraints:
#
#   * Small enough to scp over a VPN. The full power array P is
#     (n_freq, nx, ny, nz) float32 -- gigabytes. The 2-D profiles at just the
#     resonant peaks are a few MB, which is all you need to plot modes.
#   * Loadable on a machine that does not have this codebase. Only NumPy is
#     required; array names are spelled out and units are in the names.
#   * Self-describing. Every parameter that affects the numbers is stored in
#     the file, so a .npz found six months later is still interpretable.

def profile_axes(args) -> dict:
    """
    Work out what the two axes of a mode profile actually *are*.

    This is not cosmetic. ``get_spatial_profile`` reduces a fixed axis index
    (0 for 'Z', 1 for 'Y', 2 for 'X'), but which physical axis sits at that
    index depends on ``axis_order``:

        axis_order='xyz'  ->  P spatial dims are (x, y, z)
        axis_order='zyx'  ->  P spatial dims are (z, y, x)

    With the GUI's default 'xyz' ordering, asking for ``--avg-axis Z`` reduces
    index 0, which is **x**, not z. That is the long-standing transpose issue
    described in cluster/README_CLUSTER.md. Rather than silently mislabelling
    the plot, we record what was requested *and* what actually happened, so a
    bundle is honest about its own contents.
    """
    dims = ("x", "y", "z") if args.axis_order == "xyz" else ("z", "y", "x")
    reduce_idx = {"Z": 0, "Y": 1, "X": 2, "None": 0}[args.avg_axis]
    remaining = [d for i, d in enumerate(dims) if i != reduce_idx]

    row_axis, col_axis = remaining[0], remaining[1]

    # apply_orientation rotates before the array is stored. A 90 or 270 degree
    # rotation swaps rows and columns; 180 does not.
    if int(args.rotate) % 360 in (90, 270):
        row_axis, col_axis = col_axis, row_axis

    return {
        "avg_axis_requested": args.avg_axis,
        "avg_axis_effective": dims[reduce_idx],
        "profile_row_axis":   row_axis,
        "profile_col_axis":   col_axis,
        "axes_match":         (args.avg_axis == "None"
                               or dims[reduce_idx] == args.avg_axis.lower()),
    }


def save_download_bundle(
    path: Path,
    f: np.ndarray,
    P_int: np.ndarray,
    peaks: list[dict],
    profiles: list[np.ndarray],
    args,
    sim_dir: Path,
    grid_shape: tuple,
    full_P: np.ndarray | None = None,
) -> Path:
    """Write the compact (or full) .npz. Returns the path written."""
    axes = profile_axes(args)
    if not axes["axes_match"]:
        log.warning(
            "--avg-axis %s reduced the '%s' axis, because axis_order=%s puts "
            "%s at that index. Profile axes are (%s, %s). See the axis-order "
            "note in cluster/README_CLUSTER.md.",
            args.avg_axis, axes["avg_axis_effective"], args.axis_order,
            axes["avg_axis_effective"],
            axes["profile_row_axis"], axes["profile_col_axis"])

    meta = {
        "component":      args.component,
        "dt_s":           args.dt,
        "t_start_s":      args.t_start,
        "t_end_s":        args.t_end,
        "avg_axis":       args.avg_axis,
        "view_plane":     (axes["profile_col_axis"] + axes["profile_row_axis"]).upper(),
        **axes,
        "axis_order":     args.axis_order,
        "rotation_cw_deg": args.rotate,
        "flip_h":         bool(args.flip_h),
        "flip_v":         bool(args.flip_v),
        "f_min_ghz":      args.f_min,
        "f_max_ghz":      args.f_max,
        "grid_shape":     list(grid_shape),
        "sim_dir":        str(sim_dir),
        "sim_name":       sim_dir.name,
        "sim_label":      bundle_stem(sim_dir, getattr(args, "bundle_name", None)),
        "created_utc":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slurm_job_id":   os.environ.get("SLURM_JOB_ID", ""),
        "host":           _hostname(),
        "tool_version":   "mumax-postproc-cli/headless",
    }

    arrays = {
        # Frequency axis, shared by spectrum and peaks.
        "f_hz":            f.astype(np.float64),
        "f_ghz":           (f / 1e9).astype(np.float64),
        # Spatially integrated FFT amplitude -- the FMR spectrum.
        "spectrum":        P_int.astype(np.float64),
        # Detected resonances.
        "peak_freqs_hz":   np.array([p["f_peak"] for p in peaks], dtype=np.float64),
        "peak_freqs_ghz":  np.array([p["f_peak"] / 1e9 for p in peaks], dtype=np.float64),
        "peak_indices":    np.array([p["pk_idx"] for p in peaks], dtype=np.int64),
        "mode_numbers":    np.array([p["mode"] for p in peaks], dtype=np.int64),
        # (n_peaks, ny, nx) -- one 2-D spatial map per mode, already oriented.
        "profiles":        (np.stack(profiles).astype(np.float32)
                            if profiles else np.zeros((0, 0, 0), np.float32)),
        # JSON rather than many 0-d string arrays: one thing to read, and it
        # survives round-tripping through np.load without dtype surprises.
        "metadata_json":   np.array(json.dumps(meta, indent=2)),
    }

    if full_P is not None:
        arrays["power_full"] = full_P          # (n_freq, *grid) float32

    np.savez_compressed(path, **arrays)
    size = path.stat().st_size
    log.info("wrote %s  (%s)", path, _human(size))
    return path


# ---------------------------------------------------------------------------
# Sub-command: probe
# ---------------------------------------------------------------------------

def cmd_probe(args) -> int:
    grid = probe_grid(args.sim_dir)
    est  = estimate_memory_gb(
        grid["n_frames"], grid["nx"], grid["ny"], grid["nz"],
        grid["vdim"], chunk_points=args.chunk_points,
    )
    print()
    print(f"  Simulation dir : {Path(args.sim_dir).resolve()}")
    print(f"  OVF frames     : {grid['n_frames']}")
    print(f"  Grid           : {grid['nx']} x {grid['ny']} x {grid['nz']} "
          f"(vdim {grid['vdim']})")
    print(f"  Spatial points : {grid['nx'] * grid['ny'] * grid['nz']:,}")
    print()
    print("  Estimated peak memory (GiB)")
    for k in ("m_raw_gb", "component_gb", "power_gb", "fft_chunk_gb", "total_gb"):
        print(f"    {k:<16} {est[k]:>10.3f}")
    print()
    print(f"  ==> put this in your SLURM script:  #SBATCH --mem={est['recommended_gb']}G")
    print()
    if args.json:
        Path(args.json).write_text(json.dumps({**grid, **est}, indent=2))
        log.info("wrote %s", args.json)
    return 0


# ---------------------------------------------------------------------------
# Sub-command: modes  (spatial mode profiles from OVF files)
# ---------------------------------------------------------------------------

def cmd_modes(args) -> int:
    _banner()
    sim_dir = Path(args.sim_dir).expanduser().resolve()
    out = _resolve_outdir(args, sim_dir)
    log.info("simulation : %s", sim_dir)
    log.info("results    : %s", out)
    t0 = time.time()

    # ── 1. load ───────────────────────────────────────────────────────
    cache = mp.fft_cache_path(sim_dir, args.component, args.dt,
                              args.t_start, args.t_end)
    if cache.exists() and not args.force:
        log.info("Reusing FFT cache: %s", cache.name)
        result = mp.load_fft_result(cache)
    else:
        m_raw = load_dataset_lowmem(
            sim_dir,
            workers=args.workers or _cpus(),
            use_cache=not args.no_npy_cache,
            write_cache=not args.no_npy_cache,
            mmap=args.mmap,
            axis_order=args.axis_order,
        )
        log.info("loaded m_raw shape=%s dtype=%s  (%.1f s)",
                 m_raw.shape, m_raw.dtype, time.time() - t0)

        # ── 2. FFT ────────────────────────────────────────────────────
        result = compute_fft_chunked(
            m_raw, args.component, args.dt, args.t_start, args.t_end,
            chunk_points=args.chunk_points,
        )
        del m_raw
        # The on-disk FFT cache holds the FULL power array, which is the big
        # one. Worth keeping when you expect to rerun with different peak or
        # orientation settings; skip it with --no-fft-cache when disk is tight.
        if not args.no_fft_cache:
            try:
                mp.save_fft_result(sim_dir, args.component, args.dt,
                                   args.t_start, args.t_end, result)
            except OSError as exc:
                log.warning("could not write FFT cache: %s", exc)

    f, P, P_int = result["f"], result["P"], result["P_int"]
    log.info("FFT done: %d frequencies, P shape %s  (%.1f s total)",
             len(f), P.shape, time.time() - t0)

    make_plots = not args.no_plots

    # Every output is component-tagged. All simulations now share one results
    # folder per sim dir, so running --component My then --component Mz would
    # otherwise silently overwrite the first run's spectrum and peaks.
    tag0 = args.component

    # ── 3. integrated spectrum ────────────────────────────────────────
    export_dataframe(
        pd.DataFrame({"Frequency_Hz": f, "Frequency_GHz": f / 1e9,
                      "Integrated_Power": P_int}),
        out / f"{tag0}_spectrum.csv",
    )
    if make_plots:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(f / 1e9, P_int)
        ax.set_xlim(args.f_min, args.f_max)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Integrated FFT power (arb. u.)")
        ax.set_title(f"{args.component} spectrum - {sim_dir.name}")
        style_axis(ax)
        _save_fig(fig, out / f"{tag0}_spectrum.png", args.dpi)

    # ── 4. peaks ──────────────────────────────────────────────────────
    peaks = mp.find_fmr_peaks(f, P_int, args.n_peaks, args.f_min, args.f_max)
    export_dataframe(
        pd.DataFrame([{"Mode": p["mode"],
                       "f_peak_Hz": p["f_peak"],
                       "f_peak_GHz": p["f_peak"] / 1e9,
                       "Integrated_Power": float(P_int[p["pk_idx"]])}
                      for p in peaks]),
        out / f"{tag0}_peaks.csv",
    )
    for p in peaks:
        log.info("mode %d : %.4f GHz", p["mode"], p["f_peak"] / 1e9)

    # ── 5. spatial profiles ───────────────────────────────────────────
    images: list[Path] = []
    profiles: list[np.ndarray] = []
    for p in peaks:
        prof = mp.get_spatial_profile(P, p["pk_idx"], args.avg_axis)
        prof = mp.apply_orientation(prof, args.rotate, args.flip_h, args.flip_v)
        profiles.append(prof)

        tag = f"{tag0}_mode{p['mode']}_{p['f_peak'] / 1e9:.3f}GHz"
        if not args.no_csv:
            np.savetxt(out / f"{tag}_profile.csv", prof, delimiter=",")

        if make_plots:
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            im = ax.imshow(prof, origin="lower", aspect="auto", cmap=args.cmap)
            fig.colorbar(im, ax=ax, label="FFT amplitude (arb. u.)")
            ax.set_title(f"Mode {p['mode']} - {p['f_peak'] / 1e9:.3f} GHz "
                         f"[{mp.AVG_TO_VIEW.get(args.avg_axis, 'XY')}]")
            ax.set_xlabel("x (cells)")
            ax.set_ylabel("y (cells)")
            img = out / f"{tag}_profile.png"
            _save_fig(fig, img, args.dpi)
            images.append(img)

    # ── 6. the download bundle ────────────────────────────────────────
    if args.npz_mode != "none":
        stem = bundle_stem(sim_dir, args.bundle_name)
        bundle = out / f"{stem}_{args.component}_modes.npz"
        include_full = args.npz_mode in ("full", "both")
        if include_full:
            gib = P.nbytes / 1024 ** 3
            log.info("including the full power array in the bundle (%.2f GiB "
                     "uncompressed) - this is what makes the file big", gib)
        save_download_bundle(
            bundle, f, P_int, peaks, profiles, args, sim_dir,
            grid_shape=P.shape[1:],
            full_P=P if include_full else None,
        )
        log.info("")
        log.info("DOWNLOAD THIS FILE and plot it on your own machine:")
        log.info("    %s", bundle)
        log.info("    scp <user>@<cluster>:%s .", bundle)
        log.info("Load it with NumPy alone - see plot_modes.py, or the")
        log.info("'Plotting on your own PC' section of cluster/README_CLUSTER.md")
        log.info("")

    # ── 7. optional PowerPoint ────────────────────────────────────────
    if args.pptx and make_plots:
        try:
            from export.ppt_export import append_images_slide
            append_images_slide(out / args.pptx,
                                [out / f"{tag0}_spectrum.png", *images],
                                title=f"{sim_dir.name} - {args.component}")
            log.info("wrote %s", out / args.pptx)
        except ImportError as exc:
            log.warning("skipping pptx (%s). pip install python-pptx pillow", exc)
    elif args.pptx and not make_plots:
        log.warning("--pptx ignored: it needs the PNGs that --no-plots skipped")

    log.info("modes complete in %.1f s -> %s", time.time() - t0, out)
    return 0


# ---------------------------------------------------------------------------
# Sub-command: fmr  (susceptibility from table.txt)
# ---------------------------------------------------------------------------

def cmd_fmr(args) -> int:
    _banner()
    t0 = time.time()
    rc = 0

    for table in args.tables:
        table = Path(table).expanduser().resolve()
        label = args.label or table.parent.name or table.stem
        # Each table gets its own results folder next to it, so sweeping many
        # runs never has one dataset's output land on another's.
        out = _resolve_outdir(args, table.parent)
        log.info("table      : %s", table)
        log.info("results    : %s", out)
        try:
            df = load_table(table)
        except LoadError as exc:
            log.error("skipping %s: %s", table, exc)
            rc = 1
            continue

        try:
            fields, f, mFFTs = fmr_mod.calc_susceptibility(
                df, args.dt,
                interpolate=args.interpolate,
                BiasFieldDir=args.bias_dir,
                MWFieldDir=args.mw_dir,
            )
        except (KeyError, ValueError) as exc:
            log.error("skipping %s: %s", table, exc)
            rc = 1
            continue

        mag = np.abs(mFFTs)
        log.info("%s: %d fields x %d frequencies", label, len(fields), len(f))

        export_dataframe(build_heatmap_dataframe(fields, f, mag, label),
                         out / f"{label}_fmr_heatmap.csv")

        # heatmap image
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        extent = [fields.min(), fields.max(), f.min() / 1e9, f.max() / 1e9]
        im = ax.imshow(mag, origin="lower", aspect="auto",
                       extent=extent, cmap=args.cmap)
        fig.colorbar(im, ax=ax, label=r"$|\chi|$ (arb. u.)")
        ax.set_ylim(args.f_min, args.f_max)
        ax.set_xlabel("Bias field (T)")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_title(f"FMR - {label}")
        _save_fig(fig, out / f"{label}_fmr_heatmap.png", args.dpi)

        # fixed-frequency slice (absorption vs field)
        if args.f_meas is not None:
            fld, absorp = fmr_mod.get_absorption_curve(fields, mFFTs, f,
                                                       args.f_meas * 1e9)
            export_dataframe(
                build_slice_dataframe(fld, [(absorp, label)], "Field_T"),
                out / f"{label}_absorption_{args.f_meas:g}GHz.csv")
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.plot(fld, absorp, marker="o", ms=3)
            ax.set_xlabel("Bias field (T)")
            ax.set_ylabel(r"$|\chi|$ (arb. u.)")
            ax.set_title(f"{label} - absorption at {args.f_meas:g} GHz")
            style_axis(ax)
            _save_fig(fig, out / f"{label}_absorption_{args.f_meas:g}GHz.png",
                      args.dpi)

        # fixed-field slice (spectrum at one B)
        if args.b_stat is not None:
            freqs, spec = fmr_mod.get_mfft_at_field(fields, mFFTs, f, args.b_stat)
            export_dataframe(
                build_slice_dataframe(freqs / 1e9, [(spec, label)],
                                      "Frequency_GHz"),
                out / f"{label}_spectrum_{args.b_stat:g}T.csv")
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.plot(freqs / 1e9, spec)
            ax.set_xlim(args.f_min, args.f_max)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel(r"$|\chi|$ (arb. u.)")
            ax.set_title(f"{label} - spectrum at {args.b_stat:g} T")
            style_axis(ax)
            _save_fig(fig, out / f"{label}_spectrum_{args.b_stat:g}T.png",
                      args.dpi)

    log.info("fmr complete in %.1f s", time.time() - t0)
    return rc


# ---------------------------------------------------------------------------
# Sub-command: hysteresis
# ---------------------------------------------------------------------------

def cmd_hysteresis(args) -> int:
    _banner()
    datasets: list[tuple[np.ndarray, np.ndarray, str]] = []
    rc = 0

    # Unlike modes/fmr, hysteresis deliberately MERGES several runs into one
    # overlay, so there is no single "respective" simulation. Output goes with
    # the first table listed; pass --outdir to put it somewhere neutral.
    first_parent = Path(args.tables[0]).expanduser().resolve().parent
    out = _resolve_outdir(args, first_parent)
    if len(args.tables) > 1 and not args.outdir:
        log.info("merging %d datasets -> writing to the first one's results "
                 "folder: %s", len(args.tables), out)
    log.info("results    : %s", out)

    for table in args.tables:
        table = Path(table).expanduser().resolve()
        label = table.parent.name or table.stem
        try:
            df = load_table(table)
        except LoadError as exc:
            log.error("skipping %s: %s", table, exc)
            rc = 1
            continue
        missing = [c for c in (args.x_col, args.y_col) if c not in df.columns]
        if missing:
            log.error("skipping %s: missing column(s) %s. Available: %s",
                      table, missing, list(df.columns))
            rc = 1
            continue
        x, y = hyst_mod.extract_xy(df, args.x_col, args.y_col)
        datasets.append((x, y, label))
        log.info("%s: %d points", label, len(x))

    if not datasets:
        log.error("no usable datasets")
        return 1

    merged = hyst_mod.merge_datasets(datasets, args.x_col)
    export_dataframe(merged, out / "hysteresis_merged.csv")

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for x, y, label in datasets:
        ax.plot(x, y, marker=args.marker if args.marker != "none" else None,
                ms=3, label=label)
    ax.set_xlabel(args.x_col)
    ax.set_ylabel(args.y_col)
    if args.log_x:
        ax.set_xscale("log")
    if args.log_y:
        ax.set_yscale("log")
    if len(datasets) > 1:
        ax.legend()
    style_axis(ax)
    _save_fig(fig, out / "hysteresis.png", args.dpi)

    log.info("hysteresis complete -> %s", out)
    return rc


# ---------------------------------------------------------------------------
# Sub-command: run  (JSON-driven, multiple analyses in one job)
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    cfg = json.loads(Path(args.config).expanduser().read_text())
    jobs = cfg if isinstance(cfg, list) else cfg.get("jobs", [])
    if not jobs:
        log.error("config contains no jobs")
        return 1

    parser = build_parser()
    rc = 0
    for i, job in enumerate(jobs, 1):
        kind = job.pop("command", None)
        if kind not in ("modes", "fmr", "hysteresis"):
            log.error("job %d: bad or missing 'command' (%r)", i, kind)
            rc = 1
            continue
        argv = [kind]
        for key, val in job.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(val, bool):
                if val:
                    argv.append(flag)
            elif isinstance(val, (list, tuple)):
                argv.append(flag)
                argv.extend(str(v) for v in val)
            else:
                argv.extend([flag, str(val)])
        pretty = " ".join(_shell_quote(a) for a in argv)
        log.info("--- job %d/%d: cli.py %s ---", i, len(jobs), pretty)
        t0 = time.time()
        try:
            sub = parser.parse_args(argv)
            # Child jobs share the parent's handlers, so their output lands in
            # the same run log -- no second _setup_logging call.
            job_rc = int(sub.func(sub) or 0)
            rc |= job_rc
            log.info("--- job %d/%d finished rc=%d in %.1fs ---",
                     i, len(jobs), job_rc, time.time() - t0)
        except SystemExit as exc:
            log.error("job %d: bad arguments (%s)", i, exc)
            rc = 1
        except Exception:                                  # noqa: BLE001
            log.exception("job %d failed after %.1fs", i, time.time() - t0)
            rc = 1
            if not args.keep_going:
                return rc
    return rc


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="Headless MuMax3 post-processing for SLURM clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def logging_args(sp):
        """Flags every sub-command shares for command logging."""
        g = sp.add_argument_group("logging")
        g.add_argument("-v", "--verbose", action="store_true",
                       help="DEBUG level on the console (the file always gets it)")
        g.add_argument("--log-dir", default=None,
                       help="where to write logs "
                            "(default: $LOG_DIR, else <project>/logs)")
        g.add_argument("--no-log-file", action="store_true",
                       help="console only; do not write any log file")
        return sp

    def common(sp):
        sp.add_argument("--outdir", default=None,
                        help="explicit output directory. Default: a 'results' "
                             "folder inside the simulation directory, so each "
                             "simulation carries its own analysis")
        sp.add_argument("--results-subdir", default="results",
                        help="name of that folder inside the simulation dir")
        sp.add_argument("--dpi", type=int, default=300, help="figure DPI")
        sp.add_argument("--cmap", default="inferno", help="matplotlib colormap")
        return logging_args(sp)

    # -- probe -------------------------------------------------------
    sp = sub.add_parser("probe", help="report grid size + recommended --mem")
    sp.add_argument("--sim-dir", required=True)
    sp.add_argument("--chunk-points", type=int, default=1 << 20)
    sp.add_argument("--json", help="also write the report to this JSON file")
    logging_args(sp)
    sp.set_defaults(func=cmd_probe)

    # -- modes -------------------------------------------------------
    sp = common(sub.add_parser("modes", help="spatial mode profiles from OVF files"))
    sp.add_argument("--sim-dir", required=True, help="MuMax3 .out directory")
    sp.add_argument("--component", default="My", choices=mp.COMPONENT_ORDER)
    sp.add_argument("--dt", type=float, required=True, help="saving interval [s]")
    sp.add_argument("--t-start", type=float, default=0.0, help="window start [s]")
    sp.add_argument("--t-end", type=float, default=1e9, help="window end [s]")
    sp.add_argument("--n-peaks", type=int, default=5)
    sp.add_argument("--f-min", type=float, default=0.0, help="min frequency [GHz]")
    sp.add_argument("--f-max", type=float, default=50.0, help="max frequency [GHz]")
    sp.add_argument("--avg-axis", default="Z", choices=["Z", "Y", "X", "None"])
    sp.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    sp.add_argument("--flip-h", action="store_true")
    sp.add_argument("--flip-v", action="store_true")
    sp.add_argument("--workers", type=int, default=0,
                    help="OVF reader threads (0 = auto from SLURM_CPUS_PER_TASK)")
    sp.add_argument("--chunk-points", type=int, default=1 << 20,
                    help="spatial points per FFT batch (lower = less RAM)")
    sp.add_argument("--axis-order", default="zyx", choices=["zyx", "xyz"],
                    help="'zyx' = native OVF order (z,y,x), which is what "
                         "get_spatial_profile assumes and what makes "
                         "--avg-axis mean what it says. 'xyz' reproduces the "
                         "deleted GUI's transpose, for comparing against old "
                         "results only")
    sp.add_argument("--mmap", action="store_true",
                    help="memory-map m_txyz.npy instead of reading into RAM")
    sp.add_argument("--no-npy-cache", action="store_true",
                    help="do not read or write m_txyz.npy")
    sp.add_argument("--force", action="store_true",
                    help="recompute even if an FFT cache exists")
    sp.add_argument("--no-fft-cache", action="store_true",
                    help="do not write the large fft_*.npz cache next to the "
                         "simulation (saves disk; makes reruns slow)")
    sp.add_argument("--pptx", nargs="?", const="report.pptx", default=None,
                    help="also append all figures to this .pptx")

    g = sp.add_argument_group(
        "download bundle",
        "The .npz you carry off the cluster and plot on your own PC.")
    g.add_argument("--npz-mode", default="compact",
                   choices=["compact", "full", "both", "none"],
                   help="compact = spectrum + peaks + 2-D mode profiles "
                        "(a few MB, all you need to plot modes); "
                        "full = also embed the entire power array (GB); "
                        "none = skip the bundle")
    g.add_argument("--no-plots", action="store_true",
                   help="skip all PNG rendering - use when you plot on your "
                        "own machine from the .npz")
    g.add_argument("--no-csv", action="store_true",
                   help="skip the per-mode profile CSVs (they duplicate what "
                        "is already in the .npz)")
    g.add_argument("--bundle-name", default=None,
                   help="filename stem for the .npz. Default derives a stem "
                        "that stays unique across a sweep, prepending the "
                        "parent dir when the sim dir is a generic 'sim.out'")
    sp.set_defaults(func=cmd_modes)

    # -- fmr ---------------------------------------------------------
    sp = common(sub.add_parser("fmr", help="susceptibility heatmap from table.txt"))
    sp.add_argument("--tables", nargs="+", required=True)
    sp.add_argument("--dt", type=float, required=True, help="saving interval [s]")
    sp.add_argument("--bias-dir", default="y", choices=["x", "y", "z"])
    sp.add_argument("--mw-dir", default="x", choices=["x", "y", "z"])
    sp.add_argument("--interpolate", action="store_true")
    sp.add_argument("--label", default=None, help="override dataset label")
    sp.add_argument("--f-min", type=float, default=0.0, help="plot min [GHz]")
    sp.add_argument("--f-max", type=float, default=50.0, help="plot max [GHz]")
    sp.add_argument("--f-meas", type=float, default=None,
                    help="fixed-frequency slice [GHz]")
    sp.add_argument("--b-stat", type=float, default=None,
                    help="fixed-field slice [T]")
    sp.set_defaults(func=cmd_fmr)

    # -- hysteresis --------------------------------------------------
    sp = common(sub.add_parser("hysteresis", help="overlay + merged CSV export"))
    sp.add_argument("--tables", nargs="+", required=True)
    sp.add_argument("--x-col", required=True)
    sp.add_argument("--y-col", required=True)
    sp.add_argument("--log-x", action="store_true")
    sp.add_argument("--log-y", action="store_true")
    sp.add_argument("--marker", default="none")
    sp.set_defaults(func=cmd_hysteresis)

    # -- run ---------------------------------------------------------
    sp = sub.add_parser("run", help="execute a JSON list of jobs")
    sp.add_argument("--config", required=True)
    sp.add_argument("--keep-going", action="store_true",
                    help="continue after a failing job")
    logging_args(sp)
    sp.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args)
    set_origin_rcparams()

    t0 = time.time()
    _history_line("START", args)
    rc = 1
    try:
        rc = int(args.func(args) or 0)
        return rc
    except KeyboardInterrupt:
        log.error("interrupted")
        rc = 130
        return rc
    except Exception as exc:                                # noqa: BLE001
        log.exception("fatal: %s", exc)
        rc = 1
        return rc
    finally:
        # `finally` so the run is recorded even on a crash or Ctrl-C -- an
        # audit log with no END line is worse than useless, because you cannot
        # tell a still-running job from one that died.
        dur = time.time() - t0
        # "END" (not OK/FAIL) is deliberate: cluster/logging.sh uses START/END
        # to bracket a whole script and OK/FAIL for individual commands within
        # it. cli.py is a whole script, so it must close with END, or
        # `logs.sh running` -- which pairs START with END -- reports every
        # finished CLI run as still in flight. Failure is carried by rc=.
        _history_line("END", args, rc, dur)
        log.info("exit rc=%d after %.1fs", rc, dur)
        if LOG_PATHS["run"]:
            log.info("log file: %s", LOG_PATHS["run"])
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())
