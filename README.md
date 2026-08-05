# MuMax3 Post-Processing Tool — headless CLI

Spatial FMR / spin-wave mode analysis of MuMax3 output, built to run on an HPC
cluster and produce one small file you download and plot on your own PC.

**There is no GUI.** The PyQt6 interface was removed, along with its OVF video
export. Nothing in this project imports Qt or needs a display.

```
compute on the cluster  ──►  download one .npz  ──►  plot on your PC
   cli.py modes                 ~few MB              plot_modes.py
```

---

## Quick start

On the cluster:

```bash
pip install -r requirements.txt

# 1. How much memory will this need? (reads OVF headers only — instant)
python3 cli.py probe --sim-dir ~/runs/sim.out

# 2. Run the analysis
python3 cli.py modes \
    --sim-dir ~/runs/sim.out \
    --outdir  ~/runs/results \
    --component My --dt 5e-12 --t-start 3e-9 --t-end 25e-9 \
    --n-peaks 5 --f-min 0 --f-max 40 \
    --no-plots --no-csv
```

On your own PC:

```bash
scp you@cluster:~/runs/results/sim.out_My_modes.npz .
pip install numpy matplotlib
python plot_modes.py sim.out_My_modes.npz
```

For running this under SLURM on the NUS CNNL cluster — container setup, batch
scripts, job arrays, logging — see **[cluster/README_CLUSTER.md](cluster/README_CLUSTER.md)**.

---

## The download bundle

`cli.py modes` writes `<sim>_<component>_modes.npz`. This is the deliverable.
It is deliberately small: the full power array is `(n_freq, nz, ny, nx)` float32
and runs to gigabytes, but the 2-D profiles at just the resonant peaks are a few
megabytes — which is everything you need to plot modes.

| Array | Shape | Meaning |
|---|---|---|
| `f_hz`, `f_ghz` | `(n_freq,)` | frequency axis |
| `spectrum` | `(n_freq,)` | spatially integrated FFT amplitude |
| `peak_freqs_hz`, `peak_freqs_ghz` | `(n_peaks,)` | detected resonances |
| `peak_indices` | `(n_peaks,)` | index into `f_hz` |
| `mode_numbers` | `(n_peaks,)` | 1 = strongest |
| `profiles` | `(n_peaks, rows, cols)` | one 2-D spatial map per mode |
| `metadata_json` | scalar string | every parameter that produced the file |
| `power_full` | `(n_freq, …)` | only with `--npz-mode full` |

It loads with NumPy alone — this codebase is not needed:

```python
import numpy as np
d = np.load("sim.out_My_modes.npz")
f_ghz    = d["f_ghz"]
spectrum = d["spectrum"]
profiles = d["profiles"]        # profiles[0] is the strongest mode
```

`--npz-mode` controls what goes in: `compact` (default), `full` (embeds the
whole power array), `both`, or `none`.

---

## Commands

| Command | What it does |
|---|---|
| `probe` | Read OVF headers only; report grid size and the `--mem` to request. Safe on a login node. |
| `modes` | **The main one.** OVF → spatial FFT → peak detection → mode profiles → download bundle. |
| `fmr` | Susceptibility heatmap from `table.txt`. Much lighter than OVF work. |
| `hysteresis` | Overlay several `table.txt` runs, merged CSV export. |
| `run` | Execute a JSON list of the above in one job. |

`python3 cli.py <command> --help` lists every flag.

Useful `modes` flags:

- `--no-plots` — skip PNG rendering entirely (you're plotting at home)
- `--no-csv` — skip per-mode CSVs (already in the `.npz`)
- `--no-fft-cache` — don't write the large `fft_*.npz` next to the simulation
- `--mmap` — memory-map the OVF cache, for datasets larger than RAM
- `--chunk-points` — lower it if you hit OOM; it bounds the FFT transient

---

## Project layout

```
cli.py                       the entry point
plot_modes.py                run this on YOUR PC, not the cluster
processing/
  cluster_loader.py          low-memory OVF loading + chunked FFT
  mode_profile.py            component extraction, peaks, spatial profiles
  fmr.py                     susceptibility from table.txt
  hysteresis.py              table.txt helpers
  data_loader.py             table.txt → DataFrame
utils/ovf_reader.py          OVF2 binary reader
export/
  csv_export.py              CSV writers
  plot_style.py              publication ("Origin") Matplotlib style
  ppt_export.py              optional PowerPoint assembly
cluster/                     SLURM + Singularity scripts, logging, docs
```

### Where to change the physics

- `processing/mode_profile.py` — component extraction, FFT windowing, peak
  detection, spatial profile reduction
- `processing/cluster_loader.py` — the memory-efficient loader and chunked FFT
  (numerically identical to `mode_profile.compute_fft`, just allocated better)
- `processing/fmr.py` — susceptibility normalisation for the `table.txt` path

---

## Important: the axis-order default changed

`utils/ovf_reader.read_ovf` returns `(nz, ny, nx, vdim)` — the native OVF layout.
The deleted GUI additionally applied `np.transpose(data, (0,3,2,1,4))`, giving
`(T, nx, ny, nz, 3)`. Under that ordering `--avg-axis Z` reduced array index 0,
which held **x**, so it averaged over the wrong axis and returned a `(y, z)` map
where a `(y, x)` one was intended. Invisible on a cubic grid; very visible on a
thin film.

**The default is now `--axis-order zyx`**, the native and correct order, so
`--avg-axis` means what it says. Pass `--axis-order xyz` to reproduce results
generated before this change.

Every bundle records both `avg_axis_requested` and `avg_axis_effective`, and
`plot_modes.py --info` warns when they disagree — so a `.npz` is always honest
about which axes its profiles actually span.

---

## Edge cases handled

- Missing or unreadable OVF files → named in the error, not a silent partial load
- Grid mismatch between frames → rejected with both shapes reported
- Time window containing too few samples → error states dt, frame count and t_max
- Missing `table.txt` columns → error lists what is actually available
- Unequal dataset lengths in hysteresis export → outer join, NaN fill
- Duplicate time points in FMR tables → deduplicated automatically
- Datasets larger than RAM → `--mmap`
