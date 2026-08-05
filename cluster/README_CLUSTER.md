# Running this tool on the CNNL cluster (SLURM)

The CNNL tutorial PDF you have is written for **OpenPBS** (`qsub`, `#PBS`, `qstat`).
Your cluster now runs **SLURM** (`sbatch`, `#SBATCH`, `squeue`). Everything else in
that document — the Singularity container wrappers under
`/apps/containers/scripts/`, the `. /etc/profile` + `module load` dance, the
staging-to-`$TMPDIR` pattern — is unchanged and still applies.

This project is **headless** — the PyQt6 GUI was removed. The heavy lifting runs
on a node with 192 cores and up to 2 TB of RAM, and you download one small `.npz`
to plot on your own PC. No X11, no Qt, no display anywhere in the pipeline.

---

## Why this is worth doing

The two functions that limit you locally are in `processing/mode_profile.py`:

| Step | Array | Size for a 200×200×5 grid, 500 frames |
|---|---|---|
| `load_dataset` | `m_raw` float32 `(T, nx, ny, nz, 3)` | **1.1 GiB** |
| `compute_fft` | `P` `(n_freq, nz, ny, nx)` | 0.6 GiB |
| `compute_fft` transient | complex128 FFT output | **2.4 GiB** |

Scale that to a 1000×1000×10 grid with 2000 frames and you are at ~240 GiB for
`m_raw` alone — impossible on a laptop, routine on a cluster node.

On top of that, the originals waste memory unnecessarily. `load_dataset` builds a
Python list of frames, then `np.array(...)` copies it, then `np.transpose(...)`
copies again — peak RSS ≈ **2.5×** the final array. `processing/cluster_loader.py`
preallocates the output and has each worker write its slice directly, bringing
peak to ≈ **1.0×**, and chunks the FFT so the complex128 transient is bounded
regardless of grid size.

---

## What was added

```
cli.py                          the entry point — no Qt anywhere
plot_modes.py                   run this on YOUR PC, not the cluster
processing/cluster_loader.py    low-memory loader + chunked FFT + memory estimator
cluster/
  config.sh                     ← the only file you must edit
  logging.sh                    shared timestamped command logging
  logs.sh                       log viewer (history / failed / stats / tail)
  setup_venv.sh                 one-time Python environment build
  container_script.sh           runs inside Singularity (the "first script")
  submit_modes.slurm            mode-profile job (the heavy one)
  submit_array.slurm            job array — many simulations at once
  submit_batch.slurm            JSON-driven FMR / hysteresis / mixed
  interactive.sh                interactive container shell on a compute node
  jobs.example.json             config template for submit_batch.slurm
  README_CLUSTER.md             this file
logs/                           created on first run; see "Command logging"
```

`processing/fmr.py`, `processing/hysteresis.py`, `utils/ovf_reader.py` and
`export/csv_export.py` are untouched from the original project — the CLI calls
exactly the same functions, so the numbers are unchanged.

---

## OpenPBS → SLURM translation

Use this to convert the tutorial's Chapter 4–6 recipes.

### Commands

| Task | OpenPBS (tutorial) | SLURM (now) |
|---|---|---|
| Submit a batch job | `qsub script.sh` | `sbatch script.sh` |
| Interactive session | `qsub -I -l ncpus=4 -l mem=1gb` | `srun --cpus-per-task=4 --mem=1G --pty bash` |
| Interactive + X11 | `qsub -I -X` | not needed — this project is headless |
| List your jobs | `qstat` | `squeue -u $USER` |
| Job detail | `qstat -f JobID` | `scontrol show job JobID` |
| Why is it queued? | `qstat -f` → comment field | `squeue -u $USER -o "%.18i %.9T %R"` |
| Cancel a job | `qdel --force JobID` | `scancel JobID` |
| Cancel everything | — | `scancel -u $USER` |
| Node list + status | `pbsnodes -ajSL` | `sinfo -N -o "%N %P %t %c %m %G"` |
| Partitions summary | — | `sinfo -s` |
| Post-run accounting | — | `sacct -j JobID --format=JobID,Elapsed,MaxRSS,State` |
| Load scheduler module | `ml pbspro` | usually none needed; try `module av slurm` |

`squeue` state codes map to `qstat`'s: `R`=running, `PD`=pending (was `Q`),
`CG`=completing (was `E`).

### Directives

| OpenPBS | SLURM |
|---|---|
| `#PBS -N name` | `#SBATCH --job-name=name` |
| `#PBS -j oe` | `#SBATCH --output=logs/%x_%j.out` **and** `--error=` the same path |
| `#PBS -l ncpus=16` | `#SBATCH --cpus-per-task=16` |
| `#PBS -l mem=64gb` | `#SBATCH --mem=64G` |
| `#PBS -l walltime=72:00:00` | `#SBATCH --time=72:00:00` |
| `#PBS -l cudamem=1gb` | `#SBATCH --gres=gpu:1` (whole GPU; SLURM has no VRAM-size request) |
| `#PBS -l host=NodeName` | `#SBATCH --nodelist=NodeName` |
| `#PBS -J 1-40` | `#SBATCH --array=1-40` |
| `#PBS -q queue` | `#SBATCH --partition=partition` |
| `#PBS -M you@nus.edu` + `-m ae` | `#SBATCH --mail-user=...` + `--mail-type=END,FAIL` |

### Environment variables

| OpenPBS | SLURM |
|---|---|
| `$PBS_JOBID` | `$SLURM_JOB_ID` |
| `$PBS_JOBNAME` | `$SLURM_JOB_NAME` |
| `$PBS_O_WORKDIR` | `$SLURM_SUBMIT_DIR` |
| `$PBS_ARRAY_INDEX` | `$SLURM_ARRAY_TASK_ID` |
| `$NCPUS` | `$SLURM_CPUS_PER_TASK` |
| `$TMPDIR` | site-dependent — `$SLURM_TMPDIR`, `$TMPDIR`, or none |

⚠️ **`$TMPDIR` is the one real gotcha.** OpenPBS guarantees it; SLURM does not
unless the admin configured it. `config.sh` has a `pick_scratch()` helper that
tries `$SLURM_TMPDIR`, `$TMPDIR`, `/scratch/$USER`, then `/tmp`. Confirm with
Dr. Fong which node-local scratch you should be using — that is the difference
between reading OVFs off fast local NVMe and hammering the shared filesystem.

---

## Setup — do this once

**1. Copy the project to the cluster.** From your laptop:

```bash
rsync -az --exclude='__pycache__' --exclude='.git' \
      "/mnt/d/Linux/Mumax3-Post-Processing-Tool For Cluster/" \
      YOUR_ID@172.20.32.127:~/Mumax3-Post-Processing-Tool/
```

(Off campus, VPN into the NUS network first, as the tutorial says.)

**2. Edit `cluster/config.sh`.** Four values are marked `<<< CHECK`:

```bash
CONTAINER_ENV="nntools"    # ls /apps/containers/scripts   to confirm
PYTHON_MODULE="pytorch/py3/2.2.0"
PROJECT_DIR="$HOME/Mumax3-Post-Processing-Tool"
SLURM_PARTITION=""         # sinfo -s   to see what exists
```

To find the right `PYTHON_MODULE`, get a shell in the container and look:

```bash
/apps/containers/scripts/nntools/current --shell
bash
. /etc/profile
module av
```

Pick any module that provides a modern `python3` — we only need its interpreter
and its optimised NumPy/SciPy build. `pytorch/py3/2.2.0` works well.

**3. Build the Python environment:**

```bash
cd ~/Mumax3-Post-Processing-Tool
bash cluster/setup_venv.sh
```

This creates `~/.venvs/mumax-postproc` with `--system-site-packages`, so the
container's tuned NumPy/SciPy are reused and only the missing pieces
(matplotlib, python-pptx, pillow) are downloaded. No PyQt6 — if you previously
ran `setup_venv.sh --with-gui`, that flag is gone.

---

## Running a job

### Step 1: size the job on the login node

```bash
python3 cli.py probe --sim-dir ~/runs/sim.out
```

Reading only OVF *headers*, so it is instant and safe on a login node. Output:

```
  OVF frames     : 500
  Grid           : 512 x 512 x 4 (vdim 3)
  Estimated peak memory (GiB)
    m_raw_gb              6.000
    component_gb          2.000
    power_gb              1.002
    fft_chunk_gb          2.400
    total_gb             11.402
  ==> put this in your SLURM script:  #SBATCH --mem=18G
```

If `python3` on the login node is too old, run the same thing through the
container: `/apps/containers/scripts/nntools/current --exec bash cluster/container_script.sh probe --sim-dir ~/runs/sim.out`.

### Step 2: edit and submit

Open `cluster/submit_modes.slurm`, set the `CONFIG` block (`SIM_DIR`,
`COMPONENT`, `DT`, `T_START`, `T_END`) and the `--mem` from step 1, then:

```bash
mkdir -p logs
sbatch cluster/submit_modes.slurm
squeue -u $USER
tail -f logs/mumax_modes_<JobID>.out
```

The script stages the simulation to node-local scratch, runs the analysis inside
the container, and rsyncs results back — including the `fft_*.npz` cache.

### Step 3: where results land

Results are written to **both** places:

1. `<SIM_DIR>/results/` — inside the simulation itself. Each simulation carries
   its own analysis, so nothing can be orphaned from the data that produced it,
   and an array of 40 jobs cannot overwrite one another.
2. `<RESULT_DIR>/<sim name>/` — a central mirror, for browsing everything in one
   place. Set `RESULT_DIR=""` in the script to skip it.

The `fft_*.npz` cache goes back beside the simulation data (not into `results/`),
so a rerun with different peak or orientation settings skips the FFT entirely.
`m_txyz.npy` is deliberately not copied back: huge, and trivially rebuilt.

Bring the bundle home:

```bash
scp YOUR_ID@172.20.32.127:~/runs/sweepA/sim.out/results/*_modes.npz .
```

Filenames disambiguate themselves. MuMax3 names every output directory
`sim.out`, so `sim_label()` in `config.sh` prepends the parent when the leaf name
is generic — `sweepA/sim.out` becomes `sweepA_sim.out_My_modes.npz`. Download a
whole sweep into one folder and nothing collides. `--bundle-name` overrides it.

Outputs are also component-tagged (`My_spectrum.csv`, `Mz_spectrum.csv`), so
analysing several components into the same `results/` folder is safe.

### Many simulations at once

```bash
ls -d ~/runs/sweep_*/*.out > cluster/simlist.txt
wc -l < cluster/simlist.txt            # say 40
sbatch --array=1-40%8 cluster/submit_array.slurm
```

`%8` caps concurrency at 8 tasks — please keep it there on a shared cluster.

### FMR / hysteresis

Copy `jobs.example.json`, edit the paths, then:

```bash
sbatch --export=ALL,CONFIG_FILE=$HOME/my_jobs.json cluster/submit_batch.slurm
```

---

## Plotting on your own PC

The cluster produces `<sim>_<component>_modes.npz`. That file is the point of
the whole exercise — everything else in the results directory is a convenience.

**1. Download it** (from a terminal on your Windows PC — PowerShell, Git Bash,
or WSL all work):

```powershell
scp YOUR_ID@172.20.32.127:~/runs/results/sim.out_My_modes.npz .
```

Or drag it across with WinSCP / MobaXterm's file browser if you prefer.

**2. Install the two libraries you need** — not this project, just:

```powershell
pip install numpy matplotlib
```

**3. Plot it** with the bundled helper, which needs nothing else from this repo:

```powershell
python plot_modes.py sim.out_My_modes.npz            # spectrum + all modes
python plot_modes.py sim.out_My_modes.npz --info     # what is in the file
python plot_modes.py sim.out_My_modes.npz --mode 1   # one mode, large
python plot_modes.py sim.out_My_modes.npz --save     # write PNGs instead
```

Or ignore the helper entirely and go straight at the arrays:

```python
import numpy as np, matplotlib.pyplot as plt

d = np.load("sim.out_My_modes.npz")
plt.plot(d["f_ghz"], d["spectrum"])
plt.xlabel("Frequency (GHz)"); plt.ylabel("Integrated FFT amplitude")
plt.show()

plt.imshow(d["profiles"][0], origin="lower", cmap="inferno")   # strongest mode
plt.title(f'{d["peak_freqs_ghz"][0]:.3f} GHz'); plt.colorbar(); plt.show()
```

`print(json.loads(str(d["metadata_json"])))` gives you every parameter that
produced the file — component, dt, time window, averaging axis, grid shape, the
SLURM job ID, and when it was made.

### Why not run the GUI on the cluster?

There is no GUI any more, but the reasoning is worth recording: pushing
Matplotlib redraws over X11 from Singapore is slow, it needs PyQt6 and working
X11 forwarding on both ends, and PyQt6 ≥ 6.5 pulls in `libxcb-cursor0` which is
not installed on the cluster and which you have no root to add. Downloading a
few-MB `.npz` and plotting locally sidesteps all of it.

### A container shell, when you need one

For poking at data or checking a module name (the tutorial's Chapter 4 workflow):

```bash
bash cluster/interactive.sh -c 8 -m 32G
```

---

## Command logging

Every script logs what it ran, when, and how it ended. Three files per job, all
in `logs/`:

| File | What it is |
|---|---|
| `run_history.log` | **Permanent audit trail.** One tab-separated line per command, appended forever, across every job you ever run. |
| `trace_<context>_<jobid>.log` | Every individual bash command that executed, timestamped. For "which step actually failed". |
| `cli_<command>_<jobid>.log` | The Python side: full DEBUG detail, even when the console only showed INFO. |

A history line looks like this:

```
2026-08-04T20:29:36+08:00  START  job=4471  host=blade07  user=e1234567  ctx=submit_modes  cmd=rsync -a /home/…/sim.out/ /scratch/…/
2026-08-04T20:31:02+08:00  OK     job=4471  host=blade07  user=e1234567  ctx=submit_modes  rc=0  dur=86s  cmd=rsync -a /home/…/sim.out/ /scratch/…/
```

Timestamps are ISO-8601 with the UTC offset, written identically by the bash and
Python halves, so lines from both interleave correctly when sorted.

### Reading the logs

Don't memorise paths — use the viewer:

```bash
bash cluster/logs.sh history 50      # last 50 commands, aligned into columns
bash cluster/logs.sh failed          # only what went wrong
bash cluster/logs.sh running         # started but never finished (or killed)
bash cluster/logs.sh show 4471       # everything recorded for one job ID
bash cluster/logs.sh stats           # runs / failures / total time per context
bash cluster/logs.sh tail            # follow the newest log live
bash cluster/logs.sh grep "MaxRSS"   # search every log
bash cluster/logs.sh clean 30        # delete traces older than 30 days
```

`clean` never touches `run_history.log`. Trace files are the bulky ones; the
history is small and worth keeping indefinitely.

### Details worth knowing

**Array jobs share one history file safely.** All tasks append to
`run_history.log` under an exclusive `flock`, so their lines cannot interleave
mid-write. Verified with 12 concurrent writers: 210 lines, zero torn. Trace
files are per-task (`<jobid>_<taskid>`) so they never collide at all.

**Cancelled jobs are still recorded.** `scancel` and walltime kills send
SIGTERM, which the scripts trap and log before exiting. Without that, a killed
job leaves a `START` with no `END` and looks like it's still running forever —
which is exactly what `logs.sh running` is for finding.

**The trace is filtered on the way out.** bash's `set -x` traces every simple
command, including the ones inside the logging functions themselves — about 75%
of raw trace volume was logging bookkeeping. `log_finish` strips those lines
once at the end, so what's left is your commands.

**Results carry their provenance.** The job's trace file is copied into
`<results>/logs/`, so when you rsync results back to your laptop, the record of
exactly how they were produced comes with them.

**Knobs.** `LOG_DIR` moves the logs; `LOG_TRACE=0` disables the trace file
(history still written); `LOG_QUIET=1` suppresses console echo. On the Python
side, `--log-dir` and `--no-log-file` do the same per invocation.

---

## Performance notes

**Don't request a GPU.** All the maths here is NumPy on the CPU. `--gres=gpu:1`
only parks you behind the MuMax3 simulation queue. `SLURM_GRES` is empty in
`config.sh` on purpose.

**Threads are pinned to your allocation.** `container_script.sh` sets
`OMP_NUM_THREADS` etc. from `$SLURM_CPUS_PER_TASK`. Without this, OpenBLAS
spawns one thread per *physical* core on the node — up to 192 — which
oversubscribes your allocation and is slower, not faster.

**More cores stop helping around 16 for OVF loading.** The loader is thread-based
and reading is I/O bound; past ~16 workers you are limited by the filesystem.
`--cpus-per-task=16` with a good scratch disk beats 64 cores off a network mount.

**Tuning `--chunk-points`.** Lower it if you hit OOM, raise it for speed. It
directly sets the FFT transient: `chunk_points × n_time_win × 24` bytes. The
default `1048576` costs ~2.4 GiB at 100 time samples.

**Analysing arrays larger than RAM.** Run once to build `m_txyz.npy`, then rerun
with `--mmap` to memory-map it instead of loading it. Slower per access, but the
dataset size stops being bounded by RAM.

---

## The axis-order fix — read this if you have old results

`utils/ovf_reader.read_ovf` returns `(nz, ny, nx, vdim)`, the native OVF layout
(x fastest, z slowest). The now-deleted GUI's `load_dataset` stacked frames to
`(T, nz, ny, nx, 3)` and then applied:

```python
data = np.transpose(data, (0, 3, 2, 1, 4))     # -> (T, nx, ny, nz, 3)
```

contradicting its own docstring. Downstream, `get_spatial_profile` reduces a
fixed array index — 0 for `avg_axis="Z"` — and under that transposed ordering
index 0 holds **x**. So `--avg-axis Z` averaged over x and returned a `(y, z)`
map where a `(y, x)` one was intended.

Demonstrated on a deliberately asymmetric 12×9×3 test grid:

| `--axis-order` | reduces | profile shape | view |
|---|---|---|---|
| `xyz` (old GUI behaviour) | **x** | `(9, 3)` = (y, z) | wrong for a thin film |
| `zyx` (**new default**) | z | `(9, 12)` = (y, x) | correct XY mode map |

Invisible on a cubic grid. Very visible on a thin film with `nz=4, nx=ny=512`.

**The default is now `zyx`.** The only reason to keep `xyz` was parity with the
GUI, and the GUI is gone. If you need to reproduce numbers generated before this
change, pass `--axis-order xyz --force`.

Every bundle records `avg_axis_requested` alongside `avg_axis_effective`, and
both `cli.py` and `plot_modes.py --info` warn when they disagree — so no `.npz`
can quietly mislabel its own axes again.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `bash: module: command not found` | `. /etc/profile` was not run inside the container. `container_script.sh` does this; if you are in a manual shell, run it yourself. (Tutorial, Common Errors #1.) |
| `sbatch: error: Invalid partition` | `SLURM_PARTITION` in `config.sh` is wrong. Run `sinfo -s`. |
| Job stuck in `PD` | `squeue -u $USER -o "%.18i %.9T %R"` — the reason is in the last column. `Resources` = waiting for a free node; `QOSMaxJobsLimit` = you have too many queued. |
| `slurmstepd: Exceeded job memory limit` | Raise `--mem` (use `cli.py probe`), or lower `--chunk-points`. |
| `No m*.ovf files found` | `SIM_DIR` points at the run directory, not the `.out` directory MuMax3 creates. |
| `ModuleNotFoundError: No module named 'numpy'` | The venv did not activate. Check `VENV_DIR` in `config.sh` and rerun `setup_venv.sh`. |
| `qt.qpa.plugin: could not load "xcb"` | You are running an old copy that still has `main.py`. The GUI was removed; use `cli.py`. |
| `.npz` is enormous | You used `--npz-mode full`, which embeds the whole power array. Use the default `compact`. |
| `plot_modes.py` says arrays are missing | You downloaded an `fft_*.npz` cache instead of the `*_modes.npz` bundle. |
| Results vanished after the job | Scratch is wiped at job end. The scripts rsync back automatically; if you edited them, keep that step. |
| Job killed at exactly the walltime | Raise `--time`. Unlike PBS, SLURM's default is often much shorter than 24 h. |

---

## Quick reference

```bash
# size a job
python3 cli.py probe --sim-dir ~/runs/sim.out

# submit / monitor / cancel
sbatch cluster/submit_modes.slurm
squeue -u $USER
scancel <JobID>
tail -f logs/mumax_modes_<JobID>.out

# what did it actually use?
sacct -j <JobID> --format=JobID,JobName%20,Elapsed,MaxRSS,MaxVMSize,State

# interactive container shell
bash cluster/interactive.sh -c 16 -m 64G

# on your own PC, after downloading the bundle
python plot_modes.py sim.out_My_modes.npz

# see all CLI options
python3 cli.py modes --help

# command logs
bash cluster/logs.sh history 50
bash cluster/logs.sh failed
bash cluster/logs.sh show <JobID>
bash cluster/logs.sh stats
```
