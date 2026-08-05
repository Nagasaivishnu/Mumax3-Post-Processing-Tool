#!/bin/bash
# =============================================================================
#  cluster/submit_modes.slurm  --  spatial mode-profile analysis (the heavy one)
# =============================================================================
#  Submit with:      sbatch cluster/submit_modes.slurm
#  Watch with:       squeue -u $USER
#  Cancel with:      scancel <JobID>
#
#  This is the SLURM equivalent of the OpenPBS "pbs_script.sh" in Chapter 6 of
#  the CNNL tutorial. See README_CLUSTER.md for the full PBS -> SLURM mapping.
# =============================================================================

#SBATCH --job-name=mumax_modes
#SBATCH --output=logs/%x_%j.out          # %x=job name, %j=job ID (combines
#SBATCH --error=logs/%x_%j.out           #   stdout+stderr, like  #PBS -j oe)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16               # OVF reading + BLAS threads
#SBATCH --mem=64G                        # <<< run `cli.py probe` to size this
#SBATCH --time=08:00:00                  # walltime, hh:mm:ss
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=CHANGEME@u.nus.edu   # <<< or delete these two lines

# NOTE: no --gres=gpu. Post-processing is pure NumPy on the CPU, so asking for
# a GPU would only make you queue behind the MuMax3 simulation jobs.
# Add a partition here if `sinfo -s` shows you need one, e.g.:
##SBATCH --partition=cpu

set -euo pipefail

# =============================================================================
#  CONFIG -- edit these
# =============================================================================
SIM_DIR="$HOME/runs/sim.out"          # MuMax3 output dir (table.txt + m*.ovf)
RESULT_DIR="$HOME/runs/results"       # where results are copied back to

COMPONENT="My"                        # Mx | My | Mz | M45 | M135
DT="5e-12"                            # MuMax3 saving interval, seconds
T_START="3e-9"                        # analysis window start, seconds
T_END="25e-9"                         # analysis window end, seconds
N_PEAKS="5"
F_MIN="0"                             # GHz
F_MAX="40"                            # GHz
AVG_AXIS="Z"                          # Z | Y | X | None
EXTRA_ARGS=(--pptx report.pptx)       # anything else to pass to cli.py

# =============================================================================
#  Everything below is generic
# =============================================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/config.sh"
# shellcheck source=/dev/null
source "${HERE}/logging.sh"

mkdir -p "${RESULT_DIR}"
log_init submit_modes "$@"

log_info "job id      : ${SLURM_JOB_ID:-none}"
log_info "node        : $(hostname)"
log_info "cpus        : ${SLURM_CPUS_PER_TASK:-?}"
log_info "mem         : ${SLURM_MEM_PER_NODE:-?} MB"
log_info "sim dir     : ${SIM_DIR}"
log_info "result dir  : ${RESULT_DIR}"
log_info "parameters  : component=${COMPONENT} dt=${DT} window=[${T_START},${T_END}] avg=${AVG_AXIS}"

config_check || true

if [[ ! -d "${SIM_DIR}" ]]; then
    log_error "SIM_DIR does not exist: ${SIM_DIR}"
    exit 1
fi

# --- Stage input onto fast node-local NVMe -----------------------------------
# OVF reading is I/O bound; the shared filesystem is the bottleneck. Copying to
# node-local scratch first is the single biggest speedup for large runs.
# (Same idea as the tutorial's rsync-to-$TMPDIR step.)
SCRATCH="$(pick_scratch)"
mkdir -p "${SCRATCH}"

# ONE combined EXIT trap: clean scratch AND close the log. Two separate `trap
# ... EXIT` calls would silently overwrite each other -- the second wins and the
# first never runs.
cleanup() {
    local rc=$?
    log_info "cleaning scratch ${SCRATCH}"
    rm -rf "${SCRATCH}"
    log_finish "${rc}"
}
trap cleanup EXIT
trap 'log_error "SIGTERM: scancel or walltime limit"; exit 143' TERM
trap 'log_error "SIGINT"; exit 130' INT

log_cmd rsync -a "${SIM_DIR}/" "${SCRATCH}/sim.out/"
log_info "staged size: $(du -sh "${SCRATCH}/sim.out" 2>/dev/null | cut -f1)"

WORK_OUT="${SCRATCH}/results"
mkdir -p "${WORK_OUT}"

# --- Export what container_script.sh needs -----------------------------------
# LOG_DIR is exported so the container run appends to the SAME history file.
export PROJECT_DIR PYTHON_MODULE VENV_DIR LOG_DIR

# --- Run the analysis inside the container -----------------------------------
# `set +e` around this call is deliberate and important: with `set -e` still
# active, a failing analysis would abort the script here, the EXIT trap would
# wipe scratch, and any partial results (plus the log-worthy exit code) would be
# lost. We want to capture the status and still copy back whatever was produced.
set +e
log_cmd "${CONTAINER_WRAPPER}" --exec bash "${HERE}/container_script.sh" \
    modes \
    --sim-dir    "${SCRATCH}/sim.out" \
    --outdir     "${WORK_OUT}" \
    --component  "${COMPONENT}" \
    --dt         "${DT}" \
    --t-start    "${T_START}" \
    --t-end      "${T_END}" \
    --n-peaks    "${N_PEAKS}" \
    --f-min      "${F_MIN}" \
    --f-max      "${F_MAX}" \
    --avg-axis   "${AVG_AXIS}" \
    --workers    "${SLURM_CPUS_PER_TASK:-8}" \
    "${EXTRA_ARGS[@]}"
STATUS=$?
set -e

# --- Copy results back BEFORE scratch is wiped -------------------------------
log_cmd rsync -a "${WORK_OUT}/" "${RESULT_DIR}/" || true

# The .npz FFT cache and m_txyz.npy live next to the simulation data. Bring the
# .npz back too -- the GUI on your laptop reads it and skips all recomputation.
# (m_txyz.npy is deliberately NOT copied: it is huge and trivially rebuilt.)
log_cmd rsync -a --include='fft_*.npz' --exclude='*' \
      "${SCRATCH}/sim.out/" "${RESULT_DIR}/" || true

# Keep a copy of this job's logs with the results, so the numbers and the
# record of how they were produced travel together when you rsync them home.
mkdir -p "${RESULT_DIR}/logs"
[[ -n "${LOG_TRACE_FILE}" && -f "${LOG_TRACE_FILE}" ]] && \
    cp "${LOG_TRACE_FILE}" "${RESULT_DIR}/logs/" 2>/dev/null || true

log_info "analysis exit status: ${STATUS}"
log_info "results: ${RESULT_DIR}"

# Report what SLURM actually used, so you can right-size --mem next time.
if command -v sacct >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    log_info "resource usage:"
    sacct -j "${SLURM_JOB_ID}" \
          --format=JobID,JobName%20,Elapsed,MaxRSS,MaxVMSize,State || true
fi

# The EXIT trap calls log_finish and cleans scratch.
exit ${STATUS}
