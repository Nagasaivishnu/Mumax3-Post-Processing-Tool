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

# Results are written to BOTH places:
#   1. ${SIM_DIR}/${RESULTS_SUBDIR}   -- each simulation carries its own analysis
#   2. ${RESULT_DIR}/<sim name>       -- one central tree to browse everything
# Set RESULT_DIR="" to skip the central mirror and keep only the per-sim copy.
RESULTS_SUBDIR="results"
RESULT_DIR="$HOME/runs/results"

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

log_init submit_modes "$@"

SIM_DIR="${SIM_DIR%/}"
SIM_NAME="$(basename "${SIM_DIR}")"
# Computed from the real path, not the scratch copy cli.py will see.
SIM_LABEL="$(sim_label "${SIM_DIR}")"
SIM_RESULTS="${SIM_DIR}/${RESULTS_SUBDIR}"
# Namespace the central mirror by simulation name, or 40 array tasks all
# rsync into the same folder and silently overwrite each other.
CENTRAL_RESULTS="${RESULT_DIR:+${RESULT_DIR}/${SIM_NAME}}"

log_info "job id      : ${SLURM_JOB_ID:-none}"
log_info "node        : $(hostname)"
log_info "cpus        : ${SLURM_CPUS_PER_TASK:-?}"
log_info "mem         : ${SLURM_MEM_PER_NODE:-?} MB"
log_info "sim dir     : ${SIM_DIR}"
log_info "label       : ${SIM_LABEL}"
log_info "results (1) : ${SIM_RESULTS}"
log_info "results (2) : ${CENTRAL_RESULTS:-<central mirror disabled>}"
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

# Stage under the simulation's REAL name, not a hardcoded "sim.out". cli.py
# names the bundle after the directory it analysed, so a hardcoded name would
# make every job in a sweep produce an identically named .npz.
STAGED="${SCRATCH}/${SIM_NAME}"
log_cmd rsync -a "${SIM_DIR}/" "${STAGED}/"
log_info "staged size: $(du -sh "${STAGED}" 2>/dev/null | cut -f1)"

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
    --sim-dir    "${STAGED}" \
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
    --bundle-name "${SIM_LABEL}" \
    "${EXTRA_ARGS[@]}"
STATUS=$?
set -e

# --- Copy results back BEFORE scratch is wiped -------------------------------
# Destination 1: the simulation's own results folder. This is the copy that
# matters -- the analysis stays with the data that produced it.
mkdir -p "${SIM_RESULTS}"
log_cmd rsync -a "${WORK_OUT}/" "${SIM_RESULTS}/" || true

# The fft_*.npz cache is written next to the simulation data on scratch, not in
# the results folder. Bring it back to the real simulation dir so a rerun with
# different peak or orientation settings skips the whole FFT.
# (m_txyz.npy is deliberately NOT copied: huge, and trivially rebuilt.)
log_cmd rsync -a --include='fft_*.npz' --exclude='*' \
      "${STAGED}/" "${SIM_DIR}/" || true

# Keep this job's trace with the results, so the numbers and the record of how
# they were produced travel together.
mkdir -p "${SIM_RESULTS}/logs"
[[ -n "${LOG_TRACE_FILE}" && -f "${LOG_TRACE_FILE}" ]] && \
    cp "${LOG_TRACE_FILE}" "${SIM_RESULTS}/logs/" 2>/dev/null || true

# Destination 2: the central mirror, for browsing everything in one place.
if [[ -n "${CENTRAL_RESULTS}" ]]; then
    mkdir -p "${CENTRAL_RESULTS}"
    log_cmd rsync -a "${SIM_RESULTS}/" "${CENTRAL_RESULTS}/" || true
fi

log_info "analysis exit status: ${STATUS}"
log_info "results in simulation dir : ${SIM_RESULTS}"
[[ -n "${CENTRAL_RESULTS}" ]] && log_info "results in central mirror : ${CENTRAL_RESULTS}"

# Report what SLURM actually used, so you can right-size --mem next time.
if command -v sacct >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    log_info "resource usage:"
    sacct -j "${SLURM_JOB_ID}" \
          --format=JobID,JobName%20,Elapsed,MaxRSS,MaxVMSize,State || true
fi

# The EXIT trap calls log_finish and cleans scratch.
exit ${STATUS}
