#!/bin/bash
# =============================================================================
#  cluster/submit_array.slurm  --  process MANY simulations in parallel
# =============================================================================
#  This is the real payoff of moving to the cluster: instead of analysing 40
#  simulation directories one at a time, SLURM runs all 40 at once on different
#  nodes. Each task writes its results into its OWN simulation directory, so
#  they cannot overwrite one another.
#
#  Usage
#  -----
#    1. Build a list of simulation directories, one per line:
#         ls -d ~/runs/sweep_*/*.out > cluster/simlist.txt
#    2. Set the array range to match the line count:
#         wc -l < cluster/simlist.txt      # e.g. 40
#    3. Submit:
#         sbatch --array=1-40%8 cluster/submit_array.slurm
#                          ^^^ the %8 caps it at 8 running at once, so you
#                              stay a good neighbour on a shared cluster.
#
#  OpenPBS equivalent was `qsub -J 1-40`; SLURM uses --array and exposes
#  $SLURM_ARRAY_TASK_ID instead of $PBS_ARRAY_INDEX.
# =============================================================================

#SBATCH --job-name=mumax_sweep
#SBATCH --output=logs/%x_%A_%a.out       # %A=array job ID, %a=task index
#SBATCH --error=logs/%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                        # per array task -- size with `probe`
#SBATCH --time=04:00:00
#SBATCH --array=1-1                      # overridden by --array on the CLI

set -euo pipefail

# =============================================================================
#  CONFIG
# =============================================================================
SIMLIST="${SIMLIST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/simlist.txt}"
# Each task writes into its own simulation's results folder, and mirrors to a
# central tree. Set RESULT_ROOT="" to keep only the per-simulation copy.
RESULTS_SUBDIR="results"
RESULT_ROOT="${RESULT_ROOT:-$HOME/runs/results}"

COMPONENT="My"
DT="5e-12"
T_START="3e-9"
T_END="25e-9"
N_PEAKS="5"
F_MIN="0"
F_MAX="40"
AVG_AXIS="Z"

# =============================================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/config.sh"
# shellcheck source=/dev/null
source "${HERE}/logging.sh"

# Every array task appends to the SAME run_history.log; logging.sh uses flock
# so their lines cannot interleave. Trace files are per-task (jobid_taskid).
log_init submit_array "$@"

if [[ ! -f "${SIMLIST}" ]]; then
    log_error "simulation list not found: ${SIMLIST}"
    log_error "create it with:  ls -d ~/runs/*/*.out > ${SIMLIST}"
    exit 1
fi

IDX="${SLURM_ARRAY_TASK_ID:-1}"
NLINES=$(wc -l < "${SIMLIST}")
if (( IDX < 1 || IDX > NLINES )); then
    log_error "array index ${IDX} outside 1..${NLINES}"
    exit 1
fi

SIM_DIR="$(sed -n "${IDX}p" "${SIMLIST}")"
SIM_DIR="${SIM_DIR%/}"
NAME="$(basename "${SIM_DIR}")"
# Destination 1: inside the simulation itself. No collision is possible here --
# each task writes only into the directory it was given.
SIM_RESULTS="${SIM_DIR}/${RESULTS_SUBDIR}"
# Computed from the real path, not the scratch copy cli.py will see.
SIM_LABEL="$(sim_label "${SIM_DIR}")"
# Destination 2: central mirror. Keep the parent directory name too, so
# sweep_A/sim.out and sweep_B/sim.out do not collide in one flat tree.
PARENT="$(basename "$(dirname "${SIM_DIR}")")"
CENTRAL_RESULTS="${RESULT_ROOT:+${RESULT_ROOT}/${PARENT}__${NAME}}"

log_info "array task  : ${IDX} / ${NLINES}"
log_info "sim dir     : ${SIM_DIR}"
log_info "label       : ${SIM_LABEL}"
log_info "results (1) : ${SIM_RESULTS}"
log_info "results (2) : ${CENTRAL_RESULTS:-<central mirror disabled>}"
log_info "node        : $(hostname)   cpus=${SLURM_CPUS_PER_TASK:-?}"

if [[ ! -d "${SIM_DIR}" ]]; then
    log_error "not a directory: ${SIM_DIR}"
    exit 1
fi

SCRATCH="$(pick_scratch)_${IDX}"
mkdir -p "${SCRATCH}"

# One combined trap -- see the note in submit_modes.slurm about why two
# separate `trap ... EXIT` calls would clobber each other.
cleanup() { local rc=$?; rm -rf "${SCRATCH}"; log_finish "${rc}"; }
trap cleanup EXIT
trap 'log_error "SIGTERM: scancel or walltime limit"; exit 143' TERM

# Stage under the simulation's real name so each task's bundle is named after
# its own simulation rather than all 40 sharing "sim.out_..._modes.npz".
STAGED="${SCRATCH}/${NAME}"
log_cmd rsync -a "${SIM_DIR}/" "${STAGED}/"

export PROJECT_DIR PYTHON_MODULE VENV_DIR LOG_DIR

# set +e so a failed task still copies back partial results before the EXIT
# trap wipes scratch. See the comment in submit_modes.slurm.
set +e
log_cmd "${CONTAINER_WRAPPER}" --exec bash "${HERE}/container_script.sh" \
    modes \
    --sim-dir   "${STAGED}" \
    --outdir    "${SCRATCH}/results" \
    --component "${COMPONENT}" \
    --dt        "${DT}" \
    --t-start   "${T_START}" \
    --t-end     "${T_END}" \
    --n-peaks   "${N_PEAKS}" \
    --f-min     "${F_MIN}" \
    --f-max     "${F_MAX}" \
    --avg-axis  "${AVG_AXIS}" \
    --workers   "${SLURM_CPUS_PER_TASK:-8}" \
    --bundle-name "${SIM_LABEL}"
STATUS=$?
set -e

mkdir -p "${SIM_RESULTS}"
log_cmd rsync -a "${SCRATCH}/results/" "${SIM_RESULTS}/" || true

# fft_*.npz goes back beside the simulation data, not into results/, so a rerun
# can find it and skip the FFT entirely.
log_cmd rsync -a --include='fft_*.npz' --exclude='*' \
      "${STAGED}/" "${SIM_DIR}/" || true

mkdir -p "${SIM_RESULTS}/logs"
[[ -n "${LOG_TRACE_FILE}" && -f "${LOG_TRACE_FILE}" ]] && \
    cp "${LOG_TRACE_FILE}" "${SIM_RESULTS}/logs/" 2>/dev/null || true

if [[ -n "${CENTRAL_RESULTS}" ]]; then
    mkdir -p "${CENTRAL_RESULTS}"
    log_cmd rsync -a "${SIM_RESULTS}/" "${CENTRAL_RESULTS}/" || true
fi

log_info "task ${IDX} finished with status ${STATUS}"
exit ${STATUS}
