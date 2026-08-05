#!/bin/bash
# =============================================================================
#  cluster/container_script.sh  --  runs INSIDE the Singularity container
# =============================================================================
#  This is the "first Bash script" of the two-script pattern described in
#  Chapter 6 of the CNNL tutorial. It is never run directly by you; the SLURM
#  script calls it via:
#
#      /apps/containers/scripts/<env>/current --exec bash container_script.sh <args>
#
#  Everything after the script name is forwarded verbatim to cli.py, so:
#
#      ... container_script.sh modes --sim-dir foo.out --dt 5e-12
#
#  becomes  `python3 cli.py modes --sim-dir foo.out --dt 5e-12`  inside the
#  container, with the module environment and venv already active.
# =============================================================================
set -euo pipefail

# --- These are exported by the calling SLURM script --------------------------
: "${PROJECT_DIR:?PROJECT_DIR not set -- must be exported by the SLURM script}"
: "${PYTHON_MODULE:?PYTHON_MODULE not set}"
: "${VENV_DIR:?VENV_DIR not set}"

# --- Logging -----------------------------------------------------------------
# LOG_DIR is inherited from the calling SLURM script, so the container run
# appends to the same run_history.log rather than starting its own.
_CS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_CS_DIR}/logging.sh" ]]; then
    # shellcheck source=/dev/null
    source "${_CS_DIR}/logging.sh"
    log_init container "$@"
    log_trap_exit
else
    # Standalone fallback so the script still works if logging.sh is missing.
    log_info()  { echo "[container] $*"; }
    log_error() { echo "[container] ERROR: $*" >&2; }
    log_cmd()   { echo "[container] RUN: $*"; "$@"; }
    log_finish() { return "${1:-0}"; }
fi

# --- Container shells start unconfigured -------------------------------------
# Without this, `module` is not a command. This is error #1 in the tutorial's
# "Some Common Errors" chapter.
. /etc/profile

log_info "hostname : $(hostname)"
log_info "module   : ${PYTHON_MODULE}"

log_cmd module load "${PYTHON_MODULE}"

# --- Activate the venv built by setup_venv.sh --------------------------------
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    log_info "venv     : ${VENV_DIR}"
else
    log_info "WARNING: no venv at ${VENV_DIR}"
    log_info "         run 'bash cluster/setup_venv.sh' first."
fi

log_info "python3  : $(command -v python3)  ($(python3 --version 2>&1))"

# --- Thread control ----------------------------------------------------------
# NumPy's BLAS will otherwise spawn one thread per *physical* core on the node,
# not per core allocated to your job -- which oversubscribes the node and makes
# everything slower. Pin it to what SLURM actually gave us.
NTHREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${NTHREADS}"
export OPENBLAS_NUM_THREADS="${NTHREADS}"
export MKL_NUM_THREADS="${NTHREADS}"
export NUMEXPR_NUM_THREADS="${NTHREADS}"
log_info "threads  : ${NTHREADS}"

# --- Headless matplotlib -----------------------------------------------------
export MPLBACKEND=Agg
# Compute nodes often have a read-only or missing HOME cache dir; point
# matplotlib's font cache somewhere writable to avoid a startup warning storm.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mplconfig_$$}"
mkdir -p "${MPLCONFIGDIR}"

# --- Run ---------------------------------------------------------------------
cd "${PROJECT_DIR}"
log_info "cwd      : $(pwd)"

# Hand cli.py the same log directory, so its Python-side log lands next to the
# bash logs and its own run lines go into the same run_history.log.
CLI_ARGS=("$@")
if [[ -n "${LOG_DIR:-}" ]]; then
    CLI_ARGS+=(--log-dir "${LOG_DIR}")
fi

set +e
log_cmd python3 cli.py "${CLI_ARGS[@]}"
status=$?
set -e

log_info "cli.py exited with status ${status}"
exit ${status}
