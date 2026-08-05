#!/bin/bash
# =============================================================================
#  cluster/submit_batch.slurm  --  run a JSON list of mixed analyses
# =============================================================================
#  For FMR heatmaps, hysteresis overlays, or any mix of analyses described in a
#  single config file. Lighter than submit_modes.slurm because table.txt work
#  needs far less RAM than OVF work.
#
#      sbatch cluster/submit_batch.slurm
#      sbatch --export=ALL,CONFIG_FILE=$HOME/my_jobs.json cluster/submit_batch.slurm
#
#  See jobs.example.json for the config format.
# =============================================================================

#SBATCH --job-name=mumax_batch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

# =============================================================================
#  CONFIG
# =============================================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${HERE}/jobs.example.json}"
RESULT_DIR="${RESULT_DIR:-$HOME/runs/results}"

# =============================================================================
# shellcheck source=/dev/null
source "${HERE}/config.sh"
# shellcheck source=/dev/null
source "${HERE}/logging.sh"

mkdir -p "${RESULT_DIR}"
log_init submit_batch "$@"
log_trap_exit

if [[ ! -f "${CONFIG_FILE}" ]]; then
    log_error "config file not found: ${CONFIG_FILE}"
    exit 1
fi

log_info "job id  : ${SLURM_JOB_ID:-none}"
log_info "node    : $(hostname)"
log_info "config  : ${CONFIG_FILE}"
# Record the config verbatim in the log. Six months from now, "which parameters
# produced this figure?" is answerable from the log alone.
log_info "config contents:"
sed 's/^/    /' "${CONFIG_FILE}"

config_check || true

export PROJECT_DIR PYTHON_MODULE VENV_DIR LOG_DIR

set +e
log_cmd "${CONTAINER_WRAPPER}" --exec bash "${HERE}/container_script.sh" \
    run --config "${CONFIG_FILE}" --keep-going
STATUS=$?
set -e

log_info "finished with status ${STATUS} -- results under ${RESULT_DIR}"

if command -v sacct >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    sacct -j "${SLURM_JOB_ID}" \
          --format=JobID,JobName%20,Elapsed,MaxRSS,State || true
fi

exit ${STATUS}
