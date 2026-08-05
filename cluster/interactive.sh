#!/bin/bash
# =============================================================================
#  cluster/interactive.sh  --  get a shell on a compute node, inside the container
# =============================================================================
#  This is the SLURM equivalent of the tutorial's Chapter 4 workflow
#  (`qsub -I` followed by `/apps/containers/scripts/<env>/current --shell`),
#  collapsed into one command.
#
#      bash cluster/interactive.sh                  # 8 cpus, 32G, 4 hours
#      bash cluster/interactive.sh -c 32 -m 256G -t 08:00:00
#
#  Options:
#    -c N     CPUs            (default 8)
#    -m SIZE  memory          (default 32G)
#    -t TIME  walltime        (default 04:00:00)
#    -w NODE  specific node   (default: let SLURM choose)
#
#  Use this for poking at data, checking a module name, or running cli.py by
#  hand. For real work use sbatch -- an interactive session dies with your SSH
#  connection, a batch job does not.
#
#  NOTE: there is no GUI mode. This project is headless; the PyQt6 interface was
#  removed. Compute here, plot on your own machine from the .npz files.
# =============================================================================
set -euo pipefail

CPUS=8
MEM=32G
TIME=04:00:00
NODE=""

while getopts "c:m:t:w:h" opt; do
    case "$opt" in
        c) CPUS="$OPTARG" ;;
        m) MEM="$OPTARG" ;;
        t) TIME="$OPTARG" ;;
        w) NODE="$OPTARG" ;;
        h) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option; try -h" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/config.sh"
# shellcheck source=/dev/null
source "${HERE}/logging.sh"

# Trace is off for interactive sessions: xtrace would scroll past everything you
# type. The audit trail still records that you opened a session, on what node,
# with what resources.
LOG_TRACE=0
log_init interactive "$@"

config_check || true

FLAGS=(--nodes=1 --ntasks=1
       --cpus-per-task="${CPUS}"
       --mem="${MEM}"
       --time="${TIME}"
       --job-name="mumax_shell")

[[ -n "$NODE"            ]] && FLAGS+=(--nodelist="$NODE")
[[ -n "$SLURM_PARTITION" ]] && FLAGS+=(--partition="$SLURM_PARTITION")
[[ -n "$SLURM_ACCOUNT"   ]] && FLAGS+=(--account="$SLURM_ACCOUNT")
[[ -n "$SLURM_GRES"      ]] && FLAGS+=(--gres="$SLURM_GRES")

log_info "resources : ${CPUS} cpus, ${MEM} mem, ${TIME} walltime${NODE:+, node ${NODE}}"
log_info "container : ${CONTAINER_WRAPPER}"

echo
echo "Available nodes (SLURM equivalent of 'pbsnodes -ajSL'):"
sinfo -N -o "%.18N %.9P %.6t %.5c %.9m %.20G" 2>/dev/null | head -25 || true
echo

export PROJECT_DIR PYTHON_MODULE VENV_DIR LOG_DIR

log_info "RUN: srun ${FLAGS[*]} --pty ${CONTAINER_WRAPPER} --shell"
echo "   Once inside, set up the environment with:"
echo "       . /etc/profile"
echo "       module load ${PYTHON_MODULE}"
echo "       source ${VENV_DIR}/bin/activate"
echo "       cd ${PROJECT_DIR}"
echo
echo "   Type 'exit' twice when done: once to leave the container, once to"
echo "   release the SLURM allocation."
echo

# `exec` replaces this shell, so log_finish would never run -- close the record
# now, before handing off.
log_finish 0 >/dev/null
exec srun "${FLAGS[@]}" --pty "${CONTAINER_WRAPPER}" --shell
