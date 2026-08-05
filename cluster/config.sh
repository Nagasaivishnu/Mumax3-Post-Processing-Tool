#!/bin/bash
# =============================================================================
#  cluster/config.sh  --  single place to edit site-specific settings
# =============================================================================
#  Every other script in this directory sources this file. If something about
#  the cluster changes (new container path, different module name, different
#  partition), change it HERE and nowhere else.
#
#  Values marked  <<< CHECK  are the ones most likely to need adjusting for
#  your account. The commands to verify each one are given in the comments.
# =============================================================================

# --- Container ---------------------------------------------------------------
# Wrapper script that launches Singularity. From the CNNL tutorial, Chapter 3.
# On the Feynman workstations this is /apps_SEEDER01/containers/scripts instead.
CONTAINER_ROOT="${CONTAINER_ROOT:-/apps/containers/scripts}"

# Which container environment. 'nntools' has the Miniconda python3 stack
# (numpy/scipy already present); 'umag' is the one that runs MuMax3 itself.
# Verify with:  ls /apps/containers/scripts
CONTAINER_ENV="${CONTAINER_ENV:-nntools}"                       # <<< CHECK

CONTAINER_WRAPPER="${CONTAINER_ROOT}/${CONTAINER_ENV}/current"

# --- Python inside the container ---------------------------------------------
# Environment Module that puts a suitable python3 on PATH inside the container.
# List what is available with:
#   /apps/containers/scripts/nntools/current --shell
#   bash; . /etc/profile; module av
# then pick one, e.g. pytorch/py3/2.2.0  (we only need its Python + numpy)
PYTHON_MODULE="${PYTHON_MODULE:-pytorch/py3/2.2.0}"             # <<< CHECK

# Virtual environment that holds the extra libraries this project needs.
# Lives in your home dir so it persists between jobs. Created by setup_venv.sh.
VENV_DIR="${VENV_DIR:-$HOME/.venvs/mumax-postproc}"

# --- Project ------------------------------------------------------------------
# Where you cloned/copied this repository on the cluster.
PROJECT_DIR="${PROJECT_DIR:-$HOME/Mumax3-Post-Processing-Tool}"  # <<< CHECK

# --- SLURM defaults -----------------------------------------------------------
# Find your partitions with:  sinfo -s
# Leave SLURM_PARTITION empty to use the cluster default partition.
SLURM_PARTITION="${SLURM_PARTITION:-}"                           # <<< CHECK
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"                               # optional

# Post-processing here is pure NumPy on the CPU -- it does NOT need a GPU.
# Leave this empty so your jobs are not stuck behind the GPU queue.
# Set to e.g. "gpu:1" only if you later add CuPy/GPU FFT.
SLURM_GRES="${SLURM_GRES:-}"

# --- Scratch ------------------------------------------------------------------
# Fast node-local NVMe. SLURM sets different variables on different sites, so
# we try them in order and fall back to /tmp. Anything left here is deleted
# when the job ends -- always rsync results back out.
#  Note the ":-" defaults everywhere: these scripts run under `set -u`, and on
#  a site that does not define SLURM_TMPDIR an unguarded reference is a fatal
#  "unbound variable" error rather than a graceful fallback.
pick_scratch() {
  local base
  for base in "${SLURM_TMPDIR:-}" "${TMPDIR:-}" "/scratch/${USER:-nobody}" "/tmp"; do
    if [[ -n "$base" && -d "$base" && -w "$base" ]]; then
      echo "${base}/mumax_${SLURM_JOB_ID:-$$}"
      return 0
    fi
  done
  echo "/tmp/mumax_$$"
}

# --- Sanity check -------------------------------------------------------------
config_check() {
  local ok=0
  if [[ ! -x "$CONTAINER_WRAPPER" && ! -f "$CONTAINER_WRAPPER" ]]; then
    echo "WARNING: container wrapper not found: $CONTAINER_WRAPPER" >&2
    echo "         run 'ls ${CONTAINER_ROOT}' and fix CONTAINER_ENV in config.sh" >&2
    ok=1
  fi
  if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "WARNING: PROJECT_DIR does not exist: $PROJECT_DIR" >&2
    ok=1
  elif [[ ! -f "$PROJECT_DIR/cli.py" ]]; then
    echo "WARNING: $PROJECT_DIR does not look like the project (no cli.py)" >&2
    ok=1
  fi
  return $ok
}
