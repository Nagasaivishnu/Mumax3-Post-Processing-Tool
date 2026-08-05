#!/bin/bash
# =============================================================================
#  cluster/setup_venv.sh  --  ONE-TIME setup of the Python environment
# =============================================================================
#  Run this once, from a LOGIN NODE. It enters the container, loads the module
#  that provides python3, and builds a virtual environment in your home
#  directory containing the libraries this project needs.
#
#      bash cluster/setup_venv.sh
#
#  The venv lives in $VENV_DIR (see config.sh) so it survives job teardown and
#  is reused by every subsequent job.
#
#  This project is headless: no PyQt6, no Qt, no X11. If you previously ran this
#  with --with-gui, that flag is gone and PyQt6 is no longer installed.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${HERE}/config.sh"
# shellcheck source=/dev/null
source "${HERE}/logging.sh"

if [[ "${1:-}" == "--with-gui" ]]; then
    echo "NOTE: --with-gui no longer exists. The PyQt6 interface was removed;" >&2
    echo "      this project is headless. Continuing without it." >&2
fi

log_init setup_venv "$@"
log_trap_exit

log_info "container : ${CONTAINER_WRAPPER}"
log_info "module    : ${PYTHON_MODULE}"
log_info "venv      : ${VENV_DIR}"

config_check || log_info "(continuing anyway -- fix the warnings above if this fails)"

# This heredoc is executed INSIDE the container. Logged as one unit because the
# pip installs inside it produce their own detailed output.
log_info "RUN: ${CONTAINER_WRAPPER} --exec bash -s  (venv build heredoc)"
"${CONTAINER_WRAPPER}" --exec bash -s <<EOF
set -euo pipefail

# The container shell starts unconfigured; this is what makes 'module' exist.
# (See "Some Common Errors" #1 in the CNNL tutorial.)
. /etc/profile

echo ">> loading module ${PYTHON_MODULE}"
module load ${PYTHON_MODULE} || {
    echo "ERROR: could not load '${PYTHON_MODULE}'." >&2
    echo "Available modules:" >&2
    module av 2>&1 | head -60 >&2
    echo "Pick one and set PYTHON_MODULE in cluster/config.sh" >&2
    exit 1
}

PY=\$(command -v python3)
echo ">> python3 = \$PY"
\$PY --version

if [[ ! -d "${VENV_DIR}" ]]; then
    echo ">> creating venv (with --system-site-packages so the container's"
    echo "   optimised numpy/scipy build is reused instead of recompiled)"
    \$PY -m venv --system-site-packages "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python3 -m pip install --upgrade pip setuptools wheel

echo ">> installing headless dependencies"
python3 -m pip install \
    "numpy>=1.23" \
    "scipy>=1.10" \
    "pandas>=1.5" \
    "matplotlib>=3.7" \
    "python-pptx>=0.6.21" \
    "pillow"

echo
echo ">> verifying imports"
python3 - <<'PYEOF'
import matplotlib
matplotlib.use("Agg")          # no display on a compute node, ever
import numpy, scipy, pandas, matplotlib.pyplot
print(f"  numpy       {numpy.__version__}")
print(f"  scipy       {scipy.__version__}")
print(f"  pandas      {pandas.__version__}")
print(f"  matplotlib  {matplotlib.__version__}")
try:
    import pptx; print(f"  python-pptx {pptx.__version__}")
except Exception as e:
    print(f"  python-pptx MISSING ({e}) -- --pptx export will be skipped")
PYEOF
EOF

log_info "Done. The venv at ${VENV_DIR} is now reused by every job."
log_info "Next:  sbatch cluster/submit_modes.slurm  (edit the CONFIG block first)"
