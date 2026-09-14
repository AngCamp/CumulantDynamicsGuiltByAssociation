#!/usr/bin/env bash
#
# setup_env.sh
# Creates a conda environment for streaming DANDI archive sessions and
# analyzing them with Pynapple, with parallelization support.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# Then activate with:
#   conda activate dandi-nap
#
# ---------------------------------------------------------------------------

set -euo pipefail

ENV_NAME="dandi-nap"

echo ">>> Creating conda environment: ${ENV_NAME}"

# --- Make sure conda is usable inside this non-interactive shell ------------
# (conda activate needs the shell hook sourced first)
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# --- Create the base environment --------------------------------------------
# Pynapple's docs recommend a fresh conda env with pip + python, then pip
# installing pynapple. We put the conda-forge scientific/parallel/streaming
# stack in via conda, then pip-install the pynapple + dandi layer on top.
#
# Python 3.11 is compatible with numba >= 0.60 (required by pynapple).
conda create -y -n "${ENV_NAME}" -c conda-forge \
    python=3.11 \
    pip \
    numpy \
    pandas \
    scipy \
    h5py \
    hdf5 \
    xarray \
    dask \
    distributed \
    joblib \
    tqdm \
    ipython \
    jupyterlab

echo ">>> Activating ${ENV_NAME} and installing pip packages"
conda activate "${ENV_NAME}"

# --- pip layer --------------------------------------------------------------
# - numba>=0.60 first, so pynapple's numba/llvmlite pin resolves cleanly
#   on Python 3.11 (per pynapple install docs).
# - pynapple: the neural analysis package.
# - dandi: DANDI CLI + Python API client.
# - pynwb: NWB reader (pynapple dependency; pinned to >=2.0).
# - remfile / fsspec / aiohttp / requests: remote streaming of NWB, so you
#   can pull one session at a time without full downloads.
python -m pip install --upgrade pip
python -m pip install \
    "numba>=0.60" \
    pynapple \
    dandi \
    dandischema \
    "pynwb>=2.0" \
    remfile \
    fsspec \
    aiohttp \
    requests

echo ""
echo ">>> Done."
echo ">>> Verifying key imports..."
python - <<'PY'
import importlib
for mod in ["pynapple", "pynwb", "dandi", "xarray", "dask", "joblib", "remfile"]:
    importlib.import_module(mod)
    print(f"  ok: {mod}")
PY

echo ""
echo "Environment '${ENV_NAME}' is ready."
echo "Activate it in a new shell with:  conda activate ${ENV_NAME}"