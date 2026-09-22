#!/usr/bin/env bash
set -euo pipefail

# Single environment for the Hydrangea / DANDI / pynapple analysis pipeline.
# This project does not need the unrelated allensdk / mne / bugeon envs.
# The actual code uses:
#   - numpy, scipy, pandas, matplotlib
#   - scikit-learn, hmmlearn
#   - pynapple, pynwb, dandi
#   - jupyter/ipykernel

ENV_NAME="hydrangea_env"
ENV_PATH="/storage/conda_envs/${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH. Activate your conda base first." >&2
  exit 1
fi

# Source conda in this non-interactive shell.
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "Environment ${ENV_NAME} already exists; reusing it."
else
  echo "Creating conda environment: ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" -c conda-forge \
    python=3.11 \
    pip \
    numpy \
    pandas \
    scipy \
    matplotlib \
    scikit-learn \
    hmmlearn \
    joblib \
    h5py \
    xarray \
    dask \
    distributed \
    tqdm \
    ipykernel \
    jupyterlab \
    numba \
    requests \
    aiohttp \
    fsspec
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install \
  pynapple \
  dandi \
  dandischema \
  pynwb \
  remfile

python -m ipykernel install --user --name="${ENV_NAME}" --display-name="${ENV_NAME}"

python - <<'PY'
import importlib
mods = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "sklearn",
    "hmmlearn",
    "pynapple",
    "pynwb",
    "dandi",
    "joblib",
]
for mod in mods:
    importlib.import_module(mod)
    print(f"OK: {mod}")
PY

echo ""
echo "Environment ready: ${ENV_NAME}"
echo "Activate it with: conda activate ${ENV_NAME}"
echo "The Jupyter kernel is registered as: ${ENV_NAME}"
jupyter kernelspec list
