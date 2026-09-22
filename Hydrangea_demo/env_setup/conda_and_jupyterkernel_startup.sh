#!/bin/bash

# Single project environment for the Hydrangea / DANDI / pynapple workflow.
# If your conda install lives somewhere else, update this one path only.
echo 'Activating conda...'
source /storage/miniconda3/bin/activate

ENV_NAME="hydrangea_env"
ENV_PATH="/storage/conda_envs/${ENV_NAME}"

# Create the environment if it does not already exist.
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
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

echo 'Done.'
echo "Kernel registered: ${ENV_NAME}"
