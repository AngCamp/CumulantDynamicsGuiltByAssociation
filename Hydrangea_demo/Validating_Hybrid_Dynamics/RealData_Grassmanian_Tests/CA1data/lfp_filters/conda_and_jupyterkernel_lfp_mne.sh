#!/bin/bash

# Env for the CA1 LFP band-power notebook (Hp_LFP_bandpower_mne_001695.ipynb in this folder).
# Same pattern as Hydrangea_demo/env_setup/conda_and_jupyterkernel_startup.sh: make a conda env,
# pip-install into it, register it as a Jupyter kernel so the notebook can select it on the server.
#   bash conda_and_jupyterkernel_lfp_mne.sh
# If your conda install is somewhere else, change this one path.
echo 'Activating conda...'
source /storage/miniconda3/bin/activate

ENV_NAME="lfp_mne_env"

# Create the env if needed.
conda create -y -n "${ENV_NAME}" python=3.11 pip || true

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install \
  numpy \
  pandas \
  scipy \
  matplotlib \
  jupyter \
  ipykernel \
  pynapple \
  pynwb \
  h5py \
  dandi \
  dandischema \
  remfile \
  mne

python -m ipykernel install --user --name="${ENV_NAME}" --display-name="${ENV_NAME}"
conda deactivate

echo 'Done.'
echo "Kernel registered: ${ENV_NAME}  (select it in the notebook's kernel menu)"
