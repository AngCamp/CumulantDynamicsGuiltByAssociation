#!/bin/bash

# Minimal setup for the Hydrangea project env.
# If your conda install is somewhere else, change this one path.
echo 'Activating conda...'
source /storage/miniconda3/bin/activate

ENV_NAME="hydrangea_env"

# Create the env if needed.
conda create -y -n "${ENV_NAME}" python=3.11 pip || true

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install \
  numpy \
  pandas \
  scipy \
  matplotlib \
  scikit-learn \
  hmmlearn \
  jupyter \
  ipykernel \
  pynapple \
  dandi \
  dandischema \
  pynwb \
  remfile

python -m ipykernel install --user --name="${ENV_NAME}" --display-name="${ENV_NAME}"
conda deactivate

echo 'Done.'
echo "Kernel registered: ${ENV_NAME}"
