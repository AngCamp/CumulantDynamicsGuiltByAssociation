#!/bin/bash

# Conda and Jupyter Kernel Startup Script for Hydrangea
# Run this at the start of each session to activate conda and register kernels.

echo 'Activating conda...'
source /storage/miniconda3/bin/activate

echo 'Registering hydrangea_env kernel...'
conda activate /storage/conda_envs/hydrangea_env
python -m ipykernel install --user --name=hydrangea_env --display-name='hydrangea_env'
conda deactivate

echo 'Registering allensdk_env kernel...'
conda activate /storage/conda_envs/allensdk_env
python -m ipykernel install --user --name=allensdk_env --display-name='allensdk_env'
conda deactivate

echo 'Registering mne_env kernel...'
conda activate /storage/conda_envs/mne_env
python -m ipykernel install --user --name=mne_env --display-name='mne_env'
conda deactivate

echo 'Registering bugeon_dynamics kernel...'
conda activate /storage/conda_envs/bugeon_dynamics
python -m ipykernel install --user --name=bugeon_dynamics --display-name='bugeon_dynamics'
conda deactivate

echo 'Creating and registering pynapple_dandi_env kernel...'
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create --yes --prefix /storage/conda_envs/pynapple_dandi_env pip python=3.11
conda activate /storage/conda_envs/pynapple_dandi_env
conda install --yes conda-forge::dandi
pip install pynapple
pip install ipykernel
python -m ipykernel install --user --name=pynapple_dandi_env --display-name='pynapple_dandi_env'
conda deactivate

echo 'Done! Kernels registered. Refresh your Jupyter page.'
echo 'Available kernels:'
jupyter kernelspec list
