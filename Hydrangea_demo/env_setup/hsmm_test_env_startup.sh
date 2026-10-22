#!/bin/bash

# Test env for pyhsmm + pyhsmm-autoregressive (HDP-AR-HSMM).
# Both repos are unmaintained since ~2018: Cython extensions, np.float etc.,
# so this pins an old Python and numpy<1.24. Expect to iterate on this.
echo 'Activating conda...'
source /storage/miniconda3/bin/activate

ENV_NAME="hsmm_test_env"

conda create -y -n "${ENV_NAME}" python=3.8 pip || true
conda activate "${ENV_NAME}"

# compilers + Eigen for pyhsmm-autoregressive's C++ extension
conda install -y -c conda-forge gxx_linux-64 eigen

python -m pip install --upgrade "pip<24" "setuptools<60" wheel
python -m pip install "numpy<1.24" "scipy<1.11" "cython<3" future nose requests
python -m pip install \
  pandas matplotlib scikit-learn jupyter ipykernel \
  pynapple dandi dandischema pynwb remfile

python -m pip install --no-build-isolation git+https://github.com/mattjj/pybasicbayes.git
python -m pip install --no-build-isolation git+https://github.com/mattjj/pyhsmm.git
python -m pip install --no-build-isolation git+https://github.com/mattjj/pyhsmm-autoregressive.git

# smoke test: import both and build a tiny model
python - <<'EOF'
import numpy as np, pyhsmm, autoregressive
from autoregressive.models import ARWeakLimitStickyHDPHMM, ARWeakLimitHDPHSMM
print("pyhsmm", pyhsmm.__file__)
print("autoregressive", autoregressive.__file__)
EOF

python -m ipykernel install --user --name="${ENV_NAME}" --display-name="${ENV_NAME}"
conda deactivate

echo 'Done.'
echo "Kernel registered: ${ENV_NAME}"
