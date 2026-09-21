"""Notebook-ready object test script for Hydrangea demo.

Notebook context:
    /Users/anguscampbell/CumulantDynamicsGuiltByAssociation/test_GonzalezHpSpatialNavTask_dandiset.ipynb
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_dynamics_core.base import HyDan

NOTEBOOK_PATH = Path("/notebooks/CumulantDynamicsGuiltByAssociation/test_GonzalezHpSpatialNavTask_dandiset.ipynb")

# Assumes the following notebook objects already exist:
#   spike_group
#   maze_ep
#   position_xy
#   spike_group_maze
#   bin_times_s
#   session_nwb

# Example usage: create a Hydrangea object and call steps in order.
obj = HyDan(
    spike_group=spike_group_maze,
    bin_size_s=0.050,
    maze_epoch=maze_ep,
    random_state=0,
    report="selected",
)

# 1) normalize
obj.normalize(method="proportion_zscore", zscore=True, restrict_to_epoch=True)
obj.normalization_report(show=False)

# 2) global reduction
obj.global_pca(n_components=10, whiten=False, standardize=True)
obj.pca_report(show=False)

# 3) build folds
obj.build_folds(n_folds=5, strategy="temporal_block_randomized", seed=0)

# 4) fit HMM models
obj.fit_hmm(
    k_values=list(range(10, 1, -1)),
    n_iter=100,
    covariance_type="diag",
    verbose=False,
    use_cv=True,
    report="selected",
)

# 5) reporting
obj.hmm_report(report="selected")
# obj.hmm_report(report="full")
# obj.hmm_report(report="none")

print("Notebook path:", NOTEBOOK_PATH)
print("Best K:", obj.best_k)
print("Model count:", len(obj.hmm_models))
