# Hydrangea demo

This folder holds the early-stage analysis object for hybrid dynamics modeling.
The design is object-oriented, modular, and ordered by analysis step.
The remote notebook path is expected at /notebooks/CumulantDynamicsGuiltByAssociation/test_GonzalezHpSpatialNavTask_dandiset.ipynb.

## structure
- hybrid_dynamics_core/
  - base.py
  - normalization.py
  - global_reduction.py
  - hmm_fitting.py
  - hmm_report.py
  - local_state_embeddings.py
- env_setup/
  - conda_and_jupyterkernel_startup.sh
- test_hydan_obj.py

## ordering
1. normalize()
2. normalization_report()
3. global_pca()
4. pca_report()
5. build_folds()
6. fit_hmm()
7. hmm_report()
8. later: find_local_embedding_windows()
