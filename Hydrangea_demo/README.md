# Hydrangea demo

Potential Acronyms: HYbrid Dynamics Regime Analysis of Neural Geometry, Embeddings, and Activity

 HYbrid Dynamics Regime Analysis of Neural Geometry, Embeddings, and Attractors (I think attractors are too )

This folder holds the early-stage analysis object for hybrid dynamics modeling.
The design is object-oriented, modular, and ordered by analysis step.
The remote notebook path is expected under /notebooks/CumulantDynamicsGuiltByAssociation/Hydrangea_demo/.
The data archive is mounted under /storage/dandi_downloads/ and includes subject folders such as sub-M01, sub-M02, sub-M03, and sub-M05.

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
5. build_folds(n_folds=5, fold_strategy="temporal_segments", shuffle_within_segments=True)
6. fit_hmm(n_states_min=2, n_states_max=10, n_folds=5, fold_strategy="temporal_segments", shuffle_within_segments=True)
7. hmm_report(report="selected")
8. later: find_local_embedding_windows()

## API notes
- `build_folds` uses a simple integer `n_folds` value, not a list of K values.
- `fit_hmm` uses a simple integer state range: `n_states_min` and `n_states_max`.
- `fold_strategy` controls how the validation sets are formed. The default is temporal segments with optional randomization inside each segment.
- `report` accepts `"full"`, `"selected"`, or `"none"`.
