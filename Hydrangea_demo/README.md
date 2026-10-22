# Hydrangea demo

Potential Acronyms: HYbrid Dynamics Regime Analysis of Neural Geometry, Embeddings, and Activity

 HYbrid Dynamics Regime Analysis of Neural Geometry, Embeddings, and Attractors (I think attractors are too )

Draft of the necessary methods and a potential pipeline: "Hybrifolds, Conley-More Indexes and Cognitive-Dynamic Symbols.pdf" (in this folder)

Alternatives: HYDRANGEAS
HYbrid Dynamic Regime Analysis of Neural GEometries, Attractors, and Symbols
We could apply the Conley-Morse Index
Original paper on combinatorial multivariate 
Key publication: Rivas, B., & Kalies, W. (2026). Conley Index Theory for Hybrid Systems. arXiv preprint arXiv:2606.18501.
-this could unify brain state and neural manifolds, by adding rigour to the way manifolds of each "state" are computed
Barcodes, for visualization or decomposing the dynamics into symbols:
Dey, T. K., Lipiński, M., & Soriano-Trigueros, M. (2026). Conley–Morse Persistence Barcode: A Homological Signature of Combinatorial Bifurcations. Foundations of Computational Mathematics, 1-59.

-To understand Conley-Morse theory, for
understanding the conly index read...
Mrozek, M. (2017). Conley–Morse–Forman theory for combinatorial multivector fields on Lefschetz complexes. Foundations of Computational Mathematics, 17(6), 1585-1633.
https://arxiv.org/abs/1506.00018

this work might be useful for understanding persistance of the DAG a conley-morse theory returns itself, this could be useful for distinguishing regimes from one another, evnetually understanding brain states:
Dey, T. K., Mrozek, M., & Slechta, R. (2022). Persistence of Conley--Morse Graphs in Combinatorial Dynamical Systems. SIAM Journal on Applied Dynamical Systems, 21(2), 817-839.
Also see: Kokubu, H., & Spendlove, K. A User’s Guide to the Conley-Morse Database.
https://chomp.rutgers.edu/Projects/survey/cmdbSurvey.pdf. 
also here: https://kellyspendlove.github.io/cmdbSurveyII.pdf

Note that for hyprid dynamics to be described, you
have to be able to identify the reset map.  This is partly an unsolved probelm in mathematics. But there are some attempts to work towards it:
https://arxiv.org/pdf/2511.08737



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

## UMAP_EDA

Briefly visually checking the manifolds of the DANDIset being used to test some of this, and quanitifying things like position encoding as well as LFP band correspondance among other things. 
