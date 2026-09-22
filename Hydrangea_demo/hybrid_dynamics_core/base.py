from .normalization import NormalizationStep
from .global_reduction import GlobalReductionStep
from .hmm_fitting import HMMFittingStep
from .hmm_report import HMMReportStep


class HybridDynamicsAnalysis(
    NormalizationStep,
    GlobalReductionStep,
    HMMFittingStep,
    HMMReportStep,
):
    """Pipeline object for spike normalization, reduction, HMM fitting, and reporting.

    Typical call order:
        hybdyn.normalize(...)
        hybdyn.global_pca(...)
        hybdyn.build_folds(...)
        hybdyn.fit_hmm(...)
        hybdyn.hmm_report(...)

    Parameters
    ----------
    spike_group : pynapple object
        Spike train object passed into the workflow. This may already be a
        restricted object, or may be a full TsGroup whose metadata and session
        context are supplied separately.
    bin_size_s : float
        Temporal bin width used for spike counts.
    maze_epoch : pynapple.IntervalSet or None
        Optional epoch restriction before fitting.
    unit_metadata : dict or None
        Optional mapping from metadata names to per-unit arrays or labels
        (e.g. {"cell_type": ..., "cell_area": ...}). This lets the analysis
        operate on real session metadata without assuming fixed variable names.
    unit_id_key : str or None
        Optional identifier for the unit key column when metadata are provided.
    region_key : str or None
        Optional metadata field used to label a region or anatomical group.
    condition_key : str or None
        Optional metadata field used to label behavioral or experimental
        conditions.
    random_state : int
        Seed used for reproducible folds and HMM initialisation.
    report : {'full', 'selected', 'none'}
        Default reporting mode for Hmm output.
    """

    def __init__(
        self,
        spike_group,
        bin_size_s=0.050,
        maze_epoch=None,
        unit_metadata=None,
        unit_id_key=None,
        region_key=None,
        condition_key=None,
        random_state=0,
        report="full",
    ):
        self.spike_group = spike_group
        self.bin_size_s = float(bin_size_s)
        self.maze_epoch = maze_epoch
        self.unit_metadata = unit_metadata if unit_metadata is not None else {}
        self.unit_id_key = unit_id_key
        self.region_key = region_key
        self.condition_key = condition_key
        self.random_state = random_state
        self.report = report

        self._source_spike_group = None
        self.spike_matrix = None
        self.bin_times_s = None
        self.unit_ids = None
        self.normalization_method = None

        self.pca_model = None
        self.pca_scores = None
        self.pca_variance_ratio = None

        self.fold_idx = None
        self.hmm_models = {}
        self.hmm_scores = None
        self.best_k = None
        self.best_hmm = None
