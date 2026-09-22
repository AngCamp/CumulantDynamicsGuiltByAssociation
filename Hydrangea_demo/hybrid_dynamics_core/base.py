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
        obj.normalize(...)
        obj.global_pca(...)
        obj.build_folds(...)
        obj.fit_hmm(...)
        obj.hmm_report(...)

    Parameters
    ----------
    spike_group : pynapple object
        Spike train object passed into the workflow.
    bin_size_s : float
        Temporal bin width used for spike counts.
    maze_epoch : pynapple.IntervalSet or None
        Optional epoch restriction before fitting.
    random_state : int
        Seed used for reproducible folds and HMM initialisation.
    report : {'full', 'selected', 'none'}
        Default reporting mode for Hmm output.
    """

    def __init__(self, spike_group, bin_size_s=0.050, maze_epoch=None, random_state=0, report="full"):
        self.spike_group = spike_group
        self.bin_size_s = float(bin_size_s)
        self.maze_epoch = maze_epoch
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
