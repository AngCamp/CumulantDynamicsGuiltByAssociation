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


HyDan = HybridDynamicsAnalysis
