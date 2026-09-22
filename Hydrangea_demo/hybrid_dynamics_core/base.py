from .normalization import NormalizationStep
from .embeddings import EmbeddingStep
from .hmm_fitting import HMMFittingStep


class HybridDynamicsAnalysis(
    NormalizationStep,
    EmbeddingStep,
    HMMFittingStep,
):
    """Pipeline object for session-wide embeddings, HMM fitting, and local state embeddings.

    Typical call order:
        hybdyn.normalize(...)
        hybdyn.global_embedding(...)
        hybdyn.fit_hmm(...)
        hybdyn.local_embedding(...)
        hybdyn.report_embeddings(...)
        hybdyn.hmm_report(...)

    The global embedding is the session-wide representation that feeds the HMM.
    Local embeddings are computed later from subsets of the same session, usually
    after HMM states are available, so the same embedding machinery can be reused
    across the workflow.
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
        self.raw_counts = None
        self.neuron_totals = None
        self.unit_metadata_filtered = {}

        self.embedding_results = None
        self.embedding_model = None
        self.embedding_scores = None
        self.embedding_variance_ratio = None
        self.embedding_scope = None
        self.embedding_label = None

        self.global_embedding_results = None
        self.global_embedding_model = None
        self.global_embedding_scores = None
        self.global_embedding_variance_ratio = None

        self.local_embedding_results = {}
        self.local_embedding_model = None
        self.local_embedding_scores = None
        self.local_embedding_variance_ratio = None
        self.local_embedding_label = None

        self.fold_idx = None
        self.hmm_models = {}
        self.hmm_scores = None
        self.best_k = None
        self.best_hmm = None
