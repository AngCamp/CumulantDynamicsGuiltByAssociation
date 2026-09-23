from .metadata import MetadataStep
from .spiking_and_behaviour_eda import SpikingBehaviorEDAStep
from .normalization import NormalizationStep
from .embeddings import EmbeddingStep
from .hmm_fitting import HMMFittingStep


class HybridDynamicsAnalysis(
    MetadataStep,
    SpikingBehaviorEDAStep,
    NormalizationStep,
    EmbeddingStep,
    HMMFittingStep,
):
    """Pipeline object for session-wide embeddings, HMM fitting, and local state embeddings.

    Typical call order:
        hybdyn.add_continuous_behavior(...) / hybdyn.add_discrete_events(...)
        hybdyn.spiking_behavior_report(...)
        hybdyn.normalize(...)
        hybdyn.global_embedding(...)
        hybdyn.fit_states(...)
        hybdyn.local_embedding(...)
        hybdyn.report_embeddings(...)
        hybdyn.hmm_report(...)

    The global embedding is the session-wide representation that feeds the HMM.
    Local embeddings are computed later from subsets of the same session, usually
    after HMM states are available, so the same embedding machinery can be reused
    across the workflow.

    Metadata (unit, trial, condition, region tables) and behaviour channels are
    registered up front. They are not used by the fitting steps, but they are
    carried through unit filtering and bin alignment so the discovered states can
    later be related to anatomy, cell class, trial structure, and behaviour.
    """

    def __init__(
        self,
        spike_group,
        bin_size_s=0.050,
        maze_epoch=None,
        unit_metadata=None,
        trial_table=None,
        condition_table=None,
        region_table=None,
        unit_id_key=None,
        region_key=None,
        cell_type_key=None,
        condition_key=None,
        random_state=0,
        report="full",
        embedding_method="pca",
        state_discovery_method="gaussian_hmm",
        session_id=None,
        mouse_id=None,
    ):
        self.spike_group = spike_group
        self.bin_size_s = float(bin_size_s)
        self.maze_epoch = maze_epoch
        self.unit_id_key = unit_id_key
        self.region_key = region_key
        self.cell_type_key = cell_type_key
        self.condition_key = condition_key
        self.random_state = random_state
        self.report = report
        self.embedding_method = str(embedding_method).lower()
        self.state_discovery_method = str(state_discovery_method).lower()
        self.analysis_stage = "State_EDA"

        if self.embedding_method not in {"pca", "cca"}:
            raise ValueError("embedding_method must be one of: 'pca' or 'cca'.")
        if self.state_discovery_method not in {"gaussian_hmm"}:
            raise ValueError("state_discovery_method must be one of: 'gaussian_hmm'.")

        self._init_metadata(
            spike_group=spike_group,
            unit_metadata=unit_metadata,
            trial_table=trial_table,
            condition_table=condition_table,
            region_table=region_table,
        )
        self._init_behavior()

        self._source_spike_group = None
        self.spike_matrix = None
        self.bin_times_s = None
        self.unit_ids = None
        self.normalization_method = None
        self.raw_counts = None
        self.neuron_totals = None

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
        self.state_labels_ = None

        self.analysis_checkpoints = {
            "normalized": False,
            "global_embedding": False,
            "states_discovered": False,
            "local_embedding": False,
        }

        self.sessions = []
        self.register_session(
            session_id=session_id,
            mouse_id=mouse_id,
            spike_group=spike_group,
            maze_epoch=maze_epoch,
            unit_metadata=unit_metadata,
            note="primary",
        )

    def register_session(self, session_id=None, mouse_id=None, spike_group=None, maze_epoch=None, unit_metadata=None, note=None):
        """Register session-level metadata for later multi-session analysis."""
        entry = {
            "session_id": session_id,
            "mouse_id": mouse_id,
            "has_spike_group": spike_group is not None,
            "has_maze_epoch": maze_epoch is not None,
            "has_unit_metadata": unit_metadata is not None,
            "note": note,
        }
        self.sessions.append(entry)
        return entry

    def _mark_checkpoint(self, key):
        if key in self.analysis_checkpoints:
            self.analysis_checkpoints[key] = True

    def set_analysis_stage(self, stage):
        """Set high-level workflow stage for cross-session tracking."""
        allowed = {"State_EDA", "OT_LINKING", "COMPLETED"}
        if stage not in allowed:
            raise ValueError(f"stage must be one of: {sorted(allowed)}")
        self.analysis_stage = stage
        return self.analysis_stage

    def _current_analysis_point(self):
        if self.analysis_checkpoints.get("local_embedding", False):
            return "Local embeddings computed"
        if self.analysis_checkpoints.get("states_discovered", False):
            return "States discovered (HMM fit)"
        if self.analysis_checkpoints.get("global_embedding", False):
            return "Global embedding computed"
        if self.analysis_checkpoints.get("normalized", False):
            return "Normalized observations ready"
        return "Initialized"

    def describe(self, as_text=True):
        """Summarize cohort scope, locked methods, and progress state."""
        unique_mice = sorted({s["mouse_id"] for s in self.sessions if s.get("mouse_id") is not None})
        summary = {
            "n_sessions": len(self.sessions),
            "n_mice": len(unique_mice),
            "mouse_ids": unique_mice,
            "state_discovery_method": self.state_discovery_method,
            "embedding_method": self.embedding_method,
            "analysis_stage": self.analysis_stage,
            "analysis_point": self._current_analysis_point(),
            "checkpoints": dict(self.analysis_checkpoints),
            "region_key": self.resolved_region_key(filtered=False),
            "cell_type_key": self.resolved_cell_type_key(filtered=False),
            "n_continuous_behavior": len(self.continuous_behavior),
            "n_discrete_event_sets": len(self.discrete_events),
        }

        if as_text:
            print("HybridDynamicsAnalysis summary")
            print("-" * 36)
            print(f"sessions: {summary['n_sessions']}")
            print(f"mice: {summary['n_mice']} -> {summary['mouse_ids']}")
            print(f"state_discovery_method: {summary['state_discovery_method']}")
            print(f"embedding_method: {summary['embedding_method']}")
            print(f"region_key: {summary['region_key'] or 'none'}")
            print(f"cell_type_key: {summary['cell_type_key'] or 'none'}")
            print(
                f"behaviour: {summary['n_continuous_behavior']} continuous, "
                f"{summary['n_discrete_event_sets']} event sets"
            )
            print(f"analysis_stage: {summary['analysis_stage']}")
            print(f"analysis_point: {summary['analysis_point']}")
        return summary
