"""Placeholder for local state embedding logic.

This step is intentionally left empty for now. It will be built in a later notebook
once the HMM normalization and model-selection workflow has stabilized.
"""


class LocalStateEmbeddingStep:
    """Placeholder for future local-state embedding work.

    This step is intentionally left empty while the global HMM selection and time
    segmentation workflow is still being designed. Once the proper windows are
    chosen, this class can be extended to compute local embeddings around each
    selected time point or region.
    """

    def find_local_embedding_windows(self, *args, **kwargs):
        """Placeholder for selecting local time windows for future embedding work."""
        raise NotImplementedError("Local state embedding design pending.")
