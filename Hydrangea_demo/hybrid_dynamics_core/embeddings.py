import numpy as np
import matplotlib.pyplot as plt


class EmbeddingStep:
    """Shared embedding step for session-wide and state-local analyses.

    The same embedding machinery is used to fit the global session embedding
    that feeds the HMM and to refit embeddings inside individual HMM states
    later on. PCA is the first implemented embedding method; CCA is reserved
    in the API for later cross-region work.
    """

    def _fit_pca_embedding(self, matrix, n_components=10, whiten=False, standardize=True):
        if matrix is None:
            raise ValueError("No matrix provided for embedding.")

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(matrix, dtype=float)
        if X.ndim != 2:
            raise ValueError("Embedding input must be a 2D array.")
        if X.shape[0] < 2 or X.shape[1] < 1:
            raise ValueError("Not enough data for embedding.")

        if standardize:
            X = StandardScaler().fit_transform(X)

        n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
        if n_comp < 1:
            raise ValueError("Not enough dimensions for PCA.")

        pca = PCA(n_components=n_comp, whiten=whiten, random_state=self.random_state)
        scores = pca.fit_transform(X)
        return scores, pca

    def _fit_embedding_matrix(
        self,
        matrix,
        method="pca",
        n_components=10,
        whiten=False,
        standardize=True,
        paired_matrix=None,
    ):
        method = method.lower()
        if method == "pca":
            scores, model = self._fit_pca_embedding(
                matrix,
                n_components=n_components,
                whiten=whiten,
                standardize=standardize,
            )
            return scores, model

        if method == "cca":
            raise NotImplementedError(
                "CCA embedding is reserved for future paired regional inputs."
            )

        raise ValueError("method must be one of: 'pca' or 'cca'.")

    def _resolve_embedding_result(self, scope="global", label=None):
        if scope == "global":
            if self.global_embedding_model is None:
                raise ValueError("Run global_embedding() first.")
            return self.global_embedding_results

        if scope == "local":
            if not self.local_embedding_results:
                raise ValueError("Run local_embedding() first.")
            if label is None:
                label = next(reversed(self.local_embedding_results))
            if label not in self.local_embedding_results:
                raise KeyError(f"Unknown local embedding label: {label}")
            return self.local_embedding_results[label]

        raise ValueError("scope must be 'global' or 'local'.")

    def global_embedding(self, method="pca", n_components=10, whiten=False, standardize=True, paired_matrix=None):
        """Fit the session-wide embedding that will feed the HMM."""
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")

        scores, model = self._fit_embedding_matrix(
            self.spike_matrix,
            method=method,
            n_components=n_components,
            whiten=whiten,
            standardize=standardize,
            paired_matrix=paired_matrix,
        )

        result = {
            "method": method.lower(),
            "model": model,
            "scores": scores,
            "variance_ratio": model.explained_variance_ratio_,
            "scope": "global",
            "label": None,
        }
        self.global_embedding_results = result
        self.global_embedding_model = model
        self.global_embedding_scores = scores
        self.global_embedding_variance_ratio = model.explained_variance_ratio_
        self.embedding_results = result
        self.embedding_model = model
        self.embedding_scores = scores
        self.embedding_variance_ratio = model.explained_variance_ratio_
        self.embedding_scope = "global"
        self.embedding_label = None
        return scores, model

    def local_embedding(
        self,
        sample_mask=None,
        sample_indices=None,
        state_labels=None,
        state_value=None,
        method="pca",
        n_components=10,
        whiten=False,
        standardize=True,
        label=None,
        paired_matrix=None,
    ):
        """Fit an embedding on a local subset of the session.

        The subset can be supplied directly with `sample_mask` or
        `sample_indices`, or indirectly from HMM state labels via
        `state_labels` and `state_value`.
        """
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")

        X = self.spike_matrix
        if sample_mask is not None and sample_indices is not None:
            raise ValueError("Provide either sample_mask or sample_indices, not both.")
        if state_labels is not None:
            if state_value is None:
                raise ValueError("state_value is required when state_labels are provided.")
            state_labels = np.asarray(state_labels)
            if state_labels.shape[0] != X.shape[0]:
                raise ValueError("state_labels must match the number of time bins.")
            sample_mask = state_labels == state_value

        if sample_mask is not None:
            sample_mask = np.asarray(sample_mask, dtype=bool)
            if sample_mask.shape[0] != X.shape[0]:
                raise ValueError("sample_mask must match the number of time bins.")
            X = X[sample_mask]
        elif sample_indices is not None:
            sample_indices = np.asarray(sample_indices)
            X = X[sample_indices]

        scores, model = self._fit_embedding_matrix(
            X,
            method=method,
            n_components=n_components,
            whiten=whiten,
            standardize=standardize,
            paired_matrix=paired_matrix,
        )

        local_key = label if label is not None else f"local_{len(self.local_embedding_results) + 1}"
        result = {
            "method": method.lower(),
            "model": model,
            "scores": scores,
            "variance_ratio": model.explained_variance_ratio_,
            "scope": "local",
            "label": local_key,
            "sample_mask": sample_mask,
            "sample_indices": sample_indices,
            "state_labels": state_labels,
            "state_value": state_value,
        }
        self.local_embedding_results[local_key] = result
        self.local_embedding_model = model
        self.local_embedding_scores = scores
        self.local_embedding_variance_ratio = model.explained_variance_ratio_
        self.local_embedding_label = local_key
        self.embedding_results = result
        self.embedding_model = model
        self.embedding_scores = scores
        self.embedding_variance_ratio = model.explained_variance_ratio_
        self.embedding_scope = "local"
        self.embedding_label = local_key
        return scores, model

    def plot_embedding_variance(self, scope="global", label=None, ax=None, show=True):
        """Plot explained variance for a stored embedding result."""
        result = self._resolve_embedding_result(scope=scope, label=label)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure

        ax.plot(np.arange(1, len(result["variance_ratio"]) + 1), result["variance_ratio"] * 100, marker="o")
        ax.set(title=f"{scope.title()} embedding variance explained", xlabel="Component", ylabel="% variance")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_embedding_projection(self, scope="global", label=None, ax=None, show=True):
        """Plot the first two embedding dimensions when available."""
        result = self._resolve_embedding_result(scope=scope, label=label)
        scores = result["scores"]
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure

        if scores.shape[1] >= 2:
            ax.scatter(scores[:, 0], scores[:, 1], s=10, alpha=0.7)
            ax.set(title=f"{scope.title()} embedding projection", xlabel="Component 1", ylabel="Component 2")
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Need at least 2 components", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def report_embeddings(self, scope="global", label=None, show=True):
        """Display embedding variance and projection diagnostics."""
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        self.plot_embedding_variance(scope=scope, label=label, ax=ax[0], show=False)
        self.plot_embedding_projection(scope=scope, label=label, ax=ax[1], show=False)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

