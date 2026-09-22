import numpy as np
import matplotlib.pyplot as plt


class GlobalReductionStep:
    """Global dimensionality reduction step.

    This step applies a global PCA to the normalized spike matrix so the state
    finder can work with lower-dimensional embeddings before HMM fitting.
    """

    def global_pca(self, n_components=10, whiten=False, standardize=True):
        """Fit PCA to the normalized spike matrix.

        Parameters
        ----------
        n_components : int
            Number of principal components to retain.
        whiten : bool
            Whether to whiten the output features.
        standardize : bool
            Whether to z-score the matrix before PCA.
        """
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X = self.spike_matrix.copy()
        if standardize:
            X = StandardScaler().fit_transform(X)

        n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
        if n_comp < 1:
            raise ValueError("Not enough dimensions for PCA.")

        pca = PCA(n_components=n_comp, whiten=whiten, random_state=self.random_state)
        scores = pca.fit_transform(X)

        self.pca_model = pca
        self.pca_scores = scores
        self.pca_variance_ratio = pca.explained_variance_ratio_
        return scores, pca

    def pca_report(self, show=True):
        """Display PCA variance and projection diagnostics.

        Parameters
        ----------
        show : bool
            If True, display the plot immediately.
        """
        if self.pca_model is None:
            raise ValueError("Run global_pca() first.")

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(np.arange(1, len(self.pca_variance_ratio) + 1), self.pca_variance_ratio * 100, marker="o")
        ax[0].set(title="PCA variance explained", xlabel="PC", ylabel="% variance")
        ax[0].grid(alpha=0.3)

        if self.pca_scores.shape[1] >= 2:
            ax[1].scatter(self.pca_scores[:, 0], self.pca_scores[:, 1], s=10, alpha=0.7)
            ax[1].set(title="PCA projection", xlabel="PC1", ylabel="PC2")
            ax[1].grid(alpha=0.3)
        else:
            ax[1].text(0.5, 0.5, "Need at least 2 PCs", ha="center", va="center", transform=ax[1].transAxes)
            ax[1].set_axis_off()

        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax
