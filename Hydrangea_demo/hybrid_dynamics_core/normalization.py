import numpy as np
import matplotlib.pyplot as plt


class NormalizationStep:
    """Methods for transforming spike trains into a normalized observation matrix.

    This is the first step in the HybridDynamicsAnalysis pipeline. It converts
    spike times into a time-by-unit matrix using a fixed bin width, and stores
    the relevant metadata on the object for later analysis steps.
    """

    def normalize(self, method="proportion_zscore", zscore=True, smooth_sigma_bins=None, restrict_to_epoch=True):
        """Build the normalized spike matrix.

        Parameters
        ----------
        method : {'proportion_zscore', 'count_zscore', 'raw_counts'}
            Normalization method applied before the HMM.
        zscore : bool
            If True, z-score the selected matrix along units.
        smooth_sigma_bins : float or None
            Optional Gaussian smoothing width in bins.
        restrict_to_epoch : bool
            If True and a maze epoch exists, restrict spike trains before binning.
        """
        if restrict_to_epoch and self.maze_epoch is not None:
            source = self.spike_group.restrict(self.maze_epoch)
        else:
            source = self.spike_group

        self._source_spike_group = source
        self.unit_ids = np.asarray(source.index)

        if len(self.unit_ids) == 0:
            raise ValueError("No units found in selected spike group.")

        spike_arrays = [source[u].index.values for u in self.unit_ids]
        non_empty = [s for s in spike_arrays if len(s)]
        if len(non_empty) == 0:
            raise ValueError("This is empty; please return non-spiking data.")

        t_start = min(s[0] for s in non_empty)
        t_end = max(s[-1] for s in non_empty)
        edges = np.arange(t_start, t_end + self.bin_size_s, self.bin_size_s)
        self.bin_times_s = edges[:-1]

        counts = np.stack([np.histogram(spk, bins=edges)[0] for spk in spike_arrays], axis=1).astype(float)
        self.raw_counts = counts
        if counts.size == 0 or np.sum(counts) == 0:
            raise ValueError("This is empty; please return non-spiking data.")

        if method == "proportion_zscore":
            neuron_totals = np.array([spk.size for spk in spike_arrays], dtype=float)
            if np.any(neuron_totals == 0):
                raise ValueError("This is empty; please return non-spiking data.")
            prop = counts / neuron_totals
            matrix = prop
            self.neuron_totals = neuron_totals
            if zscore:
                mu = prop.mean(axis=0)
                sigma = prop.std(axis=0)
                if np.any(sigma == 0):
                    raise ValueError("This is empty; please return non-spiking data.")
                matrix = (prop - mu) / sigma
            self.normalization_method = "proportion_zscore"

        elif method == "count_zscore":
            matrix = counts
            if zscore:
                mu = counts.mean(axis=0)
                sigma = counts.std(axis=0)
                if np.any(sigma == 0):
                    raise ValueError("This is empty; please return non-spiking data.")
                matrix = (counts - mu) / sigma
            self.normalization_method = "count_zscore"

        elif method == "raw_counts":
            matrix = counts
            self.normalization_method = "raw_counts"

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if smooth_sigma_bins is not None:
            from scipy.ndimage import gaussian_filter1d
            matrix = gaussian_filter1d(matrix, smooth_sigma_bins, axis=0)

        self.spike_matrix = matrix.astype(float)
        return self.spike_matrix.copy(), self.bin_times_s.copy()

    def normalization_report(self, show=True):
        """Print diagnostic plots for the normalized spike matrix.

        Parameters
        ----------
        show : bool
            If True, immediately display the generated plots.
        """
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")

        M = self.spike_matrix
        counts = self.raw_counts
        neuron_totals = getattr(self, "neuron_totals", np.array([]))
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))

        ax[0, 0].hist(M.ravel(), bins=100, log=True)
        ax[0, 0].set(title="Normalized values (log y)", xlabel="z", ylabel="freq")

        frac_zero = (counts == 0).mean(0)
        ax[0, 1].hist(frac_zero, bins=30)
        ax[0, 1].set(title="Fraction empty bins/neuron", xlabel="P(count=0)", ylabel="count")

        m = counts.mean(axis=0)
        v = counts.var(axis=0)
        ax[0, 2].loglog(m + 1e-3, v + 1e-6, ".", alpha=0.5)
        lims = [max(m.min() + 1e-3, 1e-3), max(m.max(), 1e-3)]
        ax[0, 2].plot(lims, lims, "k--", label="var = mean")
        ax[0, 2].set(title="Mean vs var — raw", xlabel="mean", ylabel="var")
        ax[0, 2].legend()

        ax[1, 0].hist(M.std(axis=0), bins=30)
        ax[1, 0].set(title="Per-neuron std after norm (~1)", xlabel="std", ylabel="count")

        if len(neuron_totals) > 0:
            idx = np.argsort(neuron_totals)[len(neuron_totals) // 4]
        else:
            idx = np.argsort(np.abs(M.mean(axis=0)))[len(M.T) // 2]
        ax[1, 1].hist(M[:, idx], bins=50)
        ax[1, 1].set(title=f"Neuron {idx}: normalized dist", xlabel="normalized value", ylabel="count")

        if len(neuron_totals) > 0:
            ax[1, 2].hist(neuron_totals, bins=40)
            ax[1, 2].set(title="Total spikes/neuron (divisor)", xlabel="spike count", ylabel="count")
        else:
            ax[1, 2].axis("off")
        fig.suptitle(f"Normalization: {self.normalization_method}")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax
