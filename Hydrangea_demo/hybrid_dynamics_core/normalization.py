import numpy as np
import matplotlib.pyplot as plt
import warnings


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
        if counts.size == 0 or np.sum(counts) == 0:
            raise ValueError("This is empty; please return non-spiking data.")

        neuron_totals_all = np.array([spk.size for spk in spike_arrays], dtype=float)
        keep_mask = neuron_totals_all > 0
        dropped = int((~keep_mask).sum())
        if dropped > 0:
            msg = (
                f"Dropped {dropped} neurons with no spikes in analyzed time interval."
            )

            if hasattr(source, "metadata_columns") and "cell_area" in list(source.metadata_columns):
                region_vals = np.asarray(source.get_info("cell_area"))
                if len(region_vals) == len(keep_mask):
                    regions, counts_by_region = np.unique(region_vals[~keep_mask], return_counts=True)
                    if len(regions) > 0:
                        region_txt = ", ".join(
                            f"{r}:{n}" for r, n in zip(regions.tolist(), counts_by_region.tolist())
                        )
                        msg = f"{msg} Per-region dropped counts -> {region_txt}."

            warnings.warn(msg)

        if not np.any(keep_mask):
            raise ValueError("This is empty; please return non-spiking data.")

        counts = counts[:, keep_mask]
        self.unit_ids = self.unit_ids[keep_mask]
        spike_arrays = [spk for spk, keep in zip(spike_arrays, keep_mask) if keep]
        self.raw_counts = counts
        self.unit_metadata_filtered = {}
        if hasattr(source, "metadata_columns"):
            for key in list(source.metadata_columns):
                values = np.asarray(source.get_info(key))
                if len(values) == len(keep_mask):
                    self.unit_metadata_filtered[key] = values[keep_mask]

        if method == "proportion_zscore":
            neuron_totals = np.array([spk.size for spk in spike_arrays], dtype=float)
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
        if hasattr(self, "_mark_checkpoint"):
            self._mark_checkpoint("normalized")
        return self.spike_matrix.copy(), self.bin_times_s.copy()

    def _safe_hist(self, ax_obj, values, bins=30, log=False, **kwargs):
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ax_obj.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax_obj.transAxes)
            return

        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if np.isclose(vmin, vmax):
            pad = max(abs(vmin) * 1e-6, 1e-6)
            ax_obj.hist(arr, bins=1, range=(vmin - pad, vmax + pad), log=log, **kwargs)
        else:
            ax_obj.hist(arr, bins=bins, log=log, **kwargs)

    def plot_raw_population_counts(self, ax=None, show=True):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure
        self._safe_hist(ax, self.raw_counts.ravel(), bins=np.arange(0, np.max(self.raw_counts) + 2) - 0.5, log=True)
        ax.set(title="Raw bin counts (population, log y)", xlabel="count", ylabel="freq")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_fraction_empty_bins(self, ax=None, show=True):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure
        frac_zero = (self.raw_counts == 0).mean(0)
        self._safe_hist(ax, frac_zero, bins=30)
        ax.set(title="Fraction empty bins/neuron", xlabel="P(count=0)", ylabel="count")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_raw_mean_variance(self, ax=None, show=True):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure
        m = self.raw_counts.mean(axis=0)
        v = self.raw_counts.var(axis=0)
        ax.loglog(m + 1e-3, v + 1e-6, ".", alpha=0.5)
        lims = [max(m.min() + 1e-3, 1e-3), max(m.max(), 1e-3)]
        ax.plot(lims, lims, "k--", label="var = mean")
        ax.set(title="Mean vs var — raw", xlabel="mean", ylabel="var")
        ax.legend()
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_normalized_population_distribution(self, ax=None, show=True):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure
        self._safe_hist(ax, self.spike_matrix.ravel(), bins=100, log=True)
        ax.set(title="Normalized values (population, log y)", xlabel="z", ylabel="freq")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_total_spikes_per_neuron(self, ax=None, show=True):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure
        neuron_totals = getattr(self, "neuron_totals", np.array([]))
        if len(neuron_totals) > 0:
            self._safe_hist(ax, neuron_totals, bins=40)
            ax.set(title="Total spikes/neuron (divisor)", xlabel="spike count", ylabel="count")
        else:
            ax.axis("off")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_normalized_distributions_by_region(self, region_key=None, show=True, min_count=1):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")

        region_key = region_key or getattr(self, "region_key", None) or "cell_area"
        regions = self.unit_metadata_filtered.get(region_key)
        if regions is None:
            raise ValueError(f"No metadata available for region key '{region_key}'.")

        unique_regions = [r for r in np.unique(regions) if np.sum(regions == r) >= min_count]
        if len(unique_regions) == 0:
            raise ValueError(f"No regions with at least {min_count} units available for '{region_key}'.")

        ncols = min(3, len(unique_regions))
        nrows = int(np.ceil(len(unique_regions) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for ax in axes.ravel():
            ax.axis("off")

        for i, region in enumerate(unique_regions):
            ax = axes.ravel()[i]
            ax.axis("on")
            mask = regions == region
            self._safe_hist(ax, self.spike_matrix[:, mask].ravel(), bins=100, log=True)
            ax.set(title=f"{region} (n={int(mask.sum())})", xlabel="z", ylabel="freq")

        fig.suptitle(f"Normalized population distribution by {region_key}")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def normalization_report(self, show=True):
        """Print diagnostic plots for the normalized spike matrix.

        Parameters
        ----------
        show : bool
            If True, immediately display the generated plots.
        """
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))

        self.plot_raw_population_counts(ax=ax[0, 0], show=False)
        self.plot_fraction_empty_bins(ax=ax[0, 1], show=False)
        self.plot_raw_mean_variance(ax=ax[0, 2], show=False)

        self.plot_normalized_population_distribution(ax=ax[1, 0], show=False)
        self.plot_total_spikes_per_neuron(ax=ax[1, 2], show=False)

        # Put a region-distribution summary into the lower middle panel to keep the report compact.
        ax[1, 1].axis("off")
        region_key = getattr(self, "region_key", None) or "cell_area"
        if region_key in self.unit_metadata_filtered:
            regions = self.unit_metadata_filtered[region_key]
            unique_regions = [r for r in np.unique(regions) if np.sum(regions == r) > 0]
            summary = "\n".join([f"{r}: n={int(np.sum(regions == r))}" for r in unique_regions])
            ax[1, 1].text(
                0.0,
                1.0,
                f"Region slices available via plot_normalized_distributions_by_region()\n\n"
                f"{region_key}\n\n{summary}",
                va="top",
            )
        else:
            ax[1, 1].text(0.0, 1.0, "No region metadata available for region slicing", va="top")

        fig.suptitle(f"Normalization: {self.normalization_method}")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax
