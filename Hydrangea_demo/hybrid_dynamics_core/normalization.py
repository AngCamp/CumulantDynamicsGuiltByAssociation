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
        neuron_totals = np.array([spk.size for spk in spike_arrays], dtype=float)
        self.neuron_totals = neuron_totals
        self.raw_counts = counts
        self.unit_metadata_filtered = {}
        if hasattr(source, "metadata_columns"):
            for key in list(source.metadata_columns):
                values = np.asarray(source.get_info(key))
                if len(values) == len(keep_mask):
                    self.unit_metadata_filtered[key] = values[keep_mask]

        if method == "proportion_zscore":
            prop = counts / neuron_totals
            matrix = prop
            if zscore:
                mu = prop.mean(axis=0)
                sigma = prop.std(axis=0)
                if np.any(sigma == 0):
                    raise ValueError("This is empty; please return non-spiking data.")
                matrix = (prop - mu) / sigma
            self.normalization_method = "proportion_then_zscore" if zscore else "proportion"

        elif method == "count_zscore":
            matrix = counts
            if zscore:
                mu = counts.mean(axis=0)
                sigma = counts.std(axis=0)
                if np.any(sigma == 0):
                    raise ValueError("This is empty; please return non-spiking data.")
                matrix = (counts - mu) / sigma
            self.normalization_method = "count_then_zscore" if zscore else "count"

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

    def _plot_density(self, ax_obj, values, bins=120, center=False, color="tab:blue", lw=2):
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ax_obj.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax_obj.transAxes)
            return None, None

        if center:
            arr = arr - np.mean(arr)

        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if np.isclose(vmin, vmax):
            ax_obj.axvline(vmin, color=color, lw=lw)
            return np.array([vmin]), np.array([1.0])

        hist, edges = np.histogram(arr, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_obj.plot(centers, hist, color=color, lw=lw)
        ax_obj.fill_between(centers, hist, 0, color=color, alpha=0.2)
        return centers, hist

    def plot_raw_population_counts(self, ax=None, show=True):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure
        self._plot_density(ax, self.raw_counts.ravel(), bins=120, center=True, color="tab:blue")
        ax.axvline(0.0, color="k", lw=1, ls="--", alpha=0.6)
        ax.set(title="Raw binned counts density (centered)", xlabel="count - mean(count)", ylabel="density")
        ax.grid(alpha=0.3)
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
        self._plot_density(ax, frac_zero, bins=80, center=False, color="tab:orange")
        ax.set(title="Fraction empty bins/neuron", xlabel="P(count=0)", ylabel="density")
        ax.grid(alpha=0.3)
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
        self._plot_density(ax, self.spike_matrix.ravel(), bins=120, center=True, color="tab:green")
        ax.axvline(0.0, color="k", lw=1, ls="--", alpha=0.6)
        ax.set(title="Z-scored values density (centered)", xlabel="z - mean(z)", ylabel="density")
        ax.grid(alpha=0.3)
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
            self._plot_density(ax, neuron_totals, bins=80, center=False, color="tab:purple")
            ax.set(title="Total spikes/neuron (divisor)", xlabel="spike count", ylabel="density")
            ax.grid(alpha=0.3)
        else:
            ax.axis("off")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def _log10_firing_rate_hz(self):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if getattr(self, "neuron_totals", None) is None or len(self.neuron_totals) == 0:
            raise ValueError("No neuron totals available for firing-rate diagnostics.")
        duration_s = max(float(len(self.bin_times_s) * self.bin_size_s), 1e-9)
        rates_hz = self.neuron_totals / duration_s
        rates_hz = rates_hz[rates_hz > 0]
        if len(rates_hz) == 0:
            raise ValueError("No positive firing rates available.")
        return np.log10(rates_hz)

    def plot_log_firing_rate_density(self, ax=None, show=True):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure

        log_fr = self._log10_firing_rate_hz()
        self._plot_density(ax, log_fr, bins=100, center=False, color="tab:red")
        ax.set(title="log10 firing rate density", xlabel="log10 firing rate (Hz)", ylabel="density")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_log_firing_rate_density_by_region(self, ax=None, region_key=None, show=True):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure

        region_key = region_key or getattr(self, "region_key", None) or "cell_area"
        regions = self.unit_metadata_filtered.get(region_key)
        if regions is None:
            ax.text(0.5, 0.5, f"No region metadata for '{region_key}'", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig, ax

        duration_s = max(float(len(self.bin_times_s) * self.bin_size_s), 1e-9)
        rates_hz = self.neuron_totals / duration_s
        uniq = np.unique(regions)
        cmap = plt.get_cmap("tab10")
        for i, region in enumerate(uniq):
            mask = regions == region
            vals = rates_hz[mask]
            vals = vals[vals > 0]
            if len(vals) == 0:
                continue
            log_vals = np.log10(vals)
            centers, dens = self._plot_density(ax, log_vals, bins=80, center=False, color=cmap(i % 10), lw=1.8)
            if centers is not None and dens is not None:
                ax.lines[-1].set_label(str(region))

        ax.set(title=f"log10 firing rate density by {region_key}", xlabel="log10 firing rate (Hz)", ylabel="density")
        ax.grid(alpha=0.3)
        if len(ax.lines) > 0:
            ax.legend(fontsize=9)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_region_unit_counts(self, ax=None, region_key=None, show=True):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure

        region_key = region_key or getattr(self, "region_key", None) or "cell_area"
        regions = self.unit_metadata_filtered.get(region_key)
        if regions is None:
            ax.text(0.5, 0.5, f"No region metadata for '{region_key}'", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig, ax

        labels, counts = np.unique(regions, return_counts=True)
        order = np.argsort(counts)[::-1]
        labels = labels[order]
        counts = counts[order]

        ax.bar(labels.astype(str), counts, color="tab:cyan")
        ax.set(title=f"Units per {region_key}", xlabel=region_key, ylabel="unit count")
        ax.tick_params(axis="x", rotation=30)
        for x, y in zip(labels.astype(str), counts):
            ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=9)
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_region_celltype_counts(self, ax=None, region_key=None, cell_type_key="cell_type", show=True):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        else:
            fig = ax.figure

        region_key = region_key or getattr(self, "region_key", None) or "cell_area"
        regions = self.unit_metadata_filtered.get(region_key)
        cell_types = self.unit_metadata_filtered.get(cell_type_key)

        if regions is None or cell_types is None:
            ax.text(
                0.5,
                0.5,
                f"Missing metadata for '{region_key}' and/or '{cell_type_key}'",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return fig, ax

        region_labels = np.unique(regions)
        type_labels = np.unique(cell_types)
        count_map = {(r, c): 0 for r in region_labels for c in type_labels}
        for r, c in zip(regions, cell_types):
            count_map[(r, c)] += 1

        total_counts = np.array([np.sum(regions == r) for r in region_labels])
        type_counts = {c: np.array([count_map[(r, c)] for r in region_labels]) for c in type_labels}

        x = np.arange(len(region_labels), dtype=float)
        n_bars = len(type_labels) + 1
        width = 0.8 / n_bars
        cmap = plt.get_cmap("tab10")

        bars_total = ax.bar(x + (0 - (n_bars - 1) / 2) * width, total_counts, width, label="Total", color=cmap(0))
        ax.bar_label(bars_total, padding=2, fontsize=8)

        for i, c in enumerate(type_labels, start=1):
            offsets = x + (i - (n_bars - 1) / 2) * width
            bars = ax.bar(offsets, type_counts[c], width, label=str(c), color=cmap(i % 10))
            ax.bar_label(bars, padding=2, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in region_labels])
        ax.set(
            title=f"Unit counts by {region_key} (Total + per {cell_type_key})",
            xlabel="brain region",
            ylabel="unit count",
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2, axis="y")
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
            self._plot_density(ax, self.spike_matrix[:, mask].ravel(), bins=100, center=True, color="tab:blue")
            ax.axvline(0.0, color="k", lw=1, ls="--", alpha=0.5)
            ax.set(title=f"{region} (n={int(mask.sum())})", xlabel="z - mean(z)", ylabel="density")
            ax.grid(alpha=0.2)

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
        self.plot_normalized_population_distribution(ax=ax[0, 1], show=False)
        self.plot_raw_mean_variance(ax=ax[0, 2], show=False)

        self.plot_log_firing_rate_density(ax=ax[1, 0], show=False)
        self.plot_log_firing_rate_density_by_region(ax=ax[1, 1], show=False)
        self.plot_region_celltype_counts(ax=ax[1, 2], show=False)

        fig.suptitle(f"Normalization: {self.normalization_method}")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax
