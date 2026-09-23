import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metadata import extract_unit_metadata


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
        started = time.perf_counter()

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

        # Per-unit metadata for this source, before any unit is dropped. The
        # object may already hold a richer table (user-supplied columns merged
        # onto whatever the spike group carried); prefer it when it lines up.
        full_metadata = getattr(self, "unit_metadata", None)
        if full_metadata is None or len(full_metadata) != len(keep_mask):
            full_metadata = extract_unit_metadata(source)
        self.unit_metadata = full_metadata

        if dropped > 0:
            msg = (
                f"Dropped {dropped} neurons with no spikes in analyzed time interval."
            )

            region_key = self.resolved_region_key(filtered=False)
            if region_key is not None and len(full_metadata) == len(keep_mask):
                region_vals = np.asarray(full_metadata[region_key])
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

        if len(full_metadata) == len(keep_mask):
            self.unit_metadata_filtered = full_metadata.loc[full_metadata.index[keep_mask]]
        else:
            self.unit_metadata_filtered = full_metadata

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
        if hasattr(self, "_record_timing"):
            self._record_timing("normalize", time.perf_counter() - started)
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
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_raw_mean_variance(self, ax=None, show=True, log_scale=True):
        """Per-unit mean against variance, with the Poisson line for reference.

        `log_scale` controls whether both axes are logarithmic. Log axes spread
        the low-rate units out; linear axes show the high-rate units in
        proportion. Off-scale padding is only added on log axes, where zeros
        cannot be drawn.
        """
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure
        m = self.raw_counts.mean(axis=0)
        v = self.raw_counts.var(axis=0)

        if log_scale:
            ax.loglog(m + 1e-3, v + 1e-6, ".", alpha=0.5)
            lims = [max(m.min() + 1e-3, 1e-3), max(m.max(), 1e-3)]
        else:
            ax.plot(m, v, ".", alpha=0.5)
            lims = [0.0, float(m.max()) if m.size else 1.0]

        ax.plot(lims, lims, "k--", label="var = mean")
        scale = "log" if log_scale else "linear"
        ax.set(title=f"Mean vs var — raw ({scale})", xlabel="mean", ylabel="var")
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
        else:
            ax.axis("off")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def _firing_rate_hz(self):
        """Per-unit firing rate over the analyzed window, in Hz."""
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if getattr(self, "neuron_totals", None) is None or len(self.neuron_totals) == 0:
            raise ValueError("No neuron totals available for firing-rate diagnostics.")
        duration_s = max(float(len(self.bin_times_s) * self.bin_size_s), 1e-9)
        return self.neuron_totals / duration_s

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

    def plot_log_firing_rate_density(self, ax=None, show=True, log_scale=True, bins=60):
        """Firing-rate distribution, on a log10 or linear rate axis."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure

        if log_scale:
            values = self._log10_firing_rate_hz()
            label, title = "log10 firing rate (Hz)", "log10 firing rate density"
        else:
            values = self._firing_rate_hz()
            label, title = "firing rate (Hz)", "firing rate density"

        self._plot_density(ax, values, bins=bins, center=False, color="tab:red")
        ax.set(title=title, xlabel=label, ylabel="density")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_log_firing_rate_density_by_region(self, ax=None, region_key=None, show=True, log_scale=True, bins=40):
        """Firing-rate distribution split by region, log10 or linear."""
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.figure

        region_key = self.resolved_region_key(region_key)
        regions = self._unit_meta(region_key)
        if regions is None:
            ax.text(
                0.5,
                0.5,
                "No region annotation for these units",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return fig, ax

        rates_hz = self._firing_rate_hz()
        uniq = np.unique(regions)
        cmap = plt.get_cmap("tab10")
        for i, region in enumerate(uniq):
            mask = regions == region
            vals = rates_hz[mask]
            vals = vals[vals > 0]
            if len(vals) == 0:
                continue
            plotted = np.log10(vals) if log_scale else vals
            centers, dens = self._plot_density(ax, plotted, bins=bins, center=False, color=cmap(i % 10), lw=1.8)
            if centers is not None and dens is not None:
                ax.lines[-1].set_label(str(region))

        label = "log10 firing rate (Hz)" if log_scale else "firing rate (Hz)"
        prefix = "log10 firing rate" if log_scale else "firing rate"
        ax.set(title=f"{prefix} density by {region_key}", xlabel=label, ylabel="density")
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

        region_key = self.resolved_region_key(region_key)
        regions = self._unit_meta(region_key)
        if regions is None:
            ax.bar(["all units"], [self.spike_matrix.shape[1]], color="tab:cyan")
            ax.set(title="Analyzed units (no region annotation)", ylabel="unit count")
            fig.tight_layout()
            if show:
                plt.show()
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
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_region_celltype_counts(self, ax=None, region_key=None, cell_type_key=None, show=True):
        """Unit counts for the analyzed units, by region and cell type when annotated."""
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        return self.plot_unit_inventory(
            region_key=region_key,
            cell_type_key=cell_type_key,
            filtered=True,
            ax=ax,
            show=show,
        )

    def plot_normalized_distributions_by_region(self, region_key=None, show=True, min_count=1):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")

        region_key = self.resolved_region_key(region_key)
        regions = self._unit_meta(region_key)
        if regions is None:
            raise ValueError("No region annotation available for these units.")

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

        fig.suptitle(f"Normalized population distribution by {region_key}")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def normalization_summary(self):
        """Shape, sparsity, and rate statistics of the observation matrix."""
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        rates = self._firing_rate_hz()
        duration_s = float(len(self.bin_times_s) * self.bin_size_s)
        return pd.Series(
            {
                "method": self.normalization_method,
                "bin_size_s": self.bin_size_s,
                "n_bins": int(self.spike_matrix.shape[0]),
                "n_units": int(self.spike_matrix.shape[1]),
                "duration_s": round(duration_s, 1),
                "duration_min": round(duration_s / 60.0, 2),
                "total_spikes": int(self.raw_counts.sum()),
                "frac_empty_bins": round(float((self.raw_counts == 0).mean()), 4),
                "median_rate_hz": round(float(np.median(rates)), 3),
                "min_rate_hz": round(float(rates.min()), 4),
                "max_rate_hz": round(float(rates.max()), 3),
            },
            name="normalization",
        )

    def normalization_report(self, show=True, log_scale=True):
        """Display diagnostic plots for the normalized spike matrix.

        Display-only: returns None, so the figure handles do not echo in a
        notebook cell. Call the individual ``plot_*`` methods when you need the
        handles for further composition, or ``normalization_summary`` for the
        numbers.

        Parameters
        ----------
        show : bool
            If True, immediately display the generated plots.
        log_scale : bool
            Log axes on the rate and mean-variance panels. Log spreads out the
            low-rate units, which is usually what you want with a long tail of
            near-silent cells; turn it off when the tail is not the story.
        """
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")

        print(self.normalization_summary().to_string())

        fig, ax = plt.subplots(2, 3, figsize=(16, 9))

        self.plot_raw_population_counts(ax=ax[0, 0], show=False)
        self.plot_normalized_population_distribution(ax=ax[0, 1], show=False)
        self.plot_raw_mean_variance(ax=ax[0, 2], show=False, log_scale=log_scale)

        self.plot_log_firing_rate_density(ax=ax[1, 0], show=False, log_scale=log_scale)
        self.plot_log_firing_rate_density_by_region(ax=ax[1, 1], show=False, log_scale=log_scale)
        self.plot_region_celltype_counts(ax=ax[1, 2], show=False)

        fig.suptitle(f"Normalization: {self.normalization_method}")
        fig.tight_layout()
        if show:
            plt.show()
