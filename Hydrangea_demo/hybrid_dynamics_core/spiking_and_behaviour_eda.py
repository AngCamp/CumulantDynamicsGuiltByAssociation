"""Behaviour/stimulus containers and pre-modelling EDA for spiking data.

Every experiment ships some mixture of stimuli and behavioural measurements,
and they fall into two shapes:

``ContinuousBehavior``
    A score sampled at timepoints: position, running speed, pupil diameter, a
    lever trace, an LFP band envelope. Carries an explicit sampling rate
    descriptor (inferred from the timestamps when not supplied) because the
    behavioural rate is almost never the spike-binning rate.

``DiscreteEvents``
    Events with a time interval and optional subtypes: trials, stimulus
    presentations, reward deliveries, licks, sharp-wave ripples. Instantaneous
    events are stored as zero-width intervals.

Both know how to project themselves onto the spike-matrix bin edges, which is
the bridge from raw behaviour to the state-level analysis: once the HMM has
produced ``state_labels_`` over bins, behaviour aligned to the same bins gives
state occupancy conditioned on behaviour directly.

Quantities the recording does not provide (running speed from position, for
instance) are computed by the free functions here and registered on the object,
rather than being derived inside it. Derivation rules are dataset-specific and
belong in the analysis script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .metadata import (
    DEFAULT_CELL_TYPE_KEYS,
    DEFAULT_REGION_KEYS,
    as_metadata_table,
    extract_unit_metadata,
    resolve_metadata_key,
)


def infer_sampling_rate_hz(time):
    """Sampling rate in Hz from timestamps, or None if undefined."""
    t = np.asarray(time, dtype=float)
    if t.size < 2:
        return None
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return None
    return float(1.0 / dt)


def compute_speed_from_position(position, columns=("x", "y"), smooth_window_s=0.25):
    """Speed from position samples.

    Kept outside the analysis object because most archives do not ship a speed
    signal and the way it is derived (which columns, how much smoothing) is a
    per-dataset decision.

    Parameters
    ----------
    position : pynapple TsdFrame or pandas.DataFrame
        Position samples indexed by time in seconds.
    columns : sequence of str
        Position columns to combine into a displacement magnitude.
    smooth_window_s : float
        Width of the boxcar smoother applied to the speed trace.

    Returns
    -------
    time, speed : ndarray
    """
    t = np.asarray(position.index.values, dtype=float)
    coords = np.column_stack([np.asarray(position[c].values, dtype=float) for c in columns])

    dt = np.diff(t)
    step = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    speed = np.zeros_like(t, dtype=float)
    speed[1:] = step / np.maximum(dt, 1e-12)

    if dt.size > 0 and smooth_window_s:
        smooth_n = max(1, int(smooth_window_s / np.median(dt)))
        speed = np.convolve(speed, np.ones(smooth_n) / smooth_n, mode="same")
    return t, speed


class ContinuousBehavior:
    """A behavioural or stimulus score sampled at timepoints.

    Parameters
    ----------
    name : str
        Registry key.
    time : array-like
        Sample times in seconds.
    values : array-like
        Shape (T,) or (T, D). Multi-column signals keep their column names.
    sampling_rate_hz : float or None
        Nominal acquisition rate. Inferred from the timestamps when omitted.
    columns : sequence of str or None
        Names for the value columns.
    units : str or None
        Physical units, used for axis labels.
    """

    kind = "continuous"

    def __init__(
        self,
        name,
        time,
        values,
        sampling_rate_hz=None,
        columns=None,
        units=None,
        color=None,
        description=None,
    ):
        self.name = str(name)
        self.time = np.asarray(time, dtype=float).ravel()

        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("Continuous values must be 1D or 2D.")
        if values.shape[0] != self.time.shape[0]:
            raise ValueError(
                f"Signal '{name}': {values.shape[0]} samples but {self.time.shape[0]} timestamps."
            )
        self.values = values

        if columns is None:
            columns = [self.name] if values.shape[1] == 1 else [f"{self.name}_{i}" for i in range(values.shape[1])]
        if len(columns) != values.shape[1]:
            raise ValueError("columns must match the number of value columns.")
        self.columns = [str(c) for c in columns]

        self.sampling_rate_hz = (
            float(sampling_rate_hz) if sampling_rate_hz is not None else infer_sampling_rate_hz(self.time)
        )
        self.units = units
        self.color = color
        self.description = description

    @property
    def n_samples(self):
        return int(self.time.shape[0])

    @property
    def n_dims(self):
        return int(self.values.shape[1])

    @property
    def t_start(self):
        return float(self.time[0]) if self.n_samples else np.nan

    @property
    def t_end(self):
        return float(self.time[-1]) if self.n_samples else np.nan

    def restrict(self, start, end):
        """Return a copy limited to [start, end]."""
        mask = (self.time >= float(start)) & (self.time <= float(end))
        return ContinuousBehavior(
            self.name,
            self.time[mask],
            self.values[mask],
            sampling_rate_hz=self.sampling_rate_hz,
            columns=self.columns,
            units=self.units,
            color=self.color,
            description=self.description,
        )

    def bin_values(self, bin_edges, reduce="mean"):
        """Project the signal onto bin edges.

        Returns an (n_bins, n_dims) array. Bins with no sample are NaN, which
        keeps empty bins visible instead of silently imputing them.
        """
        edges = np.asarray(bin_edges, dtype=float)
        n_bins = len(edges) - 1
        if n_bins < 1:
            raise ValueError("bin_edges must define at least one bin.")

        out = np.full((n_bins, self.n_dims), np.nan, dtype=float)
        idx = np.searchsorted(edges, self.time, side="right") - 1
        in_range = (idx >= 0) & (idx < n_bins)
        if not np.any(in_range):
            return out

        for j in range(self.n_dims):
            column = self.values[:, j]
            valid = in_range & np.isfinite(column)
            if not np.any(valid):
                continue
            bins = idx[valid]
            vals = column[valid]
            counts = np.bincount(bins, minlength=n_bins)
            if reduce == "mean":
                totals = np.bincount(bins, weights=vals, minlength=n_bins)
                with np.errstate(invalid="ignore", divide="ignore"):
                    column_out = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
            elif reduce == "sum":
                column_out = np.where(counts > 0, np.bincount(bins, weights=vals, minlength=n_bins), np.nan)
            elif reduce == "max":
                column_out = np.full(n_bins, -np.inf)
                np.maximum.at(column_out, bins, vals)
                column_out = np.where(counts > 0, column_out, np.nan)
            else:
                raise ValueError("reduce must be one of: 'mean', 'sum', 'max'.")
            out[:, j] = column_out
        return out

    def describe(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "n_dims": self.n_dims,
            "n_samples_or_events": self.n_samples,
            "sampling_rate_hz": self.sampling_rate_hz,
            "t_start_s": self.t_start,
            "t_end_s": self.t_end,
            "subtypes": None,
            "units": self.units,
            "description": self.description,
        }


class DiscreteEvents:
    """Events occupying a time interval, with optional subtypes.

    Parameters
    ----------
    name : str
        Registry key.
    start : array-like
        Event onsets in seconds.
    end : array-like or None
        Event offsets. When omitted the events are instantaneous and stored as
        zero-width intervals.
    subtypes : array-like or None
        Per-event label (stimulus identity, trial outcome, event class).
    table : DataFrame-like or None
        Extra per-event columns kept alongside the intervals.
    """

    kind = "discrete"

    def __init__(
        self,
        name,
        start,
        end=None,
        subtypes=None,
        table=None,
        color=None,
        description=None,
    ):
        self.name = str(name)
        self.start = np.asarray(start, dtype=float).ravel()
        self.end = self.start.copy() if end is None else np.asarray(end, dtype=float).ravel()
        if self.end.shape != self.start.shape:
            raise ValueError(f"Events '{name}': start and end lengths differ.")
        if np.any(self.end < self.start):
            raise ValueError(f"Events '{name}': some intervals end before they start.")

        if subtypes is None:
            self.subtypes = None
        else:
            subtypes = np.asarray(subtypes).ravel()
            if subtypes.shape[0] != self.start.shape[0]:
                raise ValueError(f"Events '{name}': subtypes length does not match event count.")
            self.subtypes = subtypes

        self.table = as_metadata_table(table)
        self.color = color
        self.description = description

    @property
    def n_events(self):
        return int(self.start.shape[0])

    @property
    def durations(self):
        return self.end - self.start

    @property
    def is_instantaneous(self):
        return bool(np.allclose(self.durations, 0.0))

    @property
    def subtype_labels(self):
        if self.subtypes is None:
            return []
        return sorted({str(s) for s in self.subtypes})

    @property
    def t_start(self):
        return float(self.start.min()) if self.n_events else np.nan

    @property
    def t_end(self):
        return float(self.end.max()) if self.n_events else np.nan

    def restrict(self, start, end):
        """Return events overlapping [start, end], clipped to that window."""
        start, end = float(start), float(end)
        mask = (self.end >= start) & (self.start <= end)
        return DiscreteEvents(
            self.name,
            np.clip(self.start[mask], start, end),
            np.clip(self.end[mask], start, end),
            subtypes=None if self.subtypes is None else self.subtypes[mask],
            table=self.table.loc[mask] if len(self.table) == len(mask) else None,
            color=self.color,
            description=self.description,
        )

    def select(self, subtype):
        """Return only the events carrying one subtype label."""
        if self.subtypes is None:
            raise ValueError(f"Events '{self.name}' carry no subtypes.")
        mask = np.asarray([str(s) == str(subtype) for s in self.subtypes])
        return DiscreteEvents(
            f"{self.name}:{subtype}",
            self.start[mask],
            self.end[mask],
            subtypes=self.subtypes[mask],
            table=self.table.loc[mask] if len(self.table) == len(mask) else None,
            color=self.color,
            description=self.description,
        )

    def to_interval_set(self):
        """Convert to a pynapple IntervalSet for restriction and fold building."""
        import pynapple as nap

        return nap.IntervalSet(start=self.start, end=self.end)

    def bin_events(self, bin_edges, subtype=None):
        """Project events onto bin edges.

        Returns a DataFrame with ``occupancy`` (fraction of each bin covered by
        an event) and ``count`` (number of event onsets landing in the bin).
        Instantaneous events have zero occupancy, so use ``count`` for those.
        """
        edges = np.asarray(bin_edges, dtype=float)
        n_bins = len(edges) - 1
        if n_bins < 1:
            raise ValueError("bin_edges must define at least one bin.")

        start, end = self.start, self.end
        if subtype is not None:
            chosen = self.select(subtype)
            start, end = chosen.start, chosen.end

        occupancy = np.zeros(n_bins, dtype=float)
        counts = np.zeros(n_bins, dtype=float)
        widths = np.diff(edges)

        for s, e in zip(start, end):
            if e < edges[0] or s > edges[-1]:
                continue
            i0 = int(np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1))
            i1 = int(np.clip(np.searchsorted(edges, e, side="right") - 1, 0, n_bins - 1))
            span = np.arange(i0, i1 + 1)
            lo = np.maximum(s, edges[span])
            hi = np.minimum(e, edges[span + 1])
            occupancy[span] += np.clip(hi - lo, 0.0, None)
            counts[i0] += 1.0

        with np.errstate(invalid="ignore", divide="ignore"):
            occupancy = np.clip(occupancy / np.maximum(widths, 1e-12), 0.0, 1.0)

        return pd.DataFrame(
            {"occupancy": occupancy, "count": counts},
            index=pd.Index(edges[:-1], name="bin_start_s"),
        )

    def describe(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "n_dims": 1,
            "n_samples_or_events": self.n_events,
            "sampling_rate_hz": None,
            "t_start_s": self.t_start,
            "t_end_s": self.t_end,
            "subtypes": ", ".join(self.subtype_labels) if self.subtypes is not None else None,
            "units": "s" if not self.is_instantaneous else None,
            "description": self.description,
        }


class SpikingBehaviorEDAStep:
    """Behaviour registration and pre-modelling EDA.

    This step runs before normalization. Its job is to make the session legible:
    what units were recorded and how they are annotated, what behavioural and
    stimulus channels exist, and whether the spiking visibly tracks any of them.
    Judgement about bin size, epoch restriction, and normalization method is
    made from these plots.
    """

    def _init_behavior(self):
        self.continuous_behavior = {}
        self.discrete_events = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def add_continuous_behavior(
        self,
        name,
        time,
        values,
        sampling_rate_hz=None,
        columns=None,
        units=None,
        color=None,
        description=None,
        overwrite=True,
    ):
        """Register a continuous behavioural or stimulus signal."""
        if name in self.continuous_behavior and not overwrite:
            raise KeyError(f"Continuous signal '{name}' already registered.")
        signal = ContinuousBehavior(
            name,
            time,
            values,
            sampling_rate_hz=sampling_rate_hz,
            columns=columns,
            units=units,
            color=color,
            description=description,
        )
        self.continuous_behavior[signal.name] = signal
        return signal

    def add_discrete_events(
        self,
        name,
        start,
        end=None,
        subtypes=None,
        table=None,
        color=None,
        description=None,
        overwrite=True,
    ):
        """Register an event set (trials, stimuli, rewards, detected events)."""
        if name in self.discrete_events and not overwrite:
            raise KeyError(f"Event set '{name}' already registered.")
        events = DiscreteEvents(
            name,
            start,
            end=end,
            subtypes=subtypes,
            table=table,
            color=color,
            description=description,
        )
        self.discrete_events[events.name] = events
        return events

    def add_events_from_interval_set(self, name, interval_set, subtypes=None, **kwargs):
        """Register events from a pynapple IntervalSet."""
        return self.add_discrete_events(
            name,
            np.asarray(interval_set.start, dtype=float),
            np.asarray(interval_set.end, dtype=float),
            subtypes=subtypes,
            **kwargs,
        )

    def get_behavior(self, name):
        """Look up a registered signal or event set by name."""
        if name in self.continuous_behavior:
            return self.continuous_behavior[name]
        if name in self.discrete_events:
            return self.discrete_events[name]
        raise KeyError(f"No behaviour registered under '{name}'.")

    def behavior_summary(self, as_text=False):
        """Table of every registered behavioural channel."""
        rows = [sig.describe() for sig in self.continuous_behavior.values()]
        rows += [ev.describe() for ev in self.discrete_events.values()]
        if not rows:
            summary = pd.DataFrame(
                columns=[
                    "name",
                    "kind",
                    "n_dims",
                    "n_samples_or_events",
                    "sampling_rate_hz",
                    "t_start_s",
                    "t_end_s",
                    "subtypes",
                    "units",
                    "description",
                ]
            )
        else:
            summary = pd.DataFrame(rows).set_index("name")
        if as_text:
            print("\nRegistered behaviour channels:")
            print(summary if len(summary) else "  (none)")
            return None
        return summary

    # ------------------------------------------------------------------
    # alignment to spike bins
    # ------------------------------------------------------------------

    def bin_edges(self, bin_edges=None):
        """Bin edges matching the normalized spike matrix."""
        if bin_edges is not None:
            return np.asarray(bin_edges, dtype=float)
        if getattr(self, "bin_times_s", None) is None:
            raise ValueError("Run normalize() first, or pass explicit bin_edges.")
        left = np.asarray(self.bin_times_s, dtype=float)
        return np.append(left, left[-1] + float(self.bin_size_s))

    def align_behavior_to_bins(self, bin_edges=None, reduce="mean", names=None):
        """Project every registered channel onto the spike-matrix bins.

        Returns a DataFrame indexed by bin start time with one column per
        continuous dimension and, for each event set, ``<name>.occupancy`` and
        ``<name>.count``. This is the table that pairs with ``state_labels_``
        when asking how discovered states relate to behaviour.
        """
        edges = self.bin_edges(bin_edges)
        columns = {}

        for name, signal in self.continuous_behavior.items():
            if names is not None and name not in names:
                continue
            binned = signal.bin_values(edges, reduce=reduce)
            for j, column in enumerate(signal.columns):
                key = column if signal.n_dims > 1 else name
                columns[key] = binned[:, j]

        for name, events in self.discrete_events.items():
            if names is not None and name not in names:
                continue
            frame = events.bin_events(edges)
            columns[f"{name}.occupancy"] = frame["occupancy"].values
            columns[f"{name}.count"] = frame["count"].values

        return pd.DataFrame(columns, index=pd.Index(edges[:-1], name="bin_start_s"))

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------

    def _eda_spike_group(self, spike_group=None):
        group = spike_group
        if group is None:
            group = getattr(self, "_source_spike_group", None) or getattr(self, "spike_group", None)
        if group is None:
            raise ValueError("No spike group available for EDA plots.")
        return group

    def _unit_groupings(self, spike_group, region_key=None, cell_type_key=None):
        """Per-unit region and cell-type labels, filling in when absent."""
        unit_ids = np.asarray(spike_group.index)
        table = extract_unit_metadata(spike_group)
        if table.empty:
            stored = getattr(self, "unit_metadata", None)
            if stored is not None and len(stored) == len(unit_ids):
                table = stored

        region_key = resolve_metadata_key(
            table, region_key if region_key is not None else getattr(self, "region_key", None), DEFAULT_REGION_KEYS
        )
        cell_type_key = resolve_metadata_key(
            table,
            cell_type_key if cell_type_key is not None else getattr(self, "cell_type_key", None),
            DEFAULT_CELL_TYPE_KEYS,
        )

        if region_key is None:
            regions = np.array(["all units"] * len(unit_ids), dtype=object)
        else:
            regions = np.asarray(table[region_key].astype(str))
        if cell_type_key is None:
            cell_types = np.array(["unit"] * len(unit_ids), dtype=object)
        else:
            cell_types = np.asarray(table[cell_type_key].astype(str))

        return unit_ids, regions, cell_types, region_key, cell_type_key

    def _spike_times_in_window(self, spike_group, unit_id, t_start, t_end):
        times = np.asarray(spike_group[unit_id].index)
        return times[(times >= t_start) & (times <= t_end)]

    def _shade_events(self, axes, t_start, t_end, names=None, legend_handles=None):
        """Shade every registered event set onto a stack of time axes."""
        cmap = plt.get_cmap("Set2")
        for i, (name, events) in enumerate(self.discrete_events.items()):
            if names is not None and name not in names:
                continue
            window = events.restrict(t_start, t_end)
            if window.n_events == 0:
                continue
            color = events.color or cmap(i % 8)
            for ax in axes:
                for s, e in zip(window.start, window.end):
                    if np.isclose(e, s):
                        ax.axvline(s, color=color, lw=1.0, alpha=0.7)
                    else:
                        ax.axvspan(s, e, color=color, alpha=0.12, lw=0)
            if legend_handles is not None:
                legend_handles.append(Patch(facecolor=color, alpha=0.35, label=name))

    def plot_raster_with_behavior(
        self,
        names=None,
        region_key=None,
        cell_type_key=None,
        n_units_per_group=5,
        window_s=60.0,
        t_start=None,
        spike_group=None,
        show=True,
    ):
        """Spike rasters stacked over the continuous behavioural channels.

        Units are sampled evenly across whatever grouping the dataset supports:
        region and cell type when both are annotated, one of them when only one
        is, and a flat sample when neither is. Registered event sets are shaded
        across all panels.

        Parameters
        ----------
        names : sequence of str or None
            Continuous channels to plot. Defaults to all registered ones.
        n_units_per_group : int
            Units drawn per (region, cell type) group.
        window_s : float
            Width of the plotted window in seconds.
        t_start : float or None
            Window start. Defaults to the analysis epoch start.
        """
        group = self._eda_spike_group(spike_group)
        unit_ids, regions, cell_types, region_key, cell_type_key = self._unit_groupings(
            group, region_key, cell_type_key
        )

        selected = []
        for region in sorted(set(regions.tolist())):
            for cell_type in sorted(set(cell_types[regions == region].tolist())):
                mask = (regions == region) & (cell_types == cell_type)
                for uid in unit_ids[mask][:n_units_per_group]:
                    selected.append((uid, region, cell_type))
        if not selected:
            raise ValueError("No units available to plot.")

        color_by = cell_types if cell_type_key is not None else regions
        color_labels = sorted(set(color_by.tolist()))
        cmap = plt.get_cmap("tab10")
        palette = {label: cmap(i % 10) for i, label in enumerate(color_labels)}
        unit_color = {uid: palette[label] for uid, label in zip(unit_ids, color_by)}

        if t_start is None:
            t_start = self._default_window_start(group, selected)
        t_start = float(t_start)
        t_end = t_start + float(window_s)

        if names is None:
            names = list(self.continuous_behavior)
        signals = [self.continuous_behavior[n] for n in names if n in self.continuous_behavior]

        n_rows = 1 + len(signals)
        height_ratios = [max(len(selected) / 8.0, 1.2)] + [1.0] * len(signals)
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(15, 3 + 2 * len(signals)),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        axes = np.atleast_1d(axes)
        raster_ax = axes[0]

        for row, (uid, _region, _cell_type) in enumerate(selected):
            spikes = self._spike_times_in_window(group, uid, t_start, t_end)
            if spikes.size == 0:
                continue
            raster_ax.plot(
                spikes,
                np.full(spikes.size, row),
                "|",
                color=unit_color[uid],
                markersize=4,
                mew=0.6,
            )

        raster_ax.set_yticks(range(len(selected)))
        raster_ax.set_yticklabels(
            [self._unit_row_label(r, c, region_key, cell_type_key) for _u, r, c in selected], fontsize=6
        )
        raster_ax.set(
            ylabel="unit",
            title=(
                f"Spike rasters — up to {n_units_per_group} units per group "
                f"({window_s:g}s window from {t_start:.1f}s)"
            ),
        )
        raster_ax.invert_yaxis()

        for i, signal in enumerate(signals, start=1):
            window = signal.restrict(t_start, t_end)
            for j, column in enumerate(signal.columns):
                axes[i].plot(
                    window.time,
                    window.values[:, j],
                    lw=0.7,
                    color=signal.color if signal.n_dims == 1 else None,
                    label=column if signal.n_dims > 1 else None,
                )
            label = signal.name if signal.units is None else f"{signal.name} ({signal.units})"
            rate = signal.sampling_rate_hz
            axes[i].set_ylabel(f"{label}\n{rate:.1f} Hz" if rate else label, fontsize=8)
            if signal.n_dims > 1:
                axes[i].legend(fontsize=7, ncol=signal.n_dims, loc="upper right")

        event_handles = []
        self._shade_events(list(axes), t_start, t_end, legend_handles=event_handles)

        unit_handles = [Line2D([0], [0], color=palette[l], lw=2, label=str(l)) for l in color_labels]
        unit_legend = raster_ax.legend(
            handles=unit_handles, fontsize=8, loc="upper right", ncol=max(1, len(color_labels))
        )
        if event_handles:
            raster_ax.add_artist(unit_legend)
            raster_ax.legend(handles=event_handles, fontsize=8, loc="upper left", ncol=max(1, len(event_handles)))

        axes[-1].set_xlabel("time (s)")
        axes[0].set_xlim(t_start, t_end)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def _unit_row_label(self, region, cell_type, region_key, cell_type_key):
        parts = []
        if region_key is not None:
            parts.append(str(region)[:5])
        if cell_type_key is not None:
            parts.append(str(cell_type)[:5])
        return "·".join(parts) if parts else "unit"

    def _default_window_start(self, group, selected):
        epoch = getattr(self, "maze_epoch", None)
        if epoch is not None:
            try:
                return float(np.asarray(epoch.start)[0])
            except Exception:
                pass
        firsts = []
        for uid, _r, _c in selected:
            times = np.asarray(group[uid].index)
            if times.size:
                firsts.append(float(times[0]))
        return min(firsts) if firsts else 0.0

    def plot_behavior_overview(self, names=None, t_start=None, t_end=None, show=True):
        """Full-session view of every continuous channel with events shaded."""
        if names is None:
            names = list(self.continuous_behavior)
        signals = [self.continuous_behavior[n] for n in names if n in self.continuous_behavior]
        if not signals:
            raise ValueError("No continuous behaviour registered.")

        if t_start is None:
            t_start = min(s.t_start for s in signals)
        if t_end is None:
            t_end = max(s.t_end for s in signals)

        fig, axes = plt.subplots(len(signals), 1, figsize=(14, 2.2 * len(signals)), sharex=True)
        axes = np.atleast_1d(axes)

        for ax, signal in zip(axes, signals):
            window = signal.restrict(t_start, t_end)
            for j, column in enumerate(signal.columns):
                ax.plot(window.time, window.values[:, j], lw=0.6, label=column)
            label = signal.name if signal.units is None else f"{signal.name} ({signal.units})"
            ax.set_ylabel(label, fontsize=9)
            if signal.n_dims > 1:
                ax.legend(fontsize=7, ncol=signal.n_dims, loc="upper right")

        handles = []
        self._shade_events(list(axes), t_start, t_end, legend_handles=handles)
        if handles:
            axes[0].legend(handles=handles, fontsize=8, loc="upper left", ncol=max(1, len(handles)))

        axes[-1].set_xlabel("time (s)")
        fig.suptitle("Behaviour overview (full session)")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def spiking_behavior_report(
        self,
        window_s=60.0,
        n_units_per_group=5,
        region_key=None,
        cell_type_key=None,
        show=True,
    ):
        """Pre-modelling EDA: unit inventory, behaviour inventory, rasters.

        Run this before :meth:`normalize` to decide on epoch restriction, bin
        size, and which behavioural channels are worth carrying forward.

        Display-only: returns None. The tables it prints stay available through
        :meth:`metadata_summary`, :meth:`unit_inventory`, and
        :meth:`behavior_summary`.
        """
        self.describe_metadata(as_text=True)
        self.unit_inventory(
            region_key=region_key, cell_type_key=cell_type_key, filtered=False, print_table=True
        )
        self.plot_unit_inventory(
            region_key=region_key, cell_type_key=cell_type_key, filtered=False, show=show
        )
        self.behavior_summary(as_text=True)

        if self.continuous_behavior:
            self.plot_behavior_overview(show=show)
            self.plot_raster_with_behavior(
                region_key=region_key,
                cell_type_key=cell_type_key,
                n_units_per_group=n_units_per_group,
                window_s=window_s,
                show=show,
            )
