"""LFP band power on the UMAP manifold, plus high-frequency events (HFEs).

Standalone — does not depend on umap_lastday_test.py. Nothing is written to
disk; everything is print() and plt.show().

CELL 1     per day: load, compute band power, screen for movement artifacts,
           detect HFEs, read the authors' ripple annotations, plot examples of
           both, fit the four embeddings
CELL 2     plotting helpers
CELLS 3-8  one band each (delta, theta, beta, slow gamma, mid gamma, ripple):
           rows = days, columns = groupings, coloured by that band alone
CELL 9     HFEs and the authors' annotated ripples on the manifold, both
           coloured by running speed

What "power" means here: each band is bandpassed (4th-order Butterworth in
second-order-section form, zero-phase), turned into an amplitude envelope by
the Hilbert transform, averaged into the analysis bins, then z-scored. A
threshold of "5 SD" therefore means what it means in the ripple literature —
SD of the band's envelope.

Why HFE and not SWR: the events are defined on the ripple band plus multi-unit
activity, with no requirement that they avoid theta. Events riding a theta
trough are kept. That is a high-frequency event, which may or may not be a
classic sharp-wave ripple, and calling it what it is keeps the claim honest.

An HFE is:
    1. a ripple-band envelope peak above HFE_RIPPLE_Z, its extent taken out to
       HFE_EDGE_Z on either side, lasting between HFE_MIN_MS and HFE_MAX_MS;
    2. with z-scored MUA above HFE_MUA_Z somewhere in the window running from
       HFE_MUA_LEAD_S before the event start to the event end — before or
       during, never after;
    3. at a running speed below HFE_MAX_SPEED_CM_S.

The authors' own Ripple annotations are read out of the NWB and carried
alongside, so ours and theirs can be compared directly rather than trusted.

Artifact screen: movement and EMG raise every band at once, so the z-scored
envelopes stop being independent *locally* — in a one-second window they lock
together. Session-wide correlation says nothing (~0.05 here), so the statistic
is rolling. A bin is dropped when that local correlation is high, the bands are
genuinely elevated, and the ripple band is below ARTIFACT_RIPPLE_KEEP_Z, which
protects real large events. What gets flagged is plotted, speed included: if
the flagged bins are not concentrated at high speed they are probably not
movement artifacts, and the recording was simply clean.
"""

import gc
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import umap
from scipy import signal as sps
from scipy.fft import next_fast_len
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from sklearn.decomposition import PCA

# =============================================================================
# CONFIG
# =============================================================================

DOWNLOAD_DIR = Path("/storage/dandi_downloads").resolve()
SUBJECT = "M05"                   # the mouse with the most units on its last day

BIN_SIZE_S = 0.050                # analysis bin for counts, band power, covariates

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (6.0, 10.0),
    "beta": (12.0, 30.0),
    "slow_gamma": (30.0, 50.0),
    "mid_gamma": (50.0, 90.0),
    "ripple": (120.0, 200.0),
}
# one colormap per band, reused down every row so days are comparable
BAND_CMAPS = {
    "delta": "Blues",
    "theta": "Greens",
    "beta": "Purples",
    "slow_gamma": "Oranges",
    "mid_gamma": "YlOrRd",
    "ripple": "RdPu",
}
BAND_Z_RANGE = (-1.0, 4.0)        # shared colour limits, in SD of the envelope

# --- artifact rejection ------------------------------------------------------
ARTIFACT_CORR_WINDOW_S = 1.0      # window for the rolling cross-band correlation
ARTIFACT_CORR = 0.5               # mean pairwise correlation above this is suspicious
ARTIFACT_ALLBAND_Z = 1.5          # ...and the mean z across bands must exceed this
ARTIFACT_RIPPLE_KEEP_Z = 10.0     # ...unless ripple power exceeds this (a real event)
N_ARTIFACT_EXAMPLES = 4

# --- high-frequency events ---------------------------------------------------
HFE_RIPPLE_Z = 5.0                # ripple-band envelope peak threshold
HFE_EDGE_Z = 2.0                  # event extent runs out to this, either side of the peak
HFE_MIN_SEP_S = 0.05              # refractory between peaks
HFE_MIN_MS = 15.0                 # duration limits on the extent
HFE_MAX_MS = 250.0
HFE_MUA_Z = 2.0                   # MUA must exceed this...
HFE_MUA_LEAD_S = 0.025            # ...within 25 ms BEFORE the event, or during it
HFE_MUA_REGIONS = None            # None = every unit; e.g. ("CA1", "CA3") for hippocampus
HFE_MAX_SPEED_CM_S = 5.0          # speed gate at the event peak
HFE_FINE_BIN_S = 0.005            # bin for the MUA rate
HFE_SMOOTH_S = 0.010              # Gaussian sigma on that rate
HFE_MATCH_TOL_S = 0.05            # slack when matching ours to the authors' annotations
N_EXAMPLE_EVENTS = 6
EXAMPLE_WINDOW_S = 0.40

# --- embedding ---------------------------------------------------------------
UMAP_N_COMPONENTS = 3
UMAP_MIN_RATE_HZ = 0.1
UMAP_SMOOTH_S = 0.10
UMAP_PCA_COMPONENTS = 40
UMAP_MAX_BINS = 15000
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 0
GROUPINGS = ("global", "CA1", "CA3", "RSC")

# Band grids are drawn on UMAP 1 vs 2 by default — twelve 3d panels in one
# figure render slowly and read poorly. Set "3d" for the full cloud; the event
# figures in cell 9 are always 3d.
GRID_PROJECTION = "2d"

RUN_EXAMPLES = True               # artifact and event example plots in cell 1


def find_package_root(start):
    """Walk up from `start` looking for the hybrid_dynamics_core package."""
    start = Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "hybrid_dynamics_core").is_dir():
            return candidate
        nested = candidate / "Hydrangea_demo"
        if (nested / "hybrid_dynamics_core").is_dir():
            return nested
    return start


PACKAGE_ROOT = find_package_root(Path.cwd())
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from hybrid_dynamics_core.metadata import extract_unit_metadata  # noqa: E402


# =============================================================================
# STEP 1 — find the sessions and load one
# =============================================================================


def subject_session_table(root, subject):
    """Behaviour sessions for one subject, oldest first."""
    rows = []
    for path in sorted(Path(root).glob(f"sub-{subject}/*.nwb")):
        if "behavior" not in path.stem:
            continue
        ses = re.search(r"ses-([^_]+)", path.stem).group(1)
        rows.append({
            "subject": subject,
            "session": ses,
            "date": f"{ses[:4]}-{ses[4:6]}-{ses[6:8]}",
            "file": str(path.relative_to(root)),
        })
    return pd.DataFrame(rows).sort_values("session").reset_index(drop=True)


def read_labelled_intervals(path, name_hint="sleep"):
    """The SleepStates table WITH its labels, straight from the NWB.

    pynapple drops the label column on this dataset — the intervals are not
    end-sorted (state intervals and ripple events are interleaved in one
    table), so its IntervalSet constructor sorts them, warns, and discards the
    metadata. That column is where the Ripple annotations live.
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError:
        return None
    try:
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwbfile = io.read()
            table = None
            for key, obj in (nwbfile.intervals or {}).items():
                if name_hint.lower() in key.lower():
                    table = obj
                    break
            if table is None:
                for module in (nwbfile.processing or {}).values():
                    for key, obj in module.data_interfaces.items():
                        if name_hint.lower() in key.lower():
                            table = obj
                            break
            return None if table is None else table.to_dataframe()
    except Exception:
        return None


def annotated_ripples(labels, t_start, t_end):
    """The authors' Ripple intervals that fall inside the analysis epoch."""
    empty = (np.array([]), np.array([]))
    if labels is None or not len(labels):
        return empty
    col = next((c for c in labels.columns
                if str(c).lower() in ("state", "label", "sleep_state", "tags")), None)
    if col is None:
        return empty
    rip = labels[labels[col].astype(str).str.lower().str.startswith("ripple")]
    if not len(rip):
        return empty
    starts = rip["start_time"].values.astype(float)
    stops = rip["stop_time"].values.astype(float)
    inside = (stops > t_start) & (starts < t_end)
    return starts[inside], stops[inside]


def load_session(row, verbose=True):
    """Spikes as sorted arrays, unit metadata, behaviour, LFP, annotations."""
    path = (DOWNLOAD_DIR / row["file"]).resolve()
    t0 = time.perf_counter()
    nwb = nap.load_file(str(path))

    spikes_all = nwb["units"]
    position_all = nwb["Position"]
    maze_ep = nap.IntervalSet(start=float(position_all.index[0]),
                              end=float(position_all.index[-1]))
    maze_s = float(maze_ep.tot_length())
    spikes = spikes_all.restrict(maze_ep)

    meta = extract_unit_metadata(spikes_all).reindex(np.asarray(spikes.index))
    n_spikes = np.array([len(spikes[u]) for u in spikes.index], dtype=int)
    meta["n_spikes_maze"] = n_spikes
    meta["rate_maze_hz"] = n_spikes / maze_s
    sort_cols = [c for c in ("cell_area", "cell_type") if c in meta.columns]
    meta = meta.sort_values(sort_cols + ["rate_maze_hz"]) if sort_cols else meta
    spike_times = [np.sort(np.asarray(spikes[u].index.values, dtype=np.float64))
                   for u in meta.index]

    position = position_all.restrict(maze_ep)
    pos_t = np.asarray(position.index.values, dtype=float)
    pos_x = np.asarray(position["x"].values, dtype=float)

    speed_obj = nwb["Speed"].restrict(maze_ep)
    speed_t = np.asarray(speed_obj.index.values, dtype=float)
    speed_v = np.asarray(speed_obj.values, dtype=float).ravel()

    lfp_key = next(k for k in nwb.keys() if "lfp" in k.lower())
    lfp_obj = nwb[lfp_key].restrict(maze_ep)
    lfp_t = np.asarray(lfp_obj.index.values, dtype=float)
    lfp_v = np.asarray(lfp_obj.values, dtype=float).ravel()
    fs = float(1.0 / np.median(np.diff(lfp_t)))

    labels = read_labelled_intervals(path)
    annot_start, annot_stop = annotated_ripples(labels, float(maze_ep.start[0]),
                                                float(maze_ep.end[0]))

    session = {
        "label": f"{row['subject']} {row['date']}",
        "date": row["date"],
        "t_start": float(maze_ep.start[0]),
        "t_end": float(maze_ep.end[0]),
        "maze_s": maze_s,
        "meta": meta,
        "spike_times": spike_times,
        "pos_t": pos_t,
        "pos_x": pos_x,
        "speed_t": speed_t,
        "speed_v": speed_v,
        "lfp_key": lfp_key,
        "lfp_t": lfp_t,
        "lfp_v": lfp_v,
        "fs": fs,
        "annot_start": annot_start,
        "annot_stop": annot_stop,
    }
    del spikes, spikes_all, position, position_all, speed_obj, lfp_obj, nwb
    gc.collect()
    if verbose:
        print(f"loaded {row['file']}: {len(spike_times)} units, "
              f"{maze_s / 60:.1f} min, LFP @ {fs:.0f} Hz, "
              f"{len(annot_start)} annotated ripples in the epoch "
              f"({time.perf_counter() - t0:.1f}s)")
    return session


# =============================================================================
# STEP 2 — band power and the artifact screen
# =============================================================================


def bandpass_sos(values, fs, lo, hi, order=4):
    """Zero-phase Butterworth bandpass in SOS form.

    SOS rather than b/a because delta (0.5-4 Hz at 1250 Hz) sits at ~0.1% of
    Nyquist, where transfer-function coefficients are numerically fragile.
    """
    sos = sps.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, values)


def analytic_envelope(values):
    """Hilbert amplitude envelope, FFT-padded to a fast length."""
    n = len(values)
    return np.abs(sps.hilbert(values, N=next_fast_len(n)))[:n]


def bin_mean(sample_t, sample_v, edges):
    """Mean of a densely sampled signal within each bin."""
    idx = np.searchsorted(edges, sample_t, side="right") - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    idx, vals = idx[ok], sample_v[ok]
    total = np.bincount(idx, weights=vals, minlength=len(edges) - 1)
    count = np.bincount(idx, minlength=len(edges) - 1)
    return total / np.maximum(count, 1)


def zscore(values):
    return (values - np.nanmean(values)) / (np.nanstd(values) + 1e-12)


def band_power_table(lfp_t, lfp_v, fs, edges):
    """z-scored envelope for every band, on the analysis bins.

    The full-resolution ripple envelope and filtered trace come back too — the
    detector and the example plots need them before they are discarded.
    """
    band_z, ripple_full, ripple_filt = {}, None, None
    for name, (lo, hi) in BANDS.items():
        filt = bandpass_sos(lfp_v, fs, lo, hi)
        env = analytic_envelope(filt)
        band_z[name] = zscore(bin_mean(lfp_t, env, edges))
        if name == "ripple":
            ripple_full = zscore(env)
            ripple_filt = filt
        else:
            del filt, env
            gc.collect()
    return band_z, ripple_full, ripple_filt


def rolling_cross_band_corr(stacked, window_bins):
    """Mean pairwise correlation among the band envelopes, in a sliding window.

    Computed from rolling moments rather than a loop over windows: for each
    pair, corr = (E[xy] - E[x]E[y]) / (sd_x sd_y), every term a boxcar filter.
    """
    n_bands = stacked.shape[1]
    mean = np.column_stack([uniform_filter1d(stacked[:, j], window_bins, mode="nearest")
                            for j in range(n_bands)])
    msq = np.column_stack([uniform_filter1d(stacked[:, j] ** 2, window_bins, mode="nearest")
                           for j in range(n_bands)])
    sd = np.sqrt(np.maximum(msq - mean ** 2, 1e-12))

    total, n_pairs = np.zeros(len(stacked)), 0
    for a in range(n_bands):
        for b in range(a + 1, n_bands):
            cross = uniform_filter1d(stacked[:, a] * stacked[:, b],
                                     window_bins, mode="nearest")
            total += (cross - mean[:, a] * mean[:, b]) / (sd[:, a] * sd[:, b])
            n_pairs += 1
    return total / n_pairs


def artifact_mask(band_z, bin_s=BIN_SIZE_S):
    """True where a bin looks like a movement/EMG artifact.

    Three conditions, all required: the bands must be locally correlated (they
    are rising together rather than independently), they must actually be
    elevated (correlation in a quiet stretch means nothing), and the ripple
    band must be below the escape threshold — a genuine large event drags
    broadband power up with it and is not an artifact.
    """
    stacked = np.column_stack([band_z[name] for name in BANDS])
    broadband_z = stacked.mean(axis=1)
    window_bins = max(3, int(round(ARTIFACT_CORR_WINDOW_S / bin_s)))
    corr = rolling_cross_band_corr(stacked, window_bins)
    bad = ((corr > ARTIFACT_CORR)
           & (broadband_z > ARTIFACT_ALLBAND_Z)
           & (band_z["ripple"] < ARTIFACT_RIPPLE_KEEP_Z))
    return bad, stacked, corr, broadband_z


def plot_artifacts(session, centers, band_z, bad, corr, broadband_z, speed):
    """What the artifact rule caught, and whether it looks like movement."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    fig.suptitle(f"{session['label']} — movement/EMG artifact screen "
                 f"({bad.sum()} of {len(bad)} bins, {100 * bad.mean():.2f}%)")

    ax = axes[0]
    ax.hist(corr, bins=60, color="0.6")
    ax.axvline(ARTIFACT_CORR, color="C3", ls="--", lw=1)
    ax.set_xlabel(f"rolling cross-band correlation ({ARTIFACT_CORR_WINDOW_S:.0f}s)")
    ax.set_ylabel("bins")

    ax = axes[1]
    step = max(1, len(corr) // 20000)
    sc = ax.scatter(corr[::step], broadband_z[::step], c=speed[::step], s=2,
                    alpha=0.35, cmap="viridis", linewidths=0, rasterized=True)
    ax.axvline(ARTIFACT_CORR, color="C3", ls="--", lw=1)
    ax.axhline(ARTIFACT_ALLBAND_Z, color="C3", ls="--", lw=1)
    ax.set_xlabel("cross-band correlation")
    ax.set_ylabel("mean band power (z)")
    ax.set_title(f"excluded = upper right, unless ripple > "
                 f"{ARTIFACT_RIPPLE_KEEP_Z:.0f} SD", fontsize=8)
    fig.colorbar(sc, ax=ax).set_label("speed (cm/s)", fontsize=8)

    ax = axes[2]
    bins = np.linspace(0, np.nanpercentile(speed, 99.5), 40)
    ax.hist(speed[~bad], bins=bins, density=True, histtype="step", label="kept")
    if bad.any():
        ax.hist(speed[bad], bins=bins, density=True, histtype="step", label="artifact")
    ax.set_xlabel("speed (cm/s)")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    if not bad.any():
        print("  no artifact bins flagged — nothing to plot in detail")
        return

    # the longest flagged stretches, as raw LFP + the band envelopes + speed
    flips = np.flatnonzero(np.diff(bad.astype(np.int8)))
    starts = np.r_[0, flips + 1][np.r_[bad[0], bad[flips + 1]]]
    lengths = []
    for s in starts:
        run = 0
        while s + run < len(bad) and bad[s + run]:
            run += 1
        lengths.append(run)
    order = np.argsort(lengths)[::-1][:N_ARTIFACT_EXAMPLES]

    fig, axes = plt.subplots(3, len(order), figsize=(3.4 * len(order), 6.5),
                             sharex="col", squeeze=False)
    fig.suptitle(f"{session['label']} — longest flagged artifact stretches")
    for col, k in enumerate(order):
        s, run = starts[k], lengths[k]
        stop = min(s + run, len(centers) - 1)
        t0, t1 = centers[s] - 1.0, centers[stop] + 1.0

        ax = axes[0, col]
        m = (session["lfp_t"] >= t0) & (session["lfp_t"] <= t1)
        ax.plot(session["lfp_t"][m], session["lfp_v"][m], lw=0.5, color="k")
        ax.axvspan(centers[s], centers[stop], color="C3", alpha=0.15)
        ax.set_title(f"{centers[s]:.1f}s | {run * BIN_SIZE_S * 1e3:.0f} ms flagged",
                     fontsize=8)
        if col == 0:
            ax.set_ylabel("raw LFP")

        ax = axes[1, col]
        bm = (centers >= t0) & (centers <= t1)
        for name in BANDS:
            ax.plot(centers[bm], band_z[name][bm], lw=0.9, label=name)
        ax.axhline(ARTIFACT_ALLBAND_Z, color="C3", ls=":", lw=0.8)
        if col == 0:
            ax.set_ylabel("band power (z)")
            ax.legend(fontsize=6, ncol=2)

        ax = axes[2, col]
        ax.plot(centers[bm], speed[bm], lw=0.9, color="k")
        if col == 0:
            ax.set_ylabel("speed\n(cm/s)")
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# =============================================================================
# STEP 3 — high-frequency events
# =============================================================================


def population_mua(session, fine_edges, regions=HFE_MUA_REGIONS):
    """z-scored multi-unit rate.

    `regions` of None means every recorded unit, which is what MUA normally
    means. Note that high-rate units dominate the variance of a summed rate —
    on this dataset RSC interneurons fire at ~17 Hz against ~1.3 Hz for
    hippocampal pyramidal cells — so setting regions=("CA1", "CA3") gives a
    hippocampal MUA that tracks the sharp wave more directly.
    """
    meta = session["meta"]
    if regions is None:
        take = np.ones(len(meta), dtype=bool)
    else:
        take = meta["cell_area"].astype(str).isin(regions).values
    if take.sum() == 0:
        return None, 0
    counts = np.zeros(len(fine_edges) - 1)
    for st, use in zip(session["spike_times"], take):
        if use:
            counts += np.diff(np.searchsorted(st, fine_edges))
    rate = counts / (HFE_FINE_BIN_S * take.sum())
    rate = gaussian_filter1d(rate, sigma=HFE_SMOOTH_S / HFE_FINE_BIN_S, mode="nearest")
    return zscore(rate), int(take.sum())


def event_extent(envelope_z, peak, edge_z):
    """Walk out from a peak to where the envelope drops below `edge_z`."""
    start = peak
    while start > 0 and envelope_z[start] > edge_z:
        start -= 1
    stop = peak
    last = len(envelope_z) - 1
    while stop < last and envelope_z[stop] > edge_z:
        stop += 1
    return start, stop


def detect_hfe(session, ripple_z_full, mua_z, fine_centers):
    """Ripple-band events with MUA before or during them, below a speed gate.

    The MUA window runs from HFE_MUA_LEAD_S before the event start to the event
    end — never after. A population burst that only arrives once the ripple
    band has finished is not evidence that the two belong together, and would
    let any late, unrelated bump validate an event.
    """
    fs = session["fs"]
    lfp_t = session["lfp_t"]
    peaks, props = sps.find_peaks(ripple_z_full, height=HFE_RIPPLE_Z,
                                  distance=int(HFE_MIN_SEP_S * fs))
    funnel = {"candidates": len(peaks), "after_duration": 0,
              "after_mua": 0, "after_speed": 0}

    rows = []
    for peak, height in zip(peaks, props["peak_heights"]):
        i0, i1 = event_extent(ripple_z_full, peak, HFE_EDGE_Z)
        dur_ms = (i1 - i0) / fs * 1e3
        if not (HFE_MIN_MS <= dur_ms <= HFE_MAX_MS):
            continue
        funnel["after_duration"] += 1

        t_peak, t0, t1 = lfp_t[peak], lfp_t[i0], lfp_t[i1]
        if mua_z is None:
            continue
        lo = np.searchsorted(fine_centers, t0 - HFE_MUA_LEAD_S)
        hi = np.searchsorted(fine_centers, t1, side="right")
        if hi <= lo:
            continue
        window = mua_z[lo:hi]
        best = int(np.argmax(window))
        if window[best] < HFE_MUA_Z:
            continue
        funnel["after_mua"] += 1

        speed = float(np.interp(t_peak, session["speed_t"], session["speed_v"]))
        if speed > HFE_MAX_SPEED_CM_S:
            continue
        funnel["after_speed"] += 1

        t_mua = fine_centers[lo + best]
        rows.append({
            "t_peak": t_peak,
            "t_start": t0,
            "t_end": t1,
            "duration_ms": dur_ms,
            "ripple_z": float(height),
            "mua_z": float(window[best]),
            "t_mua": t_mua,
            "mua_lead_ms": (t_peak - t_mua) * 1e3,   # >0 = MUA before the peak
            "speed": speed,
        })
    return pd.DataFrame(rows), funnel


def match_to_annotations(events, annot_start, annot_stop, tol=HFE_MATCH_TOL_S):
    """Which of our events overlap an annotated ripple, and vice versa."""
    if not len(events) or not len(annot_start):
        return (np.zeros(len(events), dtype=bool),
                np.zeros(len(annot_start), dtype=bool))
    ours_start = events["t_start"].values - tol
    ours_end = events["t_end"].values + tol
    ours_hit = np.zeros(len(events), dtype=bool)
    theirs_hit = np.zeros(len(annot_start), dtype=bool)
    for j, (a0, a1) in enumerate(zip(annot_start, annot_stop)):
        overlap = (ours_start <= a1) & (ours_end >= a0)
        ours_hit |= overlap
        theirs_hit[j] = overlap.any()
    return ours_hit, theirs_hit


def _event_panel(session, ripple_filt, ripple_z_full, mua_z, fine_centers,
                 axes, col, t_center, t0_span, t1_span, title, shade=None):
    """Three stacked panels for one event: raw LFP, ripple band, MUA."""
    m = (session["lfp_t"] >= t0_span) & (session["lfp_t"] <= t1_span)
    tt = (session["lfp_t"][m] - t_center) * 1e3

    ax = axes[0, col]
    ax.plot(tt, session["lfp_v"][m], lw=0.6, color="k")
    if shade is not None:
        ax.axvspan((shade[0] - t_center) * 1e3, (shade[1] - t_center) * 1e3,
                   color="crimson", alpha=0.12)
    ax.set_title(title, fontsize=8)
    if col == 0:
        ax.set_ylabel("raw LFP")

    ax = axes[1, col]
    ax.plot(tt, ripple_filt[m], lw=0.6, color="crimson")
    ax.plot(tt, ripple_z_full[m] * np.std(ripple_filt[m]), lw=0.8, color="k", alpha=0.6)
    if col == 0:
        ax.set_ylabel(f"{BANDS['ripple'][0]:.0f}-{BANDS['ripple'][1]:.0f} Hz\n"
                      "(envelope in black)")

    ax = axes[2, col]
    fm = (fine_centers >= t0_span) & (fine_centers <= t1_span)
    ax.plot((fine_centers[fm] - t_center) * 1e3, mua_z[fm], lw=0.9, color="C0")
    ax.axhline(HFE_MUA_Z, color="C0", ls=":", lw=0.8)
    ax.axvline(0, color="crimson", lw=0.8)
    if shade is not None:
        ax.axvspan((shade[0] - HFE_MUA_LEAD_S - t_center) * 1e3,
                   (shade[1] - t_center) * 1e3, color="C0", alpha=0.10)
    ax.set_xlabel("ms from peak")
    if col == 0:
        ax.set_ylabel("MUA (z)")


def plot_hfe_examples(session, ripple_filt, ripple_z_full, mua_z, fine_centers,
                      events, matched, n_examples=N_EXAMPLE_EVENTS):
    """Our events, spread across the session rather than the biggest ones."""
    if not len(events):
        print("  no HFEs to plot")
        return
    picks = np.linspace(0, len(events) - 1, min(n_examples, len(events))).astype(int)
    half = EXAMPLE_WINDOW_S / 2
    fig, axes = plt.subplots(3, len(picks), figsize=(3.0 * len(picks), 6.5),
                             sharex="col", squeeze=False)
    fig.suptitle(f"{session['label']} — our HFEs (ripple z > {HFE_RIPPLE_Z}, "
                 f"MUA z > {HFE_MUA_Z} within {HFE_MUA_LEAD_S * 1e3:.0f} ms before "
                 f"or during, speed < {HFE_MAX_SPEED_CM_S:.0f} cm/s). "
                 "Shaded: event extent, and the MUA search window")
    for col, i in enumerate(picks):
        ev = events.iloc[i]
        tag = "matches annotation" if matched[i] else "ours only"
        _event_panel(session, ripple_filt, ripple_z_full, mua_z, fine_centers,
                     axes, col, ev["t_peak"],
                     ev["t_peak"] - half, ev["t_peak"] + half,
                     f"{ev['t_peak']:.1f}s | {ev['speed']:.1f} cm/s | "
                     f"{ev['duration_ms']:.0f} ms\nMUA lead {ev['mua_lead_ms']:.0f} ms "
                     f"| {tag}",
                     shade=(ev["t_start"], ev["t_end"]))
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_annotated_examples(session, ripple_filt, ripple_z_full, mua_z, fine_centers,
                            theirs_hit, n_examples=N_EXAMPLE_EVENTS):
    """The authors' annotated ripples, drawn exactly the same way."""
    starts, stops = session["annot_start"], session["annot_stop"]
    if not len(starts):
        print("  no annotated ripples inside the maze epoch")
        return
    picks = np.linspace(0, len(starts) - 1, min(n_examples, len(starts))).astype(int)
    half = EXAMPLE_WINDOW_S / 2
    fig, axes = plt.subplots(3, len(picks), figsize=(3.0 * len(picks), 6.5),
                             sharex="col", squeeze=False)
    fig.suptitle(f"{session['label']} — the authors' annotated ripples "
                 f"({len(starts)} in this epoch)")
    for col, i in enumerate(picks):
        mid = 0.5 * (starts[i] + stops[i])
        speed = float(np.interp(mid, session["speed_t"], session["speed_v"]))
        tag = "we found it" if theirs_hit[i] else "we missed it"
        _event_panel(session, ripple_filt, ripple_z_full, mua_z, fine_centers,
                     axes, col, mid, mid - half, mid + half,
                     f"{mid:.1f}s | {speed:.1f} cm/s | "
                     f"{(stops[i] - starts[i]) * 1e3:.0f} ms\n{tag}",
                     shade=(starts[i], stops[i]))
    fig.tight_layout()
    plt.show()
    plt.close(fig)


# =============================================================================
# STEP 4 — the embeddings
# =============================================================================


def counts_from_spike_times(spike_times, edges):
    out = np.empty((len(edges) - 1, len(spike_times)), dtype=np.float32)
    for j, st in enumerate(spike_times):
        out[:, j] = np.diff(np.searchsorted(st, edges))
    return out


def prepare_matrix(counts, bin_s):
    """counts -> sqrt -> smooth along time -> per-unit z-score."""
    X = np.sqrt(counts)
    X = gaussian_filter1d(X, sigma=UMAP_SMOOTH_S / bin_s, axis=0, mode="nearest")
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)


def subsample_index(n_rows, must_keep, max_bins=UMAP_MAX_BINS):
    """Strided subsample, with every must-keep bin forced in.

    Events are rare enough that a plain stride would drop most of them, and
    then there would be nothing to colour in cell 9.
    """
    stride = max(1, n_rows // max_bins)
    idx = np.union1d(np.arange(0, n_rows, stride), np.flatnonzero(must_keep))
    return idx.astype(int)


def run_umap(X, seed=UMAP_RANDOM_STATE):
    n_pcs = int(min(UMAP_PCA_COMPONENTS, X.shape[1], X.shape[0] - 1))
    pca = PCA(n_components=n_pcs, random_state=seed)
    scores = pca.fit_transform(X)
    reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
                        min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC, random_state=seed)
    return reducer.fit_transform(scores), float(pca.explained_variance_ratio_.sum())


def bins_covering(edges, starts, stops=None):
    """Boolean over bins: True where an event falls inside the bin."""
    hit = np.zeros(len(edges) - 1, dtype=bool)
    if starts is None or len(starts) == 0:
        return hit
    if stops is None:
        stops = starts
    for a, b in zip(np.atleast_1d(starts), np.atleast_1d(stops)):
        lo = np.searchsorted(edges, a, side="right") - 1
        hi = np.searchsorted(edges, b, side="right") - 1
        lo, hi = max(lo, 0), min(hi, len(hit) - 1)
        if hi >= lo:
            hit[lo:hi + 1] = True
    return hit


def process_day(row):
    """Everything for one recording, reduced to what the figures need."""
    session = load_session(row)
    label = session["label"]
    edges = np.arange(session["t_start"], session["t_end"], BIN_SIZE_S)
    centers = edges[:-1] + BIN_SIZE_S / 2

    # --- band power ---------------------------------------------------------
    t0 = time.perf_counter()
    band_z, ripple_z_full, ripple_filt = band_power_table(
        session["lfp_t"], session["lfp_v"], session["fs"], edges)
    print(f"  band power: {len(BANDS)} bands in {time.perf_counter() - t0:.1f}s")

    # --- artifacts ----------------------------------------------------------
    speed = np.interp(centers, session["speed_t"], session["speed_v"])
    bad, stacked, corr, broadband_z = artifact_mask(band_z)
    session_corr = np.corrcoef(stacked.T)[np.triu_indices(len(BANDS), k=1)]
    print(f"  cross-band correlation: session-wide mean {session_corr.mean():.2f}, "
          f"rolling ({ARTIFACT_CORR_WINDOW_S:.0f}s) median {np.median(corr):.2f}, "
          f"95th pct {np.percentile(corr, 95):.2f}, max {corr.max():.2f}")
    print(f"  artifact bins dropped: {bad.sum()} of {len(bad)} "
          f"({100 * bad.mean():.2f}%)")
    if bad.any():
        print(f"  speed in flagged bins: median {np.median(speed[bad]):.2f} cm/s "
              f"vs {np.median(speed[~bad]):.2f} cm/s elsewhere")
    if RUN_EXAMPLES:
        plot_artifacts(session, centers, band_z, bad, corr, broadband_z, speed)
    keep = ~bad

    # --- high-frequency events ----------------------------------------------
    fine_edges = np.arange(session["t_start"], session["t_end"], HFE_FINE_BIN_S)
    fine_centers = fine_edges[:-1] + HFE_FINE_BIN_S / 2
    mua_z, n_mua_units = population_mua(session, fine_edges)
    events, funnel = detect_hfe(session, ripple_z_full, mua_z, fine_centers)
    ours_hit, theirs_hit = match_to_annotations(events, session["annot_start"],
                                                session["annot_stop"])

    minutes = session["maze_s"] / 60
    print(f"  MUA from {n_mua_units} units "
          f"({'all regions' if HFE_MUA_REGIONS is None else '+'.join(HFE_MUA_REGIONS)})")
    print(f"  HFE funnel: {funnel['candidates']} ripple peaks > {HFE_RIPPLE_Z} SD "
          f"-> {funnel['after_duration']} within {HFE_MIN_MS:.0f}-{HFE_MAX_MS:.0f} ms "
          f"-> {funnel['after_mua']} with MUA > {HFE_MUA_Z} SD in window "
          f"-> {funnel['after_speed']} below {HFE_MAX_SPEED_CM_S:.0f} cm/s "
          f"({len(events) / minutes:.2f}/min)")
    if len(events):
        print(f"  duration: median {events['duration_ms'].median():.0f} ms | "
              f"MUA lead: median {events['mua_lead_ms'].median():.0f} ms | "
              f"speed: median {events['speed'].median():.2f} cm/s")
    print(f"  annotated ripples in epoch: {len(session['annot_start'])} | "
          f"ours matching one: {int(ours_hit.sum())}/{len(events)} | "
          f"theirs we recovered: {int(theirs_hit.sum())}/{len(session['annot_start'])}")

    if RUN_EXAMPLES:
        plot_hfe_examples(session, ripple_filt, ripple_z_full, mua_z,
                          fine_centers, events, ours_hit)
        plot_annotated_examples(session, ripple_filt, ripple_z_full, mua_z,
                                fine_centers, theirs_hit)
    del ripple_filt, ripple_z_full, mua_z
    gc.collect()

    # --- covariates ---------------------------------------------------------
    track_x = np.interp(centers, session["pos_t"], session["pos_x"])
    hfe_bin = bins_covering(edges, events["t_start"].values if len(events) else None,
                            events["t_end"].values if len(events) else None)
    annot_bin = bins_covering(edges, session["annot_start"], session["annot_stop"])

    # --- embeddings ---------------------------------------------------------
    counts = counts_from_spike_times(session["spike_times"], edges)
    X = prepare_matrix(counts, BIN_SIZE_S)[keep]
    meta = session["meta"]
    region = meta["cell_area"].astype(str).values
    alive = meta["rate_maze_hz"].values > UMAP_MIN_RATE_HZ

    hfe_kept, annot_kept = hfe_bin[keep], annot_bin[keep]
    sub = subsample_index(X.shape[0], hfe_kept | annot_kept)

    embeddings = {}
    for name in GROUPINGS:
        mask = alive if name == "global" else (alive & (region == name))
        if mask.sum() < 5:
            print(f"  {name}: {int(mask.sum())} units, skipped")
            continue
        t_fit = time.perf_counter()
        emb, var = run_umap(X[np.ix_(sub, np.flatnonzero(mask))])
        embeddings[name] = emb
        print(f"  {name:7s}: {int(mask.sum()):3d} units, {len(sub)} bins, "
              f"PCA kept {100 * var:.1f}%, {time.perf_counter() - t_fit:.0f}s")

    day = {
        "label": label,
        "date": session["date"],
        "embeddings": embeddings,
        "band_z": {name: band_z[name][keep][sub] for name in BANDS},
        "speed": speed[keep][sub],
        "track_x": track_x[keep][sub],
        "hfe": hfe_kept[sub],
        "annot": annot_kept[sub],
        "events": events,
        "n_annotated": int(len(session["annot_start"])),
        "n_matched": int(ours_hit.sum()),
        "n_recovered": int(theirs_hit.sum()),
        "n_bins": len(sub),
        "frac_artifact": float(bad.mean()),
    }
    del session, counts, X, band_z
    gc.collect()
    return day


sessions = subject_session_table(DOWNLOAD_DIR, SUBJECT)
print(f"{SUBJECT}: {len(sessions)} behaviour sessions")
print(sessions[["date", "file"]].to_string(index=False))

days = []
for _, row in sessions.iterrows():
    print("\n" + "=" * 78)
    print(row["subject"], row["date"])
    print("=" * 78)
    days.append(process_day(row))

print("\n" + "=" * 78)
print("summary")
print("=" * 78)
print(pd.DataFrame([{
    "date": d["date"],
    "bins_embedded": d["n_bins"],
    "frac_artifact": round(d["frac_artifact"], 4),
    "hfe": len(d["events"]),
    "hfe_bins": int(d["hfe"].sum()),
    "annotated": d["n_annotated"],
    "annotated_bins": int(d["annot"].sum()),
    "ours_matching_theirs": d["n_matched"],
    "theirs_recovered": d["n_recovered"],
    "median_speed_at_hfe": (round(d["events"]["speed"].median(), 2)
                            if len(d["events"]) else np.nan),
} for d in days]).to_string(index=False))


# %% ===========================================================================
# CELL 2 — plotting helpers (instant; run before any band cell)
# ==============================================================================
# One band per cell from here on, so a single band can be re-rendered without
# redrawing the other five. Each band figure is rows = days, columns = the four
# groupings, coloured by that band alone on one shared scale — a band that
# shifts between days shows as a change in colour, not a change in scale.


def scatter_panel(ax, emb, colour, cmap, vmin, vmax, projection):
    if projection == "3d":
        return ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=colour, cmap=cmap,
                          vmin=vmin, vmax=vmax, s=1.2, alpha=0.55,
                          linewidths=0, rasterized=True)
    return ax.scatter(emb[:, 0], emb[:, 1], c=colour, cmap=cmap,
                      vmin=vmin, vmax=vmax, s=1.5, alpha=0.6,
                      linewidths=0, rasterized=True)


def band_figure(band, days, projection=GRID_PROJECTION):
    """One band: rows = days, columns = groupings, one shared colour scale."""
    rows = [d for d in days if d["embeddings"]]
    if not rows:
        print(f"{band}: nothing to plot")
        return
    groupings = [g for g in GROUPINGS if any(g in d["embeddings"] for d in rows)]
    subplot_kw = {"projection": "3d"} if projection == "3d" else {}
    fig, axes = plt.subplots(len(rows), len(groupings),
                             figsize=(3.2 * len(groupings), 3.1 * len(rows)),
                             squeeze=False, subplot_kw=subplot_kw)
    lo, hi = BANDS[band]
    fig.suptitle(f"{SUBJECT} — {band} ({lo:.0f}-{hi:.0f} Hz) power on the manifold "
                 f"(z, {BAND_Z_RANGE[0]:.0f} to {BAND_Z_RANGE[1]:.0f} SD)")

    for r, day in enumerate(rows):
        for c, grouping in enumerate(groupings):
            ax = axes[r, c]
            if grouping not in day["embeddings"]:
                ax.axis("off")
                continue
            sc = scatter_panel(ax, day["embeddings"][grouping], day["band_z"][band],
                               BAND_CMAPS[band], BAND_Z_RANGE[0], BAND_Z_RANGE[1],
                               projection)
            if r == 0:
                ax.set_title(grouping, fontsize=10)
            if c == 0:
                ax.set_ylabel(day["date"], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if projection == "3d":
                ax.set_zticks([])
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6,
                 pad=0.02).set_label(f"{band} power (z)", fontsize=9)
    plt.show()
    plt.close(fig)


# %% ===========================================================================
# CELL 3 — delta
# ==============================================================================

band_figure("delta", days)


# %% ===========================================================================
# CELL 4 — theta
# ==============================================================================

band_figure("theta", days)


# %% ===========================================================================
# CELL 5 — beta
# ==============================================================================

band_figure("beta", days)


# %% ===========================================================================
# CELL 6 — slow gamma
# ==============================================================================

band_figure("slow_gamma", days)


# %% ===========================================================================
# CELL 7 — mid gamma
# ==============================================================================

band_figure("mid_gamma", days)


# %% ===========================================================================
# CELL 8 — ripple band
# ==============================================================================

band_figure("ripple", days)


# %% ===========================================================================
# CELL 9 — events on the manifold: ours and the authors'
# ==============================================================================
# Two rows per grouping. Top: the authors' annotated ripples. Bottom: our HFEs.
# Every bin in grey underneath, the event bins on top coloured by running speed
# at the event — the question is whether events landing in one part of the
# manifold are the ones the animal was still for.


def event_figure(grouping, days):
    rows = [d for d in days if grouping in d["embeddings"]]
    if not rows:
        return
    fig, axes = plt.subplots(2, len(rows), figsize=(4.8 * len(rows), 8.8),
                             squeeze=False, subplot_kw={"projection": "3d"})
    fig.suptitle(f"{SUBJECT} — {grouping} — event bins coloured by speed\n"
                 f"top: authors' annotated ripples | bottom: our HFEs "
                 f"(speed < {HFE_MAX_SPEED_CM_S:.0f} cm/s by construction)")

    pooled = [d["speed"][d[key]] for d in rows for key in ("annot", "hfe")
              if d[key].any()]
    speeds = np.concatenate(pooled) if pooled else np.array([0.0, 1.0])
    vmin, vmax = float(np.min(speeds)), float(max(np.max(speeds), 1e-3))

    for r, key in enumerate(("annot", "hfe")):
        for c, day in enumerate(rows):
            ax = axes[r, c]
            emb, hit = day["embeddings"][grouping], day[key]
            ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c="0.82", s=1.0,
                       alpha=0.35, linewidths=0, rasterized=True)
            if hit.any():
                sc = ax.scatter(emb[hit, 0], emb[hit, 1], emb[hit, 2],
                                c=day["speed"][hit], cmap="viridis",
                                vmin=vmin, vmax=vmax, s=34,
                                edgecolors="k", linewidths=0.4)
                fig.colorbar(sc, ax=ax, shrink=0.55,
                             pad=0.08).set_label("speed (cm/s)", fontsize=8)
            kind = "annotated" if key == "annot" else "HFE"
            ax.set_title(f"{day['date']} — {int(hit.sum())} {kind} bins", fontsize=9)
            ax.set_xlabel("UMAP 1", fontsize=8)
            ax.set_ylabel("UMAP 2", fontsize=8)
            ax.set_zlabel("UMAP 3", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


for grouping in GROUPINGS:
    event_figure(grouping, days)

# Event-level summary: durations, how far the MUA leads, and the speed the
# events happen at.
all_events = pd.concat([d["events"].assign(date=d["date"]) for d in days
                        if len(d["events"])], ignore_index=True)
if len(all_events):
    print("\nHFEs, all days")
    print(all_events.groupby("date")[["ripple_z", "mua_z", "duration_ms",
                                      "mua_lead_ms", "speed"]]
          .median().round(2).to_string())

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    fig.suptitle(f"{SUBJECT} — high-frequency events")
    axes[0].hist(all_events["duration_ms"], bins=30, color="C0")
    axes[0].set_xlabel("event duration (ms)")
    axes[0].set_ylabel("events")
    axes[1].hist(all_events["mua_lead_ms"], bins=30, color="C1")
    axes[1].axvline(0, color="crimson", lw=1)
    axes[1].set_xlabel("MUA lead (ms; >0 = before the peak)")
    axes[2].hist(all_events["speed"], bins=30, color="C2")
    axes[2].axvline(HFE_MAX_SPEED_CM_S, color="C3", ls=":", lw=1)
    axes[2].set_xlabel("speed at peak (cm/s)")
    axes[3].scatter(all_events["mua_z"], all_events["ripple_z"], s=8, alpha=0.5, c="k")
    axes[3].set_xlabel("MUA (z)")
    axes[3].set_ylabel("ripple-band (z)")
    fig.tight_layout()
    plt.show()
    plt.close(fig)
