"""LFP band power on the UMAP manifold, plus our own sharp-wave ripples.

Standalone — does not depend on umap_lastday_test.py. Three notebook cells,
split at the "CELL 2" and "CELL 3" banners. Nothing is written to disk.

CELL 1  per day: load, compute band power, drop movement artifacts, detect
        sharp-wave ripples, plot example events, fit the four embeddings
CELL 2  band grids: one figure per grouping (global / CA1 / CA3 / RSC),
        rows = days, columns = bands, each panel the same manifold coloured
        by that band's power
CELL 3  the SWR figures: manifold coloured by running speed at our detected
        sharp-wave ripple bins

What "power" means here: each band is bandpassed (4th-order Butterworth in
second-order-section form, zero-phase), turned into an amplitude envelope by
the Hilbert transform, averaged into the analysis bins, then z-scored. So a
threshold of "5 SD" means what it means in the ripple literature — SD of the
band's envelope, not of raw power.

Two things this does that the first script did not:

1. Artifact rejection. Movement and EMG raise every band at once. A bin is
   dropped when all six bands are simultaneously elevated AND the ripple band
   is below ARTIFACT_RIPPLE_KEEP_Z — the second clause protects genuine large
   ripples, which also push broadband power up.

2. Our own ripples. The dataset's Ripple annotations are almost absent from
   the maze epoch, so events are detected here instead: a ripple-band envelope
   peak above SWR_RIPPLE_Z, preceded within SWR_SW_WINDOW_S by a peak in the
   z-scored CA3 pyramidal population rate (the sharp wave). Both halves are
   required, which is what separates a sharp-wave ripple from a ripple-band
   blip.
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
from scipy.ndimage import gaussian_filter1d
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
ARTIFACT_ALLBAND_Z = 2.0          # "all bands up together": min z across bands above this
ARTIFACT_RIPPLE_KEEP_Z = 10.0     # ...unless ripple power exceeds this (a real event)

# --- our sharp-wave ripple detector ------------------------------------------
SWR_RIPPLE_Z = 5.0                # minimum ripple-band envelope peak
SWR_MIN_SEP_S = 0.05              # refractory between ripple peaks
SWR_SW_WINDOW_S = 0.20            # how far before the ripple to look for the sharp wave
SWR_SW_Z = 2.0                    # minimum CA3 pyramidal population-rate peak
SWR_FINE_BIN_S = 0.010            # bin for the CA3 population rate
SWR_SMOOTH_S = 0.015              # Gaussian sigma on that rate
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

# 18 three-dimensional panels in one figure render slowly and read poorly, so
# the band grid is drawn on UMAP 1 vs 2 by default. Set "3d" if you want the
# full cloud in each cell of the grid; the SWR figures are always 3d.
GRID_PROJECTION = "2d"

RUN_EXAMPLES = True               # example event plots in cell 1


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


def load_session(row, verbose=True):
    """Spikes as sorted arrays, unit metadata, behaviour, and the CA1 LFP."""
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
    }
    del spikes, spikes_all, position, position_all, speed_obj, lfp_obj, nwb
    gc.collect()
    if verbose:
        print(f"loaded {row['file']}: {len(spike_times)} units, "
              f"{maze_s / 60:.1f} min, LFP @ {fs:.0f} Hz "
              f"({time.perf_counter() - t0:.1f}s)")
    return session


# =============================================================================
# STEP 2 — band power, artifacts, and sharp-wave ripples
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

    The full-resolution ripple envelope and filtered trace are returned too —
    the detector and the example plots need them before they are discarded.
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


def artifact_mask(band_z):
    """True where a bin looks like a movement/EMG artifact.

    Broadband co-activation is the signature: every band rises together, which
    a genuine oscillation never does. The ripple-band escape clause keeps large
    real ripples, which do drag the other bands up with them.
    """
    stacked = np.column_stack([band_z[name] for name in BANDS])
    all_band_z = stacked.min(axis=1)          # high only if EVERY band is up
    broadband = all_band_z > ARTIFACT_ALLBAND_Z
    return broadband & (band_z["ripple"] < ARTIFACT_RIPPLE_KEEP_Z), stacked


def ca3_pyramidal_rate(session, fine_edges):
    """z-scored CA3 pyramidal population rate — the sharp-wave proxy."""
    meta = session["meta"]
    is_ca3_pyr = ((meta["cell_area"].astype(str) == "CA3")
                  & (meta["cell_type"].astype(str).str.contains("Pyramidal"))).values
    if is_ca3_pyr.sum() == 0:
        return None, 0
    counts = np.zeros(len(fine_edges) - 1)
    for st, take in zip(session["spike_times"], is_ca3_pyr):
        if take:
            counts += np.diff(np.searchsorted(st, fine_edges))
    rate = counts / (SWR_FINE_BIN_S * is_ca3_pyr.sum())
    rate = gaussian_filter1d(rate, sigma=SWR_SMOOTH_S / SWR_FINE_BIN_S, mode="nearest")
    return zscore(rate), int(is_ca3_pyr.sum())


def detect_sharp_wave_ripples(session, ripple_z_full, ca3_z, fine_centers):
    """Ripple-band peak preceded by a CA3 pyramidal population peak.

    The sharp wave is the CA3 output that drives the CA1 ripple, so it leads
    it. Requiring both, in that order, is what makes this a sharp-wave ripple
    rather than any excursion of the ripple band — which on this channel is
    mostly spike leakage from the high-rate interneurons.
    """
    fs = session["fs"]
    peaks, props = sps.find_peaks(ripple_z_full, height=SWR_RIPPLE_Z,
                                  distance=int(SWR_MIN_SEP_S * fs))
    rows = []
    for peak, height in zip(peaks, props["peak_heights"]):
        t_ripple = session["lfp_t"][peak]
        lo = np.searchsorted(fine_centers, t_ripple - SWR_SW_WINDOW_S)
        hi = np.searchsorted(fine_centers, t_ripple)
        if ca3_z is None or hi <= lo:
            continue
        window = ca3_z[lo:hi]
        best = int(np.argmax(window))
        if window[best] < SWR_SW_Z:
            continue
        t_sw = fine_centers[lo + best]
        rows.append({
            "t_ripple": t_ripple,
            "t_sharpwave": t_sw,
            "lag_ms": (t_ripple - t_sw) * 1e3,
            "ripple_z": float(height),
            "sharpwave_z": float(window[best]),
            "speed": float(np.interp(t_ripple, session["speed_t"], session["speed_v"])),
        })
    events = pd.DataFrame(rows)
    return events, len(peaks)


def plot_example_events(session, ripple_filt, ripple_z_full, ca3_z, fine_centers,
                        events, n_examples=N_EXAMPLE_EVENTS):
    """A few putative events end to end: raw LFP, ripple band, sharp wave."""
    if not len(events):
        print("  no events to plot")
        return
    # spread across the session rather than showing the biggest ones
    picks = np.linspace(0, len(events) - 1, min(n_examples, len(events))).astype(int)
    half = EXAMPLE_WINDOW_S / 2
    fig, axes = plt.subplots(3, len(picks), figsize=(3.0 * len(picks), 6.5),
                             sharex="col", squeeze=False)
    fig.suptitle(f"{session['label']} — putative sharp-wave ripples "
                 f"(ripple z > {SWR_RIPPLE_Z}, CA3 pyramidal z > {SWR_SW_Z} within "
                 f"{SWR_SW_WINDOW_S * 1e3:.0f} ms before)")

    for col, i in enumerate(picks):
        ev = events.iloc[i]
        t0, t1 = ev["t_ripple"] - half, ev["t_ripple"] + half
        m = (session["lfp_t"] >= t0) & (session["lfp_t"] <= t1)
        tt = (session["lfp_t"][m] - ev["t_ripple"]) * 1e3

        ax = axes[0, col]
        ax.plot(tt, session["lfp_v"][m], lw=0.6, color="k")
        ax.set_title(f"{ev['t_ripple']:.1f}s | {ev['speed']:.1f} cm/s\n"
                     f"lag {ev['lag_ms']:.0f} ms", fontsize=8)
        if col == 0:
            ax.set_ylabel("raw LFP")

        ax = axes[1, col]
        ax.plot(tt, ripple_filt[m], lw=0.6, color="crimson")
        ax.plot(tt, ripple_z_full[m] * np.std(ripple_filt[m]), lw=0.8, color="k", alpha=0.6)
        if col == 0:
            ax.set_ylabel(f"{BANDS['ripple'][0]:.0f}-{BANDS['ripple'][1]:.0f} Hz\n"
                          "(envelope in black)")

        ax = axes[2, col]
        fm = (fine_centers >= t0) & (fine_centers <= t1)
        ax.plot((fine_centers[fm] - ev["t_ripple"]) * 1e3, ca3_z[fm], lw=0.9, color="C0")
        ax.axhline(SWR_SW_Z, color="C0", ls=":", lw=0.8)
        ax.axvline(0, color="crimson", lw=0.8)
        ax.axvline(-ev["lag_ms"], color="C0", lw=0.8, ls="--")
        ax.set_xlabel("ms from ripple peak")
        if col == 0:
            ax.set_ylabel("CA3 pyramidal\nrate (z)")

    fig.tight_layout()
    plt.show()
    plt.close(fig)


# =============================================================================
# STEP 3 — the embeddings
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
    then there would be nothing to colour in cell 3.
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
    bad, stacked = artifact_mask(band_z)
    cross_band = np.corrcoef(stacked.T)
    off_diag = cross_band[np.triu_indices(len(BANDS), k=1)]
    print(f"  cross-band correlation: mean {off_diag.mean():.2f} "
          f"(min {off_diag.min():.2f}, max {off_diag.max():.2f})")
    print(f"  artifact bins dropped: {bad.sum()} of {len(bad)} "
          f"({100 * bad.mean():.2f}%) — all bands > {ARTIFACT_ALLBAND_Z} SD "
          f"with ripple < {ARTIFACT_RIPPLE_KEEP_Z} SD")
    keep = ~bad

    # --- sharp-wave ripples -------------------------------------------------
    fine_edges = np.arange(session["t_start"], session["t_end"], SWR_FINE_BIN_S)
    fine_centers = fine_edges[:-1] + SWR_FINE_BIN_S / 2
    ca3_z, n_ca3 = ca3_pyramidal_rate(session, fine_edges)
    events, n_candidates = detect_sharp_wave_ripples(session, ripple_z_full,
                                                     ca3_z, fine_centers)
    print(f"  CA3 pyramidal units: {n_ca3}")
    print(f"  ripple-band peaks > {SWR_RIPPLE_Z} SD: {n_candidates} | "
          f"with a preceding sharp wave: {len(events)} "
          f"({len(events) / (session['maze_s'] / 60):.2f}/min)")
    if len(events):
        print(f"  lag (sharp wave -> ripple): median {events['lag_ms'].median():.0f} ms")
        print(f"  speed at event: median {events['speed'].median():.2f} cm/s, "
              f"{100 * (events['speed'] < 2.0).mean():.0f}% below 2 cm/s")

    if RUN_EXAMPLES:
        plot_example_events(session, ripple_filt, ripple_z_full, ca3_z,
                            fine_centers, events)
    del ripple_filt, ripple_z_full
    gc.collect()

    # bins holding an event, on the kept bins
    swr_bin = np.zeros(len(centers), dtype=bool)
    if len(events):
        hit = np.searchsorted(edges, events["t_ripple"].values, side="right") - 1
        hit = hit[(hit >= 0) & (hit < len(centers))]
        swr_bin[hit] = True

    # --- covariates on kept bins -------------------------------------------
    speed = np.interp(centers, session["speed_t"], session["speed_v"])
    track_x = np.interp(centers, session["pos_t"], session["pos_x"])

    # --- embeddings ---------------------------------------------------------
    counts = counts_from_spike_times(session["spike_times"], edges)
    X = prepare_matrix(counts, BIN_SIZE_S)[keep]
    meta = session["meta"]
    region = meta["cell_area"].astype(str).values
    alive = meta["rate_maze_hz"].values > UMAP_MIN_RATE_HZ

    swr_kept = swr_bin[keep]
    sub = subsample_index(X.shape[0], swr_kept)

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
        "swr": swr_kept[sub],
        "events": events,
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
    "swr_events": len(d["events"]),
    "swr_bins": int(d["swr"].sum()),
    "median_speed_at_swr": (round(d["events"]["speed"].median(), 2)
                            if len(d["events"]) else np.nan),
} for d in days]).to_string(index=False))


# %% ===========================================================================
# CELL 2 — band grids: one figure per grouping, rows = days, columns = bands
# ==============================================================================
# The same manifold in every panel of a row; only the colouring changes. Colour
# limits are shared down each column, so a band that shifts between days shows
# up as a change in colour, not just a change in scale.


def scatter_panel(ax, emb, colour, cmap, vmin, vmax, projection):
    if projection == "3d":
        return ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=colour, cmap=cmap,
                          vmin=vmin, vmax=vmax, s=1.2, alpha=0.55,
                          linewidths=0, rasterized=True)
    return ax.scatter(emb[:, 0], emb[:, 1], c=colour, cmap=cmap,
                      vmin=vmin, vmax=vmax, s=1.5, alpha=0.6,
                      linewidths=0, rasterized=True)


def band_grid(grouping, days, projection=GRID_PROJECTION):
    rows = [d for d in days if grouping in d["embeddings"]]
    if not rows:
        print(f"{grouping}: nothing to plot")
        return
    n_rows, bands = len(rows), list(BANDS)
    subplot_kw = {"projection": "3d"} if projection == "3d" else {}
    fig, axes = plt.subplots(n_rows, len(bands),
                             figsize=(2.6 * len(bands), 2.7 * n_rows),
                             squeeze=False, subplot_kw=subplot_kw)
    fig.suptitle(f"{SUBJECT} — {grouping} — manifold coloured by band power "
                 f"(z, {BAND_Z_RANGE[0]:.0f} to {BAND_Z_RANGE[1]:.0f} SD)")

    for r, day in enumerate(rows):
        emb = day["embeddings"][grouping]
        for c, band in enumerate(bands):
            ax = axes[r, c]
            sc = scatter_panel(ax, emb, day["band_z"][band], BAND_CMAPS[band],
                               BAND_Z_RANGE[0], BAND_Z_RANGE[1], projection)
            if r == 0:
                ax.set_title(f"{band}\n{BANDS[band][0]:.0f}-{BANDS[band][1]:.0f} Hz",
                             fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{day['date']}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if projection == "3d":
                ax.set_zticks([])
            if r == n_rows - 1:
                fig.colorbar(sc, ax=ax, orientation="horizontal",
                             fraction=0.05, pad=0.08).ax.tick_params(labelsize=6)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


for grouping in GROUPINGS:
    band_grid(grouping, days)


# %% ===========================================================================
# CELL 3 — our sharp-wave ripples on the manifold, coloured by speed
# ==============================================================================
# Every bin in grey, the detected sharp-wave ripple bins on top in colour. The
# colour is running speed at the event, not ripple power: the question is
# whether the events that land in one part of the manifold are the ones the
# animal was still for.


def swr_speed_figure(grouping, days):
    rows = [d for d in days if grouping in d["embeddings"]]
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(5.0 * len(rows), 4.6),
                             squeeze=False, subplot_kw={"projection": "3d"})
    fig.suptitle(f"{SUBJECT} — {grouping} — sharp-wave ripple bins, "
                 f"coloured by speed at the event")

    speeds = np.concatenate([d["speed"][d["swr"]] for d in rows if d["swr"].any()]) \
        if any(d["swr"].any() for d in rows) else np.array([0.0, 1.0])
    vmin, vmax = float(np.min(speeds)), float(max(np.max(speeds), 1e-3))

    for c, day in enumerate(rows):
        ax = axes[0, c]
        emb, hit = day["embeddings"][grouping], day["swr"]
        ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c="0.82", s=1.0,
                   alpha=0.35, linewidths=0, rasterized=True)
        if hit.any():
            sc = ax.scatter(emb[hit, 0], emb[hit, 1], emb[hit, 2],
                            c=day["speed"][hit], cmap="viridis", vmin=vmin, vmax=vmax,
                            s=34, edgecolors="k", linewidths=0.4)
            fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08).set_label("speed (cm/s)",
                                                                    fontsize=8)
        ax.set_title(f"{day['date']} — {int(hit.sum())} SWR bins", fontsize=9)
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.set_zlabel("UMAP 3", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


for grouping in GROUPINGS:
    swr_speed_figure(grouping, days)

# Event-level summary across days: does speed at our events look like the
# immobility the SWR literature expects?
all_events = pd.concat([d["events"].assign(date=d["date"]) for d in days
                        if len(d["events"])], ignore_index=True)
if len(all_events):
    print("\nsharp-wave ripple events, all days")
    print(all_events.groupby("date")[["ripple_z", "sharpwave_z", "lag_ms", "speed"]]
          .describe()[[("ripple_z", "count"), ("ripple_z", "50%"),
                       ("sharpwave_z", "50%"), ("lag_ms", "50%"),
                       ("speed", "50%")]].round(2).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    fig.suptitle(f"{SUBJECT} — detected sharp-wave ripples")
    axes[0].hist(all_events["lag_ms"], bins=30, color="C0")
    axes[0].set_xlabel("sharp wave -> ripple lag (ms)")
    axes[0].set_ylabel("events")
    axes[1].hist(all_events["speed"], bins=30, color="C2")
    axes[1].axvline(2.0, color="C3", ls=":", lw=1)
    axes[1].set_xlabel("speed at ripple peak (cm/s)")
    axes[2].scatter(all_events["sharpwave_z"], all_events["ripple_z"],
                    s=8, alpha=0.5, c="k")
    axes[2].set_xlabel("sharp wave (CA3 pyramidal z)")
    axes[2].set_ylabel("ripple-band z")
    fig.tight_layout()
    plt.show()
    plt.close(fig)
