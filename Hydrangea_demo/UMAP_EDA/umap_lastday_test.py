"""UMAP / SWR exploratory analysis — gather, characterize, embed, view.

Three notebook cells, split at the "CELL 2" and "CELL 3" banners:
    cell 1  gather and characterize
    cell 2  the shared session loader and the UMAP sweep (the slow one)
    cell 3  the scrollable viewer
Each cell depends on the ones before it. No files are written — everything is
print() and plt.show().

CELL 1 does
    1. inventory the data mount (same paths as test_hydan_obj.py)
    2. take the LAST behaviour session of every subject, count units, pick the
       subject whose last day has the most units
    3. load that session's maze epoch: spikes, unit metadata, position,
       speed (shipped + derived), CA1 LFP, sleep states
    4. profile EVERY session of that subject: LFP spectrum, band power,
       spectrogram, rough ripple-event rate, and firing rate by region x
       cell type — so day-to-day stability is visible before any modelling
    5. population-level spiking views for the chosen session
    6. a filter report: what each candidate inclusion rule would cost

CELL 2 does
    7. the UMAP sweep: per session, one global embedding plus one per region,
       3 components each — 12 figures for 3 sessions x 4 groupings.

CELL 3 does
    8. a scrollable viewer: position (x and y as two lines), speed, the
       author's state/ripple labels as a bar, and a region x cell-type raster,
       with a session dropdown. One session is held in memory at a time.

The hydrangea package is used only for two small helpers (unit metadata
extraction, speed from position); nothing here depends on the analysis object.
"""

import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

from scipy import signal as sps

# numpy 2 renamed trapz; keep working on both
_trapz = getattr(np, "trapezoid", None) or np.trapz

# =============================================================================
# CONFIG
# =============================================================================

# Same discovery + mount as test_hydan_obj.py: works from the repo root, from
# Hydrangea_demo/, locally or on the cloud box.
DOWNLOAD_DIR = Path(os.environ.get("HYDRANGEA_DATA_DIR", "/storage/dandi_downloads")).resolve()

LFP_SNIPPET_S = 3.0        # length of the raw-LFP snippet in the diagnostic plot
PSD_MAX_HZ = 250.0         # x-limit for the LFP power spectrum panel
SPEC_MAX_HZ = 30.0         # spectrogram is plotted over delta..beta, where the structure is
WINDOW_S = 30.0            # width of the population raster / LFP window view

# LFP bands. Ripple range follows the CA1 convention used by the dataset's own
# "Best_Ripple_channel" label; slow/mid gamma are split because they dissociate
# with CA3 vs entorhinal input.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (6.0, 10.0),
    "beta": (12.0, 30.0),
    "slow_gamma": (30.0, 50.0),
    "mid_gamma": (50.0, 90.0),
    "ripple": (120.0, 200.0),
}

# Rough ripple detector (see detect_ripple_events) — a first pass for counting,
# NOT a publication detector: no speed gating, no multi-unit confirmation.
RIPPLE_Z = 3.0
RIPPLE_MIN_MS = 20.0
RIPPLE_MAX_MS = 200.0

# Candidate filters, reported (not applied) in the filter section.
MAX_POS_JUMP_CM = 10.0     # frame-to-frame displacement above this is a tracking glitch
IMMOBILE_CM_S = 2.0        # SWRs live below this; theta/running above RUN_CM_S
RUN_CM_S = 5.0
RATE_THRESHOLDS_HZ = (0.0, 0.01, 0.1, 0.25, 0.5, 1.0)

# Stage 2 settings, fixed now so the gather step can check the environment can
# actually run them. The embedding goes down to 3 components.
UMAP_N_COMPONENTS = 3
UMAP_BIN_SIZE_S = 0.050    # observation bin for the spike matrix UMAP will see
UMAP_RANDOM_STATE = 0

# SCRIPT-CONSTRUCTION ONLY — every block these gate is print/plot, see step 5.
SHOW_DIAGNOSTICS = True    # chosen-session inventory, metadata, behaviour, LFP
SHOW_CROSS_SESSION = True  # all sessions of the chosen subject compared
SHOW_POPULATION = True     # whole-population spiking views
SHOW_FILTER_REPORT = True  # what each candidate filter would cost


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
from hybrid_dynamics_core.spiking_and_behaviour_eda import (  # noqa: E402
    compute_speed_from_position,
    infer_sampling_rate_hz,
)


# =============================================================================
# STEP 1 — inventory the mount
# =============================================================================
# Identifiers come out of the BIDS-style filename:
#     sub-M01_ses-20240313T100000_behavior+ecephys.nwb
# The session stamp sorts lexicographically in time order, so "last day" is
# just the max session string per subject.


def parse_session_name(nwb_path, root):
    """Subject, session, date, and modalities from an NWB filename."""
    nwb_path = Path(nwb_path)
    stem = nwb_path.stem

    subject = re.search(r"(?:^|_)sub-([^_]+)", stem)
    session = re.search(r"(?:^|_)ses-([^_]+)", stem)
    subject = subject.group(1) if subject else nwb_path.parent.name.replace("sub-", "")
    session = session.group(1) if session else "unknown"

    date = None
    if "T" in session:
        raw_date = session.split("T", 1)[0]
        if len(raw_date) == 8 and raw_date.isdigit():
            date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"

    modalities = stem.split("_")[-1].split("+")
    return {
        "subject": subject,
        "session": session,
        "date": date,
        "modalities": "+".join(modalities),
        "has_behavior": "behavior" in modalities,
        "size_mb": round(nwb_path.stat().st_size / 1e6, 1),
        "file": str(nwb_path.relative_to(root)),
    }


def session_inventory(root):
    """Table of every NWB recording under the data mount."""
    rows = [parse_session_name(p, root) for p in sorted(Path(root).rglob("*.nwb"))]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["subject", "session"]).reset_index(drop=True)


def last_day_per_subject(inventory):
    """The latest behaviour session for each subject, one row per subject."""
    behavior = inventory[inventory["has_behavior"]]
    idx = behavior.groupby("subject")["session"].idxmax()
    return behavior.loc[idx].sort_values("subject").reset_index(drop=True)


inventory = session_inventory(DOWNLOAD_DIR)
last_days = last_day_per_subject(inventory)


# =============================================================================
# STEP 2 — loaders and signal helpers
# =============================================================================
# One definition of what the maze epoch is and where each signal lives, shared
# by the survey, the full load, and the per-session profiler.


def find_key(nwb, *patterns):
    """First NWB key matching any of the (case-insensitive) substrings."""
    keys = list(nwb.keys())
    for pattern in patterns:
        for key in keys:
            if pattern.lower() in key.lower():
                return key
    return None


def maze_epoch_from(position):
    """Analysis epoch = the position-tracked span (same rule as the demo)."""
    return nap.IntervalSet(start=float(position.index[0]), end=float(position.index[-1]))


def as_time_values(obj):
    """Timestamps + values as plain numpy, forcing any lazy NWB load."""
    t = np.asarray(obj.index.values, dtype=float)
    v = np.asarray(obj.values, dtype=float)
    return t, (v.ravel() if v.ndim == 2 and v.shape[1] == 1 else v)


def read_sleep_states_pynwb(path, name_hint="sleep"):
    """Sleep-state intervals WITH their labels, read straight from the NWB.

    pynapple drops the label column on this dataset: the state intervals are not
    end-sorted, so its IntervalSet constructor sorts them, warns, and discards
    the metadata — which is exactly the column that says wake vs NREM vs REM.
    Going through pynwb keeps the table intact.
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
            if table is None:
                return None
            return table.to_dataframe()
    except Exception:
        return None


def clean_position(pos_t, pos_xy, max_jump_cm=MAX_POS_JUMP_CM):
    """Drop teleport frames, then interpolate across the holes.

    The raw tracking has occasional single-frame jumps of the order of the whole
    track, which is why speed derived from raw position peaks in the thousands
    of cm/s. Anything moving more than `max_jump_cm` between 25 Hz frames is
    treated as a tracking failure rather than a real movement.
    """
    xy = pos_xy.copy()
    step = np.r_[0.0, np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))]
    bad = step > max_jump_cm
    xy[bad] = np.nan
    for col in range(xy.shape[1]):
        v = xy[:, col]
        ok = np.isfinite(v)
        if ok.sum() >= 2:
            xy[:, col] = np.interp(pos_t, pos_t[ok], v[ok])
    return xy, int(bad.sum()), float(step.max())


def band_power(freq, psd, lo, hi):
    """Integrated PSD over [lo, hi]."""
    m = (freq >= lo) & (freq <= hi)
    return float(_trapz(psd[m], freq[m])) if m.sum() > 1 else np.nan


def lfp_psd(values, fs, seg_s=4.0):
    """Welch PSD over `seg_s`-second segments."""
    return sps.welch(values, fs=fs, nperseg=int(seg_s * fs))


def bandpass(values, fs, lo, hi, order=4):
    b_, a_ = sps.butter(order, [lo, hi], btype="band", fs=fs)
    return sps.filtfilt(b_, a_, values)


def detect_ripple_events(t, values, fs, z_thresh=RIPPLE_Z,
                         min_ms=RIPPLE_MIN_MS, max_ms=RIPPLE_MAX_MS, smooth_ms=8.0):
    """Rough ripple-band excursions: bandpass -> moving-RMS envelope -> z > thr.

    Deliberately crude — no speed gating, no MUA confirmation, no per-event
    frequency check. It exists to answer "is the ripple band alive on this
    channel, and roughly how often", not to produce an event set to analyze.
    Returns event times plus the z-scored envelope for plotting.
    """
    lo, hi = BANDS["ripple"]
    filt = bandpass(values, fs, lo, hi)
    win = max(1, int(smooth_ms * 1e-3 * fs))
    env = np.sqrt(np.convolve(filt ** 2, np.ones(win) / win, mode="same"))
    z = (env - env.mean()) / (env.std() + 1e-12)

    above = z > z_thresh
    edges = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if above.size and above[0]:
        starts = np.r_[0, starts]
    if above.size and above[-1]:
        ends = np.r_[ends, above.size]
    n = min(len(starts), len(ends))
    starts, ends = starts[:n], ends[:n]

    dur_ms = (ends - starts) / fs * 1e3
    keep = (dur_ms >= min_ms) & (dur_ms <= max_ms)
    starts, ends = starts[keep], ends[keep]
    peaks = np.array([s + int(np.argmax(z[s:e])) for s, e in zip(starts, ends)], dtype=int)
    return {
        "t_peak": t[peaks] if len(peaks) else np.array([]),
        "t_start": t[starts] if len(starts) else np.array([]),
        "t_end": t[np.maximum(ends - 1, 0)] if len(ends) else np.array([]),
        "z": z,
        "n": int(len(starts)),
    }


def survey_row(row):
    """Unit counts and epoch length for one session, without keeping the file."""
    nwb = nap.load_file(str((DOWNLOAD_DIR / row["file"]).resolve()))
    spikes = nwb["units"]
    position = nwb["Position"]
    maze_ep = maze_epoch_from(position)
    spikes_maze = spikes.restrict(maze_ep)
    n_active = int(sum(len(spikes_maze[u]) > 0 for u in spikes_maze.index))

    meta = extract_unit_metadata(spikes)
    region_col = "cell_area" if "cell_area" in meta.columns else None

    return {
        "subject": row["subject"],
        "session": row["session"],
        "date": row["date"],
        "n_units": len(spikes),
        "n_units_active_in_maze": n_active,
        "maze_min": round(float(maze_ep.tot_length() / 60), 1),
        "n_regions": int(meta[region_col].nunique()) if region_col else 0,
        "lfp_key": find_key(nwb, "lfp"),
        "nwb_keys": ", ".join(nwb.keys()),
        "file": row["file"],
    }


def load_session_bundle(row, verbose=True):
    """Everything the UMAP needs for one session, restricted to the maze epoch.

    Returns a dict rather than an object: this is a scratch stage, and a dict
    keeps the notebook namespace explicit about what was actually loaded.
    """
    path = (DOWNLOAD_DIR / row["file"]).resolve()
    t0 = time.perf_counter()
    nwb = nap.load_file(str(path))

    # --- spikes + unit metadata --------------------------------------------
    spikes_all = nwb["units"]
    position_all = nwb["Position"]
    maze_ep = maze_epoch_from(position_all)
    maze_s = float(maze_ep.tot_length())
    spikes = spikes_all.restrict(maze_ep)

    unit_meta = extract_unit_metadata(spikes_all)
    n_spikes_maze = np.array([len(spikes[u]) for u in spikes.index], dtype=int)
    unit_meta = unit_meta.reindex(np.asarray(spikes.index))
    unit_meta["n_spikes_maze"] = n_spikes_maze
    unit_meta["rate_maze_hz"] = n_spikes_maze / maze_s

    # --- behaviour ----------------------------------------------------------
    position = position_all.restrict(maze_ep)
    pos_t = np.asarray(position.index.values, dtype=float)
    pos_xy = np.column_stack([
        np.asarray(position["x"].values, dtype=float),
        np.asarray(position["y"].values, dtype=float),
    ])
    pos_xy_clean, n_jumps, max_jump = clean_position(pos_t, pos_xy)

    speed_key = find_key(nwb, "speed")
    speed_nwb_t, speed_nwb_v = (None, None)
    if speed_key is not None:
        speed_nwb_t, speed_nwb_v = as_time_values(nwb[speed_key].restrict(maze_ep))
    # Two derived versions: raw (to show the tracking glitches) and jump-filtered
    # (which is what actually tracks the shipped channel).
    speed_der_t, speed_der_v = compute_speed_from_position(position, columns=("x", "y"),
                                                           smooth_window_s=0.25)
    clean_frame = pd.DataFrame({"x": pos_xy_clean[:, 0], "y": pos_xy_clean[:, 1]}, index=pos_t)
    _, speed_clean_v = compute_speed_from_position(clean_frame, columns=("x", "y"),
                                                   smooth_window_s=0.25)

    # --- LFP ----------------------------------------------------------------
    lfp_key = find_key(nwb, "lfp")
    lfp_t, lfp_v, lfp_hz = None, None, None
    if lfp_key is not None:
        lfp_t, lfp_v = as_time_values(nwb[lfp_key].restrict(maze_ep))
        lfp_hz = infer_sampling_rate_hz(lfp_t)

    # --- sleep states -------------------------------------------------------
    # pynapple's version (labels dropped, intervals merged) and the pynwb one
    # (labels intact) are both kept so the loss is visible rather than silent.
    sleep_key = find_key(nwb, "sleepstate", "sleep")
    sleep_states = nwb[sleep_key] if sleep_key is not None else None
    sleep_table = read_sleep_states_pynwb(path)

    bundle = {
        "subject": row["subject"],
        "session": row["session"],
        "date": row["date"],
        "path": path,
        "nwb": nwb,
        "nwb_keys": list(nwb.keys()),
        "maze_ep": maze_ep,
        "maze_s": maze_s,
        "spikes": spikes,
        "spikes_all": spikes_all,
        "unit_meta": unit_meta,
        "pos_t": pos_t,
        "pos_xy": pos_xy,
        "pos_xy_clean": pos_xy_clean,
        "pos_n_jumps": n_jumps,
        "pos_max_jump_cm": max_jump,
        "pos_hz": infer_sampling_rate_hz(pos_t),
        "speed_key": speed_key,
        "speed_nwb_t": speed_nwb_t,
        "speed_nwb_v": speed_nwb_v,
        "speed_der_t": speed_der_t,
        "speed_der_v": speed_der_v,
        "speed_clean_v": speed_clean_v,
        "lfp_key": lfp_key,
        "lfp_t": lfp_t,
        "lfp_v": lfp_v,
        "lfp_hz": lfp_hz,
        "sleep_key": sleep_key,
        "sleep_states": sleep_states,
        "sleep_table": sleep_table,
        "load_s": time.perf_counter() - t0,
    }
    if verbose:
        print(f"loaded {row['file']} in {bundle['load_s']:.1f}s")
    return bundle


def session_profile(row, verbose=True):
    """Per-session LFP and firing-rate summary, keeping no bulk arrays.

    The raw LFP for one session is ~3M samples; the profiler reduces it to a
    PSD, a low-frequency spectrogram, a ripple-power time course and an event
    count, then drops it. That keeps a loop over sessions bounded in memory.
    """
    b = load_session_bundle(row, verbose=False)
    label = f"{b['subject']} {b['date']}"
    t0 = time.perf_counter()

    profile = {
        "subject": b["subject"],
        "session": b["session"],
        "date": b["date"],
        "label": label,
        "maze_min": b["maze_s"] / 60,
        "n_units": len(b["unit_meta"]),
    }

    # --- firing rate by region x cell type ---------------------------------
    meta = b["unit_meta"]
    if {"cell_area", "cell_type"} <= set(meta.columns):
        rates = (meta.groupby(["cell_area", "cell_type"])["rate_maze_hz"]
                 .agg(n_units="size", mean_hz="mean", median_hz="median")
                 .reset_index())
    else:
        rates = pd.DataFrame()
    if len(rates):
        rates.insert(0, "date", b["date"])
        rates.insert(0, "subject", b["subject"])
    profile["rate_table"] = rates

    # behaviour summary is cheap and worth having per session
    spd = b["speed_nwb_v"] if b["speed_nwb_v"] is not None else b["speed_clean_v"]
    profile["frac_immobile"] = float(np.mean(spd < IMMOBILE_CM_S))
    profile["frac_running"] = float(np.mean(spd > RUN_CM_S))
    profile["median_speed"] = float(np.nanmedian(spd))

    # --- LFP ----------------------------------------------------------------
    if b["lfp_v"] is not None:
        fs = b["lfp_hz"]
        values = b["lfp_v"]
        freq, psd = lfp_psd(values, fs)
        total = band_power(freq, psd, 0.5, PSD_MAX_HZ)
        profile["psd_f"] = freq
        profile["psd"] = psd
        profile["lfp_sd"] = float(np.std(values))
        profile["lfp_hz"] = fs
        for name, (lo, hi) in BANDS.items():
            p = band_power(freq, psd, lo, hi)
            profile[f"abs_{name}"] = p
            profile[f"rel_{name}"] = p / total if total else np.nan
        profile["theta_delta"] = profile["abs_theta"] / profile["abs_delta"]

        nper = int(2.0 * fs)
        f_s, t_s, sxx = sps.spectrogram(values, fs=fs, nperseg=nper, noverlap=nper // 2)
        t_s = t_s + b["lfp_t"][0]
        keep = (f_s >= 0.5) & (f_s <= SPEC_MAX_HZ)
        profile["spec_f"] = f_s[keep]
        profile["spec_t"] = t_s
        profile["spec_db"] = 10 * np.log10(sxx[keep] + 1e-12)
        rip = (f_s >= BANDS["ripple"][0]) & (f_s <= BANDS["ripple"][1])
        rip_p = sxx[rip].mean(axis=0)
        profile["rip_t"] = t_s
        profile["rip_z"] = (rip_p - rip_p.mean()) / (rip_p.std() + 1e-12)

        events = detect_ripple_events(b["lfp_t"], values, fs)
        profile["swr_n"] = events["n"]
        profile["swr_per_min"] = events["n"] / (b["maze_s"] / 60)
        profile["swr_t"] = events["t_peak"]

    if verbose:
        print(f"  profiled {label}: {profile['n_units']} units, "
              f"{profile['maze_min']:.1f} min, {time.perf_counter() - t0:.1f}s")
    return profile


# =============================================================================
# STEP 3 — survey the last days, pick the mouse, load it
# =============================================================================

survey = pd.DataFrame([survey_row(r) for _, r in last_days.iterrows()])
survey = survey.sort_values("n_units_active_in_maze", ascending=False).reset_index(drop=True)

chosen_row = survey.iloc[0]
chosen_subject = chosen_row["subject"]
bundle = load_session_bundle(
    last_days[last_days["subject"] == chosen_subject].iloc[0]
)


# =============================================================================
# STEP 4 — profile every session of the chosen subject
# =============================================================================
# Day-to-day comparison, so it is clear whether the last day is representative
# before anything is fit to it. Position is not re-plotted here; what varies
# across days and matters for the embedding is the LFP state and the firing
# rates, not the track geometry.

subject_sessions = inventory[(inventory["subject"] == chosen_subject)
                             & inventory["has_behavior"]].reset_index(drop=True)
print(f"\nprofiling {len(subject_sessions)} sessions for {chosen_subject}...")
profiles = [session_profile(r) for _, r in subject_sessions.iterrows()]


# =============================================================================
# STEP 5 — SCRIPT-CONSTRUCTION ONLY: gather + display
# -----------------------------------------------------------------------------
# NOTE: everything below is here only so we can SEE the data while building the
# script — it prints and plots what was loaded above and computes nothing that
# anything downstream needs. Comment it out (or set the SHOW_* flags to False)
# once the UMAP stage is written.
# =============================================================================

if SHOW_DIAGNOSTICS:
    b = bundle
    rule = "=" * 78

    # --- 1. where we are ----------------------------------------------------
    print(rule)
    print("1. mount and inventory")
    print(rule)
    print("package root:", PACKAGE_ROOT)
    print("data mount:  ", DOWNLOAD_DIR)
    print(f"{len(inventory)} recordings | {inventory['subject'].nunique()} subjects | "
          f"{inventory['size_mb'].sum() / 1000:.1f} GB")
    print("\nlast behaviour session per subject:")
    print(last_days[["subject", "session", "date", "size_mb", "file"]].to_string(index=False))

    # --- 2. the survey that picked the mouse --------------------------------
    print("\n" + rule)
    print("2. last-day survey, ranked by units active in the maze")
    print(rule)
    print(survey[["subject", "session", "date", "n_units", "n_units_active_in_maze",
                  "maze_min", "n_regions", "lfp_key"]].to_string(index=False))
    print(f"\n-> chosen: {b['subject']} / {b['session']} ({b['date']})")
    print("   NWB keys:", b["nwb_keys"])

    # --- 3. what each NWB object actually is --------------------------------
    print("\n" + rule)
    print("3. NWB object types")
    print(rule)
    for key in b["nwb_keys"]:
        try:
            obj = b["nwb"][key]
            shape = getattr(obj, "shape", None)
            n = len(obj)
            print(f"  {key:32s} {type(obj).__name__:12s} len={n:<10d} shape={shape}")
        except Exception as exc:  # some NWB fields do not load through pynapple
            print(f"  {key:32s} <could not load: {type(exc).__name__}: {exc}>")

    # --- 4. epoch + spiking -------------------------------------------------
    print("\n" + rule)
    print("4. maze epoch and spiking")
    print(rule)
    rec_start = float(min(b["spikes_all"][u].index[0]
                          for u in b["spikes_all"].index if len(b["spikes_all"][u])))
    rec_end = float(max(b["spikes_all"][u].index[-1]
                        for u in b["spikes_all"].index if len(b["spikes_all"][u])))
    print(f"recording span : {rec_start:8.1f} - {rec_end:8.1f}s  ({(rec_end - rec_start) / 60:.1f} min)")
    print(f"maze epoch     : {float(b['maze_ep'].start[0]):8.1f} - "
          f"{float(b['maze_ep'].end[0]):8.1f}s  ({b['maze_s'] / 60:.1f} min)")
    print(f"units          : {len(b['spikes_all'])} total, "
          f"{int((b['unit_meta']['n_spikes_maze'] > 0).sum())} with spikes in the maze")
    print(f"spikes in maze : {int(b['unit_meta']['n_spikes_maze'].sum()):,}")

    # --- 5. unit metadata ---------------------------------------------------
    print("\n" + rule)
    print("5. unit metadata")
    print(rule)
    meta = b["unit_meta"]
    print(f"shape {meta.shape} | columns: {', '.join(meta.columns)}")
    print("\nmaze firing rate (Hz) by region:")
    if "cell_area" in meta.columns:
        print(meta.groupby("cell_area")["rate_maze_hz"]
              .describe()[["count", "min", "50%", "mean", "max"]].round(3).to_string())
    if {"cell_area", "cell_type"} <= set(meta.columns):
        print("\nunits by region x cell_type:")
        print(pd.crosstab(meta["cell_area"], meta["cell_type"], margins=True, margins_name="Total"))

    # --- 6. behaviour -------------------------------------------------------
    print("\n" + rule)
    print("6. behaviour")
    print(rule)
    print(f"position     : {len(b['pos_t'])} samples @ {b['pos_hz']:.1f} Hz | "
          f"x {b['pos_xy'][:, 0].min():.1f}-{b['pos_xy'][:, 0].max():.1f}, "
          f"y {b['pos_xy'][:, 1].min():.1f}-{b['pos_xy'][:, 1].max():.1f} cm")
    print(f"tracking glitches: {b['pos_n_jumps']} frames move > {MAX_POS_JUMP_CM} cm "
          f"(max single-frame jump {b['pos_max_jump_cm']:.1f} cm)")
    if b["speed_nwb_v"] is not None:
        print(f"speed ('{b['speed_key']}', shipped): median {np.nanmedian(b['speed_nwb_v']):.2f}, "
              f"max {np.nanmax(b['speed_nwb_v']):.1f} cm/s")
    print(f"speed (derived, raw position)   : median {np.median(b['speed_der_v']):.2f}, "
          f"max {b['speed_der_v'].max():.1f} cm/s")
    print(f"speed (derived, jump-filtered)  : median {np.median(b['speed_clean_v']):.2f}, "
          f"max {b['speed_clean_v'].max():.1f} cm/s")
    if b["speed_nwb_v"] is not None and len(b["speed_nwb_v"]) == len(b["speed_der_v"]):
        ok = np.isfinite(b["speed_nwb_v"]) & np.isfinite(b["speed_der_v"])
        print(f"corr(shipped, raw derived)      = "
              f"{np.corrcoef(b['speed_nwb_v'][ok], b['speed_der_v'][ok])[0, 1]:.3f}")
        ok2 = np.isfinite(b["speed_nwb_v"]) & np.isfinite(b["speed_clean_v"])
        print(f"corr(shipped, jump-filtered)    = "
              f"{np.corrcoef(b['speed_nwb_v'][ok2], b['speed_clean_v'][ok2])[0, 1]:.3f}")

    # --- 7. LFP -------------------------------------------------------------
    print("\n" + rule)
    print("7. LFP")
    print(rule)
    if b["lfp_v"] is None:
        print("no LFP-like key found in this NWB")
    else:
        gaps = np.diff(b["lfp_t"])
        print(f"key           : {b['lfp_key']}")
        print(f"samples       : {len(b['lfp_v']):,} @ {b['lfp_hz']:.1f} Hz "
              f"({len(b['lfp_v']) / b['lfp_hz'] / 60:.1f} min)")
        print(f"maze coverage : {len(b['lfp_v']) / b['lfp_hz'] / b['maze_s'] * 100:.1f}% of the epoch")
        print(f"sample gaps   : median {np.median(gaps) * 1e3:.3f} ms, "
              f"max {gaps.max() * 1e3:.3f} ms, "
              f"{int((gaps > 2 * np.median(gaps)).sum())} gaps > 2x median")
        print(f"amplitude     : mean {np.nanmean(b['lfp_v']):.1f}, sd {np.nanstd(b['lfp_v']):.1f}, "
              f"range {np.nanmin(b['lfp_v']):.1f} to {np.nanmax(b['lfp_v']):.1f} "
              f"| NaNs {int(np.isnan(b['lfp_v']).sum())}")

    # --- 8. sleep states ----------------------------------------------------
    # The pynapple copy loses the labels (see read_sleep_states_pynwb); the
    # pynwb table is the one to use if states are ever a modelling covariate.
    print("\n" + rule)
    print("8. sleep states")
    print(rule)
    ss = b["sleep_states"]
    if ss is not None:
        print(f"pynapple '{b['sleep_key']}': {len(ss)} intervals, "
              f"{float(ss.tot_length()) / 60:.1f} min (labels dropped on load)")
    tbl = b["sleep_table"]
    if tbl is None or not len(tbl):
        print("pynwb read: no labelled state table recovered")
    else:
        print(f"pynwb read: {len(tbl)} rows | columns: {', '.join(map(str, tbl.columns))}")
        label_col = next((c for c in tbl.columns
                          if tbl[c].dtype == object or str(c).lower() in
                          ("label", "state", "tags", "sleep_state")), None)
        if label_col is not None:
            tbl = tbl.copy()
            tbl["_label"] = tbl[label_col].astype(str)
            tbl["_dur"] = tbl["stop_time"] - tbl["start_time"]
            m0, m1 = float(b["maze_ep"].start[0]), float(b["maze_ep"].end[0])
            overlap = np.clip(np.minimum(tbl["stop_time"], m1) - np.maximum(tbl["start_time"], m0),
                              0, None)
            summary = pd.DataFrame({
                "n_intervals": tbl.groupby("_label")["_dur"].size(),
                "total_min": tbl.groupby("_label")["_dur"].sum() / 60,
                "in_maze_min": overlap.groupby(tbl["_label"]).sum() / 60,
            })
            print(summary.round(2).to_string())

    # --- 9. plots -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"{b['subject']} {b['date']} — maze epoch ({b['maze_s'] / 60:.1f} min)")

    ax = axes[0, 0]
    ax.scatter(b["pos_xy_clean"][:, 0], b["pos_xy_clean"][:, 1], c=b["pos_t"], s=2, cmap="viridis")
    ax.set_title(f"trajectory, jump-filtered (colour = time)")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[0, 1]
    if b["speed_nwb_v"] is not None:
        ax.plot(b["speed_nwb_t"], b["speed_nwb_v"], lw=0.5, label=b["speed_key"] + " (shipped)")
    ax.plot(b["speed_der_t"], b["speed_clean_v"], lw=0.5, alpha=0.7, label="derived, jump-filtered")
    ax.axhline(IMMOBILE_CM_S, color="k", ls=":", lw=0.8)
    ax.axhline(RUN_CM_S, color="k", ls="--", lw=0.8)
    ax.set_title("running speed (raw derived omitted — tracking glitches)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cm/s")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    if b["lfp_v"] is not None:
        n_snip = int(LFP_SNIPPET_S * b["lfp_hz"])
        start = len(b["lfp_v"]) // 2
        sl = slice(start, start + n_snip)
        ax.plot(b["lfp_t"][sl], b["lfp_v"][sl], lw=0.6)
        ax.set_title(f"{b['lfp_key']} — {LFP_SNIPPET_S:.0f}s from mid-epoch")
        ax.set_xlabel("time (s)")
    else:
        ax.text(0.5, 0.5, "no LFP", ha="center", transform=ax.transAxes)

    ax = axes[1, 1]
    if b["lfp_v"] is not None:
        freq, psd = lfp_psd(b["lfp_v"], b["lfp_hz"])
        keep = (freq > 0) & (freq <= PSD_MAX_HZ)
        ax.loglog(freq[keep], psd[keep])
        for band, (lo, hi) in BANDS.items():
            ax.axvspan(lo, hi, alpha=0.12, color="C1" if band != "ripple" else "crimson")
            ax.text(np.sqrt(lo * hi), psd[keep].max(), band, ha="center", fontsize=7, rotation=90)
        ax.set_title("LFP power spectrum (maze epoch)")
        ax.set_xlabel("Hz")
        ax.set_ylabel("power")
    else:
        ax.text(0.5, 0.5, "no LFP", ha="center", transform=ax.transAxes)

    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{b['subject']} {b['date']} — unit population")

    ax = axes[0]
    if {"cell_area", "cell_type"} <= set(meta.columns):
        pd.crosstab(meta["cell_area"], meta["cell_type"]).plot(kind="bar", stacked=True, ax=ax)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no region/cell_type metadata", ha="center", transform=ax.transAxes)
    ax.set_title("units per region")
    ax.set_ylabel("n units")

    ax = axes[1]
    if "cell_area" in meta.columns:
        for region, sub in meta.groupby("cell_area"):
            r = sub["rate_maze_hz"].replace(0, np.nan).dropna()
            ax.hist(np.log10(r), bins=30, alpha=0.5, label=region)
        ax.legend(fontsize=7)
    else:
        ax.hist(np.log10(meta["rate_maze_hz"].replace(0, np.nan).dropna()), bins=40)
    ax.set_title("maze firing rate by region")
    ax.set_xlabel("log10 rate (Hz)")
    ax.set_ylabel("n units")

    ax = axes[2]
    order = meta["rate_maze_hz"].sort_values().index
    t_mid = float(b["maze_ep"].start[0]) + b["maze_s"] / 2
    win = nap.IntervalSet(start=t_mid, end=t_mid + 20)
    for row_i, unit in enumerate(order):
        ts = b["spikes"][unit].restrict(win).index.values
        if len(ts):
            ax.plot(ts, np.full(len(ts), row_i), "|", ms=2, color="k", alpha=0.6)
    ax.set_title("raster, 20s mid-epoch (units sorted by rate)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("unit (rate rank)")

    fig.tight_layout()
    plt.show()


# =============================================================================
# STEP 6 — SCRIPT-CONSTRUCTION ONLY: the chosen subject across days
# =============================================================================
# Is the last day representative? Three things decide it: the LFP state (band
# composition and how it moves through the session), the ripple rate on the
# CA1 channel, and whether region/cell-type firing rates hold still day to day.

if SHOW_CROSS_SESSION and profiles:
    rule = "=" * 78
    print("\n" + rule)
    print(f"cross-session: {chosen_subject}, {len(profiles)} sessions")
    print(rule)

    # --- LFP band table -----------------------------------------------------
    band_rows = []
    for p in profiles:
        row = {"date": p["date"], "maze_min": round(p["maze_min"], 1),
               "n_units": p["n_units"], "lfp_sd": round(p.get("lfp_sd", np.nan), 1)}
        for name in BANDS:
            row[name] = p.get(f"rel_{name}", np.nan)
        row["theta/delta"] = p.get("theta_delta", np.nan)
        row["swr_n"] = p.get("swr_n", np.nan)
        row["swr/min"] = p.get("swr_per_min", np.nan)
        row["frac_immobile"] = p.get("frac_immobile", np.nan)
        band_rows.append(row)
    band_table = pd.DataFrame(band_rows)
    print("\nrelative band power (fraction of 0.5-250 Hz), ripple events, immobility:")
    print(band_table.round(4).to_string(index=False))
    print("\n(ripple counts come from a crude z>3 envelope detector — relative "
          "comparison between days only, not an event set to analyze)")

    # --- firing rates by region x cell type, across days --------------------
    rate_tables = [p["rate_table"] for p in profiles if len(p.get("rate_table", []))]
    if rate_tables:
        rates_long = pd.concat(rate_tables, ignore_index=True)
        pivot_mean = rates_long.pivot_table(index=["cell_area", "cell_type"],
                                            columns="date", values="mean_hz")
        pivot_n = rates_long.pivot_table(index=["cell_area", "cell_type"],
                                         columns="date", values="n_units")
        print("\nmean firing rate (Hz) by region x cell type, per day:")
        print(pivot_mean.round(2).to_string())
        print("\nunit counts by region x cell type, per day:")
        print(pivot_n.fillna(0).astype(int).to_string())
    else:
        rates_long, pivot_mean, pivot_n = None, None, None

    # --- plots: spectra, bands, spectrograms, rates -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"{chosen_subject} — LFP across days")

    ax = axes[0]
    for p in profiles:
        if "psd" in p:
            keep = (p["psd_f"] > 0) & (p["psd_f"] <= PSD_MAX_HZ)
            ax.loglog(p["psd_f"][keep], p["psd"][keep], lw=1, label=p["date"])
    for lo, hi in BANDS.values():
        ax.axvspan(lo, hi, alpha=0.07, color="k")
    ax.set_title("power spectrum, maze epoch")
    ax.set_xlabel("Hz")
    ax.set_ylabel("power")
    ax.legend(fontsize=8)

    ax = axes[1]
    if len(band_table):
        plot_cols = list(BANDS)
        x = np.arange(len(plot_cols))
        width = 0.8 / max(len(band_table), 1)
        for i, (_, r) in enumerate(band_table.iterrows()):
            ax.bar(x + i * width, [r[c] for c in plot_cols], width, label=r["date"])
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(plot_cols, rotation=30, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("relative power")
        ax.set_title("band composition")
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()

    # spectrogram + ripple power, one row per day, shared colour scale
    specs = [p for p in profiles if "spec_db" in p]
    if specs:
        vmin = min(np.percentile(p["spec_db"], 5) for p in specs)
        vmax = max(np.percentile(p["spec_db"], 99) for p in specs)
        fig, axes = plt.subplots(len(specs), 2, figsize=(14, 2.6 * len(specs)),
                                 squeeze=False, gridspec_kw={"width_ratios": [3, 2]})
        fig.suptitle(f"{chosen_subject} — spectrogram (0.5-{SPEC_MAX_HZ:.0f} Hz) "
                     f"and ripple-band power, maze epoch")
        for i, p in enumerate(specs):
            ax = axes[i, 0]
            t_rel = (p["spec_t"] - p["spec_t"][0]) / 60
            ax.pcolormesh(t_rel, p["spec_f"], p["spec_db"], shading="auto",
                          vmin=vmin, vmax=vmax, cmap="magma")
            ax.set_ylabel(f"{p['date']}\nHz")
            if i == len(specs) - 1:
                ax.set_xlabel("minutes into maze epoch")

            ax = axes[i, 1]
            ax.plot((p["rip_t"] - p["rip_t"][0]) / 60, p["rip_z"], lw=0.5)
            ax.axhline(RIPPLE_Z, color="crimson", ls="--", lw=0.8)
            ax.set_ylabel("ripple power (z)")
            if "swr_per_min" in p:
                ax.set_title(f"{p['swr_n']} events, {p['swr_per_min']:.1f}/min", fontsize=9)
            if i == len(specs) - 1:
                ax.set_xlabel("minutes into maze epoch")
        fig.tight_layout()
        plt.show()

    # firing rate by region x cell type across days
    if rate_tables:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        fig.suptitle(f"{chosen_subject} — firing rate and yield across days")

        ax = axes[0]
        pivot_mean.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("mean rate (Hz)")
        ax.set_title("mean maze firing rate by region x cell type")
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend(fontsize=8, title="")

        ax = axes[1]
        pivot_n.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("n units")
        ax.set_title("units recorded")
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend(fontsize=8, title="")
        fig.tight_layout()
        plt.show()


# =============================================================================
# STEP 7 — SCRIPT-CONSTRUCTION ONLY: whole-population spiking, chosen session
# =============================================================================
# The observation matrix UMAP will see, looked at directly: every unit, binned
# at the embedding bin size, alongside the covariates that could explain its
# structure (speed, ripple power).

counts = None
if SHOW_POPULATION:
    b = bundle
    meta = b["unit_meta"]
    rule = "=" * 78
    print("\n" + rule)
    print("population activity, chosen session")
    print(rule)

    counts = b["spikes"].count(UMAP_BIN_SIZE_S, b["maze_ep"])
    count_t = np.asarray(counts.index.values, dtype=float)
    count_v = np.asarray(counts.values, dtype=np.float32)
    unit_ids = np.asarray(counts.columns)
    print(f"count matrix: {count_v.shape[0]} bins x {count_v.shape[1]} units "
          f"@ {UMAP_BIN_SIZE_S * 1e3:.0f} ms ({count_v.nbytes / 1e6:.0f} MB)")
    print(f"empty bins (no spike anywhere): {np.mean(count_v.sum(axis=1) == 0) * 100:.2f}%")
    print(f"median spikes per bin across the population: {np.median(count_v.sum(axis=1)):.0f}")

    regions = (meta["cell_area"].reindex(unit_ids).astype(str).values
               if "cell_area" in meta.columns else np.array(["all"] * len(unit_ids)))
    region_names = sorted(set(regions))

    # population rate per region, 1 s smoothing
    smooth_n = max(1, int(1.0 / UMAP_BIN_SIZE_S))
    kernel = np.ones(smooth_n) / smooth_n
    pop_rates = {}
    for region in region_names:
        idx = regions == region
        trace = count_v[:, idx].sum(axis=1) / (idx.sum() * UMAP_BIN_SIZE_S)
        pop_rates[region] = np.convolve(trace, kernel, mode="same")

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(f"{b['subject']} {b['date']} — population rate over the maze epoch")
    ax = axes[0]
    for region in region_names:
        ax.plot(count_t, pop_rates[region], lw=0.7, label=region)
    ax.set_ylabel("mean rate (Hz/unit)")
    ax.legend(fontsize=8, ncol=len(region_names))
    ax = axes[1]
    if b["speed_nwb_v"] is not None:
        ax.plot(b["speed_nwb_t"], b["speed_nwb_v"], lw=0.5, color="k")
    ax.axhline(IMMOBILE_CM_S, color="C3", ls=":", lw=0.8)
    ax.set_ylabel("speed (cm/s)")
    ax.set_xlabel("time (s)")
    fig.tight_layout()
    plt.show()

    # window view: every unit, sorted by region, with speed and ripple power
    t_mid = float(b["maze_ep"].start[0]) + b["maze_s"] / 2
    t_end = t_mid + WINDOW_S
    win = nap.IntervalSet(start=t_mid, end=t_end)
    order = np.argsort([f"{r}" for r in regions], kind="stable")

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [4, 1, 1]})
    fig.suptitle(f"{b['subject']} {b['date']} — all {len(unit_ids)} units, "
                 f"{WINDOW_S:.0f}s from mid-epoch")
    ax = axes[0]
    colors = {region: f"C{i}" for i, region in enumerate(region_names)}
    for row_i, pos in enumerate(order):
        unit = unit_ids[pos]
        ts = b["spikes"][unit].restrict(win).index.values
        if len(ts):
            ax.plot(ts, np.full(len(ts), row_i), "|", ms=2.5,
                    color=colors[regions[pos]], alpha=0.8)
    for region in region_names:
        ax.plot([], [], "|", color=colors[region], label=region)
    ax.set_ylabel("unit (grouped by region)")
    ax.legend(fontsize=8, ncol=len(region_names), loc="upper right")

    ax = axes[1]
    if b["speed_nwb_v"] is not None:
        m = (b["speed_nwb_t"] >= t_mid) & (b["speed_nwb_t"] <= t_end)
        ax.plot(b["speed_nwb_t"][m], b["speed_nwb_v"][m], lw=0.8, color="k")
    ax.axhline(IMMOBILE_CM_S, color="C3", ls=":", lw=0.8)
    ax.set_ylabel("speed\n(cm/s)")

    ax = axes[2]
    if b["lfp_v"] is not None:
        # z-scored within this window, so the 3 SD line is window-local and not
        # comparable to the session-wide counts in step 6
        m = (b["lfp_t"] >= t_mid) & (b["lfp_t"] <= t_end)
        ev = detect_ripple_events(b["lfp_t"][m], b["lfp_v"][m], b["lfp_hz"])
        ax.plot(b["lfp_t"][m], ev["z"], lw=0.5, color="crimson")
        ax.axhline(RIPPLE_Z, color="k", ls="--", lw=0.8)
        for tp in ev["t_peak"]:
            axes[0].axvline(tp, color="crimson", alpha=0.25, lw=1)
        print(f"ripple-band excursions in this {WINDOW_S:.0f}s window: {ev['n']} "
              "(marked on the raster)")
    ax.set_ylabel("ripple\npower (z)")
    ax.set_xlabel("time (s)")
    fig.tight_layout()
    plt.show()


# =============================================================================
# STEP 8 — SCRIPT-CONSTRUCTION ONLY: what would each filter cost?
# =============================================================================
# Nothing is applied here. This prints the size of each candidate inclusion rule
# so the filter set for stage 2 is a decision rather than a default.

if SHOW_FILTER_REPORT:
    b = bundle
    meta = b["unit_meta"]
    rule = "=" * 78
    print("\n" + rule)
    print("filter report (nothing applied — costs only)")
    print(rule)

    print("\nunits surviving a minimum-rate threshold:")
    rows = []
    for thr in RATE_THRESHOLDS_HZ:
        keep = meta["rate_maze_hz"] > thr
        row = {"min_rate_hz": thr, "n_units": int(keep.sum())}
        if "cell_area" in meta.columns:
            for region, sub in meta[keep].groupby("cell_area"):
                row[region] = len(sub)
        rows.append(row)
    print(pd.DataFrame(rows).fillna(0).astype({"n_units": int}).to_string(index=False))

    print("\nhigh-rate units (possible MUA / merged clusters):")
    for thr in (20.0, 30.0, 50.0):
        hot = meta[meta["rate_maze_hz"] > thr]
        by_region = hot["cell_area"].value_counts().to_dict() if "cell_area" in meta.columns else {}
        print(f"  > {thr:4.0f} Hz: {len(hot):3d} units  {by_region}")

    if {"cell_area", "cell_type"} <= set(meta.columns):
        print("\nrate imbalance across regions — the reason per-region normalization "
              "or region-wise embedding may be needed:")
        print(meta.groupby(["cell_area", "cell_type"])["rate_maze_hz"]
              .agg(n="size", mean="mean", median="median").round(2).to_string())

    print("\nbehaviour split (shipped speed):")
    spd = b["speed_nwb_v"] if b["speed_nwb_v"] is not None else b["speed_clean_v"]
    frac_imm = float(np.mean(spd < IMMOBILE_CM_S))
    frac_run = float(np.mean(spd > RUN_CM_S))
    print(f"  immobile (< {IMMOBILE_CM_S} cm/s): {frac_imm * 100:.1f}% of samples "
          f"({frac_imm * b['maze_s'] / 60:.1f} min) — where SWRs live")
    print(f"  running  (> {RUN_CM_S} cm/s): {frac_run * 100:.1f}% "
          f"({frac_run * b['maze_s'] / 60:.1f} min) — where theta lives")
    print(f"  tracking glitches dropped by the position filter: {b['pos_n_jumps']} frames")

    if counts is not None:
        print("\nobservation matrix at the planned bin size:")
        print(f"  {count_v.shape[0]} bins x {count_v.shape[1]} units, "
              f"{np.mean(count_v == 0) * 100:.1f}% of entries are zero")
        print(f"  bins with < 5 spikes across the whole population: "
              f"{np.mean(count_v.sum(axis=1) < 5) * 100:.1f}%")

    print("\n" + rule)
    print("ready for stage 2 (UMAP)")
    print(rule)
    n_bins = int(np.floor(b["maze_s"] / UMAP_BIN_SIZE_S))
    print(f"planned: {UMAP_N_COMPONENTS} components, {UMAP_BIN_SIZE_S * 1e3:.0f} ms bins "
          f"-> ~{n_bins} bins x {int((meta['n_spikes_maze'] > 0).sum())} units")
    try:
        import umap  # noqa: F401
        print(f"umap-learn {umap.__version__} available")
    except ImportError:
        print("umap-learn NOT installed — `%pip install umap-learn` in the notebook")


# %% ===========================================================================
# CELL 2 — shared session loader and the UMAP sweep
# ==============================================================================
# Split the file here for the second notebook cell. It assumes cell 1 has run
# (it reuses DOWNLOAD_DIR, the inventory and the config constants) and it
# defines the light session loader that cell 3's viewer also needs.
#
# Memory is the constraint both this cell and the next are built around. One
# session's spikes are ~50 MB and its LFP another ~26 MB, so nothing here holds
# more than a single session: the sweep frees each one before loading the next,
# and the LFP is not loaded at all — neither the sweep nor the viewer needs it.

import gc  # noqa: E402

import matplotlib.colors as mcolors  # noqa: E402
import umap  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

# umap-learn must be importable by THIS kernel — `%pip install umap-learn`
# (not `!pip`, which may target a different interpreter), then restart it.

# --- UMAP config -------------------------------------------------------------
# The embedding is built from binned counts, variance-stabilized (sqrt),
# smoothed, z-scored per unit, then reduced by PCA before UMAP. The PCA step is
# not cosmetic: UMAP on 300+ sparse columns is slow and mostly fits noise.
UMAP_MIN_RATE_HZ = 0.1            # drop near-silent units — they contribute only zeros
UMAP_SMOOTH_S = 0.10              # Gaussian sigma applied along time before embedding
UMAP_PCA_COMPONENTS = 40
UMAP_MAX_BINS = 20000             # subsample bins to keep each fit near a minute
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "cosine"            # rate-vector direction, not magnitude
UMAP_REGIONS = ("CA1", "CA3", "RSC")
RUN_UMAP = True


# =============================================================================
# STEP 9 — a light session loader shared by the viewer and the UMAP sweep
# =============================================================================
# Deliberately not load_session_bundle(): no LFP, no derived speed, no pynapple
# objects kept. Spike times come out as plain sorted numpy arrays, which makes
# both windowed rasters (searchsorted) and binning (diff of searchsorted) cheap
# and lets the NWB handle be dropped immediately.


def load_light_session(row, verbose=False):
    """Spikes as sorted arrays, behaviour, labels. No LFP, nothing lazy."""
    path = (DOWNLOAD_DIR / row["file"]).resolve()
    t0 = time.perf_counter()
    nwb = nap.load_file(str(path))

    spikes_all = nwb["units"]
    position_all = nwb["Position"]
    maze_ep = maze_epoch_from(position_all)
    maze_s = float(maze_ep.tot_length())
    spikes = spikes_all.restrict(maze_ep)

    meta = extract_unit_metadata(spikes_all).reindex(np.asarray(spikes.index))
    n_spikes = np.array([len(spikes[u]) for u in spikes.index], dtype=int)
    meta["n_spikes_maze"] = n_spikes
    meta["rate_maze_hz"] = n_spikes / maze_s

    # Units are ordered region -> cell type -> rate once, here, so the raster
    # rows and the UMAP column blocks share one ordering.
    sort_cols = [c for c in ("cell_area", "cell_type") if c in meta.columns]
    meta = meta.sort_values(sort_cols + ["rate_maze_hz"]) if sort_cols else meta
    spike_times = [np.sort(np.asarray(spikes[u].index.values, dtype=np.float64))
                   for u in meta.index]

    position = position_all.restrict(maze_ep)
    pos_t = np.asarray(position.index.values, dtype=float)
    pos_x = np.asarray(position["x"].values, dtype=float)
    pos_y = np.asarray(position["y"].values, dtype=float)

    speed_key = find_key(nwb, "speed")
    if speed_key is not None:
        speed_t, speed_v = as_time_values(nwb[speed_key].restrict(maze_ep))
    else:
        speed_t, speed_v = pos_t, np.full_like(pos_t, np.nan)

    labels = read_sleep_states_pynwb(path)

    session = {
        "subject": row["subject"],
        "date": row["date"],
        "session": row["session"],
        "label": f"{row['subject']} {row['date']}",
        "t_start": float(maze_ep.start[0]),
        "t_end": float(maze_ep.end[0]),
        "maze_s": maze_s,
        "meta": meta,
        "spike_times": spike_times,
        "pos_t": pos_t,
        "pos_x": pos_x,
        "pos_y": pos_y,
        "speed_t": speed_t,
        "speed_v": speed_v,
        "labels": labels,
    }
    del spikes, spikes_all, position, position_all, nwb
    gc.collect()
    if verbose:
        print(f"loaded {row['file']} ({len(spike_times)} units) in "
              f"{time.perf_counter() - t0:.1f}s")
    return session


def unit_groups(meta):
    """(region, cell_type) label and an RGBA colour for every unit row.

    Colour carries region; lightness within a region carries cell type, so the
    raster shows both dimensions without needing two legends.
    """
    region = (meta["cell_area"].astype(str).values if "cell_area" in meta.columns
              else np.array(["all"] * len(meta)))
    ctype = (meta["cell_type"].astype(str).values if "cell_type" in meta.columns
             else np.array(["unknown"] * len(meta)))
    regions = sorted(set(region))
    base = {r: np.array(mcolors.to_rgb(f"C{i}")) for i, r in enumerate(regions)}
    shades = {}
    for r in regions:
        types = sorted(set(ctype[region == r]))
        for j, c in enumerate(types):
            mix = 0.25 + 0.75 * (j / max(len(types) - 1, 1))  # pale -> saturated
            shades[(r, c)] = tuple(1 - mix * (1 - base[r])) + (0.9,)
    colors = [shades[(r, c)] for r, c in zip(region, ctype)]
    group = np.array([f"{r} {c}" for r, c in zip(region, ctype)])
    return region, ctype, group, colors


def label_spans(labels, t0, t1):
    """State intervals and ripple times overlapping [t0, t1]."""
    if labels is None or not len(labels):
        return {}, np.array([])
    col = next((c for c in labels.columns
                if str(c).lower() in ("state", "label", "sleep_state", "tags")), None)
    if col is None:
        return {}, np.array([])
    m = (labels["stop_time"] > t0) & (labels["start_time"] < t1)
    sub = labels[m]
    states, ripples = {}, np.array([])
    for name, grp in sub.groupby(sub[col].astype(str)):
        starts = np.maximum(grp["start_time"].values, t0)
        stops = np.minimum(grp["stop_time"].values, t1)
        if name.lower().startswith("ripple"):
            ripples = (starts + stops) / 2
        else:
            states[name] = list(zip(starts, stops - starts))
    return states, ripples



# =============================================================================
# STEP 10 — the UMAP sweep: global + per region, every session
# =============================================================================
# 3 sessions x (global, CA1, CA3, RSC) = 12 embeddings, 12 figures, 3 components
# each. Every figure shows the same point cloud three times, coloured by the
# three things that could explain its structure: track position, running speed,
# and behavioural state with ripples marked.
#
# Preprocessing per embedding: counts -> sqrt (variance stabilization) ->
# Gaussian smoothing along time -> per-unit z-score -> PCA -> UMAP. Bins are
# subsampled to UMAP_MAX_BINS *after* smoothing, so each point still summarizes
# its local neighbourhood in time.


def counts_from_spike_times(spike_times, edges):
    """Binned counts (n_bins x n_units) from sorted spike-time arrays."""
    out = np.empty((len(edges) - 1, len(spike_times)), dtype=np.float32)
    for j, st in enumerate(spike_times):
        out[:, j] = np.diff(np.searchsorted(st, edges))
    return out


def bin_covariates(session, centers):
    """Position, speed, state and ripple flags interpolated onto bin centres."""
    x = np.interp(centers, session["pos_t"], session["pos_x"])
    y = np.interp(centers, session["pos_t"], session["pos_y"])
    speed = np.interp(centers, session["speed_t"], session["speed_v"])

    ripple = np.zeros(len(centers), dtype=bool)
    labels = session["labels"]
    if labels is not None and len(labels):
        col = next((c for c in labels.columns
                    if str(c).lower() in ("state", "label", "sleep_state", "tags")), None)
        if col is not None:
            rip = labels[labels[col].astype(str).str.lower().str.startswith("ripple")]
            for s0, s1 in zip(rip["start_time"].values, rip["stop_time"].values):
                ripple |= (centers >= s0) & (centers <= s1)
    return {"x": x, "y": y, "speed": speed, "ripple": ripple,
            "immobile": speed < IMMOBILE_CM_S}


def embed_counts(counts, bin_s, seed=UMAP_RANDOM_STATE):
    """counts -> sqrt -> smooth -> z-score -> PCA -> 3-component UMAP."""
    X = np.sqrt(counts)
    sigma_bins = UMAP_SMOOTH_S / bin_s
    if sigma_bins > 0:
        X = gaussian_filter1d(X, sigma=sigma_bins, axis=0, mode="nearest")
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    stride = max(1, len(X) // UMAP_MAX_BINS)
    keep = np.arange(0, len(X), stride)
    X = X[keep]

    n_pcs = int(min(UMAP_PCA_COMPONENTS, X.shape[1], X.shape[0] - 1))
    pca = PCA(n_components=n_pcs, random_state=seed)
    scores = pca.fit_transform(X)

    reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS,
                        min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC, random_state=seed)
    emb = reducer.fit_transform(scores)
    return emb, keep, float(pca.explained_variance_ratio_.sum())


def plot_embedding(emb, cov, keep, title):
    """One figure per embedding: the same cloud coloured three ways."""
    fig = plt.figure(figsize=(15, 4.6))
    fig.suptitle(title)
    panels = [
        ("track position x (cm)", cov["x"][keep], "viridis"),
        ("log10 speed (cm/s)", np.log10(np.clip(cov["speed"][keep], 0.05, None)), "magma"),
        ("immobile (0) vs moving (1)", (~cov["immobile"][keep]).astype(float), "coolwarm"),
    ]
    for i, (label, colour, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=colour, cmap=cmap,
                        s=1.5, alpha=0.5, linewidths=0, rasterized=True)
        if i == 3:
            rip = cov["ripple"][keep]
            if rip.any():
                ax.scatter(emb[rip, 0], emb[rip, 1], emb[rip, 2], c="lime",
                           s=28, edgecolors="k", linewidths=0.4,
                           label=f"ripple bins (n={int(rip.sum())})")
                ax.legend(fontsize=8, loc="upper left")
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08).set_label(label, fontsize=8)
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.set_zlabel("UMAP 3", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


if RUN_UMAP:
    rule = "=" * 78
    print("\n" + rule)
    print(f"UMAP sweep: {len(subject_sessions)} sessions x "
          f"{1 + len(UMAP_REGIONS)} groupings = "
          f"{len(subject_sessions) * (1 + len(UMAP_REGIONS))} embeddings")
    print(f"expect ~1-2 min per embedding (random_state is set, which forces "
          f"single-threaded UMAP); lower UMAP_MAX_BINS to speed it up")
    print(rule)

    umap_summary = []
    for _, row in subject_sessions.iterrows():
        sess = load_light_session(row, verbose=True)
        edges = np.arange(sess["t_start"], sess["t_end"], UMAP_BIN_SIZE_S)
        centers = edges[:-1] + UMAP_BIN_SIZE_S / 2
        counts = counts_from_spike_times(sess["spike_times"], edges)
        cov = bin_covariates(sess, centers)

        meta = sess["meta"]
        region = (meta["cell_area"].astype(str).values if "cell_area" in meta.columns
                  else np.array(["all"] * len(meta)))
        alive = (meta["rate_maze_hz"].values > UMAP_MIN_RATE_HZ)

        groupings = [("global", alive)]
        groupings += [(r, alive & (region == r)) for r in UMAP_REGIONS]

        for name, mask in groupings:
            if mask.sum() < 5:
                print(f"  {sess['label']} {name}: only {int(mask.sum())} units, skipped")
                continue
            t0 = time.perf_counter()
            emb, keep, var = embed_counts(counts[:, mask], UMAP_BIN_SIZE_S)
            elapsed = time.perf_counter() - t0
            print(f"  {sess['label']} {name:7s}: {int(mask.sum()):3d} units, "
                  f"{len(keep)} bins, PCA kept {100 * var:.1f}%, {elapsed:.0f}s")
            umap_summary.append({"session": sess["label"], "grouping": name,
                                 "n_units": int(mask.sum()), "n_bins": len(keep),
                                 "pca_var": round(var, 3), "fit_s": round(elapsed, 1),
                                 "ripple_bins": int(cov["ripple"][keep].sum())})
            plot_embedding(emb, cov, keep,
                           f"{sess['label']} — {name} — UMAP "
                           f"({int(mask.sum())} units, {UMAP_BIN_SIZE_S * 1e3:.0f} ms bins)")
            del emb, keep
            gc.collect()

        del sess, counts, cov, edges, centers
        gc.collect()

    print("\n" + rule)
    print("UMAP sweep summary")
    print(rule)
    print(pd.DataFrame(umap_summary).to_string(index=False))


# %% ===========================================================================
# CELL 3 — the scrollable viewer
# ==============================================================================
# Split here for the third notebook cell. It needs cell 2 to have run, which is
# where load_light_session / unit_groups / label_spans are defined — set
# RUN_UMAP = False there to get the definitions without sitting through the
# 20-minute sweep. Re-running this cell alone re-renders the viewer.

VIEWER_WINDOW_S = 20.0            # initial width of the scrolling window
VIEWER_WIDTH_CHOICES = (5.0, 10.0, 20.0, 60.0, 120.0)
VIEWER_ALL_SUBJECTS = False       # True to list every subject in the dropdown
RUN_VIEWER = True

# =============================================================================
# STEP 11 — the scrollable viewer
# =============================================================================
# Four stacked panels on one clock: position as x(t) and y(t), speed, the
# author's labels as a bar, and the raster grouped by region x cell type.
# The session dropdown swaps which recording is loaded; the previous one is
# released before the next is read, so RAM holds one session, not four.

STATE_COLORS = {"WAKE": "#b0b0b0", "NREM": "#4c72b0", "REM": "#55a868"}


class MazeViewer:
    """Scrollable multi-panel view of one maze epoch at a time."""

    def __init__(self, rows, window_s=VIEWER_WINDOW_S):
        self.rows = rows.reset_index(drop=True)
        self.window_s = window_s
        self._cached_idx = None
        self._session = None

    def session(self, idx):
        """Load session `idx`, dropping whichever one is currently held."""
        if self._cached_idx == idx and self._session is not None:
            return self._session
        self._session = None          # release before allocating the next one
        gc.collect()
        self._session = load_light_session(self.rows.iloc[idx])
        self._cached_idx = idx
        return self._session

    def draw(self, idx, t_start, window_s):
        s = self.session(idx)
        t0 = float(np.clip(t_start, s["t_start"], max(s["t_start"], s["t_end"] - window_s)))
        t1 = min(t0 + window_s, s["t_end"])
        meta = s["meta"]
        _, _, group, colors = unit_groups(meta)

        fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, 1.0, 0.35, 4.5]})
        fig.suptitle(f"{s['label']} — {t0:.1f}-{t1:.1f}s of "
                     f"{s['t_start']:.0f}-{s['t_end']:.0f}s ({s['maze_s'] / 60:.1f} min)")

        # --- position: x and y as two lines on one axes --------------------
        ax = axes[0]
        pm = (s["pos_t"] >= t0) & (s["pos_t"] <= t1)
        ax.plot(s["pos_t"][pm], s["pos_x"][pm], lw=1.0, color="C0", label="x")
        ax.plot(s["pos_t"][pm], s["pos_y"][pm], lw=1.0, color="C1", label="y")
        ax.set_ylabel("position\n(cm)")
        ax.legend(fontsize=8, ncol=2, loc="upper right")

        # --- speed ----------------------------------------------------------
        ax = axes[1]
        sm = (s["speed_t"] >= t0) & (s["speed_t"] <= t1)
        ax.plot(s["speed_t"][sm], s["speed_v"][sm], lw=0.9, color="k")
        ax.axhline(IMMOBILE_CM_S, color="C3", ls=":", lw=0.8)
        ax.axhline(RUN_CM_S, color="C2", ls="--", lw=0.8)
        ax.set_ylabel("speed\n(cm/s)")

        # --- label bar: the author's states, with ripples marked ------------
        ax = axes[2]
        states, ripples = label_spans(s["labels"], t0, t1)
        for name, spans in states.items():
            ax.broken_barh(spans, (0.05, 0.9),
                           facecolors=STATE_COLORS.get(name.upper(), "#cccccc"),
                           label=name)
        for tr in ripples:
            ax.axvline(tr, color="crimson", lw=1.2, alpha=0.9)
        if len(ripples):
            ax.plot([], [], color="crimson", lw=1.2, label=f"ripple ({len(ripples)})")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel("labels", rotation=0, ha="right", va="center")
        if states or len(ripples):
            ax.legend(fontsize=7, ncol=5, loc="upper right", framealpha=0.9)

        # --- raster, grouped by region x cell type --------------------------
        ax = axes[3]
        windowed = []
        for st in s["spike_times"]:
            lo, hi = np.searchsorted(st, (t0, t1))
            windowed.append(st[lo:hi])
        ax.eventplot(windowed, lineoffsets=np.arange(len(windowed)),
                     linelengths=0.85, linewidths=0.6, colors=colors)
        # separators and one tick per region x cell-type block
        boundaries, centers, names = [], [], []
        start = 0
        for i in range(1, len(group) + 1):
            if i == len(group) or group[i] != group[start]:
                boundaries.append(i - 0.5)
                centers.append((start + i - 1) / 2)
                names.append(f"{group[start]} (n={i - start})")
                start = i
        for bnd in boundaries[:-1]:
            ax.axhline(bnd, color="k", lw=0.4, alpha=0.3)
        ax.set_yticks(centers)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_ylim(-1, len(windowed))
        ax.set_xlim(t0, t1)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("units")

        fig.tight_layout()
        plt.show()
        plt.close(fig)

    def show(self):
        """Render the widget UI, or fall back to three static windows."""
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            print("ipywidgets not available — showing three static windows instead.\n"
                  "`%pip install ipywidgets` for the scrollable version.")
            for frac in (0.1, 0.5, 0.9):
                s = self.session(0)
                self.draw(0, s["t_start"] + frac * s["maze_s"], self.window_s)
            return

        options = [(f"{r['subject']} {r['date']}", i) for i, r in self.rows.iterrows()]
        session_dd = widgets.Dropdown(options=options, value=0, description="session:")
        width_dd = widgets.Dropdown(options=[(f"{w:.0f}s", w) for w in VIEWER_WIDTH_CHOICES],
                                    value=self.window_s, description="window:")
        scrub = widgets.FloatSlider(description="start (s)", continuous_update=False,
                                    readout_format=".1f", layout=widgets.Layout(width="70%"))
        back = widgets.Button(description="◀", layout=widgets.Layout(width="45px"))
        fwd = widgets.Button(description="▶", layout=widgets.Layout(width="45px"))
        out = widgets.Output()

        quiet = {"on": False}  # suppress redraws while the slider is reconfigured

        def rescale(*_):
            s = self.session(session_dd.value)
            lo = s["t_start"]
            hi = max(s["t_start"], s["t_end"] - width_dd.value)
            quiet["on"] = True
            # widen before tightening: setting max below the current min (or the
            # reverse) is a traitlet error when sessions have disjoint clocks
            scrub.min = min(scrub.min, lo)
            scrub.max = max(scrub.max, hi)
            scrub.value = lo
            scrub.min, scrub.max = lo, hi
            scrub.step = width_dd.value / 4
            quiet["on"] = False
            refresh()

        def refresh(*_):
            if quiet["on"]:
                return
            with out:
                clear_output(wait=True)
                self.draw(session_dd.value, scrub.value, width_dd.value)

        def step(delta):
            def handler(_):
                scrub.value = float(np.clip(scrub.value + delta * width_dd.value / 2,
                                            scrub.min, scrub.max))
            return handler

        session_dd.observe(rescale, names="value")
        width_dd.observe(rescale, names="value")
        scrub.observe(refresh, names="value")
        back.on_click(step(-1))
        fwd.on_click(step(+1))

        display(widgets.VBox([widgets.HBox([session_dd, width_dd]),
                              widgets.HBox([back, fwd, scrub]), out]))
        rescale()


viewer_rows = (inventory[inventory["has_behavior"]] if VIEWER_ALL_SUBJECTS
               else subject_sessions)
viewer = MazeViewer(viewer_rows)
if RUN_VIEWER:
    viewer.show()

