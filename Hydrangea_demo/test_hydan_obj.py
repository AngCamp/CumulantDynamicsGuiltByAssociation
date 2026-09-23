"""Hydrangea demo: session EDA, then state-fitting EDA.

Runs from the notebook. `# %%` marks each cell boundary — split there.

The package root is discovered by walking up from the working directory, so
this works whether the notebook is launched from the repo root or from
Hydrangea_demo/, locally or on the cloud box where the repo lives at
/notebooks/CumulantDynamicsGuiltByAssociation. The data mount defaults to
/storage/dandi_downloads and can be overridden with HYDRANGEA_DATA_DIR.

Cells:
    1. setup and dataset inventory
    2. build the session: load, epoch, derived behaviour, object, channels
    3. spiking + behaviour EDA          <- judge epoch / units / channels / coverage
    4. normalize                        <- judge bin size and normalization method
    5. global embedding
    6. folds and the HMM sweep
    7. HMM report
    8. save, reload, summarize
"""

# %% ==========================================================================
# CELL 1 — setup and dataset inventory
# =============================================================================
# pynapple's Folder API walks the subject/session tree, so the inventory only
# has to parse identifiers out of the BIDS-style filenames:
#     sub-M01_ses-20240313T100000_behavior+ecephys.nwb
#     ^^^^^^^ ^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^
#     subject session             modalities
# Paths are shown relative to the data mount; the mount itself is printed once.

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap


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

from hybrid_dynamics_core.base import HybridDynamicsAnalysis  # noqa: E402
from hybrid_dynamics_core.spiking_and_behaviour_eda import (  # noqa: E402
    compute_speed_from_position,
)

DOWNLOAD_DIR = Path(os.environ.get("HYDRANGEA_DATA_DIR", "/storage/dandi_downloads")).resolve()
MODEL_DIR = DOWNLOAD_DIR / "hydrangea_models"
DANDISET_ID = "001695"
VERSION_ID = "0.260319.2023"

# Session to analyze. Set either to None to take the first behaviour recording.
SUBJECT = "M01"
SESSION = "20240313T100000"

BIN_SIZE_S = 0.050
EDA_WINDOW_S = 60
N_UNITS_PER_GROUP = 5

if DOWNLOAD_DIR.is_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def parse_session_name(nwb_path, root):
    """Subject, session, date, time, and modalities from an NWB filename."""
    nwb_path = Path(nwb_path)
    stem = nwb_path.stem

    subject = re.search(r"(?:^|_)sub-([^_]+)", stem)
    session = re.search(r"(?:^|_)ses-([^_]+)", stem)
    subject = subject.group(1) if subject else nwb_path.parent.name.replace("sub-", "")
    session = session.group(1) if session else "unknown"

    date, clock = None, None
    if "T" in session:
        raw_date, raw_time = session.split("T", 1)
        if len(raw_date) == 8 and raw_date.isdigit():
            date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
        if len(raw_time) >= 6 and raw_time[:6].isdigit():
            clock = f"{raw_time[:2]}:{raw_time[2:4]}:{raw_time[4:6]}"

    modalities = stem.split("_")[-1].split("+")
    return {
        "subject": subject,
        "session": session,
        "date": date,
        "time": clock,
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


project = nap.load_folder(str(DOWNLOAD_DIR))
inventory = session_inventory(DOWNLOAD_DIR)

print("Package root:", PACKAGE_ROOT)
print("Data mount:  ", DOWNLOAD_DIR)
print(
    f"{len(inventory)} recordings | {inventory['subject'].nunique()} subjects | "
    f"{inventory['size_mb'].sum() / 1000:.1f} GB"
)
inventory


# %% ==========================================================================
# CELL 2 — build the session
# =============================================================================
# Everything needed before any modelling decision, in four steps:
#
#   2a  load the chosen recording
#   2b  restrict to the analysis epoch (the position-tracked span)
#   2c  derive behaviour the archive does not ship (running speed)
#   2d  build the analysis object and register its behaviour channels
#
# Step 2c stays out of the object on purpose: how a derived channel is built
# (which columns, how much smoothing) is a per-dataset decision, whereas the
# object only needs the finished signal plus its sampling rate.

cell_started = time.perf_counter()

# --- 2a. load the recording ------------------------------------------------
print("=" * 72)
print("2a. load recording")
print("=" * 72)

candidates = inventory[inventory["has_behavior"]]
if SUBJECT is not None:
    candidates = candidates[candidates["subject"] == SUBJECT]
if SESSION is not None:
    candidates = candidates[candidates["session"] == SESSION]
if len(candidates) == 0:
    raise FileNotFoundError(f"No behaviour recording for subject={SUBJECT}, session={SESSION}")

session_row = candidates.iloc[0]
session_nwb_path = (DOWNLOAD_DIR / session_row["file"]).resolve()

t0 = time.perf_counter()
session_nwb = nap.load_file(str(session_nwb_path))
spike_group = session_nwb["units"]
position_xy = session_nwb["Position"]
print(f"loaded {session_row['file']} in {time.perf_counter() - t0:.1f}s")
print(f"NWB keys: {list(session_nwb.keys())}")
print(f"{len(spike_group)} units recorded")

# --- 2b. restrict to the analysis epoch ------------------------------------
print("\n" + "=" * 72)
print("2b. analysis epoch")
print("=" * 72)
# States are only interpretable where behaviour is observed, so the pipeline is
# restricted to the position-tracked span. Everything before and after it
# (pre-task rest, post-task handling) is discarded here rather than filtered
# later, which keeps the observation matrix and the behaviour on one clock.

recording_start = float(min(spike_group[u].index[0] for u in spike_group.index if len(spike_group[u])))
recording_end = float(max(spike_group[u].index[-1] for u in spike_group.index if len(spike_group[u])))
maze_start = float(position_xy.index[0])
maze_end = float(position_xy.index[-1])
maze_ep = nap.IntervalSet(start=maze_start, end=maze_end)
spike_group_maze = spike_group.restrict(maze_ep)

recording_s = recording_end - recording_start
maze_s = maze_end - maze_start
print(f"recording span : {recording_start:8.1f} - {recording_end:8.1f}s  ({recording_s / 60:.1f} min)")
print(f"maze epoch     : {maze_start:8.1f} - {maze_end:8.1f}s  ({maze_s / 60:.1f} min)")
print(
    f"discarding {(recording_s - maze_s) / 60:.1f} min "
    f"({100 * (1 - maze_s / recording_s):.1f}% of the recording) outside the tracked span"
)
print(f"units before/after restriction: {len(spike_group)} / {len(spike_group_maze)}")

# --- 2c. derive running speed ----------------------------------------------
print("\n" + "=" * 72)
print("2c. derived behaviour")
print("=" * 72)

speed_t, speed_vals = compute_speed_from_position(
    position_xy, columns=("x", "y"), smooth_window_s=0.25
)
print(
    f"running speed: {len(speed_t)} samples @ {1 / np.median(np.diff(speed_t)):.1f} Hz | "
    f"median {np.median(speed_vals):.2f} cm/s, max {speed_vals.max():.1f} cm/s"
)

# --- 2d. build the object and register behaviour ---------------------------
print("\n" + "=" * 72)
print("2d. analysis object")
print("=" * 72)
# Unit metadata (region, cell type, anything else on the TsGroup) is read off
# the spike group automatically and carried through unit filtering. Trial,
# condition, and region tables are optional and dataset-specific.

trials = None
try:
    trials = session_nwb["trials"]
except Exception:
    print("no trials table in this NWB")

hybdyn = HybridDynamicsAnalysis(
    spike_group=spike_group_maze,
    bin_size_s=BIN_SIZE_S,
    maze_epoch=maze_ep,
    trial_table=trials,  # an IntervalSet is coerced to a start/end table
    session_id=session_nwb_path.stem,
    mouse_id=session_row["subject"],
    embedding_method="pca",
    state_discovery_method="gaussian_hmm",
    random_state=0,
    report="selected",
)

# Continuous channels carry a sampling-rate descriptor (inferred from the
# timestamps here). Discrete channels carry a time interval and optional
# subtypes. Both project onto the observation bins later.
hybdyn.add_continuous_behavior(
    "position",
    time=position_xy.index.values,
    values=np.column_stack([position_xy["x"].values, position_xy["y"].values]),
    columns=["x", "y"],
    units="cm",
    description="tracked position on the maze",
)
hybdyn.add_continuous_behavior(
    "running_speed",
    time=speed_t,
    values=speed_vals,
    units="cm/s",
    color="k",
    description="derived from position, 0.25s boxcar",
)
hybdyn.add_events_from_interval_set(
    "maze_epoch", maze_ep, description="position-tracked span used for all fitting"
)
if trials is not None:
    hybdyn.add_events_from_interval_set("trials", trials, description="task trials")

print(
    f"object built: {len(hybdyn.continuous_behavior)} continuous channels, "
    f"{len(hybdyn.discrete_events)} event sets"
)
print(f"\ncell 2 complete in {time.perf_counter() - cell_started:.1f}s")


# %% ==========================================================================
# CELL 3 — spiking + behaviour EDA
# =============================================================================
# Human judgement: are the epoch, the unit sample, and the behavioural channels
# sensible, and does each channel actually cover the bins it will be aligned to?
#
# Prints, in order: metadata tables, unit inventory, behaviour channels,
# behaviour coverage on 50 ms bins. Then plots the unit inventory, the
# full-session behaviour overview with the raster window marked on it, and the
# raster for that window. Coverage is computed here rather than after
# normalization because behaviour binning does not depend on the spikes.

hybdyn.spiking_behavior_report(
    window_s=EDA_WINDOW_S,
    n_units_per_group=N_UNITS_PER_GROUP,
    show=True,
)


# %% ==========================================================================
# CELL 4 — normalization
# =============================================================================
# Human judgement, and worth dwelling on: is the bin size and normalization
# method appropriate for this firing-rate distribution?
#
# `proportion_zscore` divides each unit's binned counts by its total spike
# count before z-scoring, so a 30 Hz interneuron and a 0.2 Hz pyramidal cell
# contribute comparably to the observation matrix instead of the fast cells
# dominating the covariance the PCA and HMM will see.
#
# Set log_scale=False on the report to see the rate and mean-variance panels on
# linear axes — log spreads out the near-silent tail, linear shows the
# high-rate units in proportion.

hybdyn.normalize(method="proportion_zscore", zscore=True, restrict_to_epoch=True)
hybdyn.normalization_report(show=True, log_scale=True)

# Behaviour re-aligned to the final observation bins. Coverage should match what
# cell 3 reported; if it does not, the epoch or the bin size moved.
behavior_bins = hybdyn.align_behavior_to_bins()
print("\nBehaviour on the observation bins:")
hybdyn.behavior_coverage(aligned=behavior_bins)


# %% ==========================================================================
# CELL 5 — global embedding
# =============================================================================
# The session-wide PCA that the HMM observes.

hybdyn.global_embedding(method="pca", n_components=10, whiten=False, standardize=True)
cumulative = np.cumsum(hybdyn.global_embedding_variance_ratio)
print(
    f"PCA: {hybdyn.global_embedding_scores.shape[1]} components, "
    f"{100 * cumulative[-1]:.1f}% of variance retained "
    f"(PC1 {100 * hybdyn.global_embedding_variance_ratio[0]:.1f}%)"
)
hybdyn.report_embeddings(show=True)


# %% ==========================================================================
# CELL 6 — folds and the HMM sweep
# =============================================================================
# This is the slow step: K * (1 + k_fold) EM runs. `show_progress` prints which
# K and fold is running, how long each took, whether EM converged, and a rolling
# estimate of time remaining. `verbose` is hmmlearn's own per-EM-iteration
# monitor — leave it off unless a single fit refuses to converge.

hybdyn.build_folds(
    k_fold=5,
    fold_strategy="temporal_segments",
    shuffle_within_segments=True,
    seed=0,
)
print(f"{len(hybdyn.fold_idx)} folds, {len(hybdyn.fold_idx[0])} held-out bins each")

hybdyn.fit_states(
    n_states_min=2,
    n_states_max=10,
    k_fold=5,
    fold_strategy="temporal_segments",
    shuffle_within_segments=True,
    n_iter=100,
    covariance_type="diag",
    verbose=False,
    use_cv=True,
    show_progress=True,
    report="none",  # the full table is shown in cell 7
)


# %% ==========================================================================
# CELL 7 — HMM report
# =============================================================================
# Human judgement: inspect the ranked scores and choose a working K. The table
# carries the CV spread (mean/median/std/min/max across folds), per-bin
# log-likelihood so K values are comparable, EM convergence, and fit time.

hybdyn.hmm_report(report="selected")
# hybdyn.hmm_report(report="full")

# Optional next step: local embedding inside one discovered state.
# hybdyn.local_embedding(state_labels=hybdyn.state_labels_, state_value=0, method="pca", label="state_0")
# hybdyn.report_embeddings(scope="local", label="state_0", show=True)

hybdyn.hmm_score_table(sort_by="median_cv_loglik")


# %% ==========================================================================
# CELL 8 — save, reload, summarize
# =============================================================================

saved_model_path = MODEL_DIR / f"{session_nwb_path.stem}_best_hmm.pkl"
hybdyn.save_hmm_model(saved_model_path)
reloaded_artifact = hybdyn.load_hmm_model(saved_model_path)

print("Saved model:", saved_model_path.name)
print("Reloaded model states:", reloaded_artifact["n_states"])
print("Best K:", hybdyn.best_k, "| models fit:", len(hybdyn.hmm_models))
print()
hybdyn.describe(as_text=True)
hybdyn.timing_summary(as_text=True)
