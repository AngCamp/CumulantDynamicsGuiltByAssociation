"""Hydrangea demo: session EDA, then state-fitting EDA.

Runs from the notebook. The package root is discovered by walking up from the
working directory, so this works whether the notebook is launched from the repo
root or from Hydrangea_demo/, locally or on the cloud box where the repo lives
at /notebooks/CumulantDynamicsGuiltByAssociation.

The data mount defaults to /storage/dandi_downloads and can be overridden with
the HYDRANGEA_DATA_DIR environment variable.

Order of operations:
    0. locate package + data, list sessions
    1. load one session, build the analysis epoch
    2. derive behaviour the archive does not ship (running speed)
    3. build the analysis object with its metadata tables
    4. register behaviour channels (continuous signals, discrete events)
    5. spiking + behaviour EDA          <- decide bin size / epoch / channels
    6. normalize + normalization report <- decide normalization method
    7. global embedding
    8. folds + state fitting + HMM report
    9. save/reload, summarize
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap


# ---------------------------------------------------------------------------
# 0. paths
# ---------------------------------------------------------------------------

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
SUBJECTS = ["sub-M01", "sub-M02", "sub-M03", "sub-M05"]

# Session to analyze. Set to None to take the first behaviour+ecephys file found.
SESSION_FILENAME = "sub-M01/sub-M01_ses-20240313T100000_behavior+ecephys.nwb"

BIN_SIZE_S = 0.050
EDA_WINDOW_S = 60
N_UNITS_PER_GROUP = 5

if DOWNLOAD_DIR.is_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def session_inventory(download_root):
    """Table of every NWB file under the data mount."""
    rows = []
    for subject_dir in sorted(Path(download_root).glob("sub-*")):
        for nwb_file in sorted(subject_dir.rglob("*.nwb")):
            stem = nwb_file.stem
            session = stem.split("_ses-")[-1].split("_")[0] if "_ses-" in stem else "unknown"
            rows.append(
                {
                    "subject": subject_dir.name,
                    "session": session,
                    "has_behavior": "behavior" in stem,
                    "nwb_file": str(nwb_file),
                }
            )
    return pd.DataFrame(rows)


def print_session_table(download_root):
    table = session_inventory(download_root)
    if len(table) == 0:
        print(f"No NWB files found under {download_root}")
        return table
    print("\nDANDI session inventory")
    print(table.to_string(index=False))
    return table


inventory = print_session_table(DOWNLOAD_DIR)


# ---------------------------------------------------------------------------
# 1. load one session and build the analysis epoch
# ---------------------------------------------------------------------------

if SESSION_FILENAME is not None:
    session_nwb_path = (DOWNLOAD_DIR / SESSION_FILENAME).resolve()
else:
    with_behavior = inventory[inventory["has_behavior"]]
    if len(with_behavior) == 0:
        raise FileNotFoundError(f"No behaviour+ecephys NWB files under {DOWNLOAD_DIR}")
    session_nwb_path = Path(with_behavior.iloc[0]["nwb_file"]).resolve()

session_nwb = nap.load_file(str(session_nwb_path))
print("\nSession NWB path:", session_nwb_path)
print("NWB keys:", list(session_nwb.keys()))

spike_group = session_nwb["units"]
position_xy = session_nwb["Position"]

# The maze epoch is the span over which position was tracked. Spiking outside it
# is unusable for behaviour-linked state analysis, so the whole pipeline is
# restricted to it.
maze_start = float(position_xy.index[0])
maze_end = float(position_xy.index[-1])
maze_ep = nap.IntervalSet(start=maze_start, end=maze_end)
spike_group_maze = spike_group.restrict(maze_ep)

print(f"Maze epoch: {maze_start:.1f}s to {maze_end:.1f}s ({maze_end - maze_start:.1f}s)")
print("Units before/after maze restriction:", len(spike_group), "/", len(spike_group_maze))


# ---------------------------------------------------------------------------
# 2. derive behaviour the archive does not ship
# ---------------------------------------------------------------------------
# Running speed is not in the downloaded NWB, so it is computed here rather than
# inside the analysis object: how a derived behavioural channel is built is a
# per-dataset decision.

speed_t, speed_vals = compute_speed_from_position(
    position_xy, columns=("x", "y"), smooth_window_s=0.25
)
print(f"Derived running speed: {len(speed_t)} samples, median {np.median(speed_vals):.3f} units/s")


# ---------------------------------------------------------------------------
# 3. build the analysis object with its metadata tables
# ---------------------------------------------------------------------------
# Unit metadata is read off the TsGroup automatically. Trial, condition, and
# region tables are optional and dataset-specific; this session ships none of
# them, so they stay empty here and the object reports that in describe().

trial_table = None
try:
    trials = session_nwb["trials"]
    trial_table = trials  # IntervalSet is coerced to a start/end table
except Exception:
    pass

hybdyn = HybridDynamicsAnalysis(
    spike_group=spike_group_maze,
    bin_size_s=BIN_SIZE_S,
    maze_epoch=maze_ep,
    trial_table=trial_table,
    session_id=session_nwb_path.stem,
    mouse_id=session_nwb_path.stem.split("_")[0].replace("sub-", ""),
    embedding_method="pca",
    state_discovery_method="gaussian_hmm",
    random_state=0,
    report="selected",
)


# ---------------------------------------------------------------------------
# 4. register behaviour channels
# ---------------------------------------------------------------------------
# Continuous channels carry a sampling rate descriptor (inferred from the
# timestamps here). Discrete channels carry an interval and optional subtypes.

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
    "maze_epoch",
    maze_ep,
    description="position-tracked span used for all fitting",
)

if trial_table is not None:
    hybdyn.add_events_from_interval_set("trials", trials, description="task trials")


# ---------------------------------------------------------------------------
# 5. spiking + behaviour EDA
# ---------------------------------------------------------------------------
# Human judgement: are the epoch, the unit sample, and the behavioural channels
# sensible before anything is binned?

hybdyn.spiking_behavior_report(
    window_s=EDA_WINDOW_S,
    n_units_per_group=N_UNITS_PER_GROUP,
    show=True,
)


# ---------------------------------------------------------------------------
# 6. normalization
# ---------------------------------------------------------------------------
# Human judgement: is the binning and normalization method appropriate for this
# firing-rate distribution?

hybdyn.normalize(method="proportion_zscore", zscore=True, restrict_to_epoch=True)
hybdyn.normalization_report(show=True)

print("Observation matrix:", hybdyn.spike_matrix.shape, "(bins x units)")

# Behaviour projected onto the same bins as the observation matrix. This is the
# table that pairs with state_labels_ once the HMM has run.
behavior_bins = hybdyn.align_behavior_to_bins()
print("\nBehaviour aligned to bins:", behavior_bins.shape)
print(behavior_bins.head())


# ---------------------------------------------------------------------------
# 7. global embedding
# ---------------------------------------------------------------------------

hybdyn.global_embedding(method="pca", n_components=10, whiten=False, standardize=True)
hybdyn.report_embeddings(show=True)


# ---------------------------------------------------------------------------
# 8. folds, state fitting, HMM report
# ---------------------------------------------------------------------------

hybdyn.build_folds(
    k_fold=5,
    fold_strategy="temporal_segments",
    shuffle_within_segments=True,
    seed=0,
)

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
    report="selected",
)

hybdyn.hmm_report(report="selected")
# hybdyn.hmm_report(report="full")

# Optional next step: local embedding inside one discovered state.
# hybdyn.local_embedding(state_labels=hybdyn.state_labels_, state_value=0, method="pca", label="state_0")
# hybdyn.report_embeddings(scope="local", label="state_0", show=True)


# ---------------------------------------------------------------------------
# 9. save, reload, summarize
# ---------------------------------------------------------------------------

saved_model_path = MODEL_DIR / f"{session_nwb_path.stem}_best_hmm.pkl"
hybdyn.save_hmm_model(saved_model_path)
reloaded_artifact = hybdyn.load_hmm_model(saved_model_path)

print("\nSaved model path:", saved_model_path)
print("Reloaded model states:", reloaded_artifact["n_states"])
print("Package root:", PACKAGE_ROOT)
print("Download dir:", DOWNLOAD_DIR)
print("Best K:", hybdyn.best_k)
print("Model count:", len(hybdyn.hmm_models))

hybdyn.describe(as_text=True)
