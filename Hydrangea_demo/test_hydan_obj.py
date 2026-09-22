"""Hydrangea demo object test script.

This runs inside the notebook. Do not hardcode the notebook path.
The data mount is fixed at /storage/dandi_downloads.
"""

from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pynapple as nap

from hybrid_dynamics_core.base import HybridDynamicsAnalysis

DOWNLOAD_DIR = Path("/storage/dandi_downloads").resolve()
DANDISET_ID = "001695"
VERSION_ID = "0.260319.2023"
SUBJECTS = ["sub-M01", "sub-M02", "sub-M03", "sub-M05"]

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def print_session_table(download_root: Path):
    rows = []
    for subject_dir in sorted(download_root.glob("sub-*")):
        for nwb_file in sorted(subject_dir.rglob("*.nwb")):
            rel = nwb_file.relative_to(download_root)
            parts = rel.parts
            if len(parts) >= 3:
                subject = parts[0]
                session = parts[1] if len(parts) > 1 else "unknown"
                date = parts[-2] if len(parts) > 2 else "unknown"
            else:
                subject = subject_dir.name
                session = "unknown"
                date = "unknown"
            rows.append((subject, session, date, str(nwb_file)))

    if not rows:
        print(f"No NWB files found under {download_root}")
        return

    print("\nDANDI session inventory")
    print("-" * 110)
    print(f"{'subject':<12} {'session':<20} {'date':<20} {'nwb_file':<80}")
    print("-" * 110)
    for subject, session, date, path in rows:
        print(f"{subject:<12} {session:<20} {date:<20} {path:<80}")
    print("-" * 110)


print_session_table(DOWNLOAD_DIR)

# Build the same session context used in the notebook.
# Hardcoded for the known dataset in this tutorial script.
session_nwb_path = Path(
    "/storage/dandi_downloads/sub-M01/"
    "sub-M01_ses-20240313T100000_behavior+ecephys.nwb"
).resolve()
session_nwb = nap.load_file(str(session_nwb_path))

spike_group = session_nwb["units"]
position_xy = session_nwb["Position"]

maze_start = float(position_xy.index[0])
maze_end = float(position_xy.index[-1])
maze_ep = nap.IntervalSet(start=maze_start, end=maze_end)
spike_group_maze = spike_group.restrict(maze_ep)

print("Session NWB path:", session_nwb_path)
print("Units before/after maze restriction:", len(spike_group), "/", len(spike_group_maze))

# Example usage: create a Hydrangea object and call steps in order.
hybdyn = HybridDynamicsAnalysis(
    spike_group=spike_group_maze,
    bin_size_s=0.050,
    maze_epoch=maze_ep,
    random_state=0,
    report="selected",
)

# 1) normalize
hybdyn.normalize(method="proportion_zscore", zscore=True, restrict_to_epoch=True)
hybdyn.normalization_report(show=False)

# 2) global reduction
hybdyn.global_pca(n_components=10, whiten=False, standardize=True)
hybdyn.pca_report(show=False)

# 3) build folds
hybdyn.build_folds(
    k_fold=5,
    fold_strategy="temporal_segments",
    shuffle_within_segments=True,
    seed=0,
)

# 4) fit HMM models
hybdyn.fit_hmm(
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

# 5) reporting
hybdyn.hmm_report(report="selected")
# hybdyn.hmm_report(report="full")
# hybdyn.hmm_report(report="none")

print("Notebook working directory:", ROOT)
print("Download dir:", DOWNLOAD_DIR)
print("Best K:", hybdyn.best_k)
print("Model count:", len(hybdyn.hmm_models))
