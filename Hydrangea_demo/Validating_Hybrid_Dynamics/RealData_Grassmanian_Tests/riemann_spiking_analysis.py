"""
riemann_spiking_analysis.py
===========================

Standardised pipeline shared by the real-data Grassmannian notebooks
(ALM / Dandiset 000363 and hippocampus / Dandiset 001695):

    DANDI session(s) -> units + trials / epochs        (one loader per dataset)
                     -> spike counts                    (count_bin_ms)
                     -> per-neuron normalisation        (z-score across all retained bins)
                     -> one covariance per window       (mollification_bin_ms = window length)
                        and its full eigendecomposition
                     -> k selection                     (median elbow of the explained-variance
                                                         curves, or a fixed k)
                     -> points U_w in Gr(k, N): Karcher mean, log map, tangent PCA
                     -> sphere / tangent-plane / timeline figure

Every preprocessing decision is a *function handed to the object*, so the
notebook decides what is done; the only dataset-specific code is the loaders.

Three time quantities (all in ms at the interface):
    count_bin_ms            spike-count resolution: the sum of Dirac deltas becomes
                            a count vector every count_bin_ms.
    mollification_bin_ms    how long the counts are accumulated for ONE covariance
                            matrix. Each such window gives one point on Gr(k, N).
                            None in "trials" mode = one window per whole trial.
    mollification_step_ms   stride between consecutive windows (default = window,
                            i.e. non-overlapping). Windows never cross a trial /
                            epoch boundary.

Grassmann geometry helpers are ported from
GrassmanianDraftScripts/grassmann_regimes.py (bases are N x k orthonormal
matrices; log map after Bendokat, Zimmermann & Absil 2024; Karcher 1977).

Typical use (see the notebooks):

    obj = RiemannSpikingAnalysis(mode="trials", count_bin_ms=10, mollification_bin_ms=500,
                                 unit_filter=my_unit_filter, trial_filter=my_trial_filter,
                                 normalize=zscore_units, k_rule=k_median_elbow())
    obj.add_session("alm", loader=load_alm_000363, asset=asset, h5=h5f, nwb=nwb)
    obj.load("alm"); obj.status()
    obj.prepare(); obj.window_spectra(); obj.choose_k()
    obj.plot_participation_ratio(); obj.plot_explained_variance(); obj.plot_eigenvalue_spectra()
    obj.fit_grassmann(); obj.plot_grassmann()
"""

from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

__all__ = [
    "RiemannSpikingAnalysis", "SessionRecord",
    "load_alm_000363", "load_hp_001695",
    "zscore_units", "identity", "sqrt_counts", "keep_all",
    "k_median_elbow", "k_fixed", "elbow_index", "intervals_by_column", "session_key_from_path",
    "principal_angles", "geodesic_distance", "grassmann_log", "grassmann_exp",
    "extrinsic_mean", "karcher_mean",
]


# ----------------------------------------------------------------------------
# Grassmann geometry (ported from grassmann_regimes.py; bases are N x k)
# ----------------------------------------------------------------------------

def principal_angles(A, B):
    """Principal angles between span(A) and span(B), ascending, shape (..., k).

    cos from svd(A^T B), sin from svd(B - A A^T B), combined with arctan2 so
    both small and large angles are accurate (Bjorck & Golub; Knyazev & Argentati).
    """
    M = np.swapaxes(A, -1, -2) @ B
    c = np.linalg.svd(M, compute_uv=False)                     # descending
    H = B - A @ M
    s = np.linalg.svd(H, compute_uv=False)[..., ::-1]          # ascending
    return np.arctan2(np.clip(s, 0, 1), np.clip(c, 0, 1))


def geodesic_distance(A, B):
    """Arc-length distance on Gr(k, N): sqrt(sum_i theta_i^2)."""
    return np.linalg.norm(principal_angles(A, B), axis=-1)


def grassmann_log(mu, U):
    """Log map at mu (N, k) of points U (..., N, k) -> tangent vectors (..., N, k).

    Stable form (Bendokat-Zimmermann-Absil): SVD of mu^T U = W C Z^T,
    H = U Z - mu W C has orthogonal columns of norm sin(theta),
    Delta = H diag(theta / sin theta) W^T.  ||Delta||_F = geodesic distance.
    """
    M = mu.T @ U
    Wm, C, Zt = np.linalg.svd(M)
    UZ = U @ np.swapaxes(Zt, -1, -2)
    H = UZ - mu @ (Wm * C[..., None, :])
    S = np.linalg.norm(H, axis=-2)
    theta = np.arctan2(S, C)
    safe = S > 1e-12
    fac = np.where(safe, theta / np.where(safe, S, 1.0), 1.0)
    return (H * fac[..., None, :]) @ np.swapaxes(Wm, -1, -2)


def grassmann_exp(mu, D):
    """Exp map at mu (N, k) of a tangent vector D (N, k) (Edelman-Arias-Smith)."""
    Q, s, Rt = np.linalg.svd(D, full_matrices=False)
    Y = (mu @ Rt.T) * np.cos(s) @ Rt + (Q * np.sin(s)) @ Rt
    Y, _ = np.linalg.qr(Y)
    return Y


def extrinsic_mean(U, k):
    """Top-k eigenvectors of the mean projection (1/n) sum_w U_w U_w^T."""
    N = U.shape[1]
    Uf = np.ascontiguousarray(np.swapaxes(U, 0, 1)).reshape(N, -1)   # (N, n*k)
    P = (Uf @ Uf.T) / len(U)
    w, V = np.linalg.eigh(P)
    return V[:, ::-1][:, :k]


def karcher_mean(U, k, n_iter=30, tol=1e-7, chunk=4000, init=None):
    """Riemannian centre of mass of U (n, N, k). Returns (mu, gradient-norm history)."""
    mu = extrinsic_mean(U, k) if init is None else init
    hist = []
    for _ in range(n_iter):
        D = np.zeros_like(mu)
        for a in range(0, len(U), chunk):
            D += grassmann_log(mu, U[a:a + chunk]).sum(axis=0)
        D /= len(U)
        nd = float(np.linalg.norm(D))
        hist.append(nd)
        mu = grassmann_exp(mu, D)
        if nd < tol:
            break
    return mu, hist


# ----------------------------------------------------------------------------
# Preprocessing hooks: defaults the notebooks can swap for their own functions
# ----------------------------------------------------------------------------

def identity(x):
    """count_transform: leave the counts as they are."""
    return x


def sqrt_counts(counts):
    """count_transform: variance-stabilising square root of the counts."""
    return np.sqrt(counts)


def zscore_units(X):
    """normalize: z-score every neuron (column) across ALL retained bins.

    In "trials" mode the retained bins are the within-trial bins of every kept
    trial, so this is 'z-score each neuron across all trials'. Silent units get 0.
    """
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def keep_all(df):
    """unit_filter / trial_filter: keep every row."""
    return np.ones(len(df), dtype=bool)


def elbow_index(y, m=50):
    """Elbow of a descending curve: the point farthest below the chord through
    points 1..m (1-based index). Explained-variance curves are the eigenvalues
    scaled by their sum, so the elbow is the same on either."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    m = min(int(m), len(y))
    if m < 3:
        return 1
    y = y[:m]
    chord = y[0] + (y[-1] - y[0]) * np.arange(m) / (m - 1)
    return int(np.argmax(chord - y)) + 1


def k_median_elbow(m=50, k_max=None):
    """k_rule: elbow of every window's explained-variance curve, median across
    windows (rounded), optionally capped at k_max."""
    def rule(evr):
        ks = np.array([elbow_index(row, m) for row in evr])
        k = int(round(np.median(ks)))
        if k_max is not None:
            k = min(k, int(k_max))
        return max(k, 1)
    rule.__name__ = "median_elbow(m=%d%s)" % (m, "" if k_max is None else ", k_max=%d" % k_max)
    return rule


def k_fixed(K):
    """k_rule: always K."""
    def rule(evr):
        return int(K)
    rule.__name__ = "fixed(K=%d)" % K
    return rule


def intervals_by_column(trials, col, start="start_time", stop="stop_time"):
    """{value: (n, 2) array of [start, stop]} for every value of trials[col]."""
    out = {}
    for v, g in trials.groupby(trials[col].astype(str), sort=True):
        out[v] = g[[start, stop]].to_numpy(dtype=float)
    return out


# ----------------------------------------------------------------------------
# NWB / DANDI helpers (same conventions as the ALM notebook's survey cells)
# ----------------------------------------------------------------------------

def _decode(arr):
    return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                     for x in np.asarray(arr)])


def _region_name(loc):
    """000363 stores a JSON blob per electrode ({"brain_regions": "left ALM", ...});
    other datasets store the plain name."""
    s = str(loc).strip()
    if s.startswith("{"):
        try:
            return str(json.loads(s).get("brain_regions", "")).strip()
        except ValueError:
            pass
    return s


def _ragged(group, col):
    data = group[col][:]
    if col + "_index" in group:
        return np.split(data, group[col + "_index"][:-1])
    return [np.atleast_1d(v) for v in data]


def session_key_from_path(path):
    """'sub-480928/sub-480928_ses-20210128T133949_...nwb' -> '480928_20210128T133949'."""
    path = str(path)
    sub = re.search(r"sub-([^_/]+)", path)
    ses = re.search(r"ses-([^_/]+)", path)
    if sub and ses:
        return "%s_%s" % (sub.group(1), ses.group(1))
    return Path(path).stem


def _object_column(items):
    """A pandas object column holding one (ragged) array per row."""
    col = pd.Series(index=range(len(items)), dtype=object)
    for i, it in enumerate(items):
        col.iat[i] = it
    return col


# ----------------------------------------------------------------------------
# Loaders: dandi session -> standard tables. Each returns a dict with
#   units   DataFrame: unit_id, spike_times (sorted 1-D arrays), region, cell_type,
#           quality, n_spikes, rate_hz  (+ any extra per-unit metadata columns)
#   trials  DataFrame with start_time / stop_time (or None)
#   epochs  {name: (n, 2) array of [start, stop]}
#   events  {name: 1-D array of times}
#   t_range (t0, t1)
#   meta    dict
# ----------------------------------------------------------------------------

def load_alm_000363(rec, h5=None, nwb=None, units_df=None):
    """Dandiset 000363 (Chen et al. 2024), streamed from S3 with remfile.

    Pass h5= / nwb= if the notebook already opened the file (saves re-parsing);
    otherwise the asset on the record is streamed. Pass units_df= (a table with
    at least `spike_times` and `region`, e.g. the one built in the exploration
    cells) to reuse spike trains already streamed instead of reading the whole
    units table again (~hundreds of MB per session). Units carry region (from
    the electrode JSON), quality ('good' / 'unlabelled' in `classification`)
    and no cell-type label. Trials come from nwb.trials; task events from
    acquisition/BehavioralEvents.
    """
    import h5py
    import pynwb

    h5 = h5 if h5 is not None else rec.h5
    if h5 is None:
        import remfile
        if rec.asset is None:
            raise ValueError("record %r has neither an open h5 file nor a dandi asset" % rec.key)
        url = rec.asset.get_content_url(follow_redirects=1, strip_query=True)
        h5 = h5py.File(remfile.File(url), "r")
    rec.h5 = h5
    if nwb is None:
        io = pynwb.NWBHDF5IO(file=h5, load_namespaces=True)
        nwb = io.read()
        rec.io = io
    rec.raw = nwb

    if units_df is not None:                       # reuse already-streamed spike trains
        src = units_df.reset_index(drop=True)
        spikes = [np.sort(np.asarray(s, dtype=float)) for s in src["spike_times"]]
        units = pd.DataFrame({
            "unit_id": src["unit_id"].values if "unit_id" in src else np.arange(len(src)),
            "region": src["region"].astype(object).values if "region" in src else "",
            "cell_type": src["cell_type"].astype(object).values if "cell_type" in src else "",
            "quality": src["quality"].astype(object).values if "quality" in src else "",
            "n_spikes": [len(s) for s in spikes]})
        units["spike_times"] = _object_column(spikes)
        for col in src.columns:                    # keep any other per-unit columns
            if col not in units:
                units[col] = src[col].values
    else:
        ug = h5["units"]
        spikes = [np.sort(np.asarray(s, dtype=float)) for s in _ragged(ug, "spike_times")]
        loc = np.array([_region_name(s) for s in
                        _decode(h5["general/extracellular_ephys/electrodes/location"][:])])
        region = np.array([loc[int(e[0])] if len(e) else "" for e in _ragged(ug, "electrodes")],
                          dtype=object)
        quality = np.full(len(spikes), "", dtype=object)
        for cand in ("classification", "unit_quality", "quality", "label"):
            if cand in ug:
                quality = np.array([s.strip() for s in _decode(ug[cand][:])], dtype=object)
                break
        units = pd.DataFrame({"unit_id": np.asarray(ug["id"][:]),
                              "region": region,
                              "cell_type": "",                   # no cell labels in 000363
                              "quality": quality,
                              "n_spikes": [len(s) for s in spikes]})
        units["spike_times"] = _object_column(spikes)

    trials = nwb.trials.to_dataframe().reset_index(drop=True)
    t0, t1 = float(trials["start_time"].min()), float(trials["stop_time"].max())
    units["rate_hz"] = units["n_spikes"] / max(t1 - t0, 1e-9)

    events = {}
    beh = nwb.acquisition.get("BehavioralEvents")
    if beh is not None:
        for name, ts in beh.time_series.items():
            events[name] = np.asarray(ts.timestamps[:], dtype=float)

    epochs = {"trial": trials[["start_time", "stop_time"]].to_numpy(dtype=float)}
    if "trial_instruction" in trials:
        for v, iv in intervals_by_column(trials, "trial_instruction").items():
            epochs["instruction=%s" % v] = iv

    meta = {"session_description": str(nwb.session_description),
            "subject": getattr(getattr(nwb, "subject", None), "subject_id", ""),
            "units_in_file": len(units),
            "units_from": "units_df (pre-filtered table)" if units_df is not None else "file",
            "quality_values": {str(k): int(v) for k, v in
                               units["quality"].astype(str).value_counts().items()}}
    return dict(units=units, trials=trials, epochs=epochs, events=events,
                t_range=(t0, t1), meta=meta)


def load_hp_001695(rec, download_dir=None, region_col="cell_area", type_col="cell_type"):
    """Dandiset 001695 (Buzsaki lab, maze + sleep), read with pynapple from a
    local copy (downloaded into download_dir if the record has no local file).

    Units carry region (cell_area: CA1 / CA3 / RSC), cell_type (pyramidal /
    narrow / wide interneuron) and all other per-unit metadata columns.
    No trials; epochs = sleep intervals (SleepStates) and the maze epoch
    (span of the Position samples).
    """
    import pynapple as nap

    path = rec.local_path
    if path is None or not Path(path).exists():
        if rec.asset is None:
            raise FileNotFoundError("record %r: no local file and no dandi asset" % rec.key)
        if download_dir is None:
            raise ValueError("download_dir is needed to fetch %s" % rec.asset.path)
        path = Path(download_dir) / rec.asset.path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            rec.asset.download(path)
    rec.local_path = Path(path)
    data = nap.load_file(str(path))
    rec.raw = data

    tsg = data["units"]
    ids = np.asarray(tsg.index)
    spikes = [np.sort(np.asarray(tsg[u].index.values, dtype=float)) for u in ids]

    def info(col):
        try:
            return np.asarray(tsg.get_info(col))
        except Exception:
            return np.full(len(ids), "", dtype=object)

    units = pd.DataFrame({"unit_id": ids,
                          "region": info(region_col).astype(object),
                          "cell_type": info(type_col).astype(object),
                          "quality": "curated",
                          "n_spikes": [len(s) for s in spikes]})
    units["spike_times"] = _object_column(spikes)
    for col in list(getattr(tsg, "metadata_columns", [])):
        if col not in (region_col, type_col):
            units[col] = info(col)

    nonempty = [s for s in spikes if len(s)]
    t0 = float(min(s[0] for s in nonempty))
    t1 = float(max(s[-1] for s in nonempty))
    units["rate_hz"] = units["n_spikes"] / max(t1 - t0, 1e-9)

    epochs = {}
    if "SleepStates" in data.keys():
        ep = data["SleepStates"]
        epochs["sleep"] = np.column_stack([np.asarray(ep.start, float), np.asarray(ep.end, float)])
    if "Position" in data.keys():
        pos = data["Position"]
        epochs["maze"] = np.array([[float(pos.index[0]), float(pos.index[-1])]])

    meta = {"file": str(path), "pynapple_keys": list(data.keys()), "units_in_file": len(units)}
    return dict(units=units, trials=None, epochs=epochs, events={},
                t_range=(t0, t1), meta=meta)


# ----------------------------------------------------------------------------
# Session registry
# ----------------------------------------------------------------------------

@dataclass
class SessionRecord:
    """One session: where it comes from, whether it is in memory, and its tables."""
    key: str
    loader: Callable
    asset: object = None                 # dandi asset (available to load)
    path: str = ""                       # asset path inside the dandiset
    local_path: Optional[Path] = None    # local NWB copy, if any
    size_gb: float = np.nan
    loader_kwargs: dict = field(default_factory=dict)
    h5: object = None                    # open h5py.File (streaming), if any
    io: object = None                    # pynwb IO, if any
    raw: object = None                   # loader's native object (NWBFile / pynapple dict)
    loaded: bool = False
    units: Optional[pd.DataFrame] = None
    trials: Optional[pd.DataFrame] = None
    epochs: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    t_range: tuple = (np.nan, np.nan)
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Main object
# ----------------------------------------------------------------------------

class RiemannSpikingAnalysis:
    """Sessions -> windowed covariance spectra -> Gr(k, N) embedding.

    Parameters (all preprocessing decisions are functions, see module docstring)
    ----------
    mode                    "trials": bins and windows live inside each kept trial.
                            "full":   the whole recording (or `t_range`) is one
                                      stretch, optionally cut into the intervals
                                      of a named epoch.
    count_bin_ms            spike-count bin (mollifier) width.
    mollification_bin_ms    window length per covariance matrix; None = whole trial.
    mollification_step_ms   stride between windows; None = window length.
    unit_filter             f(units DataFrame) -> bool mask of units to keep.
    trial_filter            f(trials DataFrame) -> bool mask of trials to keep.
    count_transform         f(counts (T_b, N) float32) -> array, e.g. sqrt_counts.
    normalize               f(transformed counts) -> (T_b, N); default z-score per
                            neuron across all retained bins.
    k_rule                  f(evr (n_win, n_eig)) -> int; default k_median_elbow().
    k_store                 eigenvectors kept per window (k must be <= k_store).
    center_windows          subtract each window's own mean before the covariance.
    t_range                 "full" mode only: None = whole recording, (t0, t1), or
                            the name of an epoch of the session (e.g. "maze").
    """

    def __init__(self, mode="trials", count_bin_ms=10.0, mollification_bin_ms=500.0,
                 mollification_step_ms=None, unit_filter=keep_all, trial_filter=keep_all,
                 count_transform=identity, normalize=zscore_units, k_rule=None,
                 k_store=15, center_windows=True, t_range=None, verbose=True,
                 output_dir=None, run_name=None, session_cache_dir=None, use_cache=True,
                 save_figures=False):
        if mode not in ("trials", "full"):
            raise ValueError("mode must be 'trials' or 'full'")
        self.mode = mode
        self.count_bin_ms = float(count_bin_ms)
        self.mollification_bin_ms = None if mollification_bin_ms is None else float(mollification_bin_ms)
        self.mollification_step_ms = mollification_step_ms
        self.unit_filter = unit_filter
        self.trial_filter = trial_filter
        self.count_transform = count_transform
        self.normalize = normalize
        self.k_rule = k_rule if k_rule is not None else k_median_elbow()
        self.k_store = int(k_store)
        self.center_windows = bool(center_windows)
        self.t_range = t_range
        self.verbose = verbose
        self.run_name = run_name
        # caches only (nothing is written that could not be recomputed):
        #   output_dir/<session>/spectra_<hash>.npz, grassmann_<hash>.npz   per run
        #   session_cache_dir/<session>.pkl                                   loaded tables, shared by runs
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.session_cache_dir = None if session_cache_dir is None else Path(session_cache_dir)
        self.use_cache = bool(use_cache)
        self.save_figures = bool(save_figures)    # off by default: figures only display
        self.session_data: dict[str, SessionRecord] = {}
        self.analysis: dict[str, SimpleNamespace] = {}
        self.active: Optional[str] = None

    # ------------------------------------------------------------ utilities
    def _say(self, *a):
        if self.verbose:
            print(*a)

    @staticmethod
    def _fname(f):
        return getattr(f, "__name__", repr(f))

    def _save_fig(self, fig, name, key):
        """Save a figure as <output_dir>/<session>/<name>.png. No-op unless
        save_figures=True and output_dir is set."""
        if not self.save_figures or self.output_dir is None or not name:
            return None
        d = self.output_dir / str(key)
        d.mkdir(parents=True, exist_ok=True)
        path = d / ("%s.png" % name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        self._say("  saved figure:", path)
        return path

    def config(self):
        """The run configuration as a JSON-able dict."""
        return {"run_name": self.run_name, "mode": self.mode, "t_range": self.t_range,
                "count_bin_ms": self.count_bin_ms, "mollification_bin_ms": self.mollification_bin_ms,
                "mollification_step_ms": self.mollification_step_ms,
                "unit_filter": self._fname(self.unit_filter), "trial_filter": self._fname(self.trial_filter),
                "count_transform": self._fname(self.count_transform), "normalize": self._fname(self.normalize),
                "k_rule": self._fname(self.k_rule), "k_store": self.k_store,
                "center_windows": self.center_windows,
                "output_dir": None if self.output_dir is None else str(self.output_dir),
                "save_figures": self.save_figures}

    # ------------------------------------------------------------- caches
    def _data_hash(self, st):
        """Fingerprint of the prepared data + the settings that shape the spectra.
        Hashes the data itself (kept unit ids, segments, per-unit counts, a sample
        of X), so changing REGION / MIN_RATE_HZ inside a filter function is caught
        even though the function name is unchanged."""
        import hashlib
        h = hashlib.sha1()
        cfg = {k: v for k, v in self.config().items()
               if k not in ("run_name", "output_dir", "save_figures", "k_rule")}
        h.update(json.dumps(cfg, sort_keys=True, default=str).encode())
        h.update(np.asarray(st.units["unit_id"].astype(str).values).astype("U").tobytes())
        h.update(np.ascontiguousarray(st.segments, dtype=np.float64).tobytes())
        h.update(st.counts.sum(axis=0, dtype=np.int64).tobytes())
        step = max(1, st.X.shape[0] // 2000)
        h.update(np.ascontiguousarray(st.X[::step]).tobytes())
        return h.hexdigest()[:12]

    def _cache_file(self, key, stage, h):
        if not self.use_cache or self.output_dir is None:
            return None
        d = self.output_dir / str(key)
        d.mkdir(parents=True, exist_ok=True)
        return d / ("%s_%s.npz" % (stage, h))

    def describe(self):
        """Print the current configuration."""
        step = self.mollification_step_ms
        print("RiemannSpikingAnalysis", "| run_name = %r" % self.run_name)
        print("  run cache (spectra, Gr fit):", self.output_dir if self.use_cache else "off")
        print("  session cache (loaded data):", self.session_cache_dir if self.use_cache else "off")
        print("  figures                     : display only" if not self.save_figures
              else "  figures                     : displayed and saved to output_dir")
        print("  mode                  :", self.mode,
              "" if self.mode == "trials" else "(t_range=%r)" % (self.t_range,))
        print("  count_bin_ms          :", self.count_bin_ms)
        print("  mollification_bin_ms  :", self.mollification_bin_ms,
              "(one window per whole trial)" if self.mollification_bin_ms is None else "")
        print("  mollification_step_ms :", step if step is not None else "= window (non-overlapping)")
        print("  unit_filter           :", self._fname(self.unit_filter))
        print("  trial_filter          :", self._fname(self.trial_filter))
        print("  count_transform       :", self._fname(self.count_transform))
        print("  normalize             :", self._fname(self.normalize))
        print("  k_rule                :", self._fname(self.k_rule), "| k_store =", self.k_store)
        print("  center_windows        :", self.center_windows)
        print("  sessions              : %d registered, %d loaded, %d analysed, active=%r"
              % (len(self.session_data), sum(r.loaded for r in self.session_data.values()),
                 len(self.analysis), self.active))

    # ------------------------------------------------------ session registry
    def add_session(self, key, loader, asset=None, path=None, local_path=None,
                    h5=None, nwb=None, **loader_kwargs):
        """Register a session as *available*. Nothing is read until load(key)."""
        rec = SessionRecord(key=key, loader=loader, asset=asset,
                            path=path or (asset.path if asset is not None else ""),
                            local_path=None if local_path is None else Path(local_path),
                            size_gb=(asset.size / 1e9) if (asset is not None and getattr(asset, "size", None)) else np.nan,
                            loader_kwargs=dict(loader_kwargs), h5=h5)
        if nwb is not None:
            rec.loader_kwargs["nwb"] = nwb
        self.session_data[key] = rec
        if self.active is None:
            self.active = key
        return rec

    def register_dandiset(self, dandiset_id, version, loader, path_filter=None,
                          limit=None, client=None, **loader_kwargs):
        """Register every .nwb asset of a dandiset as available (no data read).
        Keep the client alive (no `with`) if you plan to load from these assets."""
        from dandi.dandiapi import DandiAPIClient
        client = client or DandiAPIClient()
        ds = client.get_dandiset(dandiset_id, version)
        n = 0
        for a in ds.get_assets(order="path"):
            if not a.path.endswith(".nwb"):
                continue
            if path_filter is not None and not path_filter(a.path):
                continue
            key = session_key_from_path(a.path)
            if key in self.session_data:
                continue
            self.add_session(key, loader, asset=a, **loader_kwargs)
            n += 1
            if limit is not None and n >= limit:
                break
        self._say("registered %d sessions from %s/%s (%d total)"
                  % (n, dandiset_id, version, len(self.session_data)))
        return n

    def load(self, key=None, refresh=False, **loader_kwargs):
        """Run the record's loader, store the standard tables, mark it loaded.

        With session_cache_dir set, the standard tables (spike trains, trials,
        epochs, events) are pickled to <session_cache_dir>/<key>.pkl on the
        first load and read back afterwards, skipping the stream / NWB parse.
        Not cached when units_df= is given (that table is already filtered).
        refresh=True ignores and rewrites the cache."""
        rec = self._rec(key)
        kw = dict(rec.loader_kwargs)
        kw.update(loader_kwargs)
        t = time.time()
        cache = None
        if self.use_cache and self.session_cache_dir is not None and kw.get("units_df") is None:
            cache = self.session_cache_dir / ("%s.pkl" % rec.key)
        if cache is not None and cache.exists() and not refresh:
            out = pd.read_pickle(cache)
            self._say("session %r read from cache %s" % (rec.key, cache))
        else:
            out = rec.loader(rec, **kw)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                pd.to_pickle({k: out.get(k) for k in ("units", "trials", "epochs", "events",
                                                      "t_range", "meta")}, cache)
                self._say("session %r cached to %s" % (rec.key, cache))
        rec.units = out["units"].reset_index(drop=True)
        rec.trials = out.get("trials")
        rec.epochs = dict(out.get("epochs") or {})
        rec.events = dict(out.get("events") or {})
        rec.t_range = tuple(out["t_range"])
        rec.meta = dict(out.get("meta") or {})
        rec.loaded = True
        self.active = rec.key
        self._say("loaded %r in %.1fs: %d units, %s trials, epochs=%s, t=[%.1f, %.1f] s"
                  % (rec.key, time.time() - t, len(rec.units),
                     "no" if rec.trials is None else len(rec.trials),
                     list(rec.epochs), rec.t_range[0], rec.t_range[1]))
        return rec

    def unload(self, key=None, drop_analysis=True):
        """Free a session's tables (and analysis) but keep it registered."""
        rec = self._rec(key)
        rec.units = rec.trials = rec.raw = None
        rec.epochs, rec.events = {}, {}
        for attr in ("io", "h5"):
            obj = getattr(rec, attr)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
                setattr(rec, attr, None)
        rec.loaded = False
        if drop_analysis:
            self.analysis.pop(rec.key, None)
        if self.active == rec.key:
            self.active = next((k for k, r in self.session_data.items() if r.loaded), None)

    def select(self, key):
        """Make `key` the session the analysis methods act on."""
        if key not in self.session_data:
            raise KeyError(key)
        self.active = key
        return self.session_data[key]

    def status(self):
        """DataFrame: what is registered, what is in memory, how far each got."""
        rows = []
        for k, r in self.session_data.items():
            st = self.analysis.get(k)
            rows.append({
                "key": k, "active": k == self.active, "loader": self._fname(r.loader),
                "path": r.path, "size_gb": r.size_gb,
                "local_file": bool(r.local_path is not None and Path(r.local_path).exists()),
                "loaded": r.loaded,
                "n_units": len(r.units) if r.loaded else None,
                "n_trials": (len(r.trials) if (r.loaded and r.trials is not None) else None),
                "epochs": ",".join(r.epochs) if r.loaded else "",
                "prepared": st is not None and hasattr(st, "X"),
                "n_windows": len(st.windows) if (st is not None and hasattr(st, "windows")) else None,
                "k": getattr(st, "k", None) if st is not None else None,
                "grassmann": st is not None and hasattr(st, "mu"),
            })
        return pd.DataFrame(rows)

    def _rec(self, key=None):
        key = self.active if key is None else key
        if key is None or key not in self.session_data:
            raise KeyError("no session %r; add_session() / register_dandiset() first" % key)
        return self.session_data[key]

    def _st(self, key=None):
        rec = self._rec(key)
        if rec.key not in self.analysis:
            raise RuntimeError("session %r: run prepare() first" % rec.key)
        return self.analysis[rec.key]

    # ------------------------------------------------- 1. units, bins, X
    def prepare(self, key=None):
        """Apply unit / trial filters, cut the recording into segments (trials or
        epochs), bin spikes at count_bin_ms, transform, normalise -> X (T_b, N)."""
        rec = self._rec(key)
        if not rec.loaded:
            self.load(rec.key)
        st = SimpleNamespace(key=rec.key)

        keep = np.asarray(self.unit_filter(rec.units), dtype=bool)
        st.units = rec.units[keep].reset_index(drop=True)
        if len(st.units) == 0:
            raise ValueError("unit_filter kept no units")

        st.trials = None
        if self.mode == "trials":
            if rec.trials is None:
                raise ValueError("session %r has no trials table; use mode='full'" % rec.key)
            tk = np.asarray(self.trial_filter(rec.trials), dtype=bool)
            st.trials = rec.trials[tk].reset_index(drop=True)
            segments = st.trials[["start_time", "stop_time"]].to_numpy(dtype=float)
        else:
            if self.t_range is None:
                segments = np.array([rec.t_range], dtype=float)
            elif isinstance(self.t_range, str):
                if self.t_range not in rec.epochs:
                    raise KeyError("epoch %r not in %s" % (self.t_range, list(rec.epochs)))
                segments = np.asarray(rec.epochs[self.t_range], dtype=float)
            else:
                segments = np.atleast_2d(np.asarray(self.t_range, dtype=float))
        st.segments = segments

        dt = self.count_bin_ms / 1000.0
        spikes = [np.asarray(s, dtype=float) for s in st.units["spike_times"]]
        blocks, centers, seg_of_bin = [], [], []
        t = time.time()
        for si, (s, e) in enumerate(segments):
            n = int(np.floor((e - s) / dt + 1e-9))
            if n < 1:
                continue
            blocks.append(_bin_segment(spikes, s, n, dt))
            centers.append(s + dt * (np.arange(n) + 0.5))
            seg_of_bin.append(np.full(n, si, dtype=np.int64))
        if not blocks:
            raise ValueError("no segment is at least one bin long")
        st.counts = np.concatenate(blocks, axis=0)                  # (T_b, N) uint16
        st.bin_centers = np.concatenate(centers)
        st.bin_seg = np.concatenate(seg_of_bin)
        X = self.count_transform(st.counts.astype(np.float32))
        st.X = np.ascontiguousarray(np.asarray(self.normalize(X), dtype=np.float32))
        self.analysis[rec.key] = st
        self.active = rec.key

        n_units_raw = len(rec.units)
        n_seg_used = len(np.unique(st.bin_seg))
        self._say("prepared %r (%s mode) in %.1fs" % (rec.key, self.mode, time.time() - t))
        self._say("  units: %d of %d kept by %s" % (len(st.units), n_units_raw, self._fname(self.unit_filter)))
        if st.trials is not None:
            self._say("  trials: %d of %d kept by %s, %d at least one bin long"
                      % (len(st.trials), len(rec.trials), self._fname(self.trial_filter), n_seg_used))
        else:
            self._say("  segments: %d (t_range=%r), %.1f min in total"
                      % (len(segments), self.t_range, np.sum(segments[:, 1] - segments[:, 0]) / 60))
        self._say("  bins: %d x %d units at %g ms (%.1f min of data); transform=%s, normalize=%s"
                  % (st.X.shape[0], st.X.shape[1], self.count_bin_ms, st.X.shape[0] * dt / 60,
                     self._fname(self.count_transform), self._fname(self.normalize)))
        return st

    # ------------------------------------------- 2. windowed eigenspectra
    def window_spectra(self, key=None, mem_budget_mb=256, min_eig_frac=1e-8):
        """One covariance per window of mollification_bin_ms (stride
        mollification_step_ms), never crossing a segment; full eigendecomposition.

        Stores per window: all eigenvalues (NaN-padded if window lengths differ),
        explained-variance fractions lambda_i / sum(lambda), participation ratio
        (sum lambda)^2 / sum lambda^2, the top k_store eigenvectors, rank, validity.
        """
        st = self._st(key)
        Tb, N = st.X.shape
        dt = self.count_bin_ms / 1000.0

        # window placement, segment by segment
        seg_ids, first = np.unique(st.bin_seg, return_index=True)
        last = np.r_[first[1:], Tb]
        starts, lens, segs = [], [], []
        for si, b0, b1 in zip(seg_ids, first, last):
            n = b1 - b0
            if self.mollification_bin_ms is None:
                if n >= 2:
                    starts.append(b0); lens.append(n); segs.append(si)
                continue
            L = int(round(self.mollification_bin_ms / self.count_bin_ms))
            step_ms = self.mollification_step_ms
            S = L if step_ms is None else max(1, int(round(step_ms / self.count_bin_ms)))
            if L < 2:
                raise ValueError("mollification_bin_ms must cover at least 2 count bins")
            if n < L:
                continue
            s0 = b0 + S * np.arange((n - L) // S + 1)
            starts.extend(s0.tolist()); lens.extend([L] * len(s0)); segs.extend([si] * len(s0))
        if not starts:
            raise ValueError("no segment is long enough for one window")
        starts = np.asarray(starts); lens = np.asarray(lens); segs = np.asarray(segs)
        win = pd.DataFrame({"start_bin": starts, "n_bins": lens, "seg": segs})
        win["t_start"] = st.bin_centers[starts] - dt / 2
        win["t_stop"] = win["t_start"] + lens * dt
        win["t_center"] = 0.5 * (win["t_start"] + win["t_stop"])
        st.windows = win
        n_win = len(win)

        r_max = int(min(lens.max(), N))
        kk_max = min(self.k_store, r_max)
        eigvals = np.full((n_win, r_max), np.nan)
        eigvecs = np.zeros((n_win, N, kk_max), dtype=np.float32)
        rank = np.zeros(n_win, dtype=int)

        t = time.time()
        st.spectra_hash = self._data_hash(st)
        cache = self._cache_file(st.key, "spectra", st.spectra_hash)
        hit = cache is not None and cache.exists()
        if hit:                                    # same data + settings: skip the eigendecompositions
            with np.load(cache) as z:
                eigvals, eigvecs, rank = z["eigvals"], z["eigvecs"], z["rank"]
            self._say("window spectra read from cache %s" % cache)
        for L in (() if hit else np.unique(lens)):
            idx_w = np.flatnonzero(lens == L)
            per_win = 8 * (3 * L * N + 2 * min(L, N) ** 2)
            chunk = max(1, int(mem_budget_mb * 2 ** 20 // per_win))
            for a in range(0, len(idx_w), chunk):
                ii = idx_w[a:a + chunk]
                take = starts[ii][:, None] + np.arange(L)[None, :]
                Xw = st.X[take].astype(np.float64)                        # (n, L, N)
                if self.center_windows:
                    Xw -= Xw.mean(axis=1, keepdims=True)
                lam, V, rk = _window_eig(Xw, kk_max, min_eig_frac)
                eigvals[ii, :lam.shape[1]] = lam
                eigvecs[ii, :, :V.shape[2]] = V
                rank[ii] = rk
        if cache is not None and not hit:
            np.savez(cache, eigvals=eigvals, eigvecs=eigvecs, rank=rank)
        st.eigvals = eigvals
        st.eigvecs = eigvecs
        st.rank = rank
        with np.errstate(invalid="ignore", divide="ignore"):
            tot = np.nansum(eigvals, axis=1)
            st.evr = eigvals / tot[:, None]                                  # explained variance
            st.participation_ratio = tot ** 2 / np.nansum(eigvals ** 2, axis=1)
        st.total_variance = tot
        st.valid = np.isfinite(tot) & (tot > 0) & (rank >= 1)
        st.elbow_per_window = np.array([elbow_index(r) if v else 0 for r, v in zip(st.evr, st.valid)])

        L_desc = ("whole trial (%d-%d bins)" % (lens.min(), lens.max()) if self.mollification_bin_ms is None
                  else "%g ms = %d bins" % (self.mollification_bin_ms, lens[0]))
        self._say("window spectra for %r in %.1fs: %d windows, window = %s, step = %s"
                  % (st.key, time.time() - t, n_win, L_desc,
                     "window" if self.mollification_step_ms is None else "%g ms" % self.mollification_step_ms))
        self._say("  N = %d units; covariance rank <= min(bins%s, N) -> median rank %d; %d windows invalid"
                  % (N, " - 1" if self.center_windows else "", int(np.median(rank)), int((~st.valid).sum())))
        if np.median(rank) < N:
            self._say("  note: windows have fewer bins than units, so each spectrum has at most"
                      " %d non-zero eigenvalues" % int(np.median(rank)))
        return win

    # --------------------------------------------------------- 3. choose k
    def choose_k(self, rule=None, key=None):
        """Subspace dimension from k_rule (or `rule`) applied to the explained-
        variance curves of all valid windows. k is capped at k_store."""
        st = self._st(key)
        rule = self.k_rule if rule is None else rule
        evr = st.evr[st.valid]
        k = int(rule(evr))
        k_cap = min(self.k_store, st.eigvecs.shape[2])
        if k > k_cap:
            self._say("  k=%d capped at k_store=%d (raise k_store and rerun window_spectra for more)" % (k, k_cap))
            k = k_cap
        st.k = max(1, k)
        st.k_rule_name = self._fname(rule)
        el = st.elbow_per_window[st.valid]
        q = np.percentile(el, [25, 50, 75])
        cum = np.nanmedian(np.nansum(evr[:, :st.k], axis=1))
        self._say("k = %d by %s | per-window elbows: median %d, IQR %d-%d | median explained"
                  " variance in top k: %.1f%%" % (st.k, st.k_rule_name, q[1], q[0], q[2], 100 * cum))
        return st.k

    # ------------------------------------------------------- spectrum plots
    def plot_participation_ratio(self, key=None, bins=40, ax=None, save_as="participation_ratio"):
        """Histogram of the participation ratio (sum lambda)^2 / sum lambda^2 over windows."""
        st = self._st(key)
        pr = st.participation_ratio[st.valid]
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(pr, bins=bins, color="0.55", edgecolor="white")
        med = np.median(pr)
        ax.axvline(med, color="k", ls="--", lw=1.2, label="median = %.1f" % med)
        ax.axvline(np.percentile(pr, 25), color="0.3", ls=":", lw=0.8)
        ax.axvline(np.percentile(pr, 75), color="0.3", ls=":", lw=0.8, label="quartiles")
        if hasattr(st, "k"):
            ax.axvline(st.k, color="tab:red", lw=1, label="k = %d" % st.k)
        ax.set_xlabel("participation ratio of the window covariance")
        ax.set_ylabel("windows")
        ax.set_title("%s: participation ratio, %d windows" % (st.key, len(pr)), fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        self._save_fig(ax.figure, save_as, st.key)
        return ax

    def plot_explained_variance(self, key=None, n_show=10, ax=None, save_as="explained_variance"):
        """Violin per component of lambda_i / sum(lambda) across windows (descending
        order, NOT cumulative); medians marked and traced by a line."""
        st = self._st(key)
        evr = st.evr[st.valid]
        n_show = int(min(n_show, evr.shape[1]))
        data = [evr[:, i][np.isfinite(evr[:, i])] for i in range(n_show)]
        n_show = sum(len(d) >= 2 for d in data)          # drop components no window reaches
        data = data[:n_show]
        pos = np.arange(1, n_show + 1)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        parts = ax.violinplot(data, positions=pos, widths=0.85, showmeans=False,
                              showmedians=False, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor("tab:blue"); b.set_edgecolor("none"); b.set_alpha(0.35)
        med = np.array([np.median(d) for d in data])
        ax.plot(pos, med, "o-", color="k", ms=4, lw=1.2, label="median across windows")
        if hasattr(st, "k"):
            ax.axvline(st.k + 0.5, color="tab:red", ls="--", lw=1, label="k = %d" % st.k)
        ax.set_xticks(pos)
        ax.set_xlabel("eigenvector rank (largest eigenvalue first)")
        ax.set_ylabel("explained variance  $\\lambda_i / \\sum_j \\lambda_j$")
        ax.set_title("%s: explained variance of the top %d components, %d windows"
                     % (st.key, n_show, len(evr)), fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        self._save_fig(ax.figure, save_as, st.key)
        return ax

    def plot_eigenvalue_spectra(self, key=None, max_lines=3000, log=True, ax=None, save_as="eigenvalue_spectra"):
        """Raw eigenvalues over all dimensions: one line per window, coloured by
        window time, median spectrum in black."""
        st = self._st(key)
        ok = np.flatnonzero(st.valid)
        if len(ok) > max_lines:
            ok = ok[np.linspace(0, len(ok) - 1, max_lines).astype(int)]
        ev = st.eigvals[ok]
        tt = st.windows["t_center"].values[ok]
        x = np.arange(1, ev.shape[1] + 1)
        if log:   # numerically-zero eigenvalues (rank-deficient windows) are not drawn
            floor = 1e-10 * np.nanmax(ev, axis=1, keepdims=True)
            ev = np.where(ev > floor, ev, np.nan)
        segs = [np.column_stack([x[np.isfinite(r)], r[np.isfinite(r)]]) for r in ev]
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        lc = LineCollection(segs, cmap="viridis", linewidths=0.4, alpha=0.35)
        lc.set_array(tt)
        ax.add_collection(lc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(st.eigvals[st.valid], axis=0)
        if log:
            med = np.where(med > 1e-10 * np.nanmax(med), med, np.nan)
        ax.plot(x, med, color="k", lw=1.5, label="median spectrum")
        if hasattr(st, "k"):
            ax.axvline(st.k, color="tab:red", ls="--", lw=1, label="k = %d" % st.k)
        ax.set_xlim(1, ev.shape[1])
        finite = ev[np.isfinite(ev)]
        if finite.size:
            ax.set_ylim(finite.min() * 0.8, finite.max() * 1.2)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("component (all %d dimensions of the window covariance)" % ev.shape[1])
        ax.set_ylabel("eigenvalue" + (" (log; zeros not drawn)" if log else ""))
        ax.set_title("%s: eigenvalue spectrum of every window (%d of %d shown)"
                     % (st.key, len(ok), int(st.valid.sum())), fontsize=10)
        cb = plt.colorbar(lc, ax=ax, pad=0.01)
        cb.set_label("window time (s)")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        plt.tight_layout()
        self._save_fig(ax.figure, save_as, st.key)
        return ax

    # ------------------------------------------------- 4. Gr(k, N) embedding
    def fit_grassmann(self, k=None, mean="karcher", n_tangent_pcs=3, karcher_iters=50,
                      karcher_tol=1e-6, chunk=4000, key=None):
        """Top-k eigenvectors of every valid window as a point of Gr(k, N); Karcher
        (or extrinsic) mean; log map at the mean; PCA of the tangent vectors;
        geodesic distance to the mean; angle between consecutive windows."""
        st = self._st(key)
        if k is None:
            if not hasattr(st, "k"):
                raise RuntimeError("run choose_k() first or pass k=")
            k = st.k
        else:
            k = int(k)
            if k != getattr(st, "k", None):
                st.k_rule_name = "given k=%d" % k
        if k > st.eigvecs.shape[2]:
            raise ValueError("k=%d > k_store=%d; rerun window_spectra with a larger k_store" % (k, st.eigvecs.shape[2]))
        tot = np.where(np.isfinite(st.total_variance), st.total_variance, 0)
        lam_k = st.eigvals[:, k - 1] if st.eigvals.shape[1] >= k else np.full(len(tot), np.nan)
        valid = st.valid & (st.rank >= k) & np.isfinite(lam_k) & (lam_k > 1e-8 * tot)
        idx = np.flatnonzero(valid)
        if len(idx) < 2:
            raise ValueError("fewer than 2 windows have rank >= k=%d" % k)
        t = time.time()
        U = st.eigvecs[idx][:, :, :k].astype(np.float64)
        U, _ = np.linalg.qr(U)                                     # re-orthonormalise (batched)

        if mean not in ("karcher", "extrinsic"):
            raise ValueError("mean must be 'karcher' or 'extrinsic'")
        # the mean is the costly step; cache it (keyed on the spectra + mean settings)
        mh = "%s_k%d_%s_%d_%g" % (getattr(st, "spectra_hash", "nohash"), k, mean, karcher_iters, karcher_tol)
        cache = self._cache_file(st.key, "grassmann_mean", mh) if hasattr(st, "spectra_hash") else None
        if cache is not None and cache.exists():
            with np.load(cache) as z:
                mu, hist = z["mu"], list(z["hist"])
            self._say("Gr mean read from cache %s" % cache)
        else:
            if mean == "karcher":
                mu, hist = karcher_mean(U, k, n_iter=karcher_iters, tol=karcher_tol, chunk=chunk)
            else:
                mu, hist = extrinsic_mean(U, k), []
            if cache is not None:
                np.savez(cache, mu=mu, hist=np.asarray(hist, dtype=float))

        N = U.shape[1]
        Delta = np.empty_like(U)
        angles = np.empty((len(U), k))
        for a in range(0, len(U), chunk):
            Delta[a:a + chunk] = grassmann_log(mu, U[a:a + chunk])
            angles[a:a + chunk] = principal_angles(mu, U[a:a + chunk])
        dist = np.linalg.norm(Delta.reshape(len(U), -1), axis=1)

        # tangent PCA (not centred: the origin is the mean subspace itself)
        T = Delta.reshape(len(U), -1)
        total_var = float((T ** 2).sum())
        p = int(min(n_tangent_pcs, min(T.shape)))
        coords, var_ratio = _top_pcs(T, p, total_var)

        # angle between consecutive windows of the same segment
        seg = st.windows["seg"].values[idx]
        vel = np.full(len(st.windows), np.nan)
        same = (seg[1:] == seg[:-1]) & (np.diff(idx) == 1)
        if same.any():
            j = np.flatnonzero(same)
            d = np.empty(len(j))
            for a in range(0, len(j), chunk):
                jj = j[a:a + chunk]
                d[a:a + chunk] = geodesic_distance(U[jj], U[jj + 1])
            vel[idx[j + 1]] = d

        st.k = k
        st.grassmann_idx = idx
        st.U = U
        st.mu = mu
        st.mean_type = mean
        st.karcher_history = hist
        st.tangent = Delta
        st.distance = _scatter(dist, idx, len(st.windows))
        st.angles = _scatter(angles, idx, len(st.windows))
        st.tangent_coords = _scatter(coords, idx, len(st.windows))
        st.tangent_var_ratio = var_ratio
        st.velocity = vel

        self._say("Gr(%d, %d) for %r in %.1fs: %d of %d windows (rank >= k)" %
                  (k, N, st.key, time.time() - t, len(idx), len(st.windows)))
        if hist:
            self._say("  Karcher mean: %d iterations, final gradient norm %.2e%s"
                      % (len(hist), hist[-1],
                         "" if hist[-1] < karcher_tol else " (not below tol=%.0e; raise karcher_iters)" % karcher_tol))
        self._say("  geodesic distance to the mean: median %.3f rad (RMS principal angle %.1f deg);"
                  " max possible = sqrt(k)*pi/2 = %.3f"
                  % (np.median(dist), np.degrees(np.median(dist) / np.sqrt(k)), np.sqrt(k) * np.pi / 2))
        self._say("  tangent PCA: top %d directions carry %s of the tangent variance"
                  % (p, ", ".join("%.1f%%" % (100 * v) for v in var_ratio)))
        if np.isfinite(vel).any():
            self._say("  angle between consecutive windows: median %.3f rad" % np.nanmedian(vel))
        else:
            self._say("  angle between consecutive windows: none (no two consecutive windows share a segment)")
        return st

    # ------------------------------------------------------------- labels
    def window_labels(self, by="epochs", key=None, default="other"):
        """Categorical label per window: by="epochs" uses the session's epoch
        intervals (later epochs override earlier ones); any other string is a
        column of the trials table (trials mode)."""
        st = self._st(key)
        rec = self._rec(key)
        n = len(st.windows)
        if by == "epochs":
            lab = np.full(n, default, dtype=object)
            c = st.windows["t_center"].values
            for name, iv in rec.epochs.items():
                for s, e in np.asarray(iv, dtype=float):
                    lab[(c >= s) & (c < e)] = name
            return lab
        if st.trials is None or by not in st.trials:
            raise KeyError("%r is not a trials column (%s)" % (by, [] if st.trials is None else list(st.trials)))
        return st.trials[by].astype(str).values[st.windows["seg"].values]

    def window_values(self, times, values, key=None, agg="mean", min_samples=1):
        """Aggregate a time series inside every window [t_start, t_stop): one
        number per window, NaN where fewer than min_samples samples fall in it.
        agg: "mean" (vectorised), "median", "max" or "min". Use it to turn
        running speed, LFP band power, pupil size ... into a colour for
        plot_grassmann(values=) / plot_tangent_grid()."""
        st = self._st(key)
        t = np.asarray(times, dtype=float).ravel()
        v = np.asarray(values, dtype=float).ravel()
        if len(t) != len(v):
            raise ValueError("times and values differ in length (%d vs %d)" % (len(t), len(v)))
        ok = np.isfinite(t) & np.isfinite(v)
        t, v = t[ok], v[ok]
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
        w = st.windows
        a = np.searchsorted(t, w["t_start"].values, side="left")
        b = np.searchsorted(t, w["t_stop"].values, side="left")
        n = b - a
        good = n >= max(int(min_samples), 1)
        out = np.full(len(w), np.nan)
        if agg == "mean":
            cs = np.r_[0.0, np.cumsum(v)]
            out[good] = (cs[b[good]] - cs[a[good]]) / n[good]
        elif agg in ("median", "max", "min"):
            f = {"median": np.median, "max": np.max, "min": np.min}[agg]
            for i in np.flatnonzero(good):
                out[i] = f(v[a[i]:b[i]])
        else:
            raise ValueError("agg must be 'mean', 'median', 'max' or 'min'")
        return out

    # -------------------------------------------------------- 5. the figure
    def plot_grassmann(self, key=None, labels=None, values=None, values_label="", values_cmap="magma",
                       values_clip=(2, 98), view=(28, -55), connect="auto",
                       cmap="viridis", max_sphere_lines=600, figsize=(15, 11), title=None,
                       save_as="grassmann"):
        """Sphere view + tangent plane + timeline (as in the Gr(3,1) demo).

        The tangent space at the Karcher mean has dimension k(N-k); the figure
        shows its top-2 principal directions. Sphere: each window's 2-D tangent
        coordinates are wrapped onto a unit sphere by the exponential map at the
        pole (= Karcher mean), with the polar angle equal to the projected
        tangent norm / sqrt(k) (an RMS principal angle), so the equator is the
        largest distance Gr(k, N) allows. Plane: the same coordinates flat.
        Colour: by time (default); `labels` = one categorical label per window
        (see window_labels()); `values` = one number per window (see
        window_values()), colour limits at the `values_clip` percentiles, NaN
        drawn grey.
        """
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

        st = self._st(key)
        if not hasattr(st, "mu"):
            raise RuntimeError("run fit_grassmann() first")
        idx = st.grassmann_idx
        k = st.k
        C = st.tangent_coords[idx, :2]
        tt = st.windows["t_center"].values
        seg = st.windows["seg"].values

        # colours
        color_label = "window time (s)"
        order = np.arange(len(idx))
        if values is not None:
            values = np.asarray(values, dtype=float).ravel()
            if len(values) != len(st.windows):
                raise ValueError("values must have one entry per window (%d)" % len(st.windows))
            fin = np.isfinite(values)
            lo, hi = (np.nanpercentile(values[fin], values_clip) if fin.any() else (0.0, 1.0))
            if not hi > lo:
                hi = lo + 1e-12
            norm = plt.Normalize(lo, hi)
            cm = plt.get_cmap(values_cmap)
            colors = np.tile(np.array([0.82, 0.82, 0.82, 1.0]), (len(values), 1))     # grey = no data
            colors[fin] = cm(norm(values[fin]))
            legend_handles = None
            color_label = (values_label or "value") + ("" if fin.all() else "   (grey: no data)")
            order = np.argsort(fin[idx], kind="stable")                                # grey drawn first
        elif labels is None:
            norm = plt.Normalize(tt.min(), tt.max())
            cm = plt.get_cmap(cmap)
            colors = cm(norm(tt))
            legend_handles = None
        else:
            labels = np.asarray(labels, dtype=object)
            if len(labels) != len(st.windows):
                raise ValueError("labels must have one entry per window (%d)" % len(st.windows))
            cats = list(dict.fromkeys(labels.tolist()))
            pal = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
            cmap_cat = {c: pal(i % pal.N) for i, c in enumerate(cats)}
            colors = np.array([cmap_cat[l] for l in labels])
            legend_handles = [Line2D([0], [0], marker="o", ls="", color=cmap_cat[c], label=str(c)) for c in cats]

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[4, 0.35, 1.1], width_ratios=[1, 1], hspace=0.35, wspace=0.15)
        ax3 = fig.add_subplot(gs[0, 0], projection="3d")
        ax2 = fig.add_subplot(gs[0, 1])
        axs = fig.add_subplot(gs[1, :])
        axv = fig.add_subplot(gs[2, :], sharex=axs)

        # ---- sphere: exp map of the 2-D tangent coordinates at the pole
        sk = np.sqrt(k)
        r = np.linalg.norm(C, axis=1)
        theta = np.minimum(r / sk, np.pi / 2)
        phi = np.arctan2(C[:, 1], C[:, 0])
        P = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        Q = np.column_stack([C[:, 0] / sk, C[:, 1] / sk, np.ones(len(C))])     # on the tangent plane z = 1
        hw = max(0.35, 1.08 * np.abs(Q[:, :2]).max())

        uu = np.linspace(0, 2 * np.pi, 80)
        vv = np.linspace(0, np.pi, 40)
        ax3.plot_surface(np.outer(np.cos(uu), np.sin(vv)), np.outer(np.sin(uu), np.sin(vv)),
                         np.outer(np.ones_like(uu), np.cos(vv)), color="0.9", alpha=0.18,
                         linewidth=0, shade=False)
        circ = np.linspace(0, 2 * np.pi, 200)
        ax3.plot(np.cos(circ), np.sin(circ), np.zeros_like(circ), color="0.45", lw=0.7, ls="--")
        for a in (0.0, np.pi / 2):                                            # two meridians
            ax3.plot(np.cos(a) * np.sin(circ), np.sin(a) * np.sin(circ), np.cos(circ), color="0.8", lw=0.4)
        plane = [[(-hw, -hw, 1), (hw, -hw, 1), (hw, hw, 1), (-hw, hw, 1)]]
        ax3.add_collection3d(Poly3DCollection(plane, facecolors="0.6", edgecolors="0.35",
                                              linewidths=0.8, alpha=0.18))
        sub = np.arange(len(P))
        if len(sub) > max_sphere_lines:
            sub = np.linspace(0, len(P) - 1, max_sphere_lines).astype(int)
        ax3.add_collection3d(Line3DCollection(np.stack([P[sub], Q[sub]], axis=1), colors="0.5",
                                              linewidths=0.3, alpha=0.3))
        ax3.scatter(Q[sub, 0], Q[sub, 1], Q[sub, 2], c=colors[idx][sub], s=4, alpha=0.5, depthshade=False)
        ax3.scatter(P[order, 0], P[order, 1], P[order, 2], c=colors[idx][order], s=8, alpha=0.9, depthshade=False)
        ax3.scatter([0], [0], [1], c="k", s=35, depthshade=False)
        ax3.text(0.06, 0.06, 1.1, "Karcher mean", fontsize=9)
        lim = max(1.15, hw * 1.05)
        ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim); ax3.set_zlim(-1.1, max(1.2, hw * 0.3 + 1))
        ax3.set_box_aspect((1, 1, 1))
        ax3.view_init(elev=view[0], azim=view[1])
        ax3.set_axis_off()
        ax3.set_title("Gr(%d, %d) through the top-2 tangent directions\n"
                      "exp map onto a sphere, pole = Karcher mean; dashed = max distance (all angles 90°)"
                      % (k, st.U.shape[1]), fontsize=9)

        # ---- tangent plane
        if connect == "auto":
            connect = len(idx) <= 3000
        if connect:
            same = (seg[idx][1:] == seg[idx][:-1]) & (np.diff(idx) == 1)
            if same.any():
                j = np.flatnonzero(same)
                ax2.add_collection(LineCollection(np.stack([C[j], C[j + 1]], axis=1),
                                                  colors="0.6", linewidths=0.4, alpha=0.5))
        ax2.scatter(C[order, 0], C[order, 1], c=colors[idx][order], s=8, alpha=0.85, linewidths=0)
        ax2.axhline(0, color="0.8", lw=0.8); ax2.axvline(0, color="0.8", lw=0.8)
        ax2.plot(0, 0, "o", color="k", ms=6)
        ax2.set_aspect("equal", adjustable="datalim")
        vr = st.tangent_var_ratio
        ax2.set_xlabel("tangent PC 1 (%.0f%% of tangent variance)" % (100 * vr[0]))
        ax2.set_ylabel("tangent PC 2 (%.0f%%)" % (100 * (vr[1] if len(vr) > 1 else 0)))
        ax2.set_title("Tangent space at the mean (log map), %d windows\norigin = mean subspace; "
                      "radius in rad, max possible sqrt(k)·pi/2 = %.2f" % (len(idx), sk * np.pi / 2), fontsize=9)
        if legend_handles is not None:
            ax2.legend(handles=legend_handles, fontsize=8, frameon=False, loc="best")
        else:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
            sm.set_array([])
            cb = plt.colorbar(sm, ax=ax2, pad=0.02, fraction=0.04)
            cb.set_label(color_label)

        # ---- timeline: label strip + angle between consecutive windows
        t0, t1 = tt.min(), tt.max()
        if values is not None:
            width = float(np.nanmedian(np.diff(tt))) if len(tt) > 1 else 1.0
            axs.bar(tt[fin], np.ones(int(fin.sum())), width=width, color=colors[fin], align="center", linewidth=0)
            axs.set_yticks([]); axs.set_ylabel("value", fontsize=8, rotation=0, ha="right", va="center")
        elif labels is None:
            axs.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap=cm,
                       extent=[t0, t1, 0, 1], interpolation="nearest")
            axs.set_yticks([]); axs.set_ylabel("time", fontsize=8, rotation=0, ha="right", va="center")
        else:
            w = st.windows
            for cidx, c in enumerate(cats):
                m = labels == c
                if not m.any():
                    continue
                spans = _runs_to_spans(w["t_start"].values[m], w["t_stop"].values[m])
                axs.broken_barh(spans, (0, 1), facecolors=cmap_cat[c], edgecolors="none")
            axs.set_yticks([]); axs.set_ylabel("label", fontsize=8, rotation=0, ha="right", va="center")
        axs.set_ylim(0, 1)
        plt.setp(axs.get_xticklabels(), visible=False)

        axv.plot(tt, st.velocity, color="0.25", lw=0.6)
        axv.axhline(np.nanmedian(st.velocity), color="tab:red", lw=0.8, ls="--")
        axv.set_xlim(t0, t1)
        axv.set_xlabel("session time (s)")
        axv.set_ylabel("angle between\nconsecutive windows (rad)", fontsize=8)
        axv.set_title("Timeline: label strip (top) and subspace velocity, geodesic angle between consecutive windows "
                      "(bottom; red dashed = median)", fontsize=9, loc="left")
        if title is None:
            title = "%s | %s mode | %g ms count bins, %s windows, k = %d (%s)" % (
                st.key, self.mode, self.count_bin_ms,
                "whole-trial" if self.mollification_bin_ms is None else "%g ms" % self.mollification_bin_ms,
                k, getattr(st, "k_rule_name", "k given"))
        if values is not None and values_label:
            title += " | colour: %s" % values_label
        if self.run_name:
            title = "[%s] %s" % (self.run_name, title)
        fig.suptitle(title, fontsize=11, y=0.995)
        self._save_fig(fig, save_as, st.key)
        return fig

    def plot_tangent_grid(self, values, key=None, ncols=3, cmap="magma", clip=(2, 98), s=4,
                          panel=(4.8, 4.3), save_as="tangent_grid"):
        """Grid of tangent-plane views (same coordinates as plot_grassmann), each
        coloured by one quantity. values: {name: per-window array}. Numeric
        arrays get a colormap (limits at the `clip` percentiles, NaN = grey);
        string/object arrays are categorical labels."""
        st = self._st(key)
        if not hasattr(st, "mu"):
            raise RuntimeError("run fit_grassmann() first")
        idx = st.grassmann_idx
        C = st.tangent_coords[idx, :2]
        names = list(values)
        n = len(names)
        if n == 0:
            raise ValueError("values is empty")
        ncols = max(1, min(int(ncols), n))
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), squeeze=False)
        for ax in axes.ravel()[n:]:
            ax.set_axis_off()
        for ax, name in zip(axes.ravel(), names):
            v = np.asarray(values[name])
            if len(v) != len(st.windows):
                raise ValueError("%r must have one entry per window (%d)" % (name, len(st.windows)))
            ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8)
            if v.dtype.kind in "OUSb":                                        # categorical
                lab = v.astype(object)[idx]
                cats = list(dict.fromkeys(lab.tolist()))
                pal = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
                for i, c in enumerate(cats):
                    m = lab == c
                    ax.scatter(C[m, 0], C[m, 1], s=s, color=pal(i % pal.N), linewidths=0, alpha=0.8,
                               label="%s (%d)" % (c, int(m.sum())))
                ax.legend(fontsize=7, frameon=False, markerscale=3)
            else:
                v = v.astype(float)[idx]
                fin = np.isfinite(v)
                ax.scatter(C[~fin, 0], C[~fin, 1], s=s, color="0.82", linewidths=0)
                if fin.any():
                    lo, hi = np.nanpercentile(v[fin], clip)
                    if not hi > lo:
                        hi = lo + 1e-12
                    sc = ax.scatter(C[fin, 0], C[fin, 1], c=v[fin], s=s, cmap=cmap, vmin=lo, vmax=hi,
                                    linewidths=0, alpha=0.85)
                    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
                    cb.ax.tick_params(labelsize=7)
                ax.text(0.02, 0.02, "%d of %d windows have data" % (int(fin.sum()), len(v)),
                        transform=ax.transAxes, fontsize=7, color="0.4")
            ax.plot(0, 0, "o", color="k", ms=4)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(name, fontsize=9)
            ax.tick_params(labelsize=7)
        vr = st.tangent_var_ratio
        fig.suptitle("%s%s | tangent plane at the Karcher mean of Gr(%d, %d); PC1 %.0f%%, PC2 %.0f%% of tangent variance"
                     % (("[%s] " % self.run_name) if self.run_name else "", st.key, st.k, st.U.shape[1],
                        100 * vr[0], 100 * (vr[1] if len(vr) > 1 else 0)), fontsize=10)
        fig.tight_layout()
        self._save_fig(fig, save_as, st.key)
        return fig

    # ----------------------------------------------------------- summary
    def summary(self, key=None):
        """One-row Series of the main numbers for the active session."""
        st = self._st(key)
        s = {"run_name": self.run_name, "session": st.key, "mode": self.mode,
             "units": st.X.shape[1], "bins": st.X.shape[0],
             "count_bin_ms": self.count_bin_ms, "window_ms": self.mollification_bin_ms,
             "step_ms": self.mollification_step_ms}
        if hasattr(st, "windows"):
            s.update({"windows": len(st.windows), "valid_windows": int(st.valid.sum()),
                      "median_rank": int(np.median(st.rank)),
                      "median_PR": float(np.median(st.participation_ratio[st.valid])),
                      "median_elbow": float(np.median(st.elbow_per_window[st.valid]))})
        if hasattr(st, "k"):
            s["k"] = st.k
            s["k_rule"] = getattr(st, "k_rule_name", "")
        if hasattr(st, "mu"):
            s.update({"windows_on_Gr": len(st.grassmann_idx),
                      "median_dist_to_mean_rad": float(np.nanmedian(st.distance)),
                      "tangent_PC_var": ", ".join("%.2f" % v for v in st.tangent_var_ratio),
                      "median_velocity_rad": float(np.nanmedian(st.velocity))})
        return pd.Series(s)


# ----------------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------------

def _bin_segment(spikes, s, n, dt):
    """Counts (n, N) for the bins [s + i dt, s + (i+1) dt), i < n. spikes sorted."""
    out = np.zeros((n, len(spikes)), dtype=np.uint16)
    e = s + n * dt
    for j, st in enumerate(spikes):
        lo = np.searchsorted(st, s, side="left")
        hi = np.searchsorted(st, e, side="left")
        if hi > lo:
            idx = np.floor((st[lo:hi] - s) / dt).astype(np.int64)
            idx = idx[(idx >= 0) & (idx < n)]
            out[:, j] = np.minimum(np.bincount(idx, minlength=n), np.iinfo(np.uint16).max)
    return out


def _window_eig(Xw, kk, min_eig_frac):
    """Eigendecomposition of the covariance of each centred window Xw (n, L, N).

    Uses the L x L Gram matrix when L <= N (eigenvectors recovered as
    Xw^T v / sqrt(w)) and the N x N scatter otherwise. Returns eigenvalues
    (n, r) descending with r = min(L, N), the top-kk eigenvectors (n, N, kk)
    (float32) and the numerical rank per window.
    """
    n, L, N = Xw.shape
    denom = max(L - 1, 1)
    if L <= N:
        G = Xw @ np.swapaxes(Xw, 1, 2)                           # (n, L, L)
        w, V = np.linalg.eigh(G)
        w = w[:, ::-1]; V = V[:, :, ::-1]
        w = np.clip(w, 0, None)
        kk = min(kk, L)
        tiny = w[:, :kk] <= min_eig_frac * np.maximum(w.sum(1, keepdims=True), 1e-300)
        scale = np.where(tiny, 0.0, 1.0 / np.sqrt(np.where(tiny, 1.0, w[:, :kk])))
        U = (np.swapaxes(Xw, 1, 2) @ V[:, :, :kk]) * scale[:, None, :]   # (n, N, kk)
        lam = w / denom
    else:
        Cm = np.swapaxes(Xw, 1, 2) @ Xw                          # (n, N, N)
        w, V = np.linalg.eigh(Cm)
        w = np.clip(w[:, ::-1], 0, None); V = V[:, :, ::-1]
        kk = min(kk, N)
        U = V[:, :, :kk]
        lam = w / denom
    rank = (lam > min_eig_frac * np.maximum(lam.sum(1, keepdims=True), 1e-300)).sum(1)
    return lam, U.astype(np.float32), rank


def _top_pcs(T, p, total_var):
    """Top-p principal directions of the (uncentred) rows of T: scores (n, p) and
    variance ratios. Lanczos (scipy svds) when it is cheaper, else full SVD."""
    if p < 1 or total_var <= 0:
        return np.zeros((len(T), max(p, 1))), np.zeros(max(p, 1))
    try:
        if p < min(T.shape) - 1 and min(T.shape) > 50:
            from scipy.sparse.linalg import svds
            u, s, vt = svds(T, k=p)
            order = np.argsort(s)[::-1]
            s, vt = s[order], vt[order]
        else:
            u, s, vt = np.linalg.svd(T, full_matrices=False)
            s, vt = s[:p], vt[:p]
    except Exception:
        u, s, vt = np.linalg.svd(T, full_matrices=False)
        s, vt = s[:p], vt[:p]
    scores = T @ vt.T
    return scores, (s ** 2) / total_var


def _scatter(values, idx, n):
    """Place per-valid-window values into an (n, ...) array with NaN elsewhere."""
    values = np.asarray(values, dtype=float)
    out = np.full((n,) + values.shape[1:], np.nan)
    out[idx] = values
    return out


def _runs_to_spans(t_start, t_stop, gap=1e-9):
    """Merge sorted, touching/overlapping [start, stop) intervals -> [(start, width)]."""
    order = np.argsort(t_start)
    s, e = t_start[order], t_stop[order]
    spans = []
    cs, ce = s[0], e[0]
    for a, b in zip(s[1:], e[1:]):
        if a <= ce + gap:
            ce = max(ce, b)
        else:
            spans.append((cs, ce - cs))
            cs, ce = a, b
    spans.append((cs, ce - cs))
    return spans
