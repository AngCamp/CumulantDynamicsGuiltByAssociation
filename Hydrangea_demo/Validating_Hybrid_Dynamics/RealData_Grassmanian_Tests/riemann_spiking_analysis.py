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
                     -> k selection                     (a k_rule: elbow of the global covariance,
                                                         median of per-window elbows, or fixed)
                     -> points U_w in Gr(k, N): Karcher mean, log map, tangent PCA

This module only computes. Figures are built in the notebooks from `obj.state()`,
the per-session namespace holding X, the windows, the spectra, k and the Gr(k, N)
fit, so plot code can be changed with the data in memory.

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
                                 normalize=zscore_units, k_rule=k_global_elbow())
    obj.add_session("alm", loader=load_alm_000363, asset=asset, h5=h5f, nwb=nwb)
    obj.load("alm"); obj.status()
    obj.prepare(); obj.window_spectra(); obj.choose_k(); obj.fit_grassmann()
    st = obj.state()            # -> the notebook's plot functions read from here
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

__all__ = [
    "RiemannSpikingAnalysis", "SessionRecord",
    "load_alm_000363", "load_hp_001695",
    "zscore_units", "identity", "sqrt_counts", "keep_all",
    "k_global_elbow", "k_median_elbow", "k_fixed", "elbow_index", "global_spectrum",
    "intervals_by_column", "session_key_from_path",
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


def elbow_index(y, n_components=None):
    """Elbow of a descending curve: the point farthest below the chord through the
    first n_components finite points (None = all of them), as a 1-based index.
    Explained-variance curves are the eigenvalues scaled by their sum, so the
    elbow is the same on either."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    m = len(y) if n_components is None else min(int(n_components), len(y))
    if m < 3:
        return 1
    y = y[:m]
    chord = y[0] + (y[-1] - y[0]) * np.arange(m) / (m - 1)
    return int(np.argmax(chord - y)) + 1


def _rule_name(base, n_components, k_max):
    parts = [s for s in ("n_components=%d" % n_components if n_components is not None else "",
                         "k_max=%d" % k_max if k_max is not None else "") if s]
    return "%s(%s)" % (base, ", ".join(parts))


def global_spectrum(X, chunk=50000):
    """Eigenvalues (descending) and explained variance of the covariance of ALL rows
    of X (bins x units): the global / full-data PCA at the run's count bin. Chunked
    so a large float32 X is never copied whole to float64."""
    X = np.asarray(X)
    T, N = X.shape
    mu = X.mean(axis=0, dtype=np.float64)
    C = np.zeros((N, N))
    for a in range(0, T, chunk):
        Z = X[a:a + chunk].astype(np.float64) - mu
        C += Z.T @ Z
    C /= max(T - 1, 1)
    lam = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
    tot = lam.sum()
    evr = lam / tot if tot > 0 else np.full_like(lam, np.nan)
    return lam, evr


def k_global_elbow(n_components=None, k_max=None):
    """k_rule (default): the elbow of the explained-variance curve of the GLOBAL
    covariance, all retained bins of X at the run's count bin (full rank). Every
    valid window gets that same k as its per-window value. The global spectrum is
    stored on st.global_eigvals / st.global_evr."""
    def rule(st):
        lam, evr = global_spectrum(st.X)
        st.global_eigvals, st.global_evr = lam, evr
        k = elbow_index(evr, n_components)
        if k_max is not None:
            k = min(k, int(k_max))
        k = max(k, 1)
        return k, np.full(int(st.valid.sum()), k, dtype=int)
    rule.__name__ = _rule_name("global_elbow", n_components, k_max)
    return rule


def k_median_elbow(n_components=None, k_max=None):
    """k_rule: elbow of every valid window's explained-variance curve, median across
    windows (rounded), optionally capped at k_max. Per-window value = each window's
    own elbow (a sampling-shape statistic when windows hold fewer bins than units)."""
    def rule(st):
        evr = st.evr[st.valid]
        ks = np.array([elbow_index(row, n_components) for row in evr], dtype=int)
        k = int(round(np.median(ks))) if len(ks) else 1
        if k_max is not None:
            k = min(k, int(k_max))
        return max(k, 1), ks
    rule.__name__ = _rule_name("median_elbow", n_components, k_max)
    return rule


def k_fixed(K):
    """k_rule: always K (the one way to override the chosen k: obj.choose_k(k_fixed(K)))."""
    def rule(st):
        return int(K), np.full(int(st.valid.sum()), int(K), dtype=int)
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
    k_rule                  f(st) -> (k, one value per valid window); default
                            k_global_elbow(). k is set only by choose_k().
    k_store                 eigenvectors kept per window (k must be <= k_store).
    center_windows          subtract each window's own mean before the covariance.
    t_range                 "full" mode only: None = whole recording, (t0, t1), or
                            the name of an epoch of the session (e.g. "maze").
    """

    def __init__(self, mode="trials", count_bin_ms=10.0, mollification_bin_ms=500.0,
                 mollification_step_ms=None, unit_filter=keep_all, trial_filter=keep_all,
                 count_transform=identity, normalize=zscore_units, k_rule=None,
                 k_store=15, center_windows=True, t_range=None, verbose=True,
                 output_dir=None, run_name=None, session_cache_dir=None, use_cache=True):
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
        self.k_rule = k_rule if k_rule is not None else k_global_elbow()
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

    def _t_range_desc(self):
        """t_range as a short JSON-able description (an interval array is summarised + hashed)."""
        tr = self.t_range
        if tr is None or isinstance(tr, str):
            return tr
        arr = np.atleast_2d(np.asarray(tr, dtype=float))
        if arr.shape == (1, 2):
            return [float(arr[0, 0]), float(arr[0, 1])]
        import hashlib
        return "intervals(n=%d, %.1f s, sha=%s)" % (len(arr), float((arr[:, 1] - arr[:, 0]).sum()),
                                                    hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:10])

    def config(self):
        """The run configuration as a JSON-able dict."""
        return {"run_name": self.run_name, "mode": self.mode, "t_range": self._t_range_desc(),
                "count_bin_ms": self.count_bin_ms, "mollification_bin_ms": self.mollification_bin_ms,
                "mollification_step_ms": self.mollification_step_ms,
                "unit_filter": self._fname(self.unit_filter), "trial_filter": self._fname(self.trial_filter),
                "count_transform": self._fname(self.count_transform), "normalize": self._fname(self.normalize),
                "k_rule": self._fname(self.k_rule), "k_store": self.k_store,
                "center_windows": self.center_windows,
                "output_dir": None if self.output_dir is None else str(self.output_dir)}

    # ------------------------------------------------------------- caches
    def _data_hash(self, st):
        """Fingerprint of the prepared data + the settings that shape the spectra.
        Hashes the data itself (kept unit ids, segments, per-unit counts, a sample
        of X), so changing REGION / MIN_RATE_HZ inside a filter function is caught
        even though the function name is unchanged."""
        import hashlib
        h = hashlib.sha1()
        cfg = {k: v for k, v in self.config().items()
               if k not in ("run_name", "output_dir", "k_rule")}
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
        print("  mode                  :", self.mode,
              "" if self.mode == "trials" else "(t_range=%r)" % (self._t_range_desc(),))
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

    def state(self, key=None):
        """The analysis namespace of a session (default: the active one): units, X,
        windows, spectra, k, the Gr(k, N) fit ... everything the notebook figures read."""
        return self._st(key)

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
                      % (len(segments), self._t_range_desc(), np.sum(segments[:, 1] - segments[:, 0]) / 60))
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
        """Subspace dimension from k_rule (or `rule`): the rule receives the analysis
        namespace and returns (k, one value per valid window). k is capped at k_store.
        This is the only place k is set; to override, call choose_k(k_fixed(K))."""
        st = self._st(key)
        rule = self.k_rule if rule is None else rule
        k, per_window = rule(st)
        k = int(k)
        k_cap = min(self.k_store, st.eigvecs.shape[2])
        if k > k_cap:
            self._say("  k=%d capped at k_store=%d (pass k_store= to the constructor and rerun"
                      " window_spectra for more)" % (k, k_cap))
            k = k_cap
        st.k = max(1, k)
        st.k_rule_name = self._fname(rule)
        per_window = np.asarray(per_window, dtype=float).ravel()
        kp = np.full(len(st.windows), np.nan)
        if len(per_window) == int(st.valid.sum()):
            kp[st.valid] = per_window
        elif len(per_window) == len(st.windows):
            kp = per_window
        st.k_per_window = kp
        q = np.nanpercentile(kp, [25, 50, 75]) if np.isfinite(kp).any() else (np.nan,) * 3
        cum = np.nanmedian(np.nansum(st.evr[st.valid][:, :st.k], axis=1))
        self._say("k = %d by %s | per-window k: median %g, IQR %g-%g | median explained"
                  " variance in top k: %.1f%%" % (st.k, st.k_rule_name, q[1], q[0], q[2], 100 * cum))
        return st.k

    # ------------------------------------------------- 4. Gr(k, N) embedding
    def fit_grassmann(self, mean="karcher", n_tangent_pcs=3, karcher_iters=50,
                      karcher_tol=1e-6, chunk=4000, key=None):
        """Every valid window as a point of Gr(k, N), k = st.k from choose_k() (the only
        place k is set; override with choose_k(k_fixed(K))).

        Windows of rank >= k: their top-k eigenvectors. Windows of rank r < k pin down
        only r directions, so they are completed with the k - r directions of the mean
        orthogonal to their own range (top left singular vectors of (I - U_r U_r^T) mu):
        the k-plane containing the window's range that lies closest to the mean. They
        are flagged in st.rank_deficient and drawn hollow by the notebook figures.
        Karcher mean (or extrinsic: quick look only, the tangent PCA is uncentred and
        only unbiased at a Karcher mean) from the full-rank windows; log map at the
        mean; PCA of the tangent vectors; geodesic distance to the mean; angle between
        consecutive windows."""
        st = self._st(key)
        if not hasattr(st, "k"):
            raise RuntimeError("run choose_k() first (override with choose_k(k_fixed(K)))")
        k = int(st.k)
        if k > st.eigvecs.shape[2]:
            raise ValueError("k=%d > k_store=%d; rerun window_spectra with a larger k_store" % (k, st.eigvecs.shape[2]))
        # st.rank counts the eigenvalues above 1e-8 of the window's total variance (the
        # threshold the old lam_k test used); windows with no spikes (rank 0) have no range
        valid = st.valid & (st.rank >= 1)
        idx = np.flatnonzero(valid)
        r_of = np.minimum(st.rank[idx], k)
        is_full = r_of >= k
        if int(is_full.sum()) < 2:
            raise ValueError("fewer than 2 windows have rank >= k=%d; lower k" % k)
        t = time.time()
        U = st.eigvecs[idx][:, :, :k].astype(np.float64)
        Q, _ = np.linalg.qr(U[is_full])                            # re-orthonormalise (batched)
        U[is_full] = Q
        Uf = U[is_full]                                            # the mean is estimated from these

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
                mu, hist = karcher_mean(Uf, k, n_iter=karcher_iters, tol=karcher_tol, chunk=chunk)
            else:
                mu, hist = extrinsic_mean(Uf, k), []
            if cache is not None:
                np.savez(cache, mu=mu, hist=np.asarray(hist, dtype=float))

        # rank-deficient windows: keep the r directions they pin down, complete with the
        # mean's directions orthogonal to them (the closest k-plane containing their range)
        for r in np.unique(r_of[~is_full]):
            sel = np.flatnonzero(~is_full & (r_of == r))
            Ur, _ = np.linalg.qr(U[sel][:, :, :r])                        # (m, N, r)
            R = mu[None] - Ur @ (np.swapaxes(Ur, 1, 2) @ mu[None])        # (I - Ur Ur^T) mu, (m, N, k)
            W = np.linalg.svd(R, full_matrices=False)[0][:, :, :k - r]    # top k-r left singular vectors
            U[sel] = np.concatenate([Ur, W], axis=2)
        rank_deficient = np.zeros(len(st.windows), dtype=bool)
        rank_deficient[idx[~is_full]] = True

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
        st.rank_deficient = rank_deficient
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

        self._say("Gr(%d, %d) for %r in %.1fs: %d of %d windows placed; %d of rank >= k, %d of lower rank"
                  " completed with the mean's directions"
                  % (k, N, st.key, time.time() - t, len(idx), len(st.windows),
                     int(is_full.sum()), int((~is_full).sum())))
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
        the notebook's plot_grassmann(obj, values=) / plot_tangent_grid(obj, ...)."""
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

    # ----------------------------------------------------------- summary
    def summary(self, key=None):
        """One-row Series of the main numbers for the active session."""
        st = self._st(key)
        s = {"run_name": self.run_name, "session": st.key, "mode": self.mode, "t_range": self._t_range_desc(),
             "units": st.X.shape[1], "bins": st.X.shape[0],
             "count_bin_ms": self.count_bin_ms, "window_ms": self.mollification_bin_ms,
             "step_ms": self.mollification_step_ms}
        if hasattr(st, "windows"):
            s.update({"windows": len(st.windows), "valid_windows": int(st.valid.sum()),
                      "median_rank": int(np.median(st.rank)),
                      "median_PR": float(np.median(st.participation_ratio[st.valid]))})
        if hasattr(st, "k"):
            s["k"] = st.k
            s["k_rule"] = getattr(st, "k_rule_name", "")
            if hasattr(st, "k_per_window") and np.isfinite(st.k_per_window).any():
                s["median_k_per_window"] = float(np.nanmedian(st.k_per_window))
        if hasattr(st, "mu"):
            s.update({"windows_on_Gr": len(st.grassmann_idx),
                      "n_rank_deficient": int(st.rank_deficient.sum()),
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
