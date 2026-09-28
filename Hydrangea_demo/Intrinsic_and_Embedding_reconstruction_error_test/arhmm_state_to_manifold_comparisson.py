"""Sticky HDP-AR-HMM states of the CA1 LFP, against the manifold embeddings.

Two cells. CELL 1 is definitions only — config, the sampler, the loaders, the
scoring and the figures — and runs nothing, so it is the thing that becomes a
module once this settles. CELL 2 is the run: pick a recording, load it, fit,
rank, plot. Testing one recording at a time for now; the sweep over all
sixteen comes later and will call the same functions.

The embeddings are not refit here. They are read back from the cache that
reconstruction_error_test.py wrote, so this is fast and the geometry it plots
is exactly the geometry that was scored there.

What CELL 2 does
    1. suggest a recording: the one whose activity-reconstruction curve sits
       closest to the average across the sixteen, so the example is typical
       rather than an outlier. RECORDING overrides the suggestion.
    2. rebuild that session's 50 ms bin grid exactly as the sweep did, and
       check it against the cached embeddings before using them
    3. band power on the CA1 LFP (low theta, high theta, gamma, ripple),
       Hilbert envelope, binned to the same 50 ms grid
    4. fit the sticky HDP-AR-HMM to those four band traces at AR order 1, so
       one lag is one 50 ms bin and the number of states is inferred
    5. rank each method's components by what dropping one does to held-out
       reconstruction MSE — not by eigenvalue order, which is a smoothness
       ordering for Laplacian eigenmaps and means nothing for reconstruction
    6. plot the manifolds and the top-ranked components against position,
       speed and band power, coloured by inferred state, and one example
       window with everything on a shared time axis

Nothing is written to disk. Everything is print() and plt.show().

A note on resolution. PCA was fit on every bin, kernel PCA and Laplacian
eigenmaps on every tenth (they build an n x n matrix and cannot take 50k bins).
So the Laplacian embedding exists at 500 ms and the PCA one at 50 ms. Anything
that compares them is evaluated on the coarse grid, and the example window
shows the Laplacian as markers to keep that visible rather than hiding it
behind interpolation.
"""

# %% ===========================================================================
# CELL 1 — config, the sampler, the loaders, the scoring, the figures
# ==============================================================================
# Definitions only. Running this cell loads nothing and fits nothing.

import gc
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import scipy.signal as sps
from matplotlib.colors import ListedColormap
from scipy.fft import next_fast_len
from scipy.linalg import solve_triangular
from scipy.stats import invwishart
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

DOWNLOAD_DIR = Path("/storage/dandi_downloads").resolve()

# Written by reconstruction_error_test.py. Must be the z-scored cache: the
# embeddings and the activity matrix rebuilt below have to agree on scaling or
# the reconstruction numbers mean nothing.
CACHE_DIR = Path("/storage/manifold_embeddings_zscoreddata")

# --- these four must match the sweep that filled CACHE_DIR --------------------
BIN_SIZE_S = 0.050
RATE_TRANSFORM = "zscore"
MIN_RATE_HZ = 0.0
EPOCH = "maze"

# --- LFP bands ----------------------------------------------------------------
# Same as UMAP_EDA/umap_power.py: the only clear peak on this channel is theta,
# sitting high, so it is split into a low and a high half.
BANDS = {
    "low_theta": (6.0, 10.0),
    "high_theta": (10.0, 12.0),
    "gamma": (30.0, 90.0),
    "ripple": (120.0, 200.0),
}
BAND_COLORS = {"low_theta": "#2ca02c", "high_theta": "#1f77b4",
               "gamma": "#ff7f0e", "ripple": "#c2338f"}
# Prefer a CA1 channel if the file names one; otherwise take the only LFP there
# is and say which it was.
LFP_REGION_HINT = "ca1"

# --- the AR-HMM ---------------------------------------------------------------
# nlags=1 on 50 ms bins is the 50 ms lag asked for: one step of the AR process
# is one manifold bin. L is the weak-limit truncation, not the answer — the
# number of occupied states is what the model infers.
ARHMM_INPUT = "log_power"         # "log_power" | "z_power"
ARHMM_NLAGS = 1
ARHMM_L = 20                      # truncation level; raise if K_hat hits it
ARHMM_ITER = 400
ARHMM_BURN_IN = 200
ARHMM_ALPHA = 1.0
ARHMM_KAPPA = 50.0                # sticky bias: bigger = longer dwell times
ARHMM_GAMMA = 1.0
ARHMM_MIN_FRAC = 0.01             # a state is "used" if it holds >= 1% of bins
ARHMM_SEED = 0
# The Gibbs sampler's state step is a Python loop over time, so cost is linear
# in bins and iterations. None fits the whole epoch; an integer takes that many
# bins from the start, kept contiguous because subsampling would break the
# meaning of one lag.
ARHMM_MAX_BINS = None

# --- component ranking --------------------------------------------------------
# Rank by what dropping a component does to held-out reconstruction MSE, which
# is the only ordering that means anything across all three methods: PCA and
# kernel PCA come out in eigenvalue order, but Laplacian eigenmaps come out in
# ascending graph-Laplacian order, which is smoothness, not importance.
K_LLE = 10
LAMBDA = 1.0
RANK_DIMS = 15                    # the working set a component is dropped from
RANK_FOLDS = 5
RANK_MAX_SAMPLES = 4000
RANK_SEED = 0
TOP_N = 3                         # components carried into the figures

METHODS = ("PCA", "KernelPCA", "Laplacian")
METHOD_COLORS = {"PCA": "#1f77b4", "KernelPCA": "#d62728",
                 "Laplacian": "#9467bd"}
MAP_METHODS = ("PCA", "Laplacian")   # which get the 3d manifold figures
MAP_MAX_POINTS = 12000               # scatter subsample, for render speed

EXAMPLE_WINDOW_S = 8.0
SUGGEST_N = 6                     # how many recordings to list as candidates


# -----------------------------------------------------------------------------
# THE STICKY HDP-AR-HMM
# -----------------------------------------------------------------------------
# Fox, Sudderth, Jordan & Willsky (2009). An autoregressive HMM with a
# hierarchical Dirichlet process prior over the transition matrix, so the
# number of states is inferred rather than fixed:
#
#     beta        ~ GEM(gamma)
#     pi_j        ~ DP(alpha + kappa, (alpha*beta + kappa*delta_j)/(alpha+kappa))
#     z_t | z_t-1 ~ pi_{z_t-1}
#     y_t         = A_{z_t} [y_{t-1}, ..., y_{t-r}, 1] + e_t,  e_t ~ N(0, Sigma)
#     (A_k, Sigma_k) ~ Matrix-Normal-Inverse-Wishart
#
# kappa is the sticky self-transition bias, which stops the model explaining
# the data with rapidly switching redundant states. Inference is a blocked
# Gibbs sampler under the weak-limit approximation of the DP: truncate at L
# states, and the ones that end up holding (almost) no data are unused.
#
# The script as supplied, with the synthetic-data and evaluation part dropped —
# that was for debugging the sampler, and the data here is real.


def make_design(Y, nlags):
    """Build lagged design matrix X (with bias column) and targets Y[nlags:]."""
    T, d = Y.shape
    lags = [Y[nlags - l - 1: T - l - 1] for l in range(nlags)]
    X = np.hstack(lags + [np.ones((T - nlags, 1))])
    return X, Y[nlags:]


def sample_mniw(X, Y, M0, K0, S0, nu0, rng):
    """Sample (A, Sigma) from the MNIW posterior given regressors X, targets Y.

    With no data (X empty) this samples from the prior.
    """
    Sxx = X.T @ X + K0
    Syx = Y.T @ X + M0 @ K0
    Syy = Y.T @ Y + M0 @ K0 @ M0.T
    Sxx_inv = np.linalg.inv(Sxx)
    Sxx_inv = (Sxx_inv + Sxx_inv.T) / 2
    Mn = Syx @ Sxx_inv
    Sn = S0 + Syy - Mn @ Syx.T
    Sn = (Sn + Sn.T) / 2 + 1e-9 * np.eye(Sn.shape[0])
    nun = nu0 + X.shape[0]

    Sigma = np.atleast_2d(invwishart.rvs(df=nun, scale=Sn, random_state=rng))
    L_sig = np.linalg.cholesky(Sigma)
    L_col = np.linalg.cholesky(Sxx_inv)
    A = Mn + L_sig @ rng.standard_normal(Mn.shape) @ L_col.T
    return A, Sigma


def ar_loglik(X, Y, A, Sigma):
    """Per-timestep Gaussian log-likelihood of Y under AR params (A, Sigma)."""
    d = Y.shape[1]
    R = Y - X @ A.T
    L = np.linalg.cholesky(Sigma)
    sol = solve_triangular(L, R.T, lower=True)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + np.sum(sol ** 2, axis=0))


def sample_states(logL, P, pi0, rng):
    """Blocked sampling of the state sequence (backward filter, forward sample)."""
    T, L = logL.shape
    lik = np.exp(logL - logL.max(axis=1, keepdims=True))
    bwd = np.ones((T, L))
    for t in range(T - 2, -1, -1):
        b = P @ (lik[t + 1] * bwd[t + 1])
        bwd[t] = b / (b.sum() + 1e-300)

    u = rng.random(T)
    z = np.empty(T, dtype=int)
    p = pi0 * lik[0] * bwd[0]
    c = np.cumsum(p)
    z[0] = min(np.searchsorted(c, u[0] * c[-1]), L - 1)
    for t in range(1, T):
        p = P[z[t - 1]] * lik[t] * bwd[t]
        c = np.cumsum(p)
        z[t] = min(np.searchsorted(c, u[t] * c[-1]), L - 1)
    return z


def safe_dirichlet(a, rng):
    g = rng.gamma(np.maximum(a, 1e-12)) + 1e-300
    return g / g.sum()


def sample_tables(N, alpha, kappa, beta, rng):
    """Chinese-restaurant-franchise table counts m_jk."""
    L = len(beta)
    M = np.zeros((L, L), dtype=int)
    for j in range(L):
        for k in range(L):
            n = int(N[j, k])
            if n == 0:
                continue
            conc = alpha * beta[k] + (kappa if j == k else 0.0)
            M[j, k] = np.sum(rng.random(n) < conc / (conc + np.arange(n)))
    return M


def sample_beta(N, alpha, kappa, gamma, beta, rng):
    """Resample global weights via auxiliary tables + sticky override variables."""
    L = len(beta)
    M = sample_tables(N, alpha, kappa, beta, rng)
    rho = kappa / (alpha + kappa)
    w = rng.binomial(np.diag(M), rho / (rho + beta * (1 - rho)))
    Mbar = M.copy()
    Mbar[np.diag_indices(L)] -= w
    return safe_dirichlet(gamma / L + Mbar.sum(axis=0), rng)


def sample_transitions(N, alpha, kappa, beta, rng):
    L = len(beta)
    P = np.empty((L, L))
    for j in range(L):
        a = alpha * beta + N[j]
        a[j] += kappa
        P[j] = safe_dirichlet(a, rng)
    return P


def fit_hdp_arhmm(Y, nlags=ARHMM_NLAGS, L=ARHMM_L, n_iter=ARHMM_ITER,
                  burn_in=ARHMM_BURN_IN, alpha=ARHMM_ALPHA, kappa=ARHMM_KAPPA,
                  gamma=ARHMM_GAMMA, min_frac=ARHMM_MIN_FRAC, standardize=True,
                  seed=ARHMM_SEED, verbose=True):
    """
    Y        : (T, d) array
    nlags    : AR order r
    L        : truncation level (max number of states)
    alpha    : transition concentration
    kappa    : sticky self-transition bias (bigger = longer state durations)
    gamma    : top-level DP concentration (bigger = more states a priori)
    min_frac : a state counts as "used" if it holds >= this fraction of timesteps
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if standardize:
        Y = (Y - Y.mean(0)) / Y.std(0)

    X, Yt = make_design(Y, nlags)
    T, d = Yt.shape
    p = X.shape[1]

    # MNIW prior
    M0 = np.zeros((d, p))
    K0 = np.eye(p)
    nu0 = d + 2
    S0 = 0.1 * np.eye(d) * (nu0 - d - 1)

    # Initialise: random blocks of states
    block = 50
    z = np.repeat(rng.integers(0, L, size=T // block + 1), block)[:T]
    beta = np.ones(L) / L

    As = np.zeros((L, d, p))
    Sigmas = np.zeros((L, d, d))

    trace_K, trace_ll, z_samples = [], [], []
    started = time.perf_counter()
    for it in range(n_iter):
        # 1. AR parameters for each state
        for k in range(L):
            idx = z == k
            As[k], Sigmas[k] = sample_mniw(X[idx], Yt[idx], M0, K0, S0, nu0, rng)

        # 2. Transition counts -> beta and transition matrix
        N = np.zeros((L, L))
        np.add.at(N, (z[:-1], z[1:]), 1)
        beta = sample_beta(N, alpha, kappa, gamma, beta, rng)
        P = sample_transitions(N, alpha, kappa, beta, rng)

        # 3. State sequence
        logL = np.column_stack([ar_loglik(X, Yt, As[k], Sigmas[k]) for k in range(L)])
        z = sample_states(logL, P, beta, rng)

        counts = np.bincount(z, minlength=L)
        K_used = int(np.sum(counts >= min_frac * T))
        ll = logL[np.arange(T), z].sum()
        trace_K.append(K_used)
        trace_ll.append(ll)
        if it >= burn_in:
            z_samples.append(z.copy())

        if verbose and (it % 25 == 0 or it == n_iter - 1):
            spent = time.perf_counter() - started
            left = spent / (it + 1) * (n_iter - it - 1)
            print(f"    iter {it:4d}/{n_iter} | used states {K_used:2d} | "
                  f"log-lik {ll:12.1f} | {spent / 60:.1f} min spent, "
                  f"~{left / 60:.1f} min left", flush=True)

    post_K = np.array(trace_K[burn_in:])
    values, freq = np.unique(post_K, return_counts=True)
    K_hat = int(values[np.argmax(freq)])

    # Representative sample: highest log-lik post-burn-in sample with K_hat states
    cands = [i for i in range(burn_in, n_iter) if trace_K[i] == K_hat]
    best = max(cands, key=lambda i: trace_ll[i])
    z_best = z_samples[best - burn_in]

    if verbose:
        print("    posterior over number of used states:")
        for v, f in zip(values, freq):
            print(f"      K = {v:2d}: {f / len(post_K):.2f}")
        print(f"    estimated number of states: {K_hat}")

    return dict(K_hat=K_hat, z=z_best, trace_K=np.array(trace_K),
                trace_ll=np.array(trace_ll), z_samples=z_samples,
                As=As, Sigmas=Sigmas, beta=beta, P=P,
                Y=Y, nlags=nlags, burn_in=burn_in)


# -----------------------------------------------------------------------------
# WHICH RECORDING
# -----------------------------------------------------------------------------


def recording_table(root=DOWNLOAD_DIR):
    """Every NWB under the download directory, same ordering as the sweep."""
    rows = []
    for path in sorted(Path(root).rglob("*.nwb")):
        stem = path.stem
        subject = re.search(r"sub-([^_]+)", stem)
        session = re.search(r"ses-([^_]+)", stem)
        ses = session.group(1) if session else "unknown"
        rows.append({
            "subject": subject.group(1) if subject else path.parent.name,
            "session": ses,
            "date": f"{ses[:4]}-{ses[4:6]}-{ses[6:8]}" if len(ses) >= 8 else ses,
            "has_behavior": "behavior" in stem,
            "file": str(path.relative_to(root)),
        })
    return (pd.DataFrame(rows).sort_values(["subject", "session"])
            .reset_index(drop=True))


def load_scores(cache_dir=CACHE_DIR):
    """The sweep's per-fold reconstruction scores."""
    path = Path(cache_dir) / "results_scores.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — run reconstruction_error_test.py first, or "
            f"point CACHE_DIR at the folder its sweep filled")
    t0 = time.perf_counter()
    scores = pd.read_csv(path, usecols=["file", "method", "k", "rec_corr"])
    print(f"read {len(scores)} scored folds from {path.name} "
          f"({time.perf_counter() - t0:.1f}s)")
    return scores


def typicality(scores, max_k=30):
    """Distance of each recording's reconstruction curve from the average.

    The activity-reconstruction figure is corr(real, rebuilt) against number of
    dimensions, one curve per recording per method. This averages the folds,
    takes the mean curve across recordings as the reference, and scores each
    recording by its mean absolute departure from it. Small = typical.
    """
    curves = (scores[scores["k"] <= max_k]
              .groupby(["file", "method", "k"], as_index=False)["rec_corr"].mean())
    reference = (curves.groupby(["method", "k"], as_index=False)["rec_corr"]
                 .mean().rename(columns={"rec_corr": "mean_corr"}))
    merged = curves.merge(reference, on=["method", "k"])
    merged["gap"] = (merged["rec_corr"] - merged["mean_corr"]).abs()
    return (merged.groupby("file", as_index=False)
            .agg(distance_from_mean=("gap", "mean"),
                 mean_rec_corr=("rec_corr", "mean"))
            .sort_values("distance_from_mean").reset_index(drop=True))


def suggest_recordings(scores, recordings, n=SUGGEST_N, require_behavior=True):
    """Rank recordings by how typical their reconstruction is, and print them.

    Position and speed are needed throughout, so the ecephys-only files cannot
    be the example however typical their reconstruction looks.
    """
    ranked = typicality(scores)
    ranked["has_behavior"] = ranked["file"].isin(
        set(recordings.loc[recordings["has_behavior"], "file"]))
    ranked = ranked.merge(recordings[["file", "subject", "date"]],
                          on="file", how="left")
    eligible = (ranked[ranked["has_behavior"]] if require_behavior
                else ranked).reset_index(drop=True)

    print("\nrecordings closest to the mean reconstruction curve "
          "(k <= 30, averaged over methods and folds):")
    print(eligible.head(n)[["subject", "date", "distance_from_mean",
                            "mean_rec_corr", "file"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    dropped = ranked[~ranked["has_behavior"]]
    if require_behavior and len(dropped):
        print(f"excluded (no behaviour, so no position or speed): "
              f"{', '.join(dropped['file'])}")
    return eligible


def resolve_recording(recordings, eligible, pattern=None):
    """The row for `pattern`, or the most typical recording when it is None."""
    if pattern is None:
        chosen = eligible["file"].iloc[0]
        print(f"\nRECORDING is None — taking the suggestion: {chosen}")
    else:
        matches = [f for f in recordings["file"] if pattern in f]
        if not matches:
            raise ValueError(f"RECORDING={pattern!r} matched none of the files")
        chosen = matches[0]
        print(f"\nRECORDING={pattern!r} -> {chosen}")
    return recordings[recordings["file"] == chosen].iloc[0]


# -----------------------------------------------------------------------------
# THE SESSION, ON THE SWEEP'S OWN BIN GRID
# -----------------------------------------------------------------------------


def transform_rates(rates, how=RATE_TRANSFORM):
    """Identical to the sweep's: per unit, over the bins of this epoch."""
    if how == "none":
        return rates
    if how == "sqrt":
        return np.sqrt(rates)
    if how == "zscore":
        mu = rates.mean(axis=0, keepdims=True)
        sd = rates.std(axis=0, keepdims=True)
        return np.divide(rates - mu, sd, out=np.zeros_like(rates, dtype=np.float64),
                         where=sd > 1e-12)
    raise ValueError(f"unknown RATE_TRANSFORM: {how}")


def bandpass_sos(values, fs, lo, hi, order=4):
    """Zero-phase Butterworth bandpass in SOS form."""
    sos = sps.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, values)


def analytic_envelope(values):
    """Hilbert amplitude envelope, FFT-padded to a fast length."""
    n = len(values)
    return np.abs(sps.hilbert(values, N=next_fast_len(n)))[:n]


def bin_mean(sample_t, sample_v, edges):
    """Mean of a densely sampled signal within each bin, NaN where empty."""
    idx = np.searchsorted(edges, sample_t, side="right") - 1
    ok = (idx >= 0) & (idx < len(edges) - 1) & np.isfinite(sample_v)
    idx, vals = idx[ok], sample_v[ok]
    total = np.bincount(idx, weights=vals, minlength=len(edges) - 1)
    count = np.bincount(idx, minlength=len(edges) - 1)
    out = np.full(len(edges) - 1, np.nan)
    hit = count > 0
    out[hit] = total[hit] / count[hit]
    return out


def zscore(values):
    return (values - np.nanmean(values)) / (np.nanstd(values) + 1e-12)


def pick_lfp_key(nwb, hint=LFP_REGION_HINT):
    """The CA1 LFP if the file names one, else whatever LFP it has."""
    lfp_keys = [k for k in nwb.keys() if "lfp" in k.lower()]
    if not lfp_keys:
        raise KeyError("no LFP series in this file")
    preferred = [k for k in lfp_keys if hint in k.lower()]
    return (preferred[0] if preferred else lfp_keys[0]), lfp_keys


def load_session(row, bin_size_s=BIN_SIZE_S, epoch_mode=EPOCH,
                 transform=RATE_TRANSFORM, bands=BANDS, verbose=True):
    """The sweep's binned matrix, plus behaviour, LFP band power and metadata.

    The bin grid is rebuilt exactly as reconstruction_error_test.load_session
    built it — same epoch, same edges, same unit filter — because the cached
    embeddings are indexed into this grid and nothing downstream would notice a
    quiet disagreement.
    """
    path = (DOWNLOAD_DIR / row["file"]).resolve()
    t0 = time.perf_counter()
    nwb = nap.load_file(str(path))
    spikes_all = nwb["units"]

    epoch, epoch_kind = None, "full"
    if epoch_mode == "maze":
        try:
            position_all = nwb["Position"]
            epoch = nap.IntervalSet(start=float(position_all.index[0]),
                                    end=float(position_all.index[-1]))
            epoch_kind = "maze"
        except Exception:
            epoch = None
    if epoch is None:
        starts = [spikes_all[u].index[0] for u in spikes_all.index if len(spikes_all[u])]
        ends = [spikes_all[u].index[-1] for u in spikes_all.index if len(spikes_all[u])]
        epoch = nap.IntervalSet(start=float(min(starts)), end=float(max(ends)))

    spikes = spikes_all.restrict(epoch)
    t_start, t_end = float(epoch.start[0]), float(epoch.end[0])
    edges = np.arange(t_start, t_end, bin_size_s)
    centers = edges[:-1] + bin_size_s / 2

    spike_times = [np.sort(np.asarray(spikes[u].index.values, dtype=np.float64))
                   for u in spikes.index]
    counts = np.empty((len(edges) - 1, len(spike_times)), dtype=np.float32)
    for j, st in enumerate(spike_times):
        counts[:, j] = np.diff(np.searchsorted(st, edges))
    rates = counts / bin_size_s
    keep = rates.mean(axis=0) > MIN_RATE_HZ
    rates = rates[:, keep]

    meta = getattr(spikes_all, "metadata", None)
    if isinstance(meta, pd.DataFrame) and "cell_area" in meta.columns:
        regions = meta["cell_area"].astype(str).values[keep]
    else:
        regions = np.array(["all"] * int(keep.sum()))

    X = np.asarray(transform_rates(rates, transform), dtype=np.float64)
    del rates, counts
    gc.collect()

    # --- behaviour -----------------------------------------------------------
    position = nwb["Position"].restrict(epoch)
    pos_t = np.asarray(position.index.values, dtype=float)
    pos_x = np.asarray(position["x"].values, dtype=float)
    try:
        pos_y = np.asarray(position["y"].values, dtype=float)
    except Exception:
        pos_y = np.full_like(pos_x, np.nan)
    speed_obj = nwb["Speed"].restrict(epoch)
    speed_t = np.asarray(speed_obj.index.values, dtype=float)
    speed_v = np.asarray(speed_obj.values, dtype=float).ravel()

    track_x = bin_mean(pos_t, pos_x, edges)
    track_y = bin_mean(pos_t, pos_y, edges)
    speed = bin_mean(speed_t, speed_v, edges)

    # --- LFP and band power --------------------------------------------------
    lfp_key, lfp_keys = pick_lfp_key(nwb)
    lfp_obj = nwb[lfp_key].restrict(epoch)
    lfp_t = np.asarray(lfp_obj.index.values, dtype=float)
    lfp_v = np.asarray(lfp_obj.values, dtype=float).ravel()
    fs = float(1.0 / np.median(np.diff(lfp_t)))

    power, power_z, raw_rows = {}, {}, []
    for name, (lo, hi) in bands.items():
        env = analytic_envelope(bandpass_sos(lfp_v, fs, lo, hi))
        binned = bin_mean(lfp_t, env, edges)
        power[name] = binned
        power_z[name] = zscore(binned)
        raw_rows.append({"band": name, "range_hz": f"{lo:.0f}-{hi:.0f}",
                         "median": np.nanmedian(binned), "mean": np.nanmean(binned),
                         "p95": np.nanpercentile(binned, 95), "sd": np.nanstd(binned)})
        del env
        gc.collect()

    session = {
        "label": f"{row['subject']} {row['date']}", "file": row["file"],
        "epoch_kind": epoch_kind, "t_start": t_start, "t_end": t_end,
        "bin_size_s": bin_size_s, "edges": edges, "centers": centers,
        "X": X, "regions": regions, "n_units": X.shape[1], "n_bins": X.shape[0],
        "track_x": track_x, "track_y": track_y, "speed": speed,
        "bands": list(bands), "power": power, "power_z": power_z,
        "power_summary": pd.DataFrame(raw_rows),
        "lfp_key": lfp_key, "lfp_keys": lfp_keys, "fs": fs,
        "rate_transform": transform,
    }
    del spikes, spikes_all, spike_times, nwb, lfp_v, lfp_t
    gc.collect()

    if verbose:
        print(f"loaded {row['file']} in {time.perf_counter() - t0:.1f}s")
        print(f"    {epoch_kind} epoch {(t_end - t_start) / 60:.1f} min | "
              f"{session['n_units']} units | {session['n_bins']} bins | regions: "
              + ", ".join(f"{r} ({(regions == r).sum()})" for r in sorted(set(regions))))
        print(f"    LFP: {lfp_key} @ {fs:.0f} Hz"
              + (f"  (of {len(lfp_keys)}: {', '.join(lfp_keys)})"
                 if len(lfp_keys) > 1 else "")
              + ("" if LFP_REGION_HINT in lfp_key.lower()
                 else f"  [no '{LFP_REGION_HINT}' in the name — check this is CA1]"))
        print("\nband power on the analysis bins (raw envelope, before z-scoring)")
        print(session["power_summary"].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
    return session


# -----------------------------------------------------------------------------
# THE CACHED EMBEDDINGS
# -----------------------------------------------------------------------------


def scale_embedding(Y, X):
    """Match the embedding's radius to the data's, as the sweep does.

    It matters because the LLE weights come from distances in embedding space:
    an embedding on a different scale changes the neighbourhood geometry the
    reconstruction depends on.
    """
    Yc = Y - np.mean(Y)
    rad_y = np.percentile(Yc, 95)
    rad_x = np.percentile(X - np.mean(X), 95)
    if not np.isfinite(rad_y) or abs(rad_y) < 1e-12:
        return Yc
    return (rad_x / rad_y) * Yc


def load_embedding(session, method, cache_dir=CACHE_DIR):
    """One cached embedding, checked against the session that was just rebuilt.

    A mismatch is refused rather than reconciled: an embedding fit on a
    different unit filter, bin size or scaling has rows that mean something
    else, and nothing downstream would notice.
    """
    stem = Path(session["file"]).stem
    path = Path(cache_dir) / (f"{stem}__{session['bin_size_s'] * 1e3:.0f}ms"
                              f"__{method}.npz")
    if not path.exists():
        return None, None, f"no cache file ({path.name})"
    with np.load(path, allow_pickle=False) as stored:
        Y = np.asarray(stored["embedding"], dtype=np.float64)
        idx = np.asarray(stored["idx_embed"], dtype=np.int64)
        n_units = int(stored["n_units"])
        transform = (str(stored["rate_transform"])
                     if "rate_transform" in stored else "?")
    if n_units != session["n_units"]:
        return None, None, (f"cache has {n_units} units, this session rebuilt "
                            f"{session['n_units']} — grids disagree, not using it")
    if idx.max() >= session["n_bins"]:
        return None, None, (f"cache indexes bin {idx.max()} of "
                            f"{session['n_bins']} — epoch or bin size differs")
    if transform not in ("?", session["rate_transform"]):
        return None, None, f"cache was fit under transform {transform!r}"
    return Y, idx, f"{Y.shape[0]} x {Y.shape[1]}, stride {int(np.median(np.diff(idx)))}"


def load_embeddings(session, methods=METHODS, cache_dir=CACHE_DIR, verbose=True):
    """Every usable cached embedding for this session, already scaled."""
    if verbose:
        print(f"\nembeddings from {cache_dir}")
    out = {}
    for method in methods:
        Y, idx, note = load_embedding(session, method, cache_dir)
        if verbose:
            print(f"    {method:10s} {note}")
        if Y is not None:
            out[method] = {"Y": Y, "idx": idx,
                           "scaled": scale_embedding(Y, session["X"][idx])}
    if not out:
        raise RuntimeError("no usable cached embeddings for this recording")

    # The coarse grid every method has in common. Kernel PCA and Laplacian
    # eigenmaps were fit on every tenth bin, so this is where they can be
    # compared to PCA and to each other without inventing samples for them.
    common = out[list(out)[0]]["idx"]
    for method in out:
        common = np.intersect1d(common, out[method]["idx"])
    if verbose:
        print(f"    common grid: {len(common)} bins "
              f"({session['bin_size_s'] * np.median(np.diff(common)) * 1e3:.0f} "
              f"ms apart)")
    return out, common


# -----------------------------------------------------------------------------
# THE AR-HMM ON THE BAND POWER
# -----------------------------------------------------------------------------


def arhmm_observations(session, how=ARHMM_INPUT):
    """The band traces the AR-HMM sees, one column per band.

    The log envelope rather than the z-scored one by default: envelopes are
    close to log-normal and the model's observation noise is Gaussian. The
    sampler standardizes whatever it is handed either way.

    Bins where the LFP gave nothing are filled from the nearest finite bin
    rather than dropped — the AR model reads consecutive rows as consecutive in
    time, so deleting one would silently glue two distant moments together.
    """
    if how == "log_power":
        Y = np.column_stack([np.log10(np.maximum(session["power"][b], 1e-12))
                             for b in session["bands"]])
    elif how == "z_power":
        Y = np.column_stack([session["power_z"][b] for b in session["bands"]])
    else:
        raise ValueError(f"unknown ARHMM_INPUT: {how}")

    finite = np.isfinite(Y).all(axis=1)
    if not finite.all():
        print(f"{(~finite).sum()} of {len(Y)} bins had no LFP samples — filled "
              f"from the nearest finite bin to keep the lag structure intact")
        good = np.flatnonzero(finite)
        nearest = good[np.clip(np.searchsorted(good, np.arange(len(Y))),
                               0, len(good) - 1)]
        Y = Y[nearest]
    return Y


def fit_states(session, how=ARHMM_INPUT, max_bins=ARHMM_MAX_BINS,
               nlags=ARHMM_NLAGS, L=ARHMM_L, n_iter=ARHMM_ITER,
               burn_in=ARHMM_BURN_IN, kappa=ARHMM_KAPPA, alpha=ARHMM_ALPHA,
               gamma=ARHMM_GAMMA, min_frac=ARHMM_MIN_FRAC, seed=ARHMM_SEED):
    """Fit the AR-HMM and map its states back onto the session's bin grid.

    The AR design drops the first `nlags` bins, so state i belongs to bin
    i+nlags. The occupied states are relabelled by how much time they hold, so
    state 0 is the most common one and the numbering carries meaning across
    every figure below.
    """
    Y_full = arhmm_observations(session, how)
    n_bins = session["n_bins"]
    hmm_bins = np.arange(n_bins if max_bins is None else min(max_bins, n_bins))
    Y = Y_full[hmm_bins]

    print(f"\nfitting the sticky HDP-AR-HMM: {len(Y)} bins x {Y.shape[1]} bands, "
          f"AR order {nlags} ({nlags * session['bin_size_s'] * 1e3:.0f} ms lag), "
          f"L={L}, kappa={kappa}, {n_iter} iterations")
    print(f"    input: {how} of {', '.join(session['bands'])}")
    if max_bins is not None and max_bins < n_bins:
        print(f"    capped at {max_bins} of {n_bins} bins, contiguous from the "
              f"start — subsampling would break what one lag means")
    print("    the state step is a Python loop over time, so this is the slow "
          "part", flush=True)

    t0 = time.perf_counter()
    arhmm = fit_hdp_arhmm(Y, nlags=nlags, L=L, n_iter=n_iter, burn_in=burn_in,
                          alpha=alpha, kappa=kappa, gamma=gamma,
                          min_frac=min_frac, seed=seed)
    print(f"    fitted in {(time.perf_counter() - t0) / 60:.1f} min")
    if arhmm["K_hat"] >= L - 1:
        print(f"    WARNING: K_hat={arhmm['K_hat']} is at the truncation level "
              f"L={L}. Raise ARHMM_L — the model wanted more states than it was "
              f"allowed.")

    z_raw = arhmm["z"]
    state_bins = hmm_bins[nlags:]
    occupancy = np.bincount(z_raw, minlength=L)
    order = np.argsort(-occupancy)
    relabel = np.full(len(occupancy), -1)
    relabel[order[occupancy[order] > 0]] = np.arange(int((occupancy > 0).sum()))
    states = np.full(n_bins, -1, dtype=int)
    states[state_bins] = relabel[z_raw]
    n_states = int(states.max()) + 1

    print(f"    {n_states} occupied states over {len(state_bins)} labelled bins "
          f"({(states < 0).sum()} bins unlabelled)")
    return {"arhmm": arhmm, "states": states, "state_bins": state_bins,
            "relabel": relabel, "n_states": n_states, "burn_in": burn_in,
            "cmap": ListedColormap(
                plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:n_states])}


def run_lengths(labels):
    """(state, start index, length) for every contiguous run."""
    change = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(labels)]))
    return labels[starts], starts, stops - starts


def state_characterization(session, fit, verbose=True):
    """What each state is: occupancy, dwell time, band power, AR dynamics."""
    states, n_states = fit["states"], fit["n_states"]
    bands = session["bands"]
    run_state, _, run_len = run_lengths(states[fit["state_bins"]])
    bin_ms = session["bin_size_s"] * 1e3

    rows = []
    for s in range(n_states):
        sel = states == s
        runs = run_len[run_state == s]
        entry = {"state": s, "bins": int(sel.sum()),
                 "occupancy_%": 100 * sel.sum() / max((states >= 0).sum(), 1),
                 "n_runs": int(len(runs)),
                 "median_dwell_ms": float(np.median(runs) * bin_ms) if len(runs) else np.nan,
                 "p90_dwell_ms": float(np.percentile(runs, 90) * bin_ms) if len(runs) else np.nan}
        for band in bands:
            entry[f"{band}_mean_z"] = float(np.nanmean(session["power_z"][band][sel]))
            entry[f"{band}_median_z"] = float(np.nanmedian(session["power_z"][band][sel]))
        entry["speed_median"] = float(np.nanmedian(session["speed"][sel]))
        # Eigenvalues of the AR matrix say what the dynamics do: modulus near 1
        # is slow decay, a complex pair is an oscillation.
        A_k = fit["arhmm"]["As"][np.flatnonzero(fit["relabel"] == s)[0]][:, :len(bands)]
        eig = np.linalg.eigvals(A_k)
        entry["max_|eig|"] = float(np.max(np.abs(eig)))
        entry["oscillatory"] = bool(np.any(np.abs(eig.imag) > 1e-6))
        rows.append(entry)
    table = pd.DataFrame(rows)

    if verbose:
        print("\nwhat each state is — occupancy, dwell time, AR dynamics, speed")
        print(table[["state", "bins", "occupancy_%", "n_runs", "median_dwell_ms",
                     "p90_dwell_ms", "max_|eig|", "oscillatory", "speed_median"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print("\nmean band power by state (z, over the whole epoch)")
        print(table[["state"] + [f"{b}_mean_z" for b in bands]]
              .to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
        print("\nmedian band power by state (z)")
        print(table[["state"] + [f"{b}_median_z" for b in bands]]
              .to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    return table


# -----------------------------------------------------------------------------
# RANKING THE COMPONENTS BY DROP-ONE RECONSTRUCTION MSE
# -----------------------------------------------------------------------------
# What each component is worth to the reconstruction, which is not the same as
# the order the method returns them in. For PCA and kernel PCA that order is by
# eigenvalue and the ranking mostly confirms it. For Laplacian eigenmaps it is
# ascending graph-Laplacian eigenvalue — a smoothness ordering — and there is
# no reason its first component should be its most useful.


def lle_reconstruct(Y_train, X_train, Y_test, k_lle=K_LLE, lam=LAMBDA):
    """Out-of-sample LLE mapping, embedding -> activity (their new_LLE_pts).

    For each test point, take its k nearest neighbours among the training
    points in embedding space, solve the locally-linear weights that rebuild it
    from them, and apply those weights to the neighbours' activity.
    """
    k_lle = int(min(k_lle, len(Y_train)))
    nn = NearestNeighbors(n_neighbors=k_lle).fit(Y_train)
    _, idx = nn.kneighbors(Y_test)

    X_rec = np.empty((len(Y_test), X_train.shape[1]), dtype=np.float64)
    ones = np.ones(k_lle)
    for i, neighbours in enumerate(idx):
        Z = Y_train[neighbours] - Y_test[i]
        G = Z @ Z.T
        trace = np.trace(G)
        G.flat[:: k_lle + 1] += lam * (trace if trace > 0 else 1.0) / k_lle
        w = np.linalg.solve(G, ones)
        w /= w.sum()
        X_rec[i] = w @ X_train[neighbours]
    return X_rec


def cv_mse(Y_sub, X, splits):
    """Held-out mean squared error of the LLE reconstruction.

    Cross-validated because the LLE mapping rebuilds a point from its
    neighbours: query it against the set it was built from and every point
    finds itself, which measures memorization rather than structure.
    """
    total, count = 0.0, 0
    for train_idx, test_idx in splits:
        X_rec = lle_reconstruct(Y_sub[train_idx], X[train_idx], Y_sub[test_idx])
        resid = X[test_idx] - X_rec
        total += float((resid ** 2).sum())
        count += resid.size
    return total / count


def rank_components(session, embeddings, methods=METHODS, dims=RANK_DIMS,
                    folds=RANK_FOLDS, max_samples=RANK_MAX_SAMPLES,
                    seed=RANK_SEED, top_n=TOP_N, verbose=True):
    """Order each method's components by the MSE cost of dropping them.

    Reconstruct from the first `dims` components, then again from the same set
    with one removed; the increase in held-out MSE is that component's
    contribution. Returns {method: component order} and the long frame.
    """
    if verbose:
        print(f"\nranking components by drop-one held-out MSE ({dims} dims, "
              f"{folds} folds, <= {max_samples} samples)")
    ranking, frames = {}, []
    for method in methods:
        if method not in embeddings:
            continue
        idx = embeddings[method]["idx"]
        take = np.arange(len(idx))
        if len(take) > max_samples:
            take = np.unique(np.linspace(0, len(idx) - 1, max_samples).astype(int))
        Y_rank = embeddings[method]["scaled"][take]
        X_rank = session["X"][idx[take]]
        k = int(min(dims, Y_rank.shape[1]))

        splits = list(KFold(n_splits=folds, shuffle=True,
                            random_state=seed).split(X_rank))
        t0 = time.perf_counter()
        base = cv_mse(Y_rank[:, :k], X_rank, splits)
        contributions = np.empty(k)
        for j in range(k):
            cols = [c for c in range(k) if c != j]
            contributions[j] = cv_mse(Y_rank[:, cols], X_rank, splits) - base
        order = np.argsort(-contributions)

        frames.append(pd.DataFrame({
            "method": method, "component": np.arange(k),
            "delta_mse": contributions,
            "rank": np.argsort(np.argsort(-contributions))}))
        ranking[method] = order
        if verbose:
            print(f"    {method:10s} base MSE {base:.4f} | top {top_n}: "
                  + ", ".join(f"#{c} (+{contributions[c]:.4f})"
                              for c in order[:top_n])
                  + f" | {time.perf_counter() - t0:.0f}s", flush=True)
    return ranking, pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# FIGURES
# -----------------------------------------------------------------------------


def colour_limits(values, lo=2, hi=98):
    """Percentile limits that survive NaNs and constant input."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-9:
        vmin, vmax = float(finite.min()), float(finite.max())
    if vmax - vmin < 1e-9:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return float(vmin), float(vmax)


def plot_points(embeddings, method, n_points=MAP_MAX_POINTS, seed=0):
    """A subsample of a method's bins, as (rows into the embedding, bin ids)."""
    idx = embeddings[method]["idx"]
    take = np.arange(len(idx))
    if len(take) > n_points:
        take = np.sort(np.random.default_rng(seed)
                       .choice(len(idx), n_points, replace=False))
    return take, idx[take]


def plot_arhmm(session, fit, state_table):
    """Sampler traces, the state sequence, dwell times, band power, transitions.

    The transition matrix has its diagonal zeroed: kappa makes self-transitions
    dominate so completely that nothing else would be visible on the same
    colour scale, and the interesting question is where a state goes when it
    does leave.
    """
    states, n_states, cmap = fit["states"], fit["n_states"], fit["cmap"]
    arhmm, bands = fit["arhmm"], session["bands"]
    centers, bin_ms = session["centers"], session["bin_size_s"] * 1e3
    lab = states[fit["state_bins"]]
    run_state, _, run_len = run_lengths(lab)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f"{session['label']} — sticky HDP-AR-HMM on the "
                 f"{session['lfp_key']} band power")

    ax = axes[0, 0]
    ax.plot(arhmm["trace_K"], lw=1, color="0.3")
    ax.axvline(fit["burn_in"], color="k", ls="--", lw=0.9, label="burn-in")
    ax.set_xlabel("Gibbs iteration")
    ax.set_ylabel("used states")
    ax.set_title(f"states per iteration (estimate: {arhmm['K_hat']})", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(arhmm["trace_ll"], lw=1, color="0.3")
    ax.axvline(fit["burn_in"], color="k", ls="--", lw=0.9)
    ax.set_xlabel("Gibbs iteration")
    ax.set_ylabel("log-likelihood")
    ax.set_title("log-likelihood", fontsize=10)

    ax = axes[1, 0]
    ax.imshow(lab[None, :], aspect="auto", interpolation="nearest", cmap=cmap,
              vmin=-0.5, vmax=n_states - 0.5,
              extent=[centers[fit["state_bins"]][0] - centers[0],
                      centers[fit["state_bins"]][-1] - centers[0], 0, 1])
    ax.set_yticks([])
    ax.set_xlabel("time into epoch (s)")
    ax.set_title("inferred state sequence", fontsize=10)

    ax = axes[1, 1]
    for s in range(n_states):
        runs = run_len[run_state == s] * bin_ms
        if len(runs) < 2:
            continue
        ax.hist(runs, bins=np.logspace(np.log10(bin_ms),
                                       np.log10(max(runs.max(), 200)), 30),
                histtype="step", lw=1.4, label=f"state {s}", color=cmap(s))
    ax.set_xscale("log")
    ax.set_xlabel("dwell time (ms)")
    ax.set_ylabel("runs")
    ax.set_title("how long each state lasts", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2, 0]
    width = 0.8 / max(n_states, 1)
    for s in range(n_states):
        vals = [state_table.loc[s, f"{b}_median_z"] for b in bands]
        ax.bar(np.arange(len(bands)) + s * width - 0.4 + width / 2, vals,
               width=width, color=cmap(s), label=f"{s}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands, rotation=20, fontsize=8)
    ax.set_ylabel("median power (z)")
    ax.set_title("band power that characterizes each state", fontsize=10)
    ax.legend(fontsize=7, ncol=2, title="state", title_fontsize=7)

    ax = axes[2, 1]
    trans = np.zeros((n_states, n_states))
    np.add.at(trans, (lab[:-1], lab[1:]), 1)
    np.fill_diagonal(trans, 0)
    trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(trans, cmap="magma", vmin=0)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("P(next | leaving)", fontsize=8)
    ax.set_xlabel("to state")
    ax.set_ylabel("from state")
    ax.set_title("where a state goes when it leaves", fontsize=10)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_ranking(session, rank_table, dims=RANK_DIMS, top_n=TOP_N):
    """Drop-one MSE cost per component, one panel per method."""
    methods = list(dict.fromkeys(rank_table["method"]))
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4),
                             squeeze=False)
    fig.suptitle(f"{session['label']} — what dropping one component costs the "
                 f"reconstruction (from {dims} dims)")
    for c, method in enumerate(methods):
        ax = axes[0, c]
        sub = rank_table[rank_table["method"] == method]
        colours = ["#d62728" if r < top_n else METHOD_COLORS.get(method, "0.4")
                   for r in sub["rank"]]
        ax.bar(sub["component"], sub["delta_mse"], color=colours)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("component (as the method returns it)")
        if c == 0:
            ax.set_ylabel("increase in held-out MSE when dropped")
        ax.set_title(f"{method} — top {top_n} in red", fontsize=10)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_manifold(session, embeddings, fit, ranking, method,
                  n_points=MAP_MAX_POINTS):
    """The manifold in its top three components, coloured four ways.

    Top three by the drop-one ranking, not components 1-3. For PCA those
    usually coincide; for Laplacian eigenmaps they generally do not, which is
    the whole reason the ranking exists.
    """
    comps = ranking[method][:3]
    take, bins_here = plot_points(embeddings, method, n_points)
    coords = embeddings[method]["scaled"][np.ix_(take, comps)]
    n_states, cmap = fit["n_states"], fit["cmap"]

    with np.errstate(divide="ignore", invalid="ignore"):
        log_speed = np.log10(np.maximum(session["speed"][bins_here], 1e-2))
    layers = [
        ("track position x (cm)", session["track_x"][bins_here], "cividis", None),
        ("track position y (cm)", session["track_y"][bins_here], "cividis", None),
        ("log10 speed (cm/s)", log_speed, "magma", None),
        ("AR-HMM state", fit["states"][bins_here].astype(float), cmap,
         (-0.5, n_states - 0.5)),
    ]

    fig = plt.figure(figsize=(5.0 * len(layers), 4.6))
    fig.suptitle(f"{session['label']} — {method} — components "
                 + ", ".join(f"#{c}" for c in comps)
                 + " (top 3 by drop-one MSE)")
    for i, (title, values, cmap_i, limits) in enumerate(layers, start=1):
        ax = fig.add_subplot(1, len(layers), i, projection="3d")
        vmin, vmax = limits if limits else colour_limits(values)
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values,
                        cmap=cmap_i, s=2.0, alpha=0.55, vmin=vmin, vmax=vmax,
                        linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.10)
        cbar.set_label(title, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax.set_xlabel(f"#{comps[0]}", fontsize=8)
        ax.set_ylabel(f"#{comps[1]}", fontsize=8)
        ax.set_zlabel(f"#{comps[2]}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_covariates(session, embeddings, fit, ranking, method, top_n=TOP_N,
                    n_points=MAP_MAX_POINTS):
    """Top-ranked components against position, speed and every band.

    This is where a component either lines up with something measurable or does
    not: one that tracks position paints a gradient, one that only tracks state
    paints blocks of colour.
    """
    covariates = [("track x (cm)", session["track_x"]),
                  ("track y (cm)", session["track_y"]),
                  ("speed (cm/s)", session["speed"])]
    covariates += [(f"{b} power (z)", session["power_z"][b])
                   for b in session["bands"]]

    comps = ranking[method][:top_n]
    take, bins_here = plot_points(embeddings, method, n_points)
    coords = embeddings[method]["scaled"][np.ix_(take, comps)]
    state_here = fit["states"][bins_here]

    fig, axes = plt.subplots(len(comps), len(covariates),
                             figsize=(2.9 * len(covariates), 2.7 * len(comps)),
                             squeeze=False)
    fig.suptitle(f"{session['label']} — {method} — top {top_n} components "
                 f"against the covariates, coloured by AR-HMM state")
    for r, comp in enumerate(comps):
        score = coords[:, r]
        for c, (name, series) in enumerate(covariates):
            ax = axes[r, c]
            values = series[bins_here]
            ok = np.isfinite(values) & np.isfinite(score)
            ax.scatter(values[ok], score[ok], c=state_here[ok], cmap=fit["cmap"],
                       vmin=-0.5, vmax=fit["n_states"] - 0.5, s=1.5, alpha=0.4,
                       linewidths=0)
            if r == len(comps) - 1:
                ax.set_xlabel(name, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"component #{comp}", fontsize=8)
            ax.tick_params(labelsize=7)
            if ok.sum() > 10:
                ax.set_title(f"r = {np.corrcoef(values[ok], score[ok])[0, 1]:+.2f}",
                             fontsize=7)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_example_window(session, embeddings, fit, ranking, state_table,
                        start_s=None, window_s=EXAMPLE_WINDOW_S, top_n=TOP_N,
                        methods=MAP_METHODS):
    """A few seconds with everything on one time axis, shaded by state.

    start_s=None takes the window holding the most state changes, so the
    example shows switching rather than a quiet stretch. A method fit on every
    tenth bin is drawn as markers: joining those points with a line would imply
    a resolution it does not have.
    """
    states, cmap = fit["states"], fit["cmap"]
    centers, n_bins = session["centers"], session["n_bins"]
    window_bins = int(round(window_s / session["bin_size_s"]))

    if start_s is None:
        labelled = states >= 0
        changes = np.zeros(n_bins)
        changes[1:] = ((np.diff(states) != 0) & labelled[1:]
                       & labelled[:-1]).astype(float)
        density = np.convolve(changes, np.ones(window_bins), mode="valid")
        start_bin = int(np.argmax(density))
        print(f"\nstart_s is None — taking the busiest window: "
              f"{density[start_bin]:.0f} state changes starting "
              f"{centers[start_bin] - centers[0]:.1f}s into the epoch")
    else:
        start_bin = int(np.searchsorted(centers - centers[0], start_s))
    stop_bin = min(start_bin + window_bins, n_bins)
    win = np.arange(start_bin, stop_bin)
    t_win = centers[win] - centers[start_bin]

    rows = 3 + len(methods)
    fig, axes = plt.subplots(rows, 1, figsize=(13, 2.0 * rows), sharex=True)
    fig.suptitle(f"{session['label']} — {window_s:.0f} s from "
                 f"{centers[start_bin] - centers[0]:.1f}s into the epoch")

    run_s, run_i, run_n = run_lengths(states[win])
    for ax in axes:
        for s, i0, n in zip(run_s, run_i, run_n):
            if s < 0:
                continue
            ax.axvspan(t_win[i0], t_win[min(i0 + n, len(t_win) - 1)],
                       color=cmap(s), alpha=0.18, lw=0)

    ax = axes[0]
    ax.plot(t_win, session["track_x"][win], lw=1.4, color="#1f77b4", label="x")
    ax.plot(t_win, session["track_y"][win], lw=1.4, color="#2ca02c", label="y")
    ax.set_ylabel("position (cm)", fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    ax = axes[1]
    ax.plot(t_win, session["speed"][win], lw=1.4, color="0.2")
    ax.set_ylabel("speed (cm/s)", fontsize=9)

    ax = axes[2]
    for band in session["bands"]:
        ax.plot(t_win, session["power_z"][band][win], lw=1.2,
                color=BAND_COLORS.get(band, "0.4"), label=band)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_ylabel("band power (z)", fontsize=9)
    ax.legend(fontsize=7, ncol=len(session["bands"]), loc="upper right")

    for ax, method in zip(axes[3:], methods):
        if method not in embeddings or method not in ranking:
            ax.set_ylabel(f"{method}\n(not available)", fontsize=9)
            continue
        idx = embeddings[method]["idx"]
        here = (idx >= start_bin) & (idx < stop_bin)
        t_here = centers[idx[here]] - centers[start_bin]
        stride = int(np.median(np.diff(idx)))
        for j, comp in enumerate(ranking[method][:top_n]):
            shade = plt.get_cmap("viridis")(j / max(top_n - 1, 1))
            if stride <= 1:
                ax.plot(t_here, embeddings[method]["scaled"][here, comp], lw=1.2,
                        color=shade, label=f"#{comp}")
            else:
                ax.plot(t_here, embeddings[method]["scaled"][here, comp], "o-",
                        ms=4, lw=0.8, alpha=0.8, color=shade, label=f"#{comp}")
        ax.set_ylabel(f"{method}\nscore"
                      + (f"\n({stride * session['bin_size_s'] * 1e3:.0f} ms)"
                         if stride > 1 else ""), fontsize=9)
        ax.legend(fontsize=7, ncol=top_n, loc="upper right")

    axes[-1].set_xlabel("time in window (s)")
    axes[-1].set_xlim(t_win[0], t_win[-1])
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    present = sorted({int(s) for s in states[win] if s >= 0})
    print(f"states in this window: {present}")
    print(state_table[state_table["state"].isin(present)]
          [["state", "occupancy_%", "median_dwell_ms"]
           + [f"{b}_median_z" for b in session["bands"]]]
          .to_string(index=False, float_format=lambda v: f"{v:+.2f}"))


print("definitions loaded — run CELL 2 to analyse a recording")


# %% ===========================================================================
# CELL 2 — one recording, end to end
# ==============================================================================
# The knobs that change between runs sit here; everything else is a default in
# CELL 1. Start with RUN_MAX_BINS set low — the sampler's state step is a
# Python loop over time, so the full 40-minute epoch at 50 ms is the expensive
# part of this script by a wide margin.

RUN_RECORDING = None          # None = the suggestion; else a substring
RUN_MAX_BINS = 15000          # AR-HMM bins, contiguous from the start; None = all
RUN_ITER = 400                # Gibbs iterations
RUN_BURN_IN = 200
RUN_KAPPA = 50.0              # sticky bias: bigger = longer dwell times
RUN_L = 20                    # truncation; raise it if K_hat lands on L-1
RUN_WINDOW_START_S = None     # None = the busiest window in the session
RUN_METHODS = ("PCA", "Laplacian")

print("=" * 78)
print("HDP-AR-HMM states against the cached manifold embeddings")
print("=" * 78)

# 1. which recording
recordings = recording_table()
scores = load_scores()
eligible = suggest_recordings(scores, recordings)
row = resolve_recording(recordings, eligible, RUN_RECORDING)

# 2. the session on the sweep's bin grid, with behaviour and band power
session = load_session(row)

# 3. the embeddings the sweep already computed
embeddings, common_idx = load_embeddings(session)

# 4. the AR-HMM on the band power
fit = fit_states(session, max_bins=RUN_MAX_BINS, n_iter=RUN_ITER,
                 burn_in=RUN_BURN_IN, kappa=RUN_KAPPA, L=RUN_L)
state_table = state_characterization(session, fit)
plot_arhmm(session, fit, state_table)

# 5. what each component is worth to the reconstruction
ranking, rank_table = rank_components(session, embeddings)
plot_ranking(session, rank_table)

# 6. the manifolds and the components, against behaviour and band power
for method in RUN_METHODS:
    if method in embeddings and method in ranking:
        plot_manifold(session, embeddings, fit, ranking, method)
        plot_covariates(session, embeddings, fit, ranking, method)

# 7. one window with everything on a shared time axis
plot_example_window(session, embeddings, fit, ranking, state_table,
                    start_s=RUN_WINDOW_START_S, methods=RUN_METHODS)
