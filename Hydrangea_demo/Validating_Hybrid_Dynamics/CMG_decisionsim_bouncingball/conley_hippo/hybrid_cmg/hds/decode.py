"""
hds.decode -- the decoding pipeline, written to mirror the user's notebook:

  fine bins (STEP_MS, within trial)  -> window_counts(window w, step s)
  -> normalize: divide each bin by its population total, z-score each neuron
  -> corr_svd: eigendecomposition of the correlation matrix
  -> elbow_k: variance-explained elbow (max distance to the chord of the
     sorted eigenvalue curve, over the first K_MAX components)
  -> scores x = Z @ vec[:, :K]
  -> quotients: Lp = |x_{t+1}-x_t|/dt,  Lv = |x_{t+1}-2x_t+x_{t-1}|/dt^2

Trials are the simulated segments; no window or difference crosses a trial.
"""

import numpy as np
import warnings

STEP_MS = 10.0
K_MAX_R = 30


def fine_bins(segs, base):
    """Aggregate base-resolution counts to STEP_MS fine bins.
    Returns X (T, N) float32, trial id (T,), time-in-trial of bin centre (T,)."""
    f = int(round(STEP_MS / 1000.0 / base))
    Xs, tr, tt = [], [], []
    for i, c in enumerate(segs):
        n = (len(c) // f) * f
        Xi = c[:n].reshape(-1, f, c.shape[1]).sum(1).astype(np.float32)
        Xs.append(Xi)
        tr.append(np.full(len(Xi), i))
        tt.append((np.arange(len(Xi)) + 0.5) * STEP_MS / 1000.0)
    return np.vstack(Xs), np.concatenate(tr), np.concatenate(tt)


def window_counts(X, trial, tfine, w_ms, s_ms, centers=None):
    """Counts in windows of w_ms stepped by s_ms within each trial (user's
    function). If `centers` (dict trial -> fine-bin centre indices) is given,
    windows are centred on those fine bins instead (used to give every
    window size identical time stamps)."""
    nw, ns = int(round(w_ms / STEP_MS)), int(round(s_ms / STEP_MS))
    out, tt, tr = [], [], []
    for ti in np.unique(trial):
        m = np.flatnonzero(trial == ti)
        c = np.vstack([np.zeros((1, X.shape[1]), np.float64),
                       np.cumsum(X[m], 0, dtype=np.float64)])
        if centers is None:
            if len(m) < nw + 2 * ns:
                continue
            st = np.arange(0, len(m) - nw + 1, ns)
            tcen = tfine[m][st] - STEP_MS / 2000 + w_ms / 2000
        else:
            j = centers[ti]
            st = j - nw // 2
            tcen = tfine[m][j]
        out.append((c[st + nw] - c[st]).astype(np.float32))
        tt.append(tcen)
        tr.append(np.full(len(st), ti))
    return np.vstack(out), np.concatenate(tt), np.concatenate(tr)


def normalize(X, eps=1e-9, mode="per_neuron"):
    """Spike-count normalization.

    mode="per_neuron" (the intended one): divide each NEURON by its own total
    recorded spikes, then z-score that neuron across time. Dividing by a
    per-neuron constant and then z-scoring is algebraically the same as
    z-scoring, so this reduces to a per-neuron z-score; it is written out in
    full for clarity.

    mode="per_bin": divide each TIME BIN by the population total in that bin,
    then z-score. This removes the population firing rate, so it should only
    be used when the rate itself is a confound rather than signal.
    """
    X = np.asarray(X, np.float64)
    if mode == "per_bin":
        Y = X / np.maximum(X.sum(1, keepdims=True), eps)
    else:
        Y = X / np.maximum(X.sum(0, keepdims=True), eps)
    mu, sd = Y.mean(0), Y.std(0)
    sd = np.where(sd < eps, 1.0, sd)
    return ((Y - mu) / sd).astype(np.float32)


def corr_svd(X):
    """Normalize, then eigendecompose the correlation matrix.
    Returns eigenvalues (desc), eigenvectors (N, N), normalized data Z."""
    Z = normalize(X)
    C = (Z.T @ Z) / len(Z)
    ev, vec = np.linalg.eigh(C.astype(np.float64))
    o = np.argsort(ev)[::-1]
    return ev[o], vec[:, o].astype(np.float32), Z


def elbow_k(ev, kmax=K_MAX_R):
    """Variance-explained elbow: point of the sorted spectrum (first kmax)
    farthest from the chord joining its endpoints (axes scaled to [0,1])."""
    v = np.asarray(ev[:kmax], float)
    x = np.linspace(0, 1, len(v))
    y = (v - v.min()) / (v.max() - v.min() + 1e-12)
    # chord from (0,1) to (1,0): distance ∝ |x + y - 1|
    d = np.abs(x + y - 1) / np.sqrt(2)
    return int(np.argmax(d) + 1)


def participation_ratio(ev, kmax=None):
    v = np.asarray(ev if kmax is None else ev[:kmax], float)
    v = np.clip(v, 0, None)
    return v.sum() ** 2 / (v ** 2).sum()


def scores(X, K=None):
    ev, vec, Z = corr_svd(X)
    K = K or elbow_k(ev)
    return Z @ vec[:, :K], K, ev


def quotients_from_scores(x, tr, dt):
    sp = np.r_[False, tr[1:] == tr[:-1]]
    sn = np.r_[tr[:-1] == tr[1:], False]
    Lp = np.full(len(x), np.nan)
    Lp[1:] = np.linalg.norm(np.diff(x, axis=0), axis=1) / dt
    Lp[~sp] = np.nan
    Lv = np.full(len(x), np.nan)
    Lv[1:-1] = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=1) / dt ** 2
    Lv[~(sp & sn)] = np.nan
    return Lp, Lv


def quotients(X, tr, dt):
    x, K, ev = scores(X)
    Lp, Lv = quotients_from_scores(x, tr, dt)
    return Lp, Lv, K, x


def aligned(sig, tt, tr, ev_t, ev_tr, dt, win):
    """Event-aligned matrix (n_events, n_offsets) of sig (NaN outside trial)."""
    off = np.arange(int(np.floor(win[0] / dt)), int(np.ceil(win[1] / dt)) + 1)
    # per-trial search
    M = np.full((len(ev_t), len(off)), np.nan)
    for k, (t, ti) in enumerate(zip(ev_t, ev_tr)):
        m = np.flatnonzero(tr == ti)
        if not len(m):
            continue
        j = np.searchsorted(tt[m], t)
        idx = j + off
        ok = (idx >= 0) & (idx < len(m))
        M[k, ok] = sig[m[idx[ok]]]
    return off * dt, M


def nanmed(M, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(M, axis)


def nanmean_sem(M, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = np.nanmean(M, axis)
        n = np.sum(~np.isnan(M), axis)
        se = np.nanstd(M, axis) / np.sqrt(np.maximum(n, 1))
    return mu, se


class LipschitzGuard:
    """Guard detector from a bound on the RATE OF CHANGE of the time-varying
    Lipschitz estimate, computed on the delay-embedded state D(t).

      Lp(t)   = |D(t+1) - D(t)| / dt
      Lv(t)   = |D(t+1) - 2 D(t) + D(t-1)| / dt^2
      Lhat(t) = mean(Lv) / mean(Lp) over a sliding window of `window_s`
                (ratio of window means: the Lipschitz-type quotient)
      rate(t) = |log Lhat(t + W/2) - log Lhat(t - W/2)| / W
                (change of log-Lipschitz across the window, per second)

    A guard event fires where rate(t) > max_rate (optionally also where
    Lhat > L_max). Peaks closer than `min_gap_s` are merged. The detector is
    OFF by default (enabled=False): it only runs when called explicitly, and
    routing uses it only if enabled=True.
    """

    def __init__(self, window_s=0.1, max_rate=8.0, L_max=None, min_gap_s=0.3,
                 enabled=False):
        self.window_s, self.max_rate = window_s, max_rate
        self.L_max, self.min_gap_s, self.enabled = L_max, min_gap_s, enabled

    def trace(self, D, tr, dt):
        n = len(D)
        sp = np.r_[False, tr[1:] == tr[:-1]]
        sn = np.r_[tr[:-1] == tr[1:], False]
        Lp = np.full(n, np.nan)
        Lp[1:] = np.linalg.norm(np.diff(D, axis=0), axis=1) / dt
        Lp[~sp] = np.nan
        Lv = np.full(n, np.nan)
        Lv[1:-1] = np.linalg.norm(D[2:] - 2 * D[1:-1] + D[:-2], axis=1) / dt ** 2
        Lv[~(sp & sn)] = np.nan
        w = max(2, int(round(self.window_s / dt)))
        Lhat = np.full(n, np.nan)
        rate = np.full(n, np.nan)
        for ti in np.unique(tr):
            m = np.flatnonzero(tr == ti)
            if len(m) < 3 * w:
                continue
            a, v = Lp[m], Lv[m]
            ok = ~(np.isnan(a) | np.isnan(v))
            ca = np.concatenate([[0], np.cumsum(np.where(ok, a, 0))])
            cv = np.concatenate([[0], np.cumsum(np.where(ok, v, 0))])
            cn = np.concatenate([[0], np.cumsum(ok)])
            i = np.arange(len(m))
            lo = np.clip(i - w // 2, 0, len(m))
            hi = np.clip(i + w // 2 + 1, 0, len(m))
            cnt = np.maximum(cn[hi] - cn[lo], 1)
            lh = ((cv[hi] - cv[lo]) / cnt) / np.maximum((ca[hi] - ca[lo]) / cnt, 1e-12)
            Lhat[m] = lh
            r = np.full(len(m), np.nan)
            h = w // 2
            r[h:-h] = np.abs(np.log(lh[2 * h:] + 1e-12) - np.log(lh[:-2 * h] + 1e-12)) \
                / (2 * h * dt)
            rate[m] = r
        return Lhat, rate

    def detect(self, D, tr, tt, dt):
        Lhat, rate = self.trace(D, tr, dt)
        flag = rate > self.max_rate
        if self.L_max is not None:
            flag |= Lhat > self.L_max
        ev = []
        for ti in np.unique(tr):
            m = np.flatnonzero((tr == ti) & flag)
            last = -1e9
            for i in m:
                if tt[i] - last > self.min_gap_s:
                    ev.append((ti, tt[i]))
                    last = tt[i]
        return ev, Lhat, rate


def procrustes_resid(Xc, Yc):
    """Normalized orthogonal-Procrustes residual ||X R - Y||_F / ||Y||_F."""
    U_, S_, Vt_ = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    num = (Xc ** 2).sum() + (Yc ** 2).sum() - 2 * S_.sum()
    return np.sqrt(max(num, 0.0)) / np.linalg.norm(Yc)
