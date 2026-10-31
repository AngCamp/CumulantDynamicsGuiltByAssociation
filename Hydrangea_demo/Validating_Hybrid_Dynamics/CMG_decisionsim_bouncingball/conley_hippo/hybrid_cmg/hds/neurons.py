"""
hds.neurons -- readout from latent state to Poisson spiking.

  lambda_i(t) = r0_i * rate_gain[mode] * exp( g * w_i . phi(x(t)) + eta(t) )

  r0_i   lognormal baseline (median set by `median`)
  w_i    unit tuning directions in the 3D latent
  g      tuning gain (calibrated with N so the noise floor sits ~30 ms)
  eta    shared fast gain noise (from the simulator)
  phi    geometry DISTORTION applied before tuning (does not change dynamics,
         only how the attractor is laid out in neural space):
            Identity, RadialStretch, Shear, SineWarp, Compose
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

# ------------------------------- distortions ---------------------------------


class Identity:
    def __call__(self, X):
        return X


class RadialStretch:
    """r -> r * (1 + a cos(m theta)): squashes rings into m-lobed shapes."""

    def __init__(self, a=0.3, m=2):
        self.a, self.m = a, m

    def __call__(self, X):
        Y = X.copy()
        th = np.arctan2(X[:, 1], X[:, 0])
        f = 1 + self.a * np.cos(self.m * th)
        Y[:, 0] *= f
        Y[:, 1] *= f
        return Y


class Shear:
    def __init__(self, M):
        self.M = np.asarray(M, float)

    def __call__(self, X):
        return X @ self.M.T


class SineWarp:
    """x_j += a sin(b x_k): smooth nonlinear bend."""

    def __init__(self, a=0.2, b=3.0, j=1, k=0):
        self.a, self.b, self.j, self.k = a, b, j, k

    def __call__(self, X):
        Y = X.copy()
        Y[:, self.j] += self.a * np.sin(self.b * X[:, self.k])
        return Y


class Compose:
    def __init__(self, warps):
        self.warps = warps

    def __call__(self, X):
        for w in self.warps:
            X = w(X)
        return X

# --------------------------------- readout ------------------------------------


class Readout:
    def __init__(self, n=1200, gain=1.7, median=7.0, spread=0.6, warp=None,
                 seed=1):
        r = np.random.default_rng(seed)
        W = r.normal(size=(n, 3))
        self.W = W / np.linalg.norm(W, axis=1, keepdims=True)
        self.r0 = r.lognormal(np.log(median), spread, n)
        self.g = gain
        self.n = n
        self.warp = warp or Identity()

    def expected_rates(self, X, eta, mode_idx, system=None, idx=None):
        W = self.W if idx is None else self.W[idx]
        r0 = self.r0 if idx is None else self.r0[idx]
        rg = 1.0
        if system is not None:
            rgs = np.array([system.modes[m].rate_gain
                            for m in system.mode_names])
            rg = rgs[mode_idx][:, None]
        return r0 * rg * np.exp(self.g * self.warp(X) @ W.T + eta[:, None])

    def group(self, direction, frac=0.03):
        """Indices of neurons most aligned with a latent direction -- the
        'certain neurons' a guard listens to."""
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        score = self.W @ d
        k = max(1, int(frac * self.n))
        return np.argsort(-score)[:k]

    def spikes(self, rec, system, seed=2):
        """Poisson counts at the recording's base resolution, per trial."""
        r = np.random.default_rng(seed)
        base = rec['base']
        out = []
        for tr in range(rec['X'].shape[0]):
            lam = self.expected_rates(rec['X'][tr], rec['eta'][tr],
                                      rec['mode'][tr], system)
            out.append(r.poisson(lam * base).astype(np.uint8))
        return out


def poisson_control(segs, base, smooth_ms=100.0, seed=0):
    """User's control: per trial, Gaussian-smooth the counts (sigma =
    smooth_ms) to a slow rate estimate and resample Poisson -- keeps slow rate
    structure, destroys fast structure."""
    r = np.random.default_rng(seed)
    sig = smooth_ms / 1000.0 / base
    out = []
    for c in segs:
        rate = gaussian_filter1d(c.astype(np.float32), sig, axis=0,
                                 mode="nearest")
        out.append(r.poisson(np.clip(rate, 0, None)).astype(np.uint8))
    return out
