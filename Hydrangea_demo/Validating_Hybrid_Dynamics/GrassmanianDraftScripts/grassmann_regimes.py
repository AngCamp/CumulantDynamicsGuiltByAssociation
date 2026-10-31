"""
grassmann_regimes.py
====================

Multiscale Grassmannian regime segmentation of spike trains (numpy + scipy only).

Three time quantities
---------------------
    dt         binning resolution. Turns the spike trains (sums of Dirac deltas)
               into a continuous-valued N x T_b count matrix: the "map" to a
               smooth space. Samples inside each window are these bins.
    delta_tau  the time step. Windows are placed every delta_tau, and the
               smallest window equals delta_tau.
    win_tau    window lengths, win_tau_j = scales[j] * delta_tau.

For scale m (window = m * delta_tau) a fresh PCA is computed for EVERY placement
shifted by one delta_tau (not a re-use of one PCA). So each time step is covered
by m windows at scale m: one at the smallest scale, two at 2*delta_tau, etc.

Pipeline
--------
    1. bin spikes at dt                          -> X  (N x T_b)
    2. for each scale m, each placement p        -> U_{m,p} in Gr(k, N)
       (top-k eigenvectors of the window covariance)
    3. per scale, centre of mass of {U_{m,p}}    -> mu_m (Karcher or extrinsic mean)
       optionally R > 1 reference centres via Grassmannian k-means
    4. distance of every window to the centre(s) -> d_g = ||principal angles||
    5. per time step, aggregate the m covering windows (mean/median/max/min)
                                                 -> feature matrix F (B x n_features)
    6. cluster F: spectral (kNN, self-tuning) or hierarchical (scipy linkage)

Also: lipschitz_sweep(window, lags), the structure function
    D(lag) = median_t d_g(S(t), S(t+lag)),  L(lag) = D(lag)/lag,
with a knee detector to pick delta_tau.

Complexity (notation: T_b = T/dt bins, l = delta_tau/dt bins per step,
B = T/delta_tau steps, L_j = m_j*l, k components, R references, I Karcher iters)
------------------------------------------------------------------------------
    binning                O(n_spikes + N*T_b)
    step scatter matrices  O(N^2 * T_b)          once, shared by all cov-path scales
    cov-path scale (L>=N)  O(B*N^3)              per scale, INDEPENDENT of window length
    gram-path scale (L<N)  O(B*(N*L^2 + L^3))    per scale
    centre (Karcher)       O(I*B*N*k^2)          per scale
    distances              O(R*B*N*k^2)          per scale
    features               O(B*sum_j m_j)
    spectral clustering    O(B log B) kNN + O(B*n_neighbors) per Lanczos iteration
    hierarchical           O(M^2) time and memory, M = min(B, max_hier_points),
                           + O(B log M) 1-NN assignment of the rest
    memory                 counts N*T_b bytes (uint8), bases S*B*N*k*4 bytes
Measured (2-core cloud VM, numpy 2.5 / scipy 1.18; time ~ param^p):
    duration T          total p = 1.09 (every stage ~linear)
    neurons N           bases 1.21, centres 1.46, distances 0.32  (N = 25..200)
    window length       0.02 on the covariance path (200 ms .. 3.2 s)
    bins per step 1/dt  0.31 (only the shared scatter-matrix build grows)
    k                   bases 0.02, centres 1.01, distances 1.25
    spectral vs B       1.05;  hierarchical without subsampling vs B 2.83
    wall clock          100 neurons, 10 min, 5 scales to 800 ms, dt 5 ms:
                        ~56 s (Karcher) / ~47 s (extrinsic); 1 h ~ 5-6 min.
    The centre step (Grassmannian k-means + Karcher) is the largest cost;
    mean="extrinsic" cuts it by about a third with the same accuracy.

References that informed the method
-----------------------------------
 * Edelman, Arias & Smith (1998) "The geometry of algorithms with orthogonality
   constraints", SIAM J. Matrix Anal. Appl. -- Grassmann exp map / geodesics.
 * Absil, Mahony & Sepulchre (2004) "Riemannian geometry of Grassmann manifolds
   with a view on algorithmic computation", Acta Appl. Math.; and their book
   "Optimization Algorithms on Matrix Manifolds" (2008).
 * Bendokat, Zimmermann & Absil (2024) "A Grassmann manifold handbook: basic
   geometry and computational aspects", Adv. Comput. Math. (arXiv:2011.13699)
   -- the numerically stable log map used here.
 * Bjorck & Golub (1973) "Numerical methods for computing angles between linear
   subspaces"; Knyazev & Argentati (2002) SIAM J. Sci. Comput. -- sine/cosine
   combination for accurate small principal angles.
 * Karcher (1977) "Riemannian center of mass and mollifier smoothing".
 * Turaga, Veeraraghavan, Srivastava & Chellappa (2011) "Statistical
   computations on Grassmann and Stiefel manifolds...", IEEE TPAMI -- extrinsic
   vs intrinsic means, clustering on the Grassmannian.
 * Hamm & Lee (2008) "Grassmann discriminant analysis", ICML -- projection
   (chordal) metric, P = U U^T embedding.
 * Ng, Jordan & Weiss (2001) "On spectral clustering"; Zelnik-Manor & Perona
   (2004) "Self-tuning spectral clustering"; von Luxburg (2007) "A tutorial on
   spectral clustering".
 * Ward (1963); Mullner (2011) "Modern hierarchical, agglomerative clustering
   algorithms" (arXiv:1109.2378) -- what scipy.cluster.hierarchy implements.
 * Satopaa et al. (2011) "Finding a 'Kneedle' in a haystack" -- knee detection.
 * Cunningham & Yu (2014) "Dimensionality reduction for large-scale neural
   recordings", Nat. Neurosci. -- binning/smoothing spikes before PCA.
 * Lindeberg (1994) "Scale-Space Theory in Computer Vision" -- multiscale view.

Usage
-----
    model = MultiscaleGrassmannRegimes(dt=1e-3, delta_tau=0.05,
                                       scales=(1, 2, 4, 8, 16), k=3,
                                       clustering="spectral", n_clusters=3)
    model.fit(spike_times)          # list of 1-D arrays, one per neuron
    model.labels_, model.step_times_, model.features_
    model.cluster(method="hierarchical", n_clusters=4)   # re-cluster, no recompute
    model.lipschitz_sweep(window=0.05, lags=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05])

    python grassmann_regimes.py --test     # synthetic recovery test
    python grassmann_regimes.py --bench    # runtime scaling
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans2
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------
# Grassmannian geometry helpers (batched; bases are N x k orthonormal matrices)
# ----------------------------------------------------------------------------


def principal_angles(A, B):
    """Principal angles between span(A) and span(B), ascending, shape (..., k).

    A, B: (..., N, k) orthonormal, broadcastable. Uses cos from svd(A^T B) and
    sin from svd(B - A A^T B) combined with arctan2, which is accurate for both
    small and large angles.
    """
    M = np.swapaxes(A, -1, -2) @ B
    c = np.linalg.svd(M, compute_uv=False)                     # descending
    H = B - A @ M
    s = np.linalg.svd(H, compute_uv=False)[..., ::-1]          # ascending
    return np.arctan2(np.clip(s, 0, 1), np.clip(c, 0, 1))


def geodesic_distance(A, B):
    """Arc-length (geodesic) distance on Gr(k, N): sqrt(sum theta_i^2)."""
    return np.linalg.norm(principal_angles(A, B), axis=-1)


def grassmann_log(mu, U):
    """Log map at mu (N,k) of points U (..., N, k). Returns tangent (..., N, k).

    Stable form (Bendokat-Zimmermann-Absil): SVD of mu^T U = W C Z^T,
    H = U Z - mu W C has orthogonal columns of norm sin(theta),
    Delta = H diag(theta/sin theta) W^T.
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
    """Exp map at mu (N,k) of tangent D (N,k) (Edelman-Arias-Smith)."""
    Q, s, Rt = np.linalg.svd(D, full_matrices=False)
    Y = (mu @ Rt.T) * np.cos(s) @ Rt + (Q * np.sin(s)) @ Rt
    Y, _ = np.linalg.qr(Y)
    return Y


def extrinsic_mean(U, k, weights=None):
    """Top-k eigenvectors of the mean projection (1/n) sum U U^T."""
    if weights is None:
        P = np.einsum("wnk,wmk->nm", U, U) / len(U)
    else:
        P = np.einsum("w,wnk,wmk->nm", weights, U, U) / weights.sum()
    w, V = np.linalg.eigh(P)
    return V[:, ::-1][:, :k]


def karcher_mean(U, k, n_iter=20, tol=1e-6, chunk=20000, init=None):
    """Riemannian centre of mass of points U (n, N, k) on Gr(k, N)."""
    mu = extrinsic_mean(U, k) if init is None else init
    for _ in range(n_iter):
        D = np.zeros_like(mu)
        for a in range(0, len(U), chunk):
            D += grassmann_log(mu, U[a:a + chunk]).sum(axis=0)
        D /= len(U)
        mu = grassmann_exp(mu, D)
        if np.linalg.norm(D) < tol:
            break
    return mu


def grassmann_kmeans(U, n_refs, k, n_iter=25, rng=None):
    """Lloyd k-means on Gr(k,N) with the chordal metric; extrinsic-mean centres.

    chordal^2(U, mu) = k - ||mu^T U||_F^2. Returns centres (n_refs, N, k), labels.
    """
    rng = np.random.default_rng(rng)
    n = len(U)
    cents = [U[rng.integers(n)]]
    for _ in range(1, n_refs):                                  # k-means++
        d2 = np.min([k - np.sum((c.T @ U) ** 2, axis=(1, 2)) for c in cents], axis=0)
        d2 = np.clip(d2, 0, None)
        p = d2 / d2.sum() if d2.sum() > 0 else None
        cents.append(U[rng.choice(n, p=p)])
    cents = np.stack(cents)
    lab = np.zeros(n, int)
    for it in range(n_iter):
        sim = np.stack([np.sum((c.T @ U) ** 2, axis=(1, 2)) for c in cents], axis=1)
        new = np.argmax(sim, axis=1)
        if it > 0 and np.array_equal(new, lab):
            break
        lab = new
        for r in range(n_refs):
            if np.any(lab == r):
                cents[r] = extrinsic_mean(U[lab == r], k)
    return cents, lab


# ----------------------------------------------------------------------------
# Spike binning
# ----------------------------------------------------------------------------


def bin_spikes(spike_times, dt, neuron_ids=None, t_start=None, t_stop=None):
    """Bin spikes into an (N, T_b) count matrix (compact integer dtype).

    spike_times : list of 1-D arrays (one per neuron), or a single 1-D array of
                  spike times together with `neuron_ids` (same length).
    """
    if neuron_ids is not None:
        spike_times = np.asarray(spike_times, float)
        neuron_ids = np.asarray(neuron_ids)
        ids = np.unique(neuron_ids)
        trains = [spike_times[neuron_ids == i] for i in ids]
    elif isinstance(spike_times, np.ndarray) and spike_times.ndim == 1 and spike_times.dtype != object:
        trains = [spike_times.astype(float)]
    else:
        trains = [np.asarray(s, float).ravel() for s in spike_times]
    allt = [s for s in trains if s.size]
    if t_start is None:
        t_start = min(s.min() for s in allt) if allt else 0.0
    if t_stop is None:
        t_stop = max(s.max() for s in allt) + dt if allt else dt
    Tb = int(np.floor((t_stop - t_start) / dt + 1e-9))
    counts = np.zeros((len(trains), Tb), np.uint16)
    for n, s in enumerate(trains):
        idx = np.floor((s - t_start) / dt).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < Tb)]
        c = np.bincount(idx, minlength=Tb)
        counts[n] = np.minimum(c, np.iinfo(np.uint16).max)
    if counts.max(initial=0) < 256:
        counts = counts.astype(np.uint8)
    return counts, float(t_start)


# ----------------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------------


class MultiscaleGrassmannRegimes:
    """Multiscale sliding-window PCA -> Grassmannian distances -> regime clusters.

    Parameters
    ----------
    dt : float              bin width (s) used to turn spikes into counts. Default
                            5 ms: count covariance SNR grows ~ linearly with
                            dt (signal ~ (rate*dt)^2, Poisson noise ~ rate*dt)
                            up to the timescale of the shared fluctuations; in
                            the synthetic test 1 ms bins fail and 5 ms succeed.
    delta_tau : float       time step (s); also the smallest window.
    scales : ints           window lengths as multiples of delta_tau.
    k : int                 subspace dimension (number of PCs kept).
    transform : 'sqrt'|'none'  variance-stabilising transform of counts.
    zscore : bool           divide each neuron by its global std (PCA ~ correlation).
    mean : 'karcher'|'extrinsic'   centre of mass per scale.
    n_references : int|None 1 = distance to the single centre of mass per scale.
                            Blind to regimes that sit equidistant from that
                            centre (in the test, 3 regimes -> ARI ~ 0.1).
                            R > 1 = distances to R centres from Grassmannian
                            k-means per scale. None (default) = n_clusters.
    features : 'distance'|'angles'  per reference, d_g (1 number) or the k
                            principal angles.
    aggregate : 'mean'|'median'|'max'|'min'  how the m windows covering a step
                            are combined.
    standardize : bool      z-score feature columns before clustering.
    clustering : 'spectral'|'hierarchical'
    n_clusters : int
    linkage_method : str    scipy linkage method ('ward', 'average', ...).
    max_hier_points : int   hierarchical clustering runs on at most this many
                            steps (O(M^2) memory); the rest get 1-NN labels.
    n_neighbors, local_scale_k : kNN graph size and self-tuning scale neighbour.
    min_dwell : int         merge label runs shorter than this many steps.
    min_eig_frac : float    window invalid if lambda_k < min_eig_frac * trace.
    keep_bases : bool       keep per-window bases (needed for re-centring).
    mem_budget_mb : float   chunk size target.
    random_state : int
    """

    def __init__(self, dt=5e-3, delta_tau=0.05, scales=(1, 2, 4, 8, 16), k=3,
                 transform="sqrt", zscore=True, mean="karcher", karcher_iters=20,
                 karcher_tol=1e-7, n_references=None, features="distance",
                 aggregate="mean", standardize=True, clustering="spectral",
                 n_clusters=3, linkage_method="ward", max_hier_points=5000,
                 n_neighbors=15, local_scale_k=7, min_dwell=1, min_eig_frac=1e-8,
                 keep_bases=True, mem_budget_mb=256, random_state=0,
                 t_start=None, t_stop=None):
        self.dt = float(dt)
        self.delta_tau = float(delta_tau)
        self.scales = tuple(int(m) for m in scales)
        self.k = int(k)
        self.transform = transform
        self.zscore = zscore
        self.mean = mean
        self.karcher_iters = karcher_iters
        self.karcher_tol = karcher_tol
        self.n_references = None if n_references is None else int(n_references)
        self.features = features
        self.aggregate = aggregate
        self.standardize = standardize
        self.clustering = clustering
        self.n_clusters = int(n_clusters)
        self.linkage_method = linkage_method
        self.max_hier_points = int(max_hier_points)
        self.n_neighbors = int(n_neighbors)
        self.local_scale_k = int(local_scale_k)
        self.min_dwell = int(min_dwell)
        self.min_eig_frac = float(min_eig_frac)
        self.keep_bases = keep_bases
        self.mem_budget_mb = float(mem_budget_mb)
        self.random_state = random_state
        self.t_start = t_start
        self.t_stop = t_stop
        self.timings_ = {}

    # ------------------------------------------------------------------ fit
    def fit(self, spike_times, neuron_ids=None):
        tim = {}
        t0 = time.perf_counter()
        self.bin(spike_times, neuron_ids)
        tim["bin"] = time.perf_counter() - t0

        ell = int(round(self.delta_tau / self.dt))
        if abs(ell * self.dt - self.delta_tau) > 1e-9 * max(1.0, self.delta_tau):
            raise ValueError("delta_tau must be an integer multiple of dt")
        if ell * min(self.scales) <= self.k:
            raise ValueError("smallest window must contain more than k bins")
        self.step_bins_ = ell
        N, Tb = self.counts_.shape
        B = Tb // ell
        if B < max(self.scales):
            raise ValueError("recording shorter than the largest window")
        self.n_steps_ = B
        self.step_times_ = self.t0_ + (np.arange(B) + 0.5) * self.delta_tau

        t0 = time.perf_counter()
        bases = self._compute_bases(self.counts_, ell, self.scales)
        tim["bases"] = time.perf_counter() - t0

        self.scale_info_ = {}
        t0 = time.perf_counter()
        for m in self.scales:
            U, lam, valid = bases[m]
            info = dict(window_s=m * self.delta_tau, n_windows=len(U), valid=valid,
                        eigvals=lam,
                        eigengap=np.divide(lam[:, self.k - 1], lam[:, self.k],
                                           out=np.full(len(lam), np.inf),
                                           where=lam[:, self.k] > 0),
                        window_times=self.t0_ + (np.arange(len(U)) + m / 2) * self.delta_tau)
            if self.keep_bases:
                info["bases"] = U
            info["_U"] = U
            self.scale_info_[m] = info
        self._compute_centres()
        tim["centres"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._compute_window_features()
        tim["distances"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._build_step_features()
        tim["features"] = time.perf_counter() - t0
        if not self.keep_bases:
            for info in self.scale_info_.values():
                info.pop("_U", None)

        t0 = time.perf_counter()
        self.cluster()
        tim["cluster"] = time.perf_counter() - t0
        self.timings_ = tim
        return self

    # ------------------------------------------------------------- binning
    def bin(self, spike_times, neuron_ids=None):
        self.counts_, self.t0_ = bin_spikes(spike_times, self.dt, neuron_ids,
                                            self.t_start, self.t_stop)
        N, Tb = self.counts_.shape
        # global per-neuron mean/std of transformed counts (chunked)
        s1 = np.zeros(N)
        s2 = np.zeros(N)
        step = max(1, int(self.mem_budget_mb * 2**20 / (8 * max(N, 1))))
        for a in range(0, Tb, step):
            x = self._transform(self.counts_[:, a:a + step])
            s1 += x.sum(1)
            s2 += (x * x).sum(1)
        mu = s1 / Tb
        var = np.maximum(s2 / Tb - mu**2, 0)
        sd = np.sqrt(var)
        sd[sd <= 0] = 1.0
        self._mu = mu
        self._inv_sd = (1.0 / sd) if self.zscore else np.ones(N)
        return self.counts_

    def _transform(self, c):
        x = c.astype(np.float64)
        if self.transform == "sqrt":
            np.sqrt(x, out=x)
        elif self.transform != "none":
            raise ValueError("transform must be 'sqrt' or 'none'")
        return x

    def _load(self, X, a, b):
        x = self._transform(X[:, a:b])
        x -= self._mu[:, None]
        x *= self._inv_sd[:, None]
        return x

    # ------------------------------------------------------ PCA per window
    def _compute_bases(self, X, ell, scales):
        """Top-k bases for every window of m steps (stride one step), all scales.

        Covariance path (L >= N): window covariance from cumulative sums of
        per-step scatter matrices -> cost per window O(N^2 + N^3), independent
        of L. Gram path (L < N): eigh of the L x L Gram matrix of the window.
        Returns {m: (U float32 (n_win,N,k), eigvals (n_win,k+1), valid bool)}.
        """
        k = self.k
        N, Tb = X.shape
        B = Tb // ell
        mmax = max(scales)
        out = {}
        for m in scales:
            nw = B - m + 1
            out[m] = (np.full((nw, N, k), np.nan, np.float32), np.zeros((nw, k + 1)),
                      np.zeros(nw, bool))
        cov_sc = [m for m in scales if m * ell >= N]
        gram_sc = [m for m in scales if m * ell < N]

        per_step = 8 * N * ell * 3
        if cov_sc:
            per_step += 8 * N * N * 3
        for m in gram_sc:
            L = m * ell
            per_step += 8 * (N * L * 2 + L * L * 2)
        chunk = int(max(1, self.mem_budget_mb * 2**20 // per_step))

        for c0 in range(0, B, chunk):
            c1 = min(B, c0 + chunk)
            e1 = min(B, c1 + mmax - 1)
            ns = e1 - c0
            seg = self._load(X, c0 * ell, e1 * ell)                # (N, ns*ell)
            if cov_sc:
                steps = seg.reshape(N, ns, ell).transpose(1, 0, 2)  # (ns, N, ell)
                css = np.zeros((ns + 1, N))
                np.cumsum(steps.sum(axis=2), axis=0, out=css[1:])
                csG = np.zeros((ns + 1, N, N))
                np.cumsum(steps @ steps.transpose(0, 2, 1), axis=0, out=csG[1:])
            for m in scales:
                p1 = min(c1, B - m + 1)
                if p1 <= c0:
                    continue
                j = np.arange(c0, p1) - c0
                L = m * ell
                if m in cov_sc:
                    S2 = csG[j + m] - csG[j]
                    sv = css[j + m] - css[j]
                    C = (S2 - sv[:, :, None] * sv[:, None, :] / L) / (L - 1)
                    w, V = np.linalg.eigh(C)
                    w = w[:, ::-1]
                    lam = w[:, :k + 1]
                    U = V[:, :, ::-1][:, :, :k]
                    total = w.clip(0).sum(1)
                else:
                    Xw = sliding_window_view(seg, L, axis=1)[:, j * ell, :].transpose(1, 0, 2)
                    Xc = Xw - Xw.mean(axis=2, keepdims=True)
                    G = Xc.transpose(0, 2, 1) @ Xc                   # (nw, L, L)
                    w, V = np.linalg.eigh(G)
                    w = w[:, ::-1]
                    V = V[:, :, ::-1]
                    lam = np.zeros((len(j), k + 1))
                    kk = min(k + 1, L)
                    lam[:, :kk] = w[:, :kk] / (L - 1)
                    sq = np.sqrt(np.clip(w[:, :k], 1e-300, None))
                    U = (Xc @ V[:, :, :k]) / sq[:, None, :]
                    total = w.clip(0).sum(1) / (L - 1)
                valid = (lam[:, k - 1] > self.min_eig_frac * np.maximum(total, 1e-300)) \
                    & np.all(np.isfinite(U), axis=(1, 2))
                Uo, lo, vo = out[m]
                Uo[c0:p1][valid] = U[valid]
                lo[c0:p1] = lam
                vo[c0:p1] = valid
        return out

    # ------------------------------------------------------------ centres
    def _compute_centres(self):
        rng = np.random.default_rng(self.random_state)
        for m, info in self.scale_info_.items():
            Uv = info["_U"][info["valid"]].astype(np.float64)
            if len(Uv) == 0:
                raise RuntimeError(f"no valid windows at scale {m}")
            R = self.n_clusters if self.n_references is None else self.n_references
            if R == 1:
                if self.mean == "karcher":
                    mu = karcher_mean(Uv, self.k, self.karcher_iters, self.karcher_tol)
                elif self.mean == "extrinsic":
                    mu = extrinsic_mean(Uv, self.k)
                else:
                    raise ValueError("mean must be 'karcher' or 'extrinsic'")
                refs = mu[None]
            else:
                refs, lab = grassmann_kmeans(Uv, R, self.k, rng=rng)
                if self.mean == "karcher":
                    refs = np.stack([karcher_mean(Uv[lab == r], self.k, self.karcher_iters,
                                                  self.karcher_tol, init=refs[r])
                                     if np.any(lab == r) else refs[r]
                                     for r in range(R)])
            info["centres"] = refs

    # ---------------------------------------------- per-window distances
    def _compute_window_features(self, chunk=20000):
        k = self.k
        for m, info in self.scale_info_.items():
            U = info["_U"]
            valid = info["valid"]
            refs = info["centres"]
            R = len(refs)
            ang = np.full((len(U), R, k), np.nan)
            idx = np.flatnonzero(valid)
            for a in range(0, len(idx), chunk):
                ii = idx[a:a + chunk]
                Ui = U[ii].astype(np.float64)
                for r in range(R):
                    ang[ii, r] = principal_angles(refs[r], Ui)
            info["angles"] = ang                                  # (n_win, R, k)
            info["distance"] = np.linalg.norm(ang, axis=2)        # (n_win, R)

    # ------------------------------------------------ per-step features
    def _build_step_features(self):
        B = self.n_steps_
        cols, names = [], []
        aggf = {"mean": np.nanmean, "median": np.nanmedian,
                "max": np.nanmax, "min": np.nanmin}[self.aggregate]
        for m, info in self.scale_info_.items():
            if self.features == "distance":
                v = info["distance"]
                nm = [f"d_g[scale={m},ref={r}]" for r in range(v.shape[1])]
            elif self.features == "angles":
                v = info["angles"].reshape(len(info["angles"]), -1)
                R = info["angles"].shape[1]
                nm = [f"theta{i + 1}[scale={m},ref={r}]" for r in range(R) for i in range(self.k)]
            else:
                raise ValueError("features must be 'distance' or 'angles'")
            pad = np.full((m - 1, v.shape[1]), np.nan)
            vp = np.concatenate([pad, v, pad], axis=0)            # (B+m-1, F)
            sw = sliding_window_view(vp, m, axis=0)               # (B, F, m)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cols.append(aggf(sw, axis=-1))
            names += nm
        F = np.concatenate(cols, axis=1)
        self.raw_features_ = F
        self.feature_names_ = names
        Fz = F.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(Fz, axis=0)
            sd = np.nanstd(Fz, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        if self.standardize:
            Fz = (Fz - mu) / sd
            Fz[~np.isfinite(Fz)] = 0.0
        else:
            bad = ~np.isfinite(Fz)
            Fz[bad] = np.broadcast_to(mu, Fz.shape)[bad]
        self.features_ = Fz

    # ---------------------------------------------------------- clustering
    def cluster(self, method=None, n_clusters=None, features=None):
        """(Re)cluster the per-step feature matrix. Returns labels (B,)."""
        method = method or self.clustering
        c = int(n_clusters or self.n_clusters)
        F = self.features_ if features is None else features
        rng = np.random.default_rng(self.random_state)
        if method == "spectral":
            lab = _spectral_clustering(F, c, self.n_neighbors, self.local_scale_k, rng)
        elif method == "hierarchical":
            lab = _hierarchical_clustering(F, c, self.linkage_method,
                                           self.max_hier_points, rng)
        else:
            raise ValueError("method must be 'spectral' or 'hierarchical'")
        if self.min_dwell > 1:
            lab = smooth_labels(lab, self.min_dwell)
        lab = _relabel_by_appearance(lab)
        self.labels_ = lab
        return lab

    # ---------------------------------------------- Lipschitz / structure
    def lipschitz_sweep(self, lags, window=None, spike_times=None, neuron_ids=None,
                        t_range=None, stat="median"):
        """Structure function of the subspace curve, to choose delta_tau.

        mode 'tied' (window=None, the default; window = lag, i.e. the
        "smallest window = time step" design):
            D(lag) = stat_t d_g(S_lag(t), S_lag(t+lag)), adjacent NON-overlapping
            windows of length lag.  Small lags: estimation noise dominates,
            D ~ eps(lag), L = D/lag blows up.  Large lags: L flattens towards the
            true speed of the curve.  The knee of log L vs log lag (Kneedle) is
            the suggested delta_tau.
        mode 'fixed' (window given): one window length, displaced by each lag.
            For lag < window the two windows overlap, so D(lag) there measures
            overlap decorrelation, not dynamics; the knee is searched only over
            lag >= window.
        """
        if spike_times is not None or not hasattr(self, "counts_"):
            self.bin(spike_times, neuron_ids)
        X = self.counts_
        if t_range is not None:
            a = max(0, int((t_range[0] - self.t0_) / self.dt))
            b = min(X.shape[1], int((t_range[1] - self.t0_) / self.dt))
            X = X[:, a:b]
        lag_bins = np.unique(np.round(np.asarray(lags) / self.dt).astype(int))
        lag_bins = lag_bins[lag_bins > 0]
        D = np.full(len(lag_bins), np.nan)
        npairs = np.zeros(len(lag_bins), int)

        def _dist(U, valid, q):
            ok = valid[:-q] & valid[q:]
            A = U[:-q][ok].astype(np.float64)
            Bm = U[q:][ok].astype(np.float64)
            d = np.empty(len(A))
            for a in range(0, len(A), 20000):
                d[a:a + 20000] = geodesic_distance(A[a:a + 20000], Bm[a:a + 20000])
            return d

        if window is None:                                   # tied: window = lag
            for i, lb in enumerate(lag_bins):
                if lb <= self.k:
                    warnings.warn(f"lag {lb * self.dt}s has <= k bins; skipped")
                    continue
                ell = lb // 4 if lb % 4 == 0 else lb         # stride lag/4 for more pairs
                m = lb // ell
                if X.shape[1] // ell < 2 * m:
                    continue
                U, _, valid = self._compute_bases(X, ell, [m])[m]
                d = _dist(U, valid, m)
                npairs[i] = len(d)
                if len(d):
                    D[i] = np.median(d) if stat == "median" else np.mean(d)
            win_s = lag_bins * self.dt
            search = np.ones(len(lag_bins), bool)
        else:
            probe = int(np.gcd.reduce(lag_bins))
            m = max(1, int(round(window / self.dt / probe)))
            if m * probe <= self.k:
                raise ValueError("window must contain more than k bins")
            U, _, valid = self._compute_bases(X, probe, [m])[m]
            for i, lb in enumerate(lag_bins):
                q = lb // probe
                if q < len(U):
                    d = _dist(U, valid, q)
                    npairs[i] = len(d)
                    if len(d):
                        D[i] = np.median(d) if stat == "median" else np.mean(d)
            win_s = np.full(len(lag_bins), m * probe * self.dt)
            search = lag_bins * self.dt >= m * probe * self.dt - 1e-12
        lag_s = lag_bins * self.dt
        Lc = D / lag_s
        knee = None
        if search.sum() >= 3:
            ks = _kneedle(np.log(lag_s[search]), np.log(Lc[search]))
            knee = None if ks is None else int(np.flatnonzero(search)[ks])
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.gradient(np.log(D), np.log(lag_s)) if len(D) > 1 else np.array([np.nan])
        return dict(lags=lag_s, window=win_s, D=D, L=Lc, n_pairs=npairs,
                    holder_slope=slope, knee_index=knee,
                    suggested_delta_tau=(float(lag_s[knee]) if knee is not None else None))


# ----------------------------------------------------------------------------
# Clustering helpers
# ----------------------------------------------------------------------------


def _kmeans_restarts(Y, c, rng, n_init=10):
    best, best_in = None, np.inf
    for _ in range(n_init):
        cent, lab = kmeans2(Y, c, iter=50, minit="++", rng=rng)
        inertia = np.sum((Y - cent[lab]) ** 2)
        if inertia < best_in:
            best, best_in = lab, inertia
    return best


def _spectral_clustering(F, c, n_neighbors, local_k, rng):
    """Ng-Jordan-Weiss on a sparse symmetric kNN graph, self-tuning scales."""
    n = len(F)
    nn = min(n_neighbors, n - 1)
    lk = min(local_k, n - 1)
    tree = cKDTree(F)
    d, idx = tree.query(F, k=max(nn, lk) + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    sig = d[:, lk - 1]
    pos = sig[sig > 0]
    floor = 1e-6 * (np.median(pos) if pos.size else 1.0)
    sig = np.maximum(sig, floor)
    rows = np.repeat(np.arange(n), nn)
    cols = idx[:, :nn].ravel()
    vals = np.exp(-d[:, :nn].ravel() ** 2 / (sig[rows] * sig[cols]))
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)
    deg = np.asarray(W.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-300))
    M = sparse.diags(dinv) @ W @ sparse.diags(dinv)
    if n <= 3000:
        _, V = np.linalg.eigh(M.toarray())
        V = V[:, -c:]
    else:
        v0 = rng.standard_normal(n)
        _, V = eigsh(M, k=c, which="LA", v0=v0, tol=1e-8, maxiter=20 * n)
    Y = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-300)
    return _kmeans_restarts(Y, c, rng)


def _hierarchical_clustering(F, c, method, max_points, rng):
    """scipy linkage on <= max_points steps, 1-NN assignment for the rest."""
    n = len(F)
    if n > max_points:
        sub = np.sort(rng.choice(n, max_points, replace=False))
    else:
        sub = np.arange(n)
    Z = linkage(F[sub], method=method)
    lab_sub = fcluster(Z, t=c, criterion="maxclust") - 1
    if len(sub) == n:
        return lab_sub
    _, j = cKDTree(F[sub]).query(F, k=1)
    return lab_sub[j]


def smooth_labels(labels, min_dwell):
    """Merge runs shorter than min_dwell into the longer neighbouring run."""
    lab = np.asarray(labels).copy()
    while True:
        change = np.flatnonzero(np.diff(lab)) + 1
        starts = np.r_[0, change]
        ends = np.r_[change, len(lab)]
        lens = ends - starts
        if len(lens) <= 1 or lens.min() >= min_dwell:
            return lab
        i = int(np.argmin(lens))
        left = lens[i - 1] if i > 0 else -1
        right = lens[i + 1] if i + 1 < len(lens) else -1
        src = i - 1 if left >= right else i + 1
        lab[starts[i]:ends[i]] = lab[starts[src]]


def _relabel_by_appearance(lab):
    _, first = np.unique(lab, return_index=True)
    order = np.argsort(first)
    mapping = np.empty(order.size, int)
    mapping[order] = np.arange(order.size)
    return mapping[np.searchsorted(np.unique(lab), lab)]


def _kneedle(x, y, min_strength=0.05):
    """Index of maximum distance below the chord of a normalised curve.

    Returns None when the curve is (nearly) straight in these coordinates, i.e.
    the largest normalised gap to the chord is below `min_strength`; e.g. when
    D(lag) is flat so L = D/lag is a pure power law with no elbow.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    xi, yi = x[ok], y[ok]
    xn = (xi - xi.min()) / (np.ptp(xi) or 1)
    yn = (yi - yi.min()) / (np.ptp(yi) or 1)
    chord = yn[0] + (yn[-1] - yn[0]) * (xn - xn[0]) / (xn[-1] - xn[0] or 1)
    dist = chord - yn                   # convex, decreasing curve lies below chord
    if dist.max() < min_strength:
        return None
    return int(np.flatnonzero(ok)[np.argmax(dist)])


# ----------------------------------------------------------------------------
# Synthetic data, metrics, test and benchmark
# ----------------------------------------------------------------------------


def adjusted_rand_index(a, b):
    from scipy.special import comb
    a = np.asarray(a)
    b = np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    cont = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(cont, (ai, bi), 1)
    sij = comb(cont, 2).sum()
    sa = comb(cont.sum(1), 2).sum()
    sb = comb(cont.sum(0), 2).sum()
    tot = comb(len(a), 2)
    exp = sa * sb / tot
    mx = 0.5 * (sa + sb)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def simulate_regime_spikes(N=40, duration=120.0, dt=1e-3, n_regimes=3, k_latent=3,
                           seg_range=(3.0, 8.0), base_rate=10.0, gain=0.8,
                           latent_tau=0.05, seed=0):
    """Poisson spikes whose shared latent factors load onto neurons differently
    in each regime (same mean rates, different correlation subspace)."""
    rng = np.random.default_rng(seed)
    Tb = int(round(duration / dt))
    # piecewise regime sequence with recurrence, no immediate repeats
    reg = np.empty(Tb, int)
    t, prev = 0, -1
    while t < Tb:
        r = rng.choice([q for q in range(n_regimes) if q != prev])
        L = int(rng.uniform(*seg_range) / dt)
        reg[t:t + L] = r
        t += L
        prev = r
    # smooth latent: AR(1) / OU with unit variance
    a = np.exp(-dt / latent_tau)
    z = np.empty((k_latent, Tb))
    eps = rng.standard_normal((k_latent, Tb)) * np.sqrt(1 - a * a)
    z[:, 0] = rng.standard_normal(k_latent)
    for i in range(1, Tb):
        z[:, i] = a * z[:, i - 1] + eps[:, i]
    Wr = rng.standard_normal((n_regimes, N, k_latent)) / np.sqrt(k_latent)
    lam = np.empty((N, Tb))
    for r in range(n_regimes):
        sel = reg == r
        # log-normal correction keeps each neuron's mean rate = base_rate in every regime
        corr = 0.5 * gain**2 * (Wr[r] ** 2).sum(1)
        lam[:, sel] = base_rate * dt * np.exp(gain * (Wr[r] @ z[:, sel]) - corr[:, None])
    counts = rng.poisson(lam)
    spikes = []
    for n in range(N):
        bins = np.repeat(np.arange(Tb), counts[n])
        spikes.append(np.sort((bins + rng.random(bins.size)) * dt))
    return spikes, reg


def run_test(seed=0):
    """Synthetic recovery test plus geometry sanity checks."""
    print("== geometry checks ==")
    rng = np.random.default_rng(seed)
    N, k = 12, 3
    A, _ = np.linalg.qr(rng.standard_normal((N, k)))
    Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    assert np.allclose(principal_angles(A, A @ Q), 0, atol=1e-7), "basis invariance"
    B, _ = np.linalg.qr(rng.standard_normal((N, k)))
    from scipy.linalg import subspace_angles
    assert np.allclose(np.sort(principal_angles(A, B)), np.sort(subspace_angles(A, B)), atol=1e-10)
    D = grassmann_log(A, B[None])[0]
    assert np.allclose(A.T @ D, 0, atol=1e-10), "tangent is horizontal"
    Bh = grassmann_exp(A, D)
    assert np.allclose(principal_angles(Bh, B), 0, atol=1e-6), "exp(log(B)) = B"
    assert np.isclose(np.linalg.norm(D), geodesic_distance(A, B)), "|log| = geodesic distance"
    print("  principal angles vs scipy, basis invariance, exp(log)=id, |log|=d_g: OK")

    sim = dict(N=60, duration=120.0, base_rate=20.0, gain=1.2, latent_tau=0.1,
               seg_range=(4.0, 10.0))
    print(f"== synthetic regimes: {sim['N']} neurons, {sim['duration']:.0f} s, "
          f"3 recurring regimes, {sim['base_rate']:.0f} Hz, equal mean rates ==")
    spikes, reg = simulate_regime_spikes(seed=seed, **sim)
    print(f"  total spikes: {sum(s.size for s in spikes):,}")
    dtau = 0.05
    gt_dt = 1e-3

    def score(mdl):
        cb = ((np.arange(mdl.n_steps_) + 0.5) * dtau / gt_dt).astype(int)
        return adjusted_rand_index(reg[cb], mdl.labels_)

    results = []
    print("  bin dt | clustering   | refs | features | ARI   | time")
    for dt in (1e-3, 5e-3):
        for clus in ("spectral", "hierarchical"):
            for nref, feat in ((1, "distance"), (3, "distance"), (3, "angles")):
                mdl = MultiscaleGrassmannRegimes(dt=dt, delta_tau=dtau, scales=(1, 2, 4, 8, 16),
                                                 k=3, n_references=nref, features=feat,
                                                 clustering=clus, n_clusters=3,
                                                 random_state=seed).fit(spikes)
                ari = score(mdl)
                tt = sum(mdl.timings_.values())
                results.append((dt, clus, nref, feat, ari, tt))
                print(f"  {dt * 1e3:4.0f}ms | {clus:12s} | {nref:4d} | {feat:8s} | {ari:5.3f} | {tt:5.2f}s")

    print("== single centre of mass with only 2 regimes (dt=5 ms, spectral) ==")
    sp2, reg2 = simulate_regime_spikes(seed=seed + 1, n_regimes=2, **sim)
    for nref in (1, 2):
        mdl = MultiscaleGrassmannRegimes(dt=5e-3, delta_tau=dtau, k=3, n_references=nref,
                                         n_clusters=2, random_state=seed).fit(sp2)
        cb = ((np.arange(mdl.n_steps_) + 0.5) * dtau / gt_dt).astype(int)
        print(f"  refs={nref}: ARI={adjusted_rand_index(reg2[cb], mdl.labels_):.3f}")

    strong = dict(N=100, duration=120.0, base_rate=30.0, gain=1.5, latent_tau=0.01,
                  seg_range=(4.0, 10.0))
    sp_s, reg_s = simulate_regime_spikes(seed=seed, **strong)
    print("== one scale at a time (dt=5 ms, spectral, refs=3) ==")
    print("   slow/weak pop: 60 neurons, 20 Hz, latent tau 100 ms")
    print("   fast/strong pop: 100 neurons, 30 Hz, latent tau 10 ms")
    for m in (1, 2, 4, 8, 16):
        row = []
        for spk_, reg_ in ((spikes, reg), (sp_s, reg_s)):
            mdl = MultiscaleGrassmannRegimes(dt=5e-3, delta_tau=dtau, scales=(m,), k=3,
                                             n_clusters=3, random_state=seed).fit(spk_)
            cb = ((np.arange(mdl.n_steps_) + 0.5) * dtau / gt_dt).astype(int)
            row.append(adjusted_rand_index(reg_[cb], mdl.labels_))
        print(f"  window {m * dtau * 1e3:5.0f} ms: ARI slow/weak={row[0]:.3f}  fast/strong={row[1]:.3f}")

    print("== tied Lipschitz sweep (window = lag, dt=5 ms, fast/strong pop, first 60 s) ==")
    mdl = MultiscaleGrassmannRegimes(dt=5e-3, delta_tau=dtau, k=3)
    sw = mdl.lipschitz_sweep([0.02, 0.04, 0.06, 0.1, 0.2, 0.4, 0.8, 1.6],
                             spike_times=sp_s, t_range=(0, 60))
    for lag, d, l, npr in zip(sw["lags"], sw["D"], sw["L"], sw["n_pairs"]):
        print(f"  lag=window={lag * 1000:6.0f} ms  D={d:6.3f} rad  L=D/lag={l:8.2f} rad/s  pairs={npr}")
    print(f"  knee -> suggested delta_tau = {sw['suggested_delta_tau']}")
    return results, sw


def _slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def run_bench():
    """Empirical runtime exponents (fit of log time vs log parameter).

    Last run (2 cores): T 1.09 total; N bases 1.21 / centres 1.46; window
    length 0.02; 1/dt 0.31; k centres 1.01 / distances 1.25; spectral B 1.05;
    full hierarchical B 2.83.
    """
    out = {}
    base = dict(dt=5e-3, delta_tau=0.05, scales=(1, 2, 4, 8, 16), k=3,
                clustering="spectral", n_clusters=3)

    def timed(spk, **kw):
        p = dict(base)
        p.update(kw)
        m = MultiscaleGrassmannRegimes(**p).fit(spk)
        return m.timings_

    print("== duration T (N=40) ==")
    Ts, tt = [30, 60, 120, 240], []
    for T in Ts:
        spk, _ = simulate_regime_spikes(N=40, duration=T, seed=1)
        t = timed(spk)
        tt.append(t)
        print(f"  T={T:4d}s  " + "  ".join(f"{a}={b:.2f}" for a, b in t.items()))
    out["T"] = {s: _slope(Ts, [t[s] for t in tt]) for s in ("bin", "bases", "centres", "distances", "cluster")}
    out["T"]["total"] = _slope(Ts, [sum(t.values()) for t in tt])

    print("== neurons N (T=60 s) ==")
    Ns, tn = [25, 50, 100, 200], []
    for N in Ns:
        spk, _ = simulate_regime_spikes(N=N, duration=60, seed=2)
        t = timed(spk)
        tn.append(t)
        print(f"  N={N:4d}  " + "  ".join(f"{a}={b:.2f}" for a, b in t.items()))
    out["N"] = {s: _slope(Ns, [t[s] for t in tn]) for s in ("bases", "centres", "distances")}

    print("== largest scale m_max (N=40, T=60 s, scales = 1,2,4..m_max) ==")
    spk, _ = simulate_regime_spikes(N=40, duration=60, seed=3)
    ms, tm = [2, 4, 8, 16, 32], []
    for mm in ms:
        sc = tuple(2**i for i in range(int(np.log2(mm)) + 1))
        t = timed(spk, scales=sc)
        tm.append(t)
        print(f"  m_max={mm:3d} ({len(sc)} scales)  bases={t['bases']:.2f}")
    out["m_max(bases, geometric scales)"] = _slope(ms, [t["bases"] for t in tm])

    print("== bin width dt (N=40, T=60 s): l = delta_tau/dt bins per step ==")
    dts, td = [0.01, 0.005, 0.0025, 0.00125], []
    for d in dts:
        t = timed(spk, dt=d)
        td.append(t["bases"])
        print(f"  dt={d * 1e3:5.2f} ms (l={int(round(0.05 / d)):3d})  bases={t['bases']:.2f}")
    out["1/dt (bases)"] = _slope([1 / d for d in dts], td)

    print("== single scale window length (N=40, T=60 s, all cov path L>=N): should be flat ==")
    Ls, tl = [4, 8, 16, 32, 64], []
    for m in Ls:
        t = timed(spk, scales=(m,))
        tl.append(t["bases"])
        print(f"  window={m * 50:4d} ms  bases={t['bases']:.3f}")
    out["window length (single scale, cov path)"] = _slope(Ls, tl)

    print("== k (N=40, T=60 s) ==")
    ks, tk = [1, 2, 4, 8], []
    for k in ks:
        t = timed(spk, k=k)
        tk.append(t)
        print(f"  k={k}  bases={t['bases']:.2f} centres={t['centres']:.2f} distances={t['distances']:.2f}")
    out["k"] = {s: _slope(ks, [t[s] for t in tk]) for s in ("bases", "centres", "distances")}

    print("== clustering vs number of steps B (synthetic 5-D features) ==")
    rng = np.random.default_rng(0)
    Bs, ts, th = [2000, 4000, 8000, 16000], [], []
    for B in Bs:
        F = np.concatenate([rng.normal(c, 1, (B // 3 + 1, 5)) for c in (0, 3, 6)])[:B]
        t0 = time.perf_counter()
        _spectral_clustering(F, 3, 15, 7, np.random.default_rng(0))
        ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _hierarchical_clustering(F, 3, "ward", 10**9, np.random.default_rng(0))
        th.append(time.perf_counter() - t0)
        print(f"  B={B:6d}  spectral={ts[-1]:.2f}s  hierarchical(full)={th[-1]:.2f}s")
    out["B spectral"] = _slope(Bs[1:], ts[1:])
    out["B hierarchical full"] = _slope(Bs, th)

    print("\n== fitted exponents: time ~ param^p ==")
    for key, v in out.items():
        if isinstance(v, dict):
            print(f"  {key}: " + ", ".join(f"{a}={b:.2f}" for a, b in v.items()))
        else:
            print(f"  {key}: {v:.2f}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if not (args.test or args.bench):
        args.test = True
    if args.test:
        run_test()
    if args.bench:
        run_bench()
