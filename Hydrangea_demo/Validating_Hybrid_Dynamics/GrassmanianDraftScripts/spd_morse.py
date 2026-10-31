"""
SPD-manifold version of the regime pipeline (soft threshold: keep all directions,
weighted by variance) with a saddle (hyperbolic) visualization.

Per window t:
  1. mollify spikes (Gaussian sigma), take the window, z-score channels
  2. shrinkage correlation  C_t = (1-rho) * S_t + rho * I     (Ledoit-Wolf-like floor)
  3. point on SPD(n).  Tangent coordinates at the Karcher (log-Euclidean) mean:
         X_t = log(C_t) - mean_t log(C_t)        (log-Euclidean metric)
     Optional affine-invariant version: log(M^-1/2 C_t M^-1/2).
  4. tangent PCA -> d dims;  density-adaptive cells (kd-tree, equal occupancy)
  5. sparse transition matrix, threshold sweep, Morse graph (SCCs + condensation)

Why a saddle:  SPD(n) with the affine-invariant metric has non-positive curvature
(a Hadamard manifold).  Its simplest slice, SPD(2) with det = 1, is the hyperbolic
plane, which embeds as one sheet of the hyperboloid  x^2 + y^2 - z^2 = -1  in
Minkowski space.  That sheet is the SPD analogue of the sphere in the Gr(3,1)
picture: geodesics bend away from each other instead of toward each other, so the
tangent plane at the mean *under*-estimates distances between far clusters (on the
sphere it over-estimates them).

Figure (spd_morse.png):
  left   : the 2x2 correlation of the two leading population PCs, per window, as
           points on the hyperboloid (saddle), with the Karcher mean, its tangent
           plane, and log-map lines.
  middle : full-n tangent coordinates (tangent PCA, 2 dims), coloured by Morse set.
  bottom : timeline of Morse-set label and SPD velocity (geodesic distance between
           consecutive windows), with ground truth strip for the synthetic data.

Usage:  python spd_morse.py                       synthetic benchmark + figure
        from spd_morse import run; run(counts, sigma_ms=10, win_ms=100, ...)
        where counts is a (T_ms x n) spike-count array at 1 ms bins.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# synthetic data (same generator as grassmann_morse.py)
# ----------------------------------------------------------------------------
def simulate(n=30, T_ms=120_000, long_ms=(1200, 2000), brief_ms=150,
             brief_rate_hz=0.15, base_rate=10.0, gain=0.9, k=2, tau_ms=20):
    names = ["A", "B", "C", "F"]
    W = [np.linalg.qr(rng.standard_normal((n, k)))[0] * np.sqrt(n / k) for _ in names]
    lab = np.empty(T_ms, dtype=int)
    t, s = 0, 0
    while t < T_ms:
        d = rng.integers(*long_ms); lab[t:t + d] = s; t += d; s = (s + 1) % 3
    for st in rng.integers(0, T_ms - brief_ms, rng.poisson(brief_rate_hz * T_ms / 1000)):
        lab[st:st + brief_ms] = 3
    a = np.exp(-1 / tau_ms)
    z = np.zeros((T_ms, k)); eps = rng.standard_normal((T_ms, k)) * np.sqrt(1 - a * a)
    for i in range(1, T_ms):
        z[i] = a * z[i - 1] + eps[i]
    drive = np.zeros((T_ms, n))
    for si in range(4):
        m = lab == si; drive[m] = z[m] @ W[si].T / np.sqrt(n / k)
    rate = base_rate * np.exp(gain * drive - 0.5 * gain ** 2) / 1000.0
    return rng.poisson(rate), lab, names


# ----------------------------------------------------------------------------
# SPD points per window
# ----------------------------------------------------------------------------
def _sym_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return np.einsum("tij,tj,tkj->tik", V, fn(w), V)


def windowed_spd(spikes, sigma_ms=10, win_ms=100, step_ms=10, shrink=None, chunk=1000):
    """Shrinkage correlation matrices (T_w x n x n) and their matrix logs."""
    rates = gaussian_filter1d(spikes.astype(np.float32), sigma_ms, axis=0)
    starts = np.arange(0, rates.shape[0] - win_ms, step_ms)
    n = rates.shape[1]
    C = np.empty((len(starts), n, n))
    m_eff = win_ms / (sigma_ms * 2.0)            # rough effective sample count
    if shrink is None:                           # Ledoit-Wolf-style constant
        shrink = float(np.clip(n / (n + m_eff), 0.05, 0.95))
    for c0 in range(0, len(starts), chunk):
        idx = starts[c0:c0 + chunk, None] + np.arange(win_ms)[None, :]
        X = rates[idx]
        X = X - X.mean(1, keepdims=True)
        X = X / (X.std(1, keepdims=True) + 1e-6)
        S = np.einsum("tki,tkj->tij", X, X) / win_ms
        C[c0:c0 + chunk] = (1 - shrink) * S + shrink * np.eye(n)
    L = _sym_fn(C, np.log)
    return C, L, starts, shrink


def tangent_coords(L, d=3, metric="log-euclidean", C=None):
    """Tangent vectors at the Karcher mean, flattened with sqrt(2) on off-diagonals
    so Euclidean distance = Frobenius distance.  Returns Z (tangent PCA), full
    tangent vectors, and distance of each point from the mean."""
    n = L.shape[1]
    Lbar = L.mean(0)
    if metric == "log-euclidean":
        X = L - Lbar
    else:  # affine-invariant, one Karcher-mean step from the log-Euclidean mean
        w, V = np.linalg.eigh(Lbar)
        Mih = V @ np.diag(np.exp(-w / 2)) @ V.T
        X = _sym_fn(np.einsum("ij,tjk,kl->til", Mih, C, Mih), np.log)
    iu = np.triu_indices(n)
    wgt = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    F = X[:, iu[0], iu[1]] * wgt
    dist = np.linalg.norm(F, axis=1)
    Fc = F - F.mean(0)
    _, sv, Vt = np.linalg.svd(Fc, full_matrices=False)
    return Fc @ Vt[:d].T, F, dist, (sv[:d] ** 2).sum() / (sv ** 2).sum()


# ----------------------------------------------------------------------------
# cells, sparse graph, Morse graph
# ----------------------------------------------------------------------------
def cells_kdtree(Z, leaf_size):
    cell = np.empty(len(Z), dtype=int)
    stack, nxt = [np.arange(len(Z))], 0
    while stack:
        ix = stack.pop()
        if len(ix) <= 2 * leaf_size:
            cell[ix] = nxt; nxt += 1; continue
        ax = np.argmax(np.ptp(Z[ix], axis=0)); med = np.median(Z[ix, ax])
        l, r = ix[Z[ix, ax] <= med], ix[Z[ix, ax] > med]
        if len(l) == 0 or len(r) == 0:
            cell[ix] = nxt; nxt += 1; continue
        stack += [l, r]
    return cell


def transition_matrix(cell):
    k = cell.max() + 1
    return sparse.coo_matrix((np.ones(len(cell) - 1), (cell[:-1], cell[1:])), shape=(k, k)).tocsr()


def morse_graph(Tm, min_count=1):
    G = Tm.copy(); G.data[G.data < min_count] = 0; G.eliminate_zeros()
    ncomp, comp = connected_components(G, directed=True, connection="strong")
    sizes = np.bincount(comp, minlength=ncomp)
    selfloop = np.zeros(ncomp, bool); selfloop[comp[G.diagonal() > 0]] = True
    morse = np.where((sizes > 1) | selfloop)[0]
    Gc = G.tocoo(); cu, cv = comp[Gc.row], comp[Gc.col]; keep = cu != cv
    return comp, morse, set(zip(cu[keep].tolist(), cv[keep].tolist()))


def describe(cell, comp, morse, truth, names, min_frac=0.02):
    wc = comp[cell]; rows = []
    for m in morse:
        sel = wc == m
        if sel.mean() < min_frac: continue
        lab = np.bincount(truth[sel], minlength=len(names)) if truth is not None else None
        rows.append((m, sel.mean(), names[lab.argmax()] if lab is not None else "?",
                     lab.max() / lab.sum() if lab is not None else np.nan))
    return sorted(rows, key=lambda r: -r[1])


# ----------------------------------------------------------------------------
# saddle (hyperboloid) picture for 2x2 correlations
# ----------------------------------------------------------------------------
def spd2_to_hyperboloid(C2):
    """det-normalised 2x2 SPD -> point (x, y, z) on the upper sheet of
    x^2 + y^2 - z^2 = -1.  Mapping: C/sqrt(det C) = [[a, b], [b, c]],
    z = (a + c)/2, x = (a - c)/2, y = b."""
    det = C2[:, 0, 0] * C2[:, 1, 1] - C2[:, 0, 1] ** 2
    Cn = C2 / np.sqrt(det)[:, None, None]
    return np.stack([(Cn[:, 0, 0] - Cn[:, 1, 1]) / 2, Cn[:, 0, 1], (Cn[:, 0, 0] + Cn[:, 1, 1]) / 2], 1)


def hyp_mean(H, iters=30):
    """Karcher mean on the hyperboloid (Minkowski product <p,q> = xx' + yy' - zz')."""
    mink = lambda p, q: p[..., 0] * q[..., 0] + p[..., 1] * q[..., 1] - p[..., 2] * q[..., 2]
    m = H.mean(0); m = m / np.sqrt(-mink(m, m))
    for _ in range(iters):
        c = -mink(H, m); c = np.maximum(c, 1 + 1e-12)
        d = np.arccosh(c)
        v = H - c[:, None] * m                       # tangent direction (unnormalised)
        vn = np.sqrt(np.maximum(mink(v, v), 1e-18))
        step = (v / vn[:, None] * d[:, None]).mean(0)
        sn = np.sqrt(max(mink(step, step), 1e-18))
        m = np.cosh(sn) * m + np.sinh(sn) * step / sn
    return m, mink


def hyp_log(H, m, mink):
    c = np.maximum(-mink(H, m), 1 + 1e-12); d = np.arccosh(c)
    v = H - c[:, None] * m; vn = np.sqrt(np.maximum(mink(v, v), 1e-18))
    return v / vn[:, None] * d[:, None], d


# ----------------------------------------------------------------------------
def run(spikes, sigma_ms=10, win_ms=100, step_ms=10, d=3, leaf=50, thresholds=(1, 2, 3, 5, 8),
        truth_ms=None, names=None, metric="log-euclidean", out="spd_morse.png"):
    C, L, starts, shrink = windowed_spd(spikes, sigma_ms, win_ms, step_ms)
    Z, F, dist, ve = tangent_coords(L, d, metric, C)
    vel = np.r_[0, np.linalg.norm(F[1:] - F[:-1], axis=1)]
    truth = None
    if truth_ms is not None:
        truth = np.array([np.bincount(truth_ms[s:s + win_ms], minlength=len(names)).argmax() for s in starts])
    print(f"windows {len(C)}  n {C.shape[1]}  shrinkage {shrink:.2f}  tangent PCA {d} dims = {ve:.0%} var")

    cell = cells_kdtree(Z, leaf); Tm = transition_matrix(cell)
    print(f"cells {Tm.shape[0]}  nnz {Tm.nnz}  sparse bytes {Tm.data.nbytes + Tm.indices.nbytes + Tm.indptr.nbytes}")
    best = None
    for c in thresholds:
        comp, morse, dag = morse_graph(Tm, c)
        rows = describe(cell, comp, morse, truth, names or [], min_frac=0.02)
        cov = sum(r[1] for r in rows)
        pur = sum(r[1] * r[3] for r in rows) / cov if (cov and truth is not None) else np.nan
        print(f"  threshold >= {c}: {len(morse)} Morse sets, {len(rows)} large, cover {cov:.0%}, "
              f"weighted purity {pur:.2f}  " + " ".join(f"{r[2]}({r[1]:.0%},{r[3]:.2f})" for r in rows[:5]))
        if rows and (best is None or len(rows) > len(best[1])):
            best = (comp, rows, c)
    comp, rows, cbest = best
    label = np.full(len(cell), -1); big = [r[0] for r in rows]
    for i, m in enumerate(big):
        label[comp[cell] == m] = i

    # ---------------- figure ----------------
    # leading two population PCs over the whole recording -> 2x2 correlation per window
    rates = gaussian_filter1d(spikes.astype(np.float32), sigma_ms, axis=0)
    Rc = rates - rates.mean(0); Rc /= Rc.std(0) + 1e-6
    _, _, Vt = np.linalg.svd(Rc[::10], full_matrices=False)
    Y = Rc @ Vt[:2].T
    C2 = np.empty((len(starts), 2, 2))
    for i, s in enumerate(starts):
        y = Y[s:s + win_ms]; y = y - y.mean(0)
        S = y.T @ y / win_ms; C2[i] = 0.8 * S + 0.2 * np.diag(np.diag(S)) + 1e-3 * np.eye(2)
    H = spd2_to_hyperboloid(C2); m, mink = hyp_mean(H); V, dh = hyp_log(H, m, mink)

    cols = plt.cm.tab10(np.arange(10))
    pc = np.where(label[:, None] >= 0, cols[np.clip(label, 0, 9) % 10], np.array([[.6, .6, .6, 1]]))
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    rmax = float(np.quantile(dh, 0.98)) * 1.5 + 0.05
    r = np.linspace(0, rmax, 30); th = np.linspace(0, 2 * np.pi, 60)
    R, TH = np.meshgrid(r, th)
    ax.plot_surface(np.sinh(R) * np.cos(TH), np.sinh(R) * np.sin(TH), np.cosh(R), alpha=.15, color="gray", linewidth=0)
    e1 = np.array([1, 0, 0.]); e1 = e1 + mink(e1, m) * m; e1 /= np.sqrt(mink(e1, e1))
    e2 = np.array([0, 1, 0.]); e2 = e2 + mink(e2, m) * m - mink(e2, e1) * e1; e2 /= np.sqrt(mink(e2, e2))
    g = np.linspace(-rmax / 2, rmax / 2, 2); G1, G2 = np.meshgrid(g, g)
    P = m + G1[..., None] * e1 + G2[..., None] * e2
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], alpha=.25, color="k", linewidth=0)
    ax.scatter(H[:, 0], H[:, 1], H[:, 2], c=pc, s=6)
    ax.scatter(*m, c="k", s=50); ax.text(m[0], m[1], m[2] + .1, "Karcher mean")
    for i in range(0, len(H), max(1, len(H) // 150)):
        tp = m + V[i]; ax.plot([H[i, 0], tp[0]], [H[i, 1], tp[1]], [H[i, 2], tp[2]], c=pc[i], lw=.4, alpha=.5)
    ax.set_title("SPD(2) slice as a saddle (hyperboloid sheet) with tangent plane at the mean", fontsize=10)
    ax.set_axis_off(); ax.view_init(25, -60)

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(Z[:, 0], Z[:, 1], c="gray", lw=.3, alpha=.5)
    ax.scatter(Z[:, 0], Z[:, 1], c=pc, s=6)
    ax.scatter(0, 0, c="k", s=50); ax.axhline(0, c="lightgray", lw=.5); ax.axvline(0, c="lightgray", lw=.5)
    ax.set_title(f"tangent space at the mean, full n (tangent PCA, coloured by Morse set, threshold>={cbest})", fontsize=10)
    ax.set_xlabel("tangent PC 1"); ax.set_ylabel("tangent PC 2")

    ax = fig.add_subplot(2, 1, 2)
    t = starts / 1000.0
    ax.imshow(label[None, :], aspect="auto", extent=[t[0], t[-1], 1.15, 1.45], cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    if truth is not None:
        ax.imshow(truth[None, :], aspect="auto", extent=[t[0], t[-1], 1.5, 1.8], cmap="Set2", interpolation="nearest")
        ax.text(t[0], 1.85, "ground truth (synthetic)", fontsize=8)
    ax.text(t[0], 1.08, "Morse-set label (gray = none)", fontsize=8, va="top")
    ax.plot(t, vel / vel.max(), c="k", lw=.5)
    ax.set_ylim(0, 1.95); ax.set_xlabel("time (s)"); ax.set_ylabel("SPD velocity (norm.)")
    ax.set_xlim(t[0], min(t[-1], t[0] + 60))
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("figure ->", out)
    return dict(C=C, Z=Z, F=F, cell=cell, Tm=Tm, label=label, vel=vel, starts=starts)


if __name__ == "__main__":
    spikes, lab, names = simulate(n=30, base_rate=30)
    run(spikes, sigma_ms=10, win_ms=300, step_ms=50, truth_ms=lab, names=names)
