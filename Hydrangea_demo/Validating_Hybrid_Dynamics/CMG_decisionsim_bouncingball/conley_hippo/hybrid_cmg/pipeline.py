"""
Decoding pipeline (one implementation, used by every simulation).

  spikes (T x N counts at base_dt, per trajectory/session segment)
    -> 1. BIN:     sliding bins of length b, advanced by step s  -> rates R
    -> 2. LATENT:  PCA on the covariance of R (bin-b covariance)  -> L(t), d_lat dims
    -> 3. EMBED:   delay embedding of L with lag ell (multiple of s), m delays
                   -> PCA to a 2D chart for gridding
    -> 4. REACH:   transitions z(t) -> z(t + ell)  (reachability at lag ell)
    -> 5. CMG:     smoothed-Ulam map on a grid, SCC Morse sets, Conley index

All times in seconds. Transitions never cross segment (trajectory) seams.
"""

import numpy as np
import cmg_lib as cl


def bin_segment(counts, base_dt, b, s):
    """counts: (T, N) spike counts at base_dt. Returns rates (K, N) in Hz and
    the base-bin index of each bin's END (the causal time stamp)."""
    B = int(round(b / base_dt))
    S = int(round(s / base_dt))
    T = counts.shape[0]
    c = np.concatenate([np.zeros((1, counts.shape[1])),
                        np.cumsum(counts, 0, dtype=np.float64)], 0)
    starts = np.arange(0, T - B + 1, S)
    R = ((c[starts + B] - c[starts]) / b).astype(np.float32)
    return R, starts + B - 1


def normalize(Rs, eps=1e-9):
    """Per-bin population normalization, then per-neuron z-score across the
    session. Rs: list of (K_i, N) binned rate arrays (one per segment)."""
    out = []
    for R in Rs:
        tot = R.sum(1, keepdims=True)
        out.append((R / np.maximum(tot, eps)).astype(np.float32))
    allR = np.concatenate(out)
    mu = allR.mean(0)
    sd = allR.std(0)
    sd[sd < eps] = 1.0
    return [((R - mu) / sd).astype(np.float32) for R in out]


def decode(segments, base_dt, b, s, ell, m=3, d_lat=3, nx=64, labels=None,
           index=True, seed=0):
    """segments: list of (T_i, N) count arrays.
    labels: optional list of (T_i, ...) ground-truth arrays at base_dt;
    returned aligned to the chart samples.
    Returns dict with chart Z, grid, map A, analysis, aligned labels."""
    assert ell >= s - 1e-12, "lag must be >= step"
    L_steps = int(round(ell / s))
    # 1. bin
    Rs, stamps = [], []
    for seg in segments:
        R, st = bin_segment(seg, base_dt, b, s)
        Rs.append(R)
        stamps.append(st)
    # 1b. normalize: divide each bin by its total population spike count,
    #     then z-score each neuron across all bins of the session
    Rs = normalize(Rs)
    # 2. latent: PCA on the bin-b covariance
    Rall = np.concatenate(Rs)
    mu, V = cl.pca_fit(Rall, d=d_lat, seed=seed, max_n=40000)
    Ls = [cl.pca_apply(R, mu, V) for R in Rs]
    # 3. delay embed per segment, lag ell
    Ds, keep_stamps = [], []
    for L, st in zip(Ls, stamps):
        if len(L) <= (m - 1) * L_steps + L_steps:
            Ds.append(np.zeros((0, d_lat * m)))
            keep_stamps.append(np.zeros(0, int))
            continue
        D = cl.delay_embed(L, m, L_steps)       # row t <-> L index t+(m-1)L
        D = D.reshape(len(D), -1)               # (T, m, d) -> (T, m*d)
        Ds.append(D)
        keep_stamps.append(st[(m - 1) * L_steps:][:len(D)])
    Dall = np.concatenate(Ds)
    mu2, V2 = cl.pca_fit(Dall, d=2, seed=seed + 1, max_n=40000)
    Zs = [cl.pca_apply(D, mu2, V2) for D in Ds]
    Z = np.concatenate(Zs)
    # 4. reachability at lag ell, within segments
    g = cl.Grid2D(Z, nx)
    src, dst, seg_id = [], [], []
    off = 0
    cells_all = g.cells(Z)
    for k, Zk in enumerate(Zs):
        n = len(Zk)
        if n > L_steps:
            c = cells_all[off:off + n]
            src.append(c[:-L_steps])
            dst.append(c[L_steps:])
        off += n
    src = np.concatenate(src)
    dst = np.concatenate(dst)
    A = cl.build_map(src, dst, g.nx, g.ny, src_dilate=0)
    A = cl.support_restrict(A, src, dst)
    # 5. CMG
    an = cl.analyze(A, g.ny, f"b={b*1e3:.0f} s={s*1e3:.0f} l={ell*1e3:.0f}",
                    index=index, verbose=False)
    out = dict(Z=Z, Zs=Zs, grid=g, cells=cells_all, A=A, an=an,
               L_steps=L_steps, latent=Ls, stamps=keep_stamps)
    if labels is not None:
        out['labels'] = np.concatenate([lab[st] for lab, st in
                                        zip(labels, keep_stamps)])
    return out


def rotation_sense(Zs, cells_all, set_cells, L_steps):
    """Mean signed angular velocity of chart transitions inside a Morse set:
    sign tells the direction of circulation in the chart."""
    sc = set(int(c) for c in set_cells)
    num, cnt = 0.0, 0
    off = 0
    pts = []
    for Zk in Zs:
        n = len(Zk)
        ck = cells_all[off:off + n]
        m = np.isin(ck, list(sc))
        pts.append(Zk[m])
        off += n
    P = np.concatenate(pts) if pts else np.zeros((0, 2))
    if len(P) < 10:
        return 0.0
    ctr = P.mean(0)
    off = 0
    for Zk in Zs:
        n = len(Zk)
        ck = cells_all[off:off + n]
        if n > L_steps:
            m = np.isin(ck[:-L_steps], list(sc))
            r = Zk[:-L_steps][m] - ctr
            dz = Zk[L_steps:][m] - Zk[:-L_steps][m]
            cr = r[:, 0] * dz[:, 1] - r[:, 1] * dz[:, 0]
            num += cr.sum()
            cnt += len(cr)
        off += n
    return num / max(cnt, 1)
