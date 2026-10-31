"""
Shared pipeline for data-driven Conley-Morse graphs on 2D charts.

Chart = 2D projection (PCA) of (possibly windowed/smoothed) delay embeddings.
Map   = smoothed-Ulam multivalued map on a uniform grid over the chart.
CMG   = strongly connected components (scipy csgraph) -> recurrent Morse sets,
        reachability order, Conley index via GF(2) relative cubical homology
        of the attracting block (index pair (closure(M), closure \\ M)).

Strategies compared throughout the sims:
  fine-1step     : window w0, lag tau0           (baseline one-step map)
  composed^k     : fine one-step map composed k times (valid outer lag-k)
  coarse-matched : window k*tau0, lag k*tau0     (lag-matched coarse latent)
  bigwin-finelag : window k*tau0, lag tau0       (large window, small step)
  finewin-biglag : window w0, lag k*tau0 direct  (raw chart, long lag)
  intersect      : composed^k  INTERSECT  coarse signal mapped in fine chart
"""

import numpy as np
from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# --------------------------- signal utilities -------------------------------

def boxcar(X, w):
    """Centered moving average along axis 0, window w samples."""
    X = np.asarray(X, float)
    if w <= 1:
        return X.copy()
    n = X.shape[0]
    c = np.concatenate([np.zeros((1,) + X.shape[1:]), np.cumsum(X, 0)], 0)
    idx = np.arange(n)
    h = w // 2
    a = np.clip(idx - h, 0, n)
    b = np.clip(idx + h + 1, 0, n)
    shape = (-1,) + (1,) * (X.ndim - 1)
    return (c[b] - c[a]) / (b - a).reshape(shape)


def pca_fit(X, d=2, max_n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(len(X), max_n), replace=False)
    mu = X[idx].mean(0)
    _, _, Vt = np.linalg.svd(X[idx] - mu, full_matrices=False)
    return mu, Vt[:d]


def pca_apply(X, mu, V):
    return (np.asarray(X, float) - mu) @ V.T


def delay_embed(y, m, step):
    """Rows: (y_t, y_{t-step}, ..., y_{t-(m-1)step}); length n-(m-1)step,
    aligned so row t corresponds to original time t+(m-1)step."""
    y = np.asarray(y, float)
    n = len(y) - (m - 1) * step
    cols = [y[(m - 1 - j) * step: (m - 1 - j) * step + n] for j in range(m)]
    return np.stack(cols, 1)


def delay_embed_lags(y, lags):
    """General delay embedding with arbitrary lag tuple (0 included or not).
    Row t corresponds to original time t + max(lags)."""
    y = np.asarray(y, float)
    L = max(lags)
    n = len(y) - L
    cols = [y[L - l: L - l + n] for l in lags]
    return np.stack(cols, 1)

# ------------------------------- the grid ------------------------------------

class Grid2D:
    def __init__(self, Z, nx, ny=None, qlo=0.5, qhi=99.5, margin=0.03):
        ny = ny or nx
        lo = np.percentile(Z, qlo, axis=0)
        hi = np.percentile(Z, qhi, axis=0)
        pad = (hi - lo) * margin + 1e-12
        self.lo, self.hi = lo - pad, hi + pad
        self.nx, self.ny = nx, ny
        self.N = nx * ny

    def cells(self, Z):
        """Cell id per row; -1 if outside the box."""
        u = (Z - self.lo) / (self.hi - self.lo)
        i = np.floor(u[:, 0] * self.nx).astype(np.int64)
        j = np.floor(u[:, 1] * self.ny).astype(np.int64)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        out = np.where(ok, i * self.ny + j, -1)
        return out

    def centers(self):
        i, j = np.divmod(np.arange(self.N), self.ny)
        cx = self.lo[0] + (i + 0.5) / self.nx * (self.hi[0] - self.lo[0])
        cy = self.lo[1] + (j + 0.5) / self.ny * (self.hi[1] - self.lo[1])
        return np.stack([cx, cy], 1)

# --------------------------- map construction --------------------------------

def build_map(src, dst, nx, ny, src_dilate=1, tgt_pad=1):
    """Smoothed-Ulam multivalued map as a sparse boolean matrix A[s, d]=1.
    src, dst: aligned integer cell arrays (-1 = outside, dropped)."""
    ok = (src >= 0) & (dst >= 0)
    if not ok.any():
        return sp.csr_matrix((nx * ny, nx * ny), dtype=bool)
    P = np.unique(np.stack([src[ok], dst[ok]], 1), axis=0)
    si, sj = np.divmod(P[:, 0], ny)
    di, dj = np.divmod(P[:, 1], ny)
    rows, cols = [], []
    R1 = range(-src_dilate, src_dilate + 1)
    R2 = range(-tgt_pad, tgt_pad + 1)
    for a in R1:
        for b in R1:
            s2i, s2j = si + a, sj + b
            mS = (s2i >= 0) & (s2i < nx) & (s2j >= 0) & (s2j < ny)
            for u in R2:
                for v in R2:
                    d2i, d2j = di + u, dj + v
                    m = mS & (d2i >= 0) & (d2i < nx) & (d2j >= 0) & (d2j < ny)
                    rows.append(s2i[m] * ny + s2j[m])
                    cols.append(d2i[m] * ny + d2j[m])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    A = sp.csr_matrix((np.ones(len(rows), bool), (rows, cols)),
                      shape=(nx * ny, nx * ny))
    A.sum_duplicates()
    return A


def compose(A, k):
    """Exactly-k-step map A^k (boolean)."""
    Ak = A.copy()
    for _ in range(k - 1):
        Ak = (Ak @ A).astype(bool)
        Ak = sp.csr_matrix(Ak)
    return Ak


def intersect(A, B):
    return sp.csr_matrix(A.multiply(B))

def support_restrict(A, src, dst):
    """Keep only edges between data-supported cells (treat the rest as OUT,
    as in the bouncing-ball computation). src/dst: raw observed cell arrays."""
    n = A.shape[0]
    sup = np.zeros(n, bool)
    ok = src >= 0
    sup[src[ok]] = True
    ok = dst >= 0
    sup[dst[ok]] = True
    D = sp.diags(sup.astype(np.int8))
    return sp.csr_matrix((D @ A @ D).astype(bool))


# ------------------------ Morse graph machinery ------------------------------

def morse_graph(A):
    """Returns (comp labels, list of recurrent component ids, members dict,
    hasse edges between recurrent comps)."""
    n = A.shape[0]
    ncomp, comp = connected_components(A, directed=True, connection='strong')
    coo = A.tocoo()
    # recurrent components: contain at least one internal edge
    internal = comp[coo.row] == comp[coo.col]
    rec_ids = np.unique(comp[coo.row[internal]])
    members = defaultdict(list)
    nonempty = np.zeros(ncomp, bool)
    # restrict membership listing to cells that have any edge (in or out)
    active = np.unique(np.concatenate([coo.row, coo.col]))
    for c in active:
        members[comp[c]].append(c)
        nonempty[comp[c]] = True
    # condensation edges
    ce = np.unique(np.stack([comp[coo.row[~internal]],
                             comp[coo.col[~internal]]], 1), axis=0) \
        if (~internal).any() else np.zeros((0, 2), int)
    succ = defaultdict(set)
    for a, b in ce:
        succ[a].add(b)

    def reaches(a, b):
        seen, st = {a}, [a]
        while st:
            x = st.pop()
            if x == b:
                return True
            for y in succ[x]:
                if y not in seen:
                    seen.add(y)
                    st.append(y)
        return False

    rec = list(rec_ids)
    if len(rec) <= 40:
        reach = [(a, b) for a in rec for b in rec if a != b and reaches(a, b)]
        hasse = [(a, b) for (a, b) in reach
                 if not any((a, c) in reach and (c, b) in reach
                            for c in rec if c not in (a, b))]
    else:                      # too many recurrent comps for pairwise BFS
        reach, hasse = [], []
    return comp, rec, members, hasse, reach


def forward_closure(A, cells):
    """All cells reachable from `cells` (including them)."""
    n = A.shape[0]
    seen = np.zeros(n, bool)
    seen[list(cells)] = True
    frontier = np.array(sorted(cells), dtype=int)
    while len(frontier):
        nxt = np.unique(A[frontier].tocoo().col)
        nxt = nxt[~seen[nxt]]
        seen[nxt] = True
        frontier = nxt
    return np.flatnonzero(seen)

# ---------------- GF(2) relative cubical homology on the grid ----------------

def _cell_edges(c, ny):
    i, j = divmod(int(c), ny)
    v00 = i * (ny + 1) + j
    v01 = v00 + 1
    v10 = (i + 1) * (ny + 1) + j
    v11 = v10 + 1
    return [(v00, v01), (v01, v11), (v10, v11), (v00, v10)]


def _gf2_rank(rows):
    pivots = {}
    r = 0
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                r += 1
                break
    return r


def betti_pair(N_cells, L_cells, ny):
    """dims of H_*(|N|, |L|; Z/2) on the uniform grid (2D squares)."""
    N_cells = set(int(c) for c in N_cells)
    L_cells = set(int(c) for c in L_cells)

    def faces(cells):
        E, V = set(), set()
        for c in cells:
            for e in _cell_edges(c, ny):
                E.add(e)
                V.update(e)
        return E, V

    EN, VN = faces(N_cells)
    EL, VL = faces(L_cells)
    F2 = [c for c in N_cells if c not in L_cells]
    E = sorted(EN - EL)
    V = sorted(VN - VL)
    eidx = {e: t for t, e in enumerate(E)}
    vidx = {v: t for t, v in enumerate(V)}

    # rank d1 via union-find with virtual basepoint for L-touching edges
    parent = list(range(len(V) + 1))
    BASE = len(V)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    r1 = 0
    for e in E:
        vs = [vidx[v] for v in e if v in vidx]
        if not vs:
            continue
        a = vs[0]
        b = vs[1] if len(vs) == 2 else BASE
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            r1 += 1

    # rank d2 via free-edge peeling + GF(2) core
    cell_rel = {}
    edge_faces = defaultdict(list)
    for c in F2:
        es = [eidx[e] for e in _cell_edges(c, ny) if e in eidx]
        cell_rel[c] = es
        for t in es:
            edge_faces[t].append(c)
    alive = set(F2)
    deg = {t: len(fs) for t, fs in edge_faces.items()}
    from collections import deque
    q = deque(t for t, d in deg.items() if d == 1)
    used = set()
    r2 = 0
    while q:
        t = q.popleft()
        if deg.get(t, 0) != 1 or t in used:
            continue
        c = next((f for f in edge_faces[t] if f in alive), None)
        if c is None:
            continue
        alive.discard(c)
        used.add(t)
        r2 += 1
        for t2 in cell_rel[c]:
            if t2 != t:
                deg[t2] -= 1
                if deg[t2] == 1:
                    q.append(t2)
    core = []
    for c in alive:
        bits = 0
        for t in cell_rel[c]:
            if t not in used:
                bits ^= (1 << t)
        core.append(bits)
    r2 += _gf2_rank(core)

    b0 = len(V) - r1
    b1 = len(E) - r1 - r2
    b2 = len(F2) - r2
    return (b0, b1, b2)


def conley_indices(A, rec, members, ny, max_sets=12, min_size=3):
    """Index of each recurrent Morse set via the pair
    (forward closure of M, closure \\ M). Computed for the largest
    max_sets sets of size >= min_size (others left out)."""
    out = {}
    todo = sorted(rec, key=lambda c: -len(members[c]))[:max_sets]
    for cc in todo:
        if len(members[cc]) < min_size:
            continue
        M = set(members[cc])
        U = set(forward_closure(A, M).tolist())
        out[cc] = betti_pair(U, U - M, ny)
    return out

# ----------------------------- one-call summary ------------------------------

def analyze(A, ny, label="", index=True, verbose=True):
    comp, rec, members, hasse, reach = morse_graph(A)
    idx = conley_indices(A, rec, members, ny) if index else {}
    if verbose:
        print(f"  [{label}] Morse sets: {len(rec)}")
        for cc in sorted(rec, key=lambda c: -len(members[c])):
            tag = idx.get(cc, "-")
            print(f"     set {cc}: {len(members[cc])} cells, index {tag}")
    return dict(comp=comp, rec=rec, members=members, hasse=hasse,
                reach=reach, index=idx)

# -------------------- strategy harness (shared by sims) ----------------------

def run_strategies(y_obs, dstep, k, m=3, w0=1, nx=96, d=2, seed=0,
                   index=True, verbose=True):
    """y_obs: scalar observable, uniformly sampled; lag 1 = one sample.
    dstep: delay-embedding step in samples. k: target coarse lag in samples.
    Returns (dict strategy -> (analysis, grid), Z_fine, cells_fine)."""
    res = {}

    def chart(y, w, dseed=0):
        ys = boxcar(y, w)
        D = delay_embed(ys, m, max(1, dstep))
        mu, V = pca_fit(D, d=d, seed=dseed)
        return pca_apply(D, mu, V), (mu, V)

    # fine chart
    Zf, basis_f = chart(y_obs, w0, seed)
    g = Grid2D(Zf, nx)
    cf = g.cells(Zf)

    def pairs(cells, lag):
        return cells[:-lag], cells[lag:]

    # 1. fine one-step
    A1 = build_map(*pairs(cf, 1), g.nx, g.ny)
    res['fine-1step'] = (analyze(A1, g.ny, 'fine-1step', index, verbose), g)

    # 2. composed^k  (outer lag-k enclosure from the fine map)
    Ak = compose(A1, k)
    res['composed^k'] = (analyze(Ak, g.ny, f'composed^{k}', index, verbose), g)

    # 3. coarse-matched: w = k, lag = k, own chart
    Zc, _ = chart(y_obs, k, seed + 1)
    gc = Grid2D(Zc, nx)
    cc_ = gc.cells(Zc)
    A3 = build_map(*pairs(cc_, k), gc.nx, gc.ny)
    res['coarse-matched'] = (analyze(A3, gc.ny, 'coarse-matched',
                                     index, verbose), gc)

    # 4. big window, fine lag (user's "large window small step")
    A4 = build_map(*pairs(cc_, 1), gc.nx, gc.ny)
    res['bigwin-finelag'] = (analyze(A4, gc.ny, 'bigwin-finelag',
                                     index, verbose), gc)

    # 5. fine window, big lag direct
    A5 = build_map(*pairs(cf, k), g.nx, g.ny)
    res['finewin-biglag'] = (analyze(A5, g.ny, 'finewin-biglag',
                                     index, verbose), g)

    # 6. intersection: composed^k  AND  coarse signal in the FINE chart, lag k
    ys_c = boxcar(y_obs, k)
    Dc = delay_embed(ys_c, m, max(1, dstep))
    Zc_on_f = pca_apply(Dc, *basis_f)
    cc_f = g.cells(Zc_on_f)
    B = build_map(*pairs(cc_f, k), g.nx, g.ny)
    A6 = intersect(Ak, B)
    res['intersect'] = (analyze(A6, g.ny, 'intersect', index, verbose), g)

    return res, Zf, cf

# =========================== N-dimensional version ===========================
# Same objects as above, for a d-dimensional chart: a uniform grid of n^d boxes
# (stored sparsely: only data-supported boxes ever get edges), a smoothed-Ulam
# map padded by one box in every direction (3^d neighbours), and GF(2) relative
# cubical homology of index pairs in any dimension. Cubes are encoded in doubled
# coordinates: box i -> 2i+1 per axis; a face replaces a coordinate by 2i or 2i+2;
# odd coordinates are the cube's interval directions, so dim = #odd coords.

import itertools


class GridND:
    def __init__(self, Z, n, qlo=0.5, qhi=99.5, margin=0.03):
        lo, hi = np.percentile(Z, qlo, axis=0), np.percentile(Z, qhi, axis=0)
        pad = (hi - lo) * margin + 1e-12
        self.lo, self.hi, self.n, self.d = lo - pad, hi + pad, n, Z.shape[1]
        self.shape = (n,) * self.d
        self.N = n ** self.d

    def multi(self, Z):
        return np.floor((Z - self.lo) / (self.hi - self.lo) * self.n).astype(np.int64)

    def cells(self, Z):
        I = self.multi(Z)
        ok = ((I >= 0) & (I < self.n)).all(1)
        out = np.full(len(Z), -1, np.int64)
        out[ok] = np.ravel_multi_index(I[ok].T, self.shape)
        return out

    def unravel(self, c):
        return np.stack(np.unravel_index(np.asarray(c, np.int64), self.shape), 1)

    def centers(self, c):
        return self.lo + (self.unravel(c) + 0.5) / self.n * (self.hi - self.lo)


def build_map_nd(src, dst, grid, tgt_pad=1):
    """Ulam map on GridND: observed edges src->dst, targets padded by tgt_pad
    boxes in every axis, then restricted to data-supported boxes."""
    ok = (src >= 0) & (dst >= 0)
    sup = np.unique(np.concatenate([src[ok], dst[ok]]))
    P = np.unique(np.stack([src[ok], dst[ok]], 1), axis=0)
    D = grid.unravel(P[:, 1])
    rows, cols = [], []
    for off in itertools.product(range(-tgt_pad, tgt_pad + 1), repeat=grid.d):
        Dn = D + np.array(off)
        m = ((Dn >= 0) & (Dn < grid.n)).all(1)
        c = np.ravel_multi_index(Dn[m].T, grid.shape)
        keep = np.isin(c, sup)
        rows.append(P[m, 0][keep]); cols.append(c[keep])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = sp.csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(grid.N, grid.N))
    A.sum_duplicates()
    return A


def betti_pair_nd(N_cells, L_cells, grid):
    """dims of H_k(|N|, |L|; Z/2), k = 0..d, for unions of grid boxes."""
    d, B = grid.d, 2 * grid.n + 1
    W = B ** np.arange(d, dtype=np.int64)
    offs = np.array(list(itertools.product((-1, 0, 1), repeat=d)), np.int64)

    def closure(cells):
        if len(cells) == 0:
            return np.zeros(0, np.int64)
        C = 2 * grid.unravel(np.fromiter(cells, np.int64)) + 1
        return np.unique(((C[:, None, :] + offs[None]) @ W).ravel())

    rel = np.setdiff1d(closure(N_cells), closure(L_cells))        # sorted codes
    if len(rel) == 0:
        return (0,) * (d + 1)
    X = (rel[:, None] // W[None]) % B                               # doubled coords
    dim = (X % 2).sum(1)
    fa, co = [], []                                                 # face index, coface index
    for j in range(d):
        odd = np.flatnonzero(X[:, j] % 2 == 1)
        for s in (-1, 1):
            f = rel[odd] + s * W[j]
            pos = np.searchsorted(rel, f)
            hit = (pos < len(rel)) & (rel[np.minimum(pos, len(rel) - 1)] == f)
            fa.append(pos[hit]); co.append(odd[hit])
    fa, co = np.concatenate(fa), np.concatenate(co)
    n = len(rel)
    Bd = sp.csr_matrix((np.ones(len(fa), np.int8), (co, fa)), shape=(n, n))   # cell -> its faces
    Cf = Bd.T.tocsr()                                                          # cell -> its cofaces
    alive = np.ones(n, bool)
    cnt = np.diff(Cf.indptr).astype(np.int64)
    # free-face collapses: a face with exactly one live coface is removed with it
    stack = list(np.flatnonzero(cnt == 1))
    while stack:
        s_ = stack.pop()
        if not alive[s_] or cnt[s_] != 1:
            continue
        cf = Cf.indices[Cf.indptr[s_]:Cf.indptr[s_ + 1]]
        t = cf[alive[cf]]
        if len(t) != 1:
            continue
        t = t[0]
        alive[s_] = alive[t] = False
        for c in (t, s_):
            for f in Bd.indices[Bd.indptr[c]:Bd.indptr[c + 1]]:
                if alive[f]:
                    cnt[f] -= 1
                    if cnt[f] == 1:
                        stack.append(f)
    # GF(2) ranks of the boundary maps on the surviving core
    core = np.flatnonzero(alive)
    nk = np.bincount(dim[core], minlength=d + 1)
    loc = {int(c): i for i, c in enumerate(core)}
    rank = np.zeros(d + 2, int)
    for k in range(1, d + 1):
        rows = []
        for c in core[dim[core] == k]:
            bits = 0
            for f in Bd.indices[Bd.indptr[c]:Bd.indptr[c + 1]]:
                if alive[f]:
                    bits ^= 1 << loc[int(f)]
            rows.append(bits)
        rank[k] = _gf2_rank(rows)
    return tuple(int(nk[k] - rank[k] - rank[k + 1]) for k in range(d + 1))


def conley_indices_nd(A, rec, members, grid, max_sets=12, min_size=3, max_cells=6000):
    """Index of each recurrent set M from the pair (U, U \\ M), U = forward
    closure of M. Sets whose U exceeds max_cells are skipped (None)."""
    out = {}
    for cc in sorted(rec, key=lambda c: -len(members[c]))[:max_sets]:
        if len(members[cc]) < min_size:
            continue
        M = set(members[cc])
        U = set(forward_closure(A, M).tolist())
        out[cc] = betti_pair_nd(U, U - M, grid) if len(U) <= max_cells else None
    return out


def downsets(nodes, reach, limit=4096):
    """Lattice of downsets of the poset (nodes, a > b iff (a, b) in reach):
    each downset contains everything reachable from its members."""
    below = {a: {b for (x, b) in reach if x == a} for a in nodes}
    out = {frozenset()}
    for k in range(1, len(nodes) + 1):
        for S in itertools.combinations(nodes, k):
            S = frozenset(S)
            if all(below[a] & set(nodes) <= S for a in S):
                out.add(S)
            if len(out) > limit:
                return None
    return sorted(out, key=len)
