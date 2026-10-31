"""Relative zigzag persistence of index pairs.

H_*(|P|, |E|) is computed as the reduced homology of the triangulated cubical set
|P| with a cone over |E|:  X(P, E) = T(P) ∪ apex * T(E).  A zigzag of pairs

    (P_0,E_0) ⊇ (P_0,E_0)∩(P_1,E_1) ⊆ (P_1,E_1) ⊇ ...      (Dey-Mrozek-Slechta)

becomes a zigzag of simplicial complexes sharing one apex, fed to Dionysus.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

try:                                   # fast zigzag of Dey & Hou (preferred)
    from pyfzz import pyfzz
    BACKEND = "fzz"
except ImportError:                    # pragma: no cover
    BACKEND = "dionysus"
try:
    import dionysus as dio
except ImportError:                    # pragma: no cover
    dio = None


# Middle term between consecutive index pairs: "intersection" is the relative
# zigzag of Dey-Mrozek-Slechta (SoCG 2020), (P,E) ⊇ (P∩P', E∩E') ⊆ (P',E');
# "union" uses the cospan (P,E) ⊆ (P∪P', E∪E') ⊇ (P',E') instead.
MIDDLE = "intersection"


def set_middle(name):
    global MIDDLE
    MIDDLE = name


def set_backend(name):
    global BACKEND
    BACKEND = name


class Triangulator:
    """Freudenthal triangulation of unions of closed grid cubes (any dimension)."""

    def __init__(self, grid):
        self.grid = grid
        d = grid.dim
        self.vsize = grid.size + 1
        self.vstrides = np.array([int(np.prod(self.vsize[i + 1:])) for i in range(d)], np.int64)
        self.apex = int(np.prod(self.vsize))
        # chains of unit offsets (as flat vertex offsets) for each permutation
        chains = []
        for perm in itertools.permutations(range(d)):
            off, c = 0, [0]
            for j in perm:
                off += self.vstrides[j]
                c.append(off)
            chains.append(c)
        self.chains = np.array(chains, np.int64)                 # (d!, d+1)
        self.faces = {k: list(itertools.combinations(range(d + 1), k)) for k in range(1, d + 2)}

    def simplices(self, cells, cone=False):
        """dict k -> (m, k) int64 array of simplices with k vertices (sorted rows)."""
        out = {}
        cells = np.asarray(cells, np.int64)
        if cells.size == 0:
            return out
        base = (self.grid.multi(cells) * self.vstrides).sum(-1)          # (n,)
        tops = (base[:, None, None] + self.chains[None]).reshape(-1, self.grid.dim + 1)
        for k, combos in self.faces.items():
            arr = np.unique(np.concatenate([tops[:, list(c)] for c in combos]), axis=0)
            if cone:
                arr = np.hstack([arr, np.full((len(arr), 1), self.apex, np.int64)])
                out[k + 1] = arr
            else:
                out[k] = arr
        if cone:
            out[1] = np.array([[self.apex]], np.int64)
        return out


def pair_complex(tri, P_mask, E_mask):
    cx = tri.simplices(np.nonzero(P_mask)[0])
    if E_mask is not None and E_mask.any():
        for k, arr in tri.simplices(np.nonzero(E_mask)[0], cone=True).items():
            cx[k] = np.vstack([cx[k], arr]) if k in cx else arr
    else:   # empty E: cone point alone gives H(X ⊔ pt) whose reduced homology = H(X)
        cx[1] = np.vstack([cx.get(1, np.empty((0, 1), np.int64)), [[tri.apex]]])
    return cx


def zigzag_barcode(tri, spaces, max_dim=None, middle=None):
    """spaces: list of (P_mask, E_mask) or None (absent), consecutive in time.
    Inserts the pairwise intersections and returns bars over *space* positions:
    list of (dim, start, end) where start/end are half-integers: an integer i means
    space i, i+0.5 means the intersection between spaces i and i+1."""
    D = tri.grid.dim
    max_dim = D if max_dim is None else max_dim
    middle = MIDDLE if middle is None else middle
    cxs = [pair_complex(tri, P, E) if P is not None and P.any() else {} for (P, E) in
           [(s if s is not None else (None, None)) for s in spaces]]
    L = 2 * len(spaces) - 1
    # global simplex ids per dimension
    sims, pres = [], []
    for k in range(1, D + 3):
        arrs = [cx[k] for cx in cxs if k in cx]
        if not arrs:
            continue
        allk = np.concatenate(arrs)
        uniq, inv = np.unique(allk, axis=0, return_inverse=True)
        inv = inv.ravel()
        pr = np.zeros((len(uniq), L), bool)
        off = 0
        for i, cx in enumerate(cxs):
            if k in cx:
                n = len(cx[k])
                pr[inv[off:off + n], 2 * i] = True
                off += n
        if middle == "union":         # X_i -> X_i ∪ X_i+1 <- X_i+1
            pr[:, 1::2] = pr[:, 0:-1:2] | pr[:, 2::2]
        else:                         # X_i <- X_i ∩ X_i+1 -> X_i+1  (SoCG 2020)
            pr[:, 1::2] = pr[:, 0:-1:2] & pr[:, 2::2]
        sims.append(uniq)
        pres.append(pr)
    if not sims:
        return [], 0
    if BACKEND == "fzz":
        bars, n = _run_fzz(sims, pres, L, max_dim)
    else:
        bars, n = _run_dionysus(sims, pres, L, D, max_dim)
    bars = [(dim, q0 / 2.0, q1 / 2.0) for (dim, q0, q1) in bars]
    return _reduce(bars, spaces), n


def _run_fzz(sims, pres, L, max_dim):
    """Fast zigzag (Dey & Hou, ESA 2022).  Ops are simplex-wise; position q is
    realised at complex index c_q (count of ops done through position q)."""
    ops, c = [], []
    for q in range(L):
        ins, dels = [], []
        for uniq, pr in zip(sims, pres):
            before = pr[:, q - 1] if q > 0 else np.zeros(len(pr), bool)
            now = pr[:, q]
            ii = np.nonzero(now & ~before)[0]
            dd = np.nonzero(before & ~now)[0]
            if ii.size:
                ins.append(uniq[ii])
            if dd.size:
                dels.append(uniq[dd])
        for arr in ins:                       # faces first (dims ascending)
            ops.extend(("i", s) for s in arr.tolist())
        for arr in reversed(dels):            # cofaces first
            ops.extend(("d", s) for s in arr.tolist())
        c.append(len(ops))
    raw = pyfzz().compute_zigzag(ops) if ops else []
    c = np.array(c)
    bars = []
    for (b, d, p) in raw:
        if p > max_dim:
            continue
        qs = np.nonzero((c >= b) & (c <= d))[0]
        if qs.size:
            bars.append((p, int(qs[0]), int(qs[-1])))
    return bars, sum(len(u) for u in sims)


def _reduce(bars, spaces):
    """Reduced homology: drop one H0 bar per maximal present run (apex component)."""
    present = [i for i, s in enumerate(spaces) if s is not None and s[0] is not None and s[0].any()]
    runs, cur = [], []
    for i in present:
        if cur and i != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    for r in runs:
        key = (0, float(r[0]), float(r[-1]))
        if key in bars:
            bars.remove(key)
    return sorted(bars)


def _run_dionysus(sims, pres, L, D, max_dim):
    # times: insertion at q - 0.5 + 0.01*dim ; deletion at q - 0.5 + 0.01*(D+3-dim)
    simplex_list, times = [], []
    for uniq, pr in zip(sims, pres):
        k = uniq.shape[1]
        padded = np.hstack([np.zeros((len(pr), 1), bool), pr])
        ch = padded[:, 1:] != padded[:, :-1]                # change at position q
        rows, qs = np.nonzero(ch)
        ins = pr[rows, qs]
        tval = qs - 0.5 + np.where(ins, 0.01 * k, 0.01 * (D + 4 - k))
        split = np.searchsorted(rows, np.arange(1, len(pr)))
        per = np.split(tval, split)
        for s, t in zip(uniq, per):
            simplex_list.append(dio.Simplex(s.tolist()))
            times.append(t.tolist())
    f = dio.Filtration(simplex_list)
    # Filtration may reorder; map times through the new order
    order = {tuple(sorted(s)): i for i, s in enumerate(simplex_list)}
    times_sorted = [times[order[tuple(sorted(s))]] for s in f]
    _, dgms, _ = dio.zigzag_homology_persistence(f, times_sorted)
    bars = []
    for dim, dg in enumerate(dgms):
        if dim > max_dim:
            continue
        for p in dg:
            q0 = int(math.ceil(p.birth))
            q1 = L - 1 if math.isinf(p.death) else int(math.ceil(p.death)) - 1
            if q1 < q0:
                continue
            bars.append((dim, q0, q1))
    return bars, len(simplex_list)


def pair_homology(tri, P, E):
    bars, _ = zigzag_barcode(tri, [(P, E)])
    D = tri.grid.dim
    return [sum(1 for b in bars if b[0] == k) for k in range(D + 1)]
