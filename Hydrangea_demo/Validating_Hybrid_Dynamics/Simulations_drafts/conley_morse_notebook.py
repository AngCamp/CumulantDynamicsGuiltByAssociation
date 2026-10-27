# =============================================================================
# conley_morse_notebook.py
#
# ONE self-contained script / notebook for the combinatorial Conley-Morse
# analysis of a 2D flow with metastable sub-orbits (an energy-regulated
# double-well: each well holds a small attracting loop at target energy H0,
# noise switches the state between wells -- a 2D caricature of Lorenz wings).
#
# Run top-to-bottom, or as cells (the "# %%" markers work in Jupyter/VS Code):
#   cell 1  LIBRARY   flow, grid, bloated time-tau cell map, Morse sets
#                     (recurrent SCCs), Conley-Morse graph, Conley indices
#                     via flow-adapted index pairs + relative cubical
#                     homology over Z/2, Morse-equation check, graph drawing
#   cell 2  FIGURE 1  six-panel summary: Morse sets + computed indices,
#                     Conley-Morse graph, occupancy heatmap per graph node,
#                     delay-embedding density, energy regulation, trajectory
#                     traced through the decomposition
#   cell 3  FIGURE 2  Forman-style combinatorial (multi)vector field on two
#                     cubical patches (saddle + right well)
#   cell 4  FIGURE 3  lambda-sweep over H0: branch diagram + Conley-Morse
#                     persistence barcode (illustration of, not the algorithm
#                     from, Dey-Lipinski-Soriano-Trigueros, FoCM 2026).
#                     ~3 min on first run; results cached in sweep_cache.pkl.
#
# Extend from here: swap in your own 2D field in make_field(), re-tune the
# grid/tau/bloat in cell 2, or add lambda-parameterized families in cell 4.
# =============================================================================
# %% ------------------------------------------------- cell 1: LIBRARY
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ----------------------------------------------------------------------------
# flow
# ----------------------------------------------------------------------------
def make_field(H0, gamma_up=12.0, gamma_down=2.5):
    """Energy-regulated double-well vector field, vectorized."""
    def field(x, y):
        H = 0.5 * y**2 + 0.25 * x**4 - 0.5 * x**2
        g = np.where(H < H0, gamma_up, gamma_down)
        return y, x - x**3 + g * (H0 - H) * y
    return field

def make_rk2(field):
    def rk2(x, y, dt):
        k1x, k1y = field(x, y)
        k2x, k2y = field(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
        return x + dt * k2x, y + dt * k2y
    return rk2

# ----------------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------------
def make_grid(NX=61, NY=61, XMIN=-1.9, XMAX=1.9, YMIN=-1.25, YMAX=1.25):
    g = SimpleNamespace(NX=NX, NY=NY, XMIN=XMIN, XMAX=XMAX, YMIN=YMIN, YMAX=YMAX)
    g.x_edges = np.linspace(XMIN, XMAX, NX + 1)
    g.y_edges = np.linspace(YMIN, YMAX, NY + 1)
    g.hx = (XMAX - XMIN) / NX
    g.hy = (YMAX - YMIN) / NY
    g.N = NX * NY
    g.EXT = NX * NY
    return g

def cell_of(g, x, y):
    i = np.floor((x - g.XMIN) / (g.XMAX - g.XMIN) * g.NX).astype(int)
    j = np.floor((y - g.YMIN) / (g.YMAX - g.YMIN) * g.NY).astype(int)
    inside = (i >= 0) & (i < g.NX) & (j >= 0) & (j < g.NY)
    return np.where(inside, j * g.NX + i, -1)

def sample_points(g, cells, fr=(0.1, 0.37, 0.63, 0.9)):
    """k x k fan of sample points inside each listed cell. Returns px, py, src."""
    fr = np.asarray(fr)
    ci, cj = cells % g.NX, cells // g.NX
    cx = g.XMIN + (ci[:, None] + fr[None, :]) * g.hx
    cy = g.YMIN + (cj[:, None] + fr[None, :]) * g.hy
    px = np.repeat(cx, len(fr), axis=1).ravel()
    py = np.tile(cy, (1, len(fr))).ravel()
    src = np.repeat(cells, len(fr) ** 2)
    return px, py, src

# ----------------------------------------------------------------------------
# bloated time-tau multivalued cell map F
# ----------------------------------------------------------------------------
def build_map(g, rk2, dt=0.01, tau_steps=250, bloat=1):
    """Edges (src_cell, dst_cell) of the combinatorial multivalued map."""
    px, py, src = sample_points(g, np.arange(g.N))
    for _ in range(tau_steps):
        px, py = rk2(px, py, dt)
    di = np.floor((px - g.XMIN) / (g.XMAX - g.XMIN) * g.NX).astype(int)
    dj = np.floor((py - g.YMIN) / (g.YMAX - g.YMIN) * g.NY).astype(int)
    offs = [(a, b) for a in range(-bloat, bloat + 1) for b in range(-bloat, bloat + 1)]
    I = np.concatenate([di + a for a, b in offs])
    J = np.concatenate([dj + b for a, b in offs])
    S = np.tile(src, len(offs))
    inside = (I >= 0) & (I < g.NX) & (J >= 0) & (J < g.NY)
    D = np.where(inside, J * g.NX + I, g.EXT)
    return np.unique(np.stack([S, D], axis=1), axis=0)

# ----------------------------------------------------------------------------
# Morse decomposition + reduced Morse graph
# ----------------------------------------------------------------------------
def morse_decomposition(g, edges):
    """Returns (cell_morse, morse_cells, morse_edges, n_exits_to_EXT)."""
    A = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                   shape=(g.N + 1, g.N + 1)).tocsr()
    _, scc = connected_components(A, directed=True, connection='strong')
    size = np.bincount(scc)
    selfloop = np.zeros(len(size), dtype=bool)
    selfloop[scc[edges[edges[:, 0] == edges[:, 1], 0]]] = True
    recurrent = (size > 1) | selfloop
    recurrent[scc[g.EXT]] = False

    def cells(s): return np.flatnonzero(scc[:g.N] == s)
    morse_sccs = sorted(np.flatnonzero(recurrent),
                        key=lambda s: (-size[s], np.mean(cells(s) % g.NX)))
    morse_id = {s: k for k, s in enumerate(morse_sccs)}
    cell_morse = np.full(g.N, -1)
    morse_cells = []
    for s, k in morse_id.items():
        cs = cells(s)
        cell_morse[cs] = k
        morse_cells.append(cs)

    cond = {}
    for u, v in edges:
        if scc[u] != scc[v]:
            cond.setdefault(scc[u], set()).add(scc[v])

    def reach(s0):
        seen, stack, out = {s0}, [s0], set()
        while stack:
            for t in cond.get(stack.pop(), ()):
                if t not in seen:
                    seen.add(t)
                    if recurrent[t]:
                        out.add(t)
                    stack.append(t)
        return out

    R = {s: reach(s) for s in morse_sccs}
    morse_edges = [(morse_id[s], morse_id[t])
                   for s in morse_sccs for t in R[s]
                   if not any(t in R[m] for m in R[s] if m != t)]
    n_exit = int((edges[:, 1] == g.EXT).sum())
    return cell_morse, morse_cells, morse_edges, n_exit

# ----------------------------------------------------------------------------
# GF(2) linear algebra + relative cubical homology of a pair of square sets
# ----------------------------------------------------------------------------
def gf2_rank(A):
    A = A.copy()
    r = 0
    nrows, ncols = A.shape
    for c in range(ncols):
        piv = np.flatnonzero(A[r:, c])
        if len(piv) == 0:
            continue
        p = r + piv[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        mask = A[:, c].copy()
        mask[r] = False
        A[mask] ^= A[r]
        r += 1
        if r == nrows:
            break
    return r

def _closed_cells(squares):
    """Vertices, edges of the closed cubical complex spanned by (i,j) squares."""
    sq = set(map(tuple, squares))
    verts, hedges, vedges = set(), set(), set()
    for (i, j) in sq:
        verts |= {(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)}
        hedges |= {(i, j), (i, j + 1)}          # horizontal edge at (i, j)
        vedges |= {(i, j), (i + 1, j)}          # vertical   edge at (i, j)
    return sq, verts, hedges, vedges

def relative_betti(P1_squares, P2_squares):
    """Betti numbers of H(cl P1, cl P2; Z/2). P2_squares subset of P1_squares."""
    sq1, v1, h1, ve1 = _closed_cells(P1_squares)
    sq2, v2, h2, ve2 = _closed_cells(P2_squares) if len(P2_squares) else \
        (set(), set(), set(), set())
    Q2 = sorted(sq1 - sq2)
    Qh = sorted(h1 - h2)
    Qv = sorted(ve1 - ve2)
    Q1 = [('h', e) for e in Qh] + [('v', e) for e in Qv]
    Q0 = sorted(v1 - v2)
    i1 = {c: n for n, c in enumerate(Q1)}
    i0 = {c: n for n, c in enumerate(Q0)}

    d2 = np.zeros((len(Q1), len(Q2)), dtype=bool)
    for n, (i, j) in enumerate(Q2):
        for c in (('h', (i, j)), ('h', (i, j + 1)),
                  ('v', (i, j)), ('v', (i + 1, j))):
            if c in i1:
                d2[i1[c], n] = True
    d1 = np.zeros((len(Q0), len(Q1)), dtype=bool)
    for n, (kind, (i, j)) in enumerate(Q1):
        ends = ((i, j), (i + 1, j)) if kind == 'h' else ((i, j), (i, j + 1))
        for vtx in ends:
            if vtx in i0:
                d1[i0[vtx], n] = True

    r2 = gf2_rank(d2) if d2.size else 0
    r1 = gf2_rank(d1) if d1.size else 0
    b2 = len(Q2) - r2
    b1 = len(Q1) - r1 - r2
    b0 = len(Q0) - r1
    return (b0, b1, b2)

def poincare_str(betti):
    terms = []
    for k, b in enumerate(betti):
        if b == 0:
            continue
        t = {0: "1", 1: "t", 2: "t²"}[k]
        terms.append(str(b) if (k == 0 and b > 1) else (t if b == 1 else f"{b}{t}"))
    return " + ".join(terms) if terms else "0"

# ----------------------------------------------------------------------------
# Conley index of a Morse set: index pair + relative homology
#
# Two constructions:
#   * equilibrium-like sets (saddle, foci): forward tube + exit squares
#   * attracting sets: a FLOW-ADAPTED ISOLATING BLOCK built from the energy
#     H (which the regulation term makes a two-sided Lyapunov function around
#     the loop): the band {c- <= H <= c+} straddling the target energy is
#     positively invariant exactly -- pumping enters through the inner shell,
#     damping through the outer -- so its exit set is empty and its topology
#     (annulus / disk) carries the index. A cell-hull cannot do this, because
#     grid cells straddling the separatrix drag the hull into the other well.
# ----------------------------------------------------------------------------
def _H(x, y):
    return 0.5 * y**2 + 0.25 * x**4 - 0.5 * x**2

def _n_components(cells):
    """Number of 8-connected components of a set of (i, j) squares."""
    cells = set(cells)
    comps, seen = 0, set()
    for c in cells:
        if c in seen:
            continue
        comps += 1
        stack = [c]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    n = (x[0] + a, x[1] + b)
                    if n in cells and n not in seen:
                        stack.append(n)
    return comps

def attracting_block_index(g, rk2, set_cells, H0, other_sets=None,
                           exit_time=2.0, dt=0.01, delta=0.02):
    """
    Index pair for an attracting Morse set at target energy H0, from a
    flow-adapted energy block (the band/sub-level sets of H are exactly
    positively invariant). Shapes:
      * point-attractor blob (H0 <= -0.25, or set contains a well bottom,
        or the set has swallowed everything near a bifurcation):
        disk {H <= hi}, possibly minus holes at remaining OTHER Morse sets
      * loop in one well (H0 < 0): annulus {lo <= H <= hi} in that well
      * merged loop around both wells (H0 > 0): global annulus
    The margin is widened until the cubical carrier is one connected piece;
    exits are tested against the SMOOTH block (H within [lo-delta, hi+delta]
    and on the correct side), not the jagged cell carrier.
    """
    ci = set_cells % g.NX
    xlo = g.XMIN + ci.min() * g.hx
    xhi = g.XMIN + (ci.max() + 1) * g.hx
    mean_x = 0.5 * (xlo + xhi)
    spans = (xlo < -0.3) and (xhi > 0.3)
    fcells = {cell_of(g, np.array([sx]), np.array([0.0]))[0] for sx in (-1., 1.)}
    scell = int(cell_of(g, np.array([0.0]), np.array([0.0]))[0])
    contains_focus = bool(fcells & set(set_cells.tolist()))
    # a set that has swallowed the saddle is a merged bifurcation chunk: its
    # block must be a disk (minus holes at remaining sets), never a band
    contains_saddle = scell in set(set_cells.tolist())

    ii, jj = np.meshgrid(np.arange(g.NX), np.arange(g.NY))
    Xc = g.XMIN + np.stack([ii, ii + 1, ii, ii + 1, ii + 0.5]) * g.hx
    Yc = g.YMIN + np.stack([jj, jj, jj + 1, jj + 1, jj + 0.5]) * g.hy
    Hs = _H(Xc, Yc)
    Hmin, Hmax = Hs.min(axis=0), Hs.max(axis=0)
    xc = g.XMIN + (ii + 0.5) * g.hx

    holes = set()
    if other_sets:
        for cs in other_sets:
            holes |= _dilate(g, cs, 2)

    shape = None
    for margin in (0.5, 0.65, 0.8, 0.92):
        if spans and H0 > 0 and not contains_saddle:   # global annulus
            lo = max(1e-4, H0 - margin * H0)
            hi = H0 + margin * (0.25 - H0)
            keep = (Hmin >= lo) & (Hmax <= hi)
            shape, well = 'band', 0
        elif (spans or contains_saddle or H0 <= -0.25
              or contains_focus):               # disk (minus holes)
            lo = -np.inf
            glob = spans or contains_saddle
            hi = (H0 + margin * (0.0 - H0)) if (H0 < 0 and not glob) \
                else (max(0.02, H0) + 0.06)
            keep = Hmax <= hi
            shape = 'disk'
            well = 0 if glob else (-1 if mean_x < 0 else 1)
        else:                                   # annulus in one well
            lo = H0 - margin * (H0 + 0.25)
            hi = H0 + margin * (0.0 - H0)
            keep = (Hmin >= lo) & (Hmax <= hi)
            shape, well = 'band', (-1 if mean_x < 0 else 1)
        if well < 0:
            keep = keep & (xc < 0)
        elif well > 0:
            keep = keep & (xc > 0)
        P1 = set(zip(ii[keep].ravel().tolist(), jj[keep].ravel().tolist()))
        P1 -= holes
        if P1 and _n_components(P1) == 1:
            break

    P1_cells = np.array(sorted(i + j * g.NX for i, j in P1))
    if len(P1_cells) == 0:
        return (0, 0, 0), SimpleNamespace(P1=[], P2=[], isolated=False,
                                          tube_time=0.0)
    # exit test against the smooth block
    px, py, src = sample_points(g, P1_cells)
    for _ in range(max(1, int(round(exit_time / dt)))):
        px, py = rk2(px, py, dt)
    Hf = _H(px, py)
    ok = (Hf <= hi + delta)
    if shape == 'band':
        ok &= (Hf >= lo - delta)
    if well < 0:
        ok &= (px < 0.05)
    elif well > 0:
        ok &= (px > -0.05)
    if holes:                                   # landing inside a hole = exit
        hc = cell_of(g, px, py)
        hole_ids = {i + j * g.NX for i, j in holes}
        ok &= ~np.isin(hc, list(hole_ids))
    exit_cells = np.unique(src[~ok])
    P2 = [(c % g.NX, c // g.NX) for c in exit_cells]
    betti = relative_betti(sorted(P1), sorted(P2))
    return betti, SimpleNamespace(P1=sorted(P1), P2=sorted(P2), isolated=True,
                                  tube_time=0.0)

def repelling_block_index(g, set_cells, lam, margin=0.6):
    """
    Flow-adapted index pair for a repelling focus at a well bottom (lam >
    -0.25 makes the bottoms unstable): P1 = energy disk {H <= c} around the
    bottom with -0.25 < c < lam. Pumping pushes H up through its entire
    boundary, so the exit set is the whole combinatorial boundary ring and
    H(P1, P2) = t^2 exactly. (Mirror image of the attracting energy block.)
    """
    ci = set_cells % g.NX
    mean_x = np.mean(g.XMIN + (ci + 0.5) * g.hx)
    sgn = -1 if mean_x < 0 else 1
    c = -0.25 + margin * (lam + 0.25)
    ii, jj = np.meshgrid(np.arange(g.NX), np.arange(g.NY))
    Xc = g.XMIN + np.stack([ii, ii + 1, ii, ii + 1, ii + 0.5]) * g.hx
    Yc = g.YMIN + np.stack([jj, jj, jj + 1, jj + 1, jj + 0.5]) * g.hy
    Hmax = _H(Xc, Yc).max(axis=0)
    xc = g.XMIN + (ii + 0.5) * g.hx
    keep = (Hmax <= c) & ((xc < 0) if sgn < 0 else (xc > 0))
    P1 = set(zip(ii[keep].ravel().tolist(), jj[keep].ravel().tolist()))
    if not P1:                                # disk under grid resolution:
        P1 = _dilate(g, set_cells, 1)         # fall back to dilated cells
    P2 = [c_ for c_ in P1
          if any((c_[0] + a, c_[1] + b) not in P1
                 for a in (-1, 0, 1) for b in (-1, 0, 1))]
    betti = relative_betti(sorted(P1), sorted(P2))
    return betti, SimpleNamespace(P1=sorted(P1), P2=sorted(P2), isolated=True,
                                  tube_time=0.0)

def equilibrium_cells(g):
    """Grid cells of the three equilibria: left bottom, saddle, right bottom."""
    return {name: int(cell_of(g, np.array([x]), np.array([0.0]))[0])
            for name, x in (('left', -1.0), ('saddle', 0.0), ('right', 1.0))}

def merge_adjacent_sets(g, morse_cells, radius=1):
    """Union Morse sets whose cells come within `radius` cells of each other
    (combinatorial Morse decompositions may always be coarsened; this absorbs
    splinter SCCs produced by near-neutral slow regions)."""
    n = len(morse_cells)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    pts = [set(zip((cs % g.NX).tolist(), (cs // g.NX).tolist()))
           for cs in morse_cells]
    dil = [_dilate(g, cs, radius) for cs in morse_cells]
    for a in range(n):
        for b in range(a + 1, n):
            if dil[a] & pts[b]:
                parent[find(a)] = find(b)
    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    merged = [np.unique(np.concatenate([morse_cells[k] for k in ks]))
              for ks in groups.values()]
    return sorted(merged, key=lambda cs: (-len(cs), float(np.mean(cs % g.NX))))

def _dilate(g, cells, r=1):
    ci, cj = cells % g.NX, cells // g.NX
    out = set()
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            I, J = ci + a, cj + b
            ok = (I >= 0) & (I < g.NX) & (J >= 0) & (J < g.NY)
            out |= set(zip(I[ok].tolist(), J[ok].tolist()))
    return out

def conley_index(g, rk2, set_cells, tube_time, exit_time,
                 dt=0.01, snapshots=10, forbidden=None, max_retry=3):
    """
    Forward-tube index pair for an equilibrium-like Morse set:
      P1 = 1-cell dilation of (set cells + cells swept by the flow over
           [0, tube_time], recorded at `snapshots` intermediate times)
      P2 = squares of P1 any of whose sample points leave P1 within exit_time
    Returns (betti, info). `forbidden` = cells of OTHER Morse sets; the tube
    is shrunk (retries) if it touches them, so the pair stays isolating.
    The time scale is also retried downward if the pair comes out degenerate
    (e.g. the quotient core disconnects for a very unstable saddle).
    Attracting sets should use attracting_block_index instead.
    """
    result = None
    for scale in (1.0, 0.65, 0.45, 0.3):
        tt, et = tube_time * scale, exit_time * scale
        fset = set()
        for attempt in range(max_retry):
            swept = set(zip((set_cells % g.NX).tolist(),
                            (set_cells // g.NX).tolist()))
            if tt > 0:
                px, py, _ = sample_points(g, set_cells)
                steps = max(1, int(round(tt / dt)))
                rec = np.unique(np.linspace(0, steps, snapshots + 1).astype(int))
                for s in range(1, steps + 1):
                    px, py = rk2(px, py, dt)
                    if s in rec:
                        c = cell_of(g, px, py)
                        c = c[c >= 0]
                        swept |= set(zip((c % g.NX).tolist(),
                                         (c // g.NX).tolist()))
            P1 = _dilate(g, np.array([i + j * g.NX for i, j in swept]), 1)

            if forbidden is not None and len(forbidden):
                fset = set(zip((forbidden % g.NX).tolist(),
                               (forbidden // g.NX).tolist()))
                if P1 & fset:
                    tt *= 0.6             # tube touched another Morse set
                    continue
            break
        isolated = not (P1 & fset)

        # exit squares: any sample point leaves P1 within exit_time
        P1_cells = np.array(sorted(i + j * g.NX for i, j in P1))
        px, py, src = sample_points(g, P1_cells)
        for _ in range(max(1, int(round(et / dt)))):
            px, py = rk2(px, py, dt)
        dest = cell_of(g, px, py)
        in_P1 = np.isin(dest, P1_cells) & (dest >= 0)
        exit_cells = np.unique(src[~in_P1])
        P2 = [(c % g.NX, c // g.NX) for c in exit_cells]

        betti = relative_betti(sorted(P1), sorted(P2))
        result = (betti, SimpleNamespace(P1=sorted(P1), P2=sorted(P2),
                                         isolated=isolated, tube_time=tt))
        if max(betti) <= 1 and 1 <= sum(betti) <= 2:   # elementary index: done
            break
    return result

def index_params_for(set_cells, g):
    """Heuristic (tube_time, exit_time) per Morse-set size/kind."""
    if len(set_cells) > 20:               # loop annulus or attracting blob
        return 0.0, 2.5
    return 1.1, 1.1                       # equilibrium-like (saddle / focus)

# ----------------------------------------------------------------------------
# Morse-equation check
# ----------------------------------------------------------------------------
def morse_equation(bettis, pX=(1, 0, 0)):
    """sum p_M(t) - p_X(t) must equal (1+t) q(t), q >= 0. Returns q or None."""
    tot = [sum(b[k] for b in bettis) - pX[k] for k in range(3)]
    # divide 'tot' (degree <= 2) by (1+t): synthetic division
    q0 = tot[0]
    q1 = tot[1] - q0
    rem = tot[2] - q1
    if rem != 0 or q0 < 0 or q1 < 0:
        return None
    return (q0, q1)

# ----------------------------------------------------------------------------
# Morse-graph drawing (Hasse diagram, rim-offset arrows, p(t) labels)
# ----------------------------------------------------------------------------
def layout_morse_graph(n, edges):
    """Longest-path layering + barycenter ordering within layers."""
    depth = {k: 0 for k in range(n)}
    for _ in range(n):
        for a, b in edges:
            depth[b] = max(depth[b], depth[a] + 1)
    maxd = max(depth.values()) if n else 0
    layers = {}
    for k in range(n):
        layers.setdefault(depth[k], []).append(k)
    # barycenter pass: order each layer by mean x of connected nodes above
    dy = max(0.72 / max(maxd, 1), 0.26)      # adaptive spacing for deep graphs
    pos = {}
    for d in sorted(layers):
        ks = layers[d]
        if d > 0:
            def bary(k):
                nb = [pos[a][0] for a, b in edges if b == k and a in pos]
                return np.mean(nb) if nb else 0.5
            ks = sorted(ks, key=bary)
        for q, k in enumerate(ks):
            pos[k] = ((q + 1) / (len(ks) + 1), 0.88 - dy * d)
    return pos, depth, maxd

def draw_morse_graph(ax, edges, occ, labels=None, colors=None, names=None,
                     min_r=0.05, max_extra=0.24):
    """
    Nodes sized by occupancy, arrows start/end ON NODE RIMS (computed in data
    coordinates, so heads are never hidden under the disks), p(t) labels
    inside big nodes / beside small ones. All radii are globally rescaled so
    that no two disks can overlap and every edge keeps room for an arrowhead.
    ax gets equal aspect and [0,1]^2 limits.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Circle

    n = len(occ)
    occ = np.asarray(occ, dtype=float)
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:max(n, 1)]
    if names is None:
        names = [f"M{k}" for k in range(n)]
    pos, depth, maxd = layout_morse_graph(n, edges)
    radii = {k: min_r + max_extra * np.sqrt(occ[k]) for k in range(n)}

    # global shrink so disks never collide and arrows keep >= 0.09 of room
    edge_set = {tuple(sorted(e)) for e in edges}
    scale = 1.0
    for a in range(n):
        for b in range(a + 1, n):
            d = np.linalg.norm(np.array(pos[a]) - np.array(pos[b]))
            gap = 0.09 if (a, b) in edge_set else 0.015
            if radii[a] + radii[b] > 0:
                scale = min(scale, max(d - gap, 0.02) / (radii[a] + radii[b]))
    radii = {k: r * scale for k, r in radii.items()}

    ax.set_aspect('equal')
    xs = [pos[k][0] for k in range(n)]; ys = [pos[k][1] for k in range(n)]
    rs = [radii[k] for k in range(n)]
    ax.set_xlim(min(x - r for x, r in zip(xs, rs)) - 0.12,
                max(x + r for x, r in zip(xs, rs)) + 0.12)
    ax.set_ylim(min(y - r for y, r in zip(ys, rs)) - 0.09,
                max(y + r for y, r in zip(ys, rs)) + 0.09)
    ax.set_axis_off()

    for a, b in edges:
        pa, pb = np.array(pos[a]), np.array(pos[b])
        u = pb - pa
        L = np.linalg.norm(u)
        if L < 1e-9:
            continue
        u = u / L
        start = pa + u * (radii[a] + 0.008)
        end = pb - u * (radii[b] + 0.010)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                     mutation_scale=14, color='0.3',
                                     lw=1.4, shrinkA=0, shrinkB=0, zorder=2))
    for k in range(n):
        r = radii[k]
        ax.add_patch(Circle(pos[k], r, color=colors[k], zorder=3))
        big = r >= 0.10
        inside = names[k]
        if big and occ[k] >= 0.005:
            inside += f"\n{occ[k]*100:.0f}%"
        if big and labels is not None:
            inside += f"\n{labels[k]}"
        ax.text(pos[k][0], pos[k][1], inside, ha='center', va='center',
                fontsize=9 if big else 7.5, color='white',
                fontweight='bold', zorder=4)
        if labels is not None and not big:
            # small node: label above (top layer), below (bottom layer),
            # or at the side (middle layers), away from the arrows
            if depth[k] == 0:
                xy, ha, va = (pos[k][0], pos[k][1] + r + 0.012), 'center', 'bottom'
            elif depth[k] == maxd:
                xy, ha, va = (pos[k][0], pos[k][1] - r - 0.012), 'center', 'top'
            else:
                side = 1 if pos[k][0] <= 0.55 else -1
                xy = (pos[k][0] + side * (r + 0.015), pos[k][1])
                ha, va = ('left' if side > 0 else 'right'), 'center'
            ax.text(*xy, labels[k], ha=ha, va=va,
                    fontsize=8.5, style='italic', zorder=4,
                    bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.0))
    return pos


# %% ------------------------------------------------- cell 2: FIGURE 1 -- Conley-Morse summary of the H0 = -0.12 system


rng = np.random.default_rng(7)

H0, DT = -0.12, 0.01
field = make_field(H0)
rk2 = make_rk2(field)
g = make_grid()

# --- combinatorial map + Morse decomposition --------------------------------
edges = build_map(g, rk2, dt=DT, tau_steps=250, bloat=1)
cell_morse, morse_cells, morse_edges, n_exit = morse_decomposition(g, edges)
n_morse = len(morse_cells)

# --- Conley indices via index pairs (P1 = forward tube, P2 = exit squares) --
bettis, pair_info = [], []
all_cells = np.concatenate(morse_cells)
for k, cs in enumerate(morse_cells):
    if len(cs) > 20:                      # attracting set: energy-band block
        b, info = attracting_block_index(g, rk2, cs, H0, dt=DT)
    else:                                 # equilibrium-like: forward tube
        tube_t, exit_t = index_params_for(cs, g)
        b, info = conley_index(g, rk2, cs, tube_t, exit_t, dt=DT,
                                  forbidden=np.setdiff1d(all_cells, cs))
    bettis.append(b)
    pair_info.append(info)

labels = [poincare_str(b) for b in bettis]
q = morse_equation(bettis, pX=(1, 0, 0))

print(f"Morse sets: {n_morse}   (cells mapping out of window: {n_exit})")
for k, (cs, b, info) in enumerate(zip(morse_cells, bettis, pair_info)):
    print(f"  M{k}: {len(cs):4d} cells   Con = ({b[0]},{b[1]},{b[2]})  "
          f"p(t) = {labels[k]:7s}  |P1|={len(info.P1):4d} |P2|={len(info.P2):3d}"
          f"  isolated={info.isolated}")
print("Conley-Morse edges:", [(f"M{a}", f"M{b}") for a, b in morse_edges])
tot = tuple(sum(b[k] for b in bettis) for k in range(3))
if q is not None:
    print(f"Morse equation:  sum p_M = {poincare_str(tot)}"
          f"  =  p_X + (1+t) q(t)  with p_X = 1,  q(t) = {poincare_str((*q, 0))}"
          f"  ->  BALANCED (q >= 0 certifies the forced connecting orbits)")
else:
    print(f"Morse equation:  sum p_M = {poincare_str(tot)}  "
          f"did NOT balance against p_X = 1 (window not a trapping region?)")

# --- long noisy trajectory: densities, delay embedding, tracing -------------
STEPS, SIGMA = 90_000, 0.19
tx, ty = np.empty(STEPS), np.empty(STEPS)
x, y = 1.2, 0.1
for t in range(STEPS):
    x, y = rk2(x, y, DT)
    y += SIGMA * np.sqrt(DT) * rng.standard_normal()
    x = np.clip(x, g.XMIN + 1e-6, g.XMAX - 1e-6)
    y = np.clip(y, g.YMIN + 1e-6, g.YMAX - 1e-6)
    tx[t], ty[t] = x, y

traj_cells = cell_of(g, tx, ty)
density = np.bincount(traj_cells, minlength=g.N).reshape(g.NY, g.NX).astype(float)
trace = cell_morse[traj_cells]
occ = np.array([(trace == k).sum() for k in range(n_morse)]) / STEPS
visits = trace[np.isin(trace, [0, 1])]
switches = int(np.sum(visits[1:] != visits[:-1]))
print(f"Loop-to-loop switches in trajectory: {switches}")

LAG = 110
emb_hist, ex_e, ey_e = np.histogram2d(tx[LAG:], tx[:-LAG], bins=70,
                                      range=[[g.XMIN, g.XMAX], [g.XMIN, g.XMAX]])

# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
morse_colors = np.array(plt.cm.tab10.colors)[[0, 2, 3, 1, 4, 5, 6, 7, 8, 9][:n_morse]]
fig = plt.figure(figsize=(15.5, 11.5))
gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 0.3], hspace=0.42, wspace=0.3)

# (A) phase space
axA = fig.add_subplot(gs[0, 0:2])
gx, gy = np.meshgrid(np.linspace(g.XMIN, g.XMAX, 26), np.linspace(g.YMIN, g.YMAX, 18))
u, v = field(gx, gy)
axA.streamplot(gx, gy, u, v, color='0.84', density=1.1, linewidth=0.7, arrowsize=0.8)
overlay = np.ma.masked_where(cell_morse.reshape(g.NY, g.NX) < 0,
                             cell_morse.reshape(g.NY, g.NX))
axA.pcolormesh(g.x_edges, g.y_edges, overlay,
               cmap=mcolors.ListedColormap(morse_colors),
               vmin=-0.5, vmax=n_morse - 0.5, alpha=0.8)
axA.plot(tx[::3], ty[::3], color='k', lw=0.25, alpha=0.3)
anchor_off = {0: (0, -0.52), 1: (0, 0.52)}
text_off = {0: (-0.2, -0.95), 1: (0.2, 0.95),
            2: (0, 0.4), 3: (-0.4, 0.3), 4: (0.4, -0.3)}
for k in range(n_morse):
    cs = morse_cells[k]
    mx = np.mean(g.XMIN + (cs % g.NX + 0.5) * g.hx)
    my = np.mean(g.YMIN + (cs // g.NX + 0.5) * g.hy)
    ax_, ay_ = anchor_off.get(k, (0, 0))
    tx_, ty_ = text_off.get(k, (0, 0))
    axA.annotate(f"M{k}: {labels[k]}", (mx + ax_, my + ay_), (mx + tx_, my + ty_),
                 fontsize=10, fontweight='bold', ha='center', va='center',
                 arrowprops=dict(arrowstyle='-', lw=0.8) if (tx_ or ty_) else None,
                 bbox=dict(fc='white', alpha=0.85, ec='none', pad=1.5))
axA.set_title("(A) Flow, Morse sets with Conley indices $p(t)$, noisy trajectory")
axA.set_xlabel("x"); axA.set_ylabel("y")

# (B) Conley-Morse graph with p(t) labels (rim-offset arrows)
axB = fig.add_subplot(gs[0, 2])
axB.set_title("(B) Conley-Morse graph\n(node = Morse set, label = Conley index)")
draw_morse_graph(axB, morse_edges, occ, labels=labels, colors=morse_colors)

# (C) occupancy density per graph node
axC = fig.add_subplot(gs[1, 0])
pc = axC.pcolormesh(g.x_edges, g.y_edges, np.ma.masked_less(density, 1),
                    norm=mcolors.LogNorm(vmin=1), cmap='magma')
fig.colorbar(pc, ax=axC, label="visits per cell (log)")
cxs = 0.5 * (g.x_edges[:-1] + g.x_edges[1:])
cys = 0.5 * (g.y_edges[:-1] + g.y_edges[1:])
for k in range(n_morse):
    axC.contour(cxs, cys, (cell_morse.reshape(g.NY, g.NX) == k).astype(float),
                levels=[0.5], colors=[morse_colors[k]], linewidths=1.5)
axC.set_facecolor('0.96')
axC.set_title("(C) Occupancy density per graph node\n(Morse sets outlined)")
axC.set_xlabel("x"); axC.set_ylabel("y")

# (D) delay-embedding density
axD = fig.add_subplot(gs[1, 1])
pd_ = axD.pcolormesh(ex_e, ey_e, np.ma.masked_less(emb_hist.T, 1),
                     norm=mcolors.LogNorm(vmin=1), cmap='viridis')
fig.colorbar(pd_, ax=axD, label="visits per bin (log)")
axD.set_facecolor('0.96')
axD.set_title(f"(D) Delay-embedding density of x(t)\n"
              f"x(t) vs x(t−{LAG*DT:.1f}) — same two metastable loops")
axD.set_xlabel("x(t)"); axD.set_ylabel(f"x(t−{LAG*DT:.1f})")

# (E) energy regulation
axE = fig.add_subplot(gs[1, 2])
tseg = slice(0, 20_000)
tt = np.arange(STEPS) * DT
axE.plot(tt[tseg], 0.5*ty[tseg]**2 + 0.25*tx[tseg]**4 - 0.5*tx[tseg]**2,
         lw=0.5, color='0.3')
axE.axhline(H0, color='crimson', ls='--', lw=1.2, label='target $H_0$ (loops)')
axE.axhline(0.0, color='navy', ls=':', lw=1.2, label='saddle energy')
axE.set_title("(E) Energy regulation:\nfast pump-up, slow die-down")
axE.set_xlabel("time"); axE.set_ylabel("H(x,y)")
axE.legend(fontsize=8, loc='upper right')

# (F) trajectory traced through the decomposition
axF = fig.add_subplot(gs[2, :])
strip = mcolors.ListedColormap(['0.9'] + [tuple(c) for c in morse_colors])
axF.imshow((trace + 1)[None, :], aspect='auto', cmap=strip,
           vmin=-0.5, vmax=n_morse + 0.5,
           extent=[0, STEPS * DT, 0, 1], interpolation='nearest')
axF.set_yticks([]); axF.set_xlabel("time")
axF.set_title(f"(F) Morse-set membership along the trajectory "
              f"(grey = transient cells; {switches} switches between the loops)")

fig.suptitle("Combinatorial Conley-Morse summary of metastable sub-orbits in an "
             "energy-regulated double-well flow", fontsize=14, y=0.995)
fig.savefig("morse_metastable.png", dpi=140, bbox_inches='tight')
plt.show()


# %% ------------------------------------------------- cell 3: FIGURE 2 -- combinatorial (multi)vector field


field = make_field(-0.12)

# ---------------------------------------------------------------- complexes
def build_patch(x0, x1, y0, y1, nx, ny):
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    verts = {(i, j): (xs[i], ys[j]) for i in range(nx + 1) for j in range(ny + 1)}
    edges = {}
    for i in range(nx):
        for j in range(ny + 1):
            edges[('h', i, j)] = ((xs[i] + xs[i + 1]) / 2, ys[j])
    for i in range(nx + 1):
        for j in range(ny):
            edges[('v', i, j)] = (xs[i], (ys[j] + ys[j + 1]) / 2)
    squares = {('q', i, j): ((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2)
               for i in range(nx) for j in range(ny)}
    return xs, ys, verts, edges, squares

def vertex_edges(v, nx, ny):
    i, j = v
    out = []
    if i < nx:  out.append((('h', i, j),     np.array([+1., 0.])))
    if i > 0:   out.append((('h', i - 1, j), np.array([-1., 0.])))
    if j < ny:  out.append((('v', i, j),     np.array([0., +1.])))
    if j > 0:   out.append((('v', i, j - 1), np.array([0., -1.])))
    return out

def edge_squares(e, nx, ny):
    kind, i, j = e
    out = []
    if kind == 'h':
        if j < ny:  out.append((('q', i, j),     np.array([0., +1.])))
        if j > 0:   out.append((('q', i, j - 1), np.array([0., -1.])))
    else:
        if i < nx:  out.append((('q', i, j),     np.array([+1., 0.])))
        if i > 0:   out.append((('q', i - 1, j), np.array([-1., 0.])))
    return out

def forman_match(verts, edges, squares, nx, ny):
    """Greedy best-score matching: vertices->edges, then edges->squares."""
    f_at = {c: np.array(field(np.array([p[0]]), np.array([p[1]]))).ravel()
            for c, p in {**verts, **edges}.items()}
    # vertices -> edges (edge claimed by the best-scoring endpoint)
    claims = {}
    for v in verts:
        for e, d in vertex_edges(v, nx, ny):
            s = float(f_at[v] @ d)
            if s > 1e-12:
                claims.setdefault(e, []).append((s, v))
    pair = {}                       # lower cell -> higher cell
    taken_e, taken_v = set(), set()
    for e, cands in sorted(claims.items(),
                           key=lambda kv: -max(s for s, _ in kv[1])):
        for s, v in sorted(cands, reverse=True):
            if v not in taken_v and e not in taken_e:
                pair[v] = e
                taken_v.add(v); taken_e.add(e)
                break
    # unmatched edges -> squares
    claims = {}
    for e in edges:
        if e in taken_e:
            continue
        for q, nrm in edge_squares(e, nx, ny):
            s = float(f_at[e] @ nrm)
            if s > 1e-12:
                claims.setdefault(q, []).append((s, e))
    taken_q = set()
    for q, cands in sorted(claims.items(),
                           key=lambda kv: -max(s for s, _ in kv[1])):
        for s, e in sorted(cands, reverse=True):
            if e not in pair and e not in taken_e and q not in taken_q:
                pair[e] = q
                taken_e.add(e); taken_q.add(q)
                break
    critical = ([v for v in verts if v not in pair]
                + [e for e in edges if e not in pair and e not in taken_e]
                + [q for q in squares if q not in taken_q])
    return pair, critical

# ------------------------------------------------------------------ drawing
BEIGE, EDGE_C = '#e8dfae', '#8a8a6a'
UP_C, DOWN_C, CRIT_C = '#d95f0e', '#52a39b', '#d13d1e'

def draw_patch(ax, x0, x1, y0, y1, nx, ny, title, down_arrows):
    xs, ys, verts, edges, squares = build_patch(x0, x1, y0, y1, nx, ny)
    pair, critical = forman_match(verts, edges, squares, nx, ny)
    posmap = {**verts, **edges, **squares}

    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=BEIGE, ec='none'))
    for x in xs: ax.plot([x, x], [y0, y1], color=EDGE_C, lw=0.7, zorder=1)
    for y in ys: ax.plot([x0, x1], [y, y], color=EDGE_C, lw=0.7, zorder=1)

    if down_arrows:                       # thin dashed: dominant -> outside faces
        for q in squares:
            _, i, j = q
            for e in (('h', i, j), ('h', i, j + 1), ('v', i, j), ('v', i + 1, j)):
                if pair.get(e) != q:      # face outside this square's multivector
                    ax.add_patch(FancyArrowPatch(
                        posmap[q], posmap[e], arrowstyle='-|>',
                        mutation_scale=7, color=DOWN_C, lw=0.8,
                        linestyle=(0, (3, 2)), shrinkA=4, shrinkB=2, zorder=2))
        for v, e in [(v, e) for v, e in pair.items() if len(v) == 2]:
            kind, i, j = e                # edge's other vertex gets a down-arrow
            other = ((i + 1, j) if kind == 'h' else (i, j + 1))
            if other == v:
                other = (i, j)
            ax.add_patch(FancyArrowPatch(
                posmap[e], posmap[other], arrowstyle='-|>', mutation_scale=7,
                color=DOWN_C, lw=0.8, linestyle=(0, (3, 2)),
                shrinkA=3, shrinkB=3, zorder=2))

    for low, high in pair.items():        # thick solid up-arrows
        ax.add_patch(FancyArrowPatch(posmap[low], posmap[high],
                                     arrowstyle='-|>', mutation_scale=13,
                                     color=UP_C, lw=2.6, shrinkA=2, shrinkB=1,
                                     zorder=4))
    def on_boundary(c):
        if c[0] == 'q':
            return False
        if len(c) == 2:                   # vertex
            i, j = c
            return i in (0, nx) or j in (0, ny)
        kind, i, j = c
        return (j in (0, ny)) if kind == 'h' else (i in (0, nx))

    for c in critical:                    # loops on critical cells
        dim = 2 if c[0] == 'q' else (1 if c[0] in ('h', 'v') else 0)
        p = posmap[c]
        if on_boundary(c):                # artifact: flow exits the patch here
            ax.add_patch(Circle(p, 0.05, fill=False, ec='0.55', lw=1.6,
                                linestyle=(0, (2, 2)), zorder=5))
        else:
            ax.add_patch(Circle(p, 0.05, fill=False, ec=CRIT_C, lw=2.6,
                                zorder=5))
            ax.annotate(f"dim {dim}", p, (p[0] + 0.035, p[1] + 0.05),
                        fontsize=8, color=CRIT_C, fontweight='bold', zorder=6)
    for v, p in verts.items():
        ax.plot(*p, 'o', ms=5, color='#5b4a86', zorder=3)

    ax.set_xlim(x0 - 0.06, x1 + 0.06)
    ax.set_ylim(y0 - 0.06, y1 + 0.06)
    ax.set_aspect('equal'); ax.set_axis_off()
    ax.set_title(title, fontsize=11)
    return critical

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(15, 6.8), gridspec_kw=dict(width_ratios=[1, 1.5]))

critL = draw_patch(axL, -0.36, 0.36, -0.24, 0.24, 6, 4,
                   "Saddle patch — up-arrows (solid), down-arrows "
                   "(dashed)\nred rings: critical cluster at the saddle "
                   "(total index χ = −1)", down_arrows=True)
critR = draw_patch(axR, 0.40, 1.60, -0.60, 0.60, 10, 10,
                   "Right-well patch — circulation of up-arrows around "
                   "the metastable loop\n(down-arrows implicit); the rotation "
                   "center stays critical", down_arrows=False)
axL.annotate("grey dashed rings = patch-boundary artifacts\n(the flow exits "
             "the window there)", (0.02, -0.02), xycoords='axes fraction',
             fontsize=8.5, color='0.4', va='top')
axR.annotate("pure rotation cannot be a gradient matching — the center "
             "cell is forced critical.\nThis is exactly why Mrozek generalizes "
             "vectors to MULTIVECTORS: a strict\nmultivector (like "
             "{BC, FG, BCGF} in his Fig. 3) can swallow the rotation.",
             (0.02, -0.02), xycoords='axes fraction', fontsize=8.5,
             color='0.4', va='top')

fig.suptitle("A combinatorial vector field built from the flow "
             "(Forman matching on the cubical complex):\n"
             "each multivector = {cell, paired coface}; unmatched cells are "
             "critical multivectors (H(V) ≠ 0)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("multivector_field.png", dpi=140, bbox_inches='tight')
ncrit = {0: 0, 1: 0, 2: 0}
for c in critL + critR:
    ncrit[2 if c[0] == 'q' else (1 if c[0] in ('h', 'v') else 0)] += 1
print("critical cells by dimension:", ncrit)
plt.show()


# %% ------------------------------------------------- cell 4: FIGURE 3 -- lambda sweep + Conley-Morse barcode (~3 min first run, then cached)


DT = 0.01
LAMS = np.round(np.arange(-0.30, 0.1801, 0.02), 3)
g = make_grid(NX=81, NY=81)          # finer than the main figure's 61:
                                        # narrows the window where the grid
                                        # cannot separate the colliding loops

TYPE_COLOR = {  # stability read off the index
    (1, 0, 0): ('#1b6b38', 'attracting point  (1)'),
    (1, 1, 0): ('#58b368', 'attracting loop  (1+t)'),
    (1, 2, 0): ('#0e7b8a', 'merged loops+saddle  (1+2t)'),
    (0, 1, 0): ('#6a51a3', 'saddle  (t)'),
    (0, 0, 1): ('#e66101', 'repelling focus  (t²)'),
}
DEG_COLOR = {0: '#2a924a', 1: '#d95f0e', 2: '#b2182b'}
DEG_LABEL = {0: '[k,0,0]', 1: '[0,k,0]', 2: '[0,0,k]'}

# ---------------------------------------------------------------- the sweep
EQ = equilibrium_cells(g)

def analyze(lam):
    rk2 = make_rk2(make_field(lam))
    edges = build_map(g, rk2, dt=DT, tau_steps=250, bloat=1)
    _, morse_cells, _, _ = morse_decomposition(g, edges)
    # absorb splinter SCCs from near-neutral slow regions (a Morse
    # decomposition may always be coarsened)
    morse_cells = merge_adjacent_sets(g, morse_cells, radius=1)
    # EQUILIBRIUM PROBES: a strong repeller leaves no cell-level recurrence,
    # so the SCC criterion can miss it entirely -- but the equilibria are
    # isolated invariant sets regardless, so include each fixed point as a
    # Morse set whenever no detected set already covers it.
    covered = set().union(*[set(cs.tolist()) for cs in morse_cells]) \
        if morse_cells else set()
    for name, c in EQ.items():
        if c not in covered:
            morse_cells.append(np.array([c]))
    # small sets clustered at the same equilibrium are one Morse set
    def centroid_x(cs):
        return g.XMIN + (np.mean(cs % g.NX) + 0.5) * g.hx
    buckets, keep = {}, []
    for cs in morse_cells:
        cx = centroid_x(cs)
        tag = None
        if len(cs) <= 20:
            for name, x0 in (('left', -1.0), ('saddle', 0.0), ('right', 1.0)):
                if abs(cx - x0) < 4 * g.hx:
                    tag = name
        if tag is None:
            keep.append(cs)
        else:
            buckets.setdefault(tag, []).append(cs)
    morse_cells = keep + [np.unique(np.concatenate(v))
                          for v in buckets.values()]
    allc = np.concatenate(morse_cells)
    out = []
    for cs in morse_cells:
        others = [o for o in morse_cells if o is not cs]
        cx = centroid_x(cs)
        near_bottom = min(abs(cx - 1.0), abs(cx + 1.0)) < 4 * g.hx
        if len(cs) > 20:
            b, info = attracting_block_index(g, rk2, cs, lam, dt=DT,
                                                other_sets=others)
        elif near_bottom and lam > -0.25:      # repelling focus
            b, info = repelling_block_index(g, cs, lam)
        else:                                  # saddle (or attracting bottom)
            # short tube: a strongly unstable saddle (trace ~ GAMMA_UP*lam)
            # shreds a long tube's core into disconnected pieces, so scale
            # the tube time with the local instability
            tt = min(0.5, 0.6 / (1 + 12 * max(lam, 0))) if abs(cx) < 0.3 \
                else 1.1
            b, info = conley_index(g, rk2, cs, tt, tt, dt=DT,
                                      forbidden=np.setdiff1d(allc, cs))
        ci = cs % g.NX
        out.append(dict(cells=set(cs.tolist()), betti=b,
                        isolated=info.isolated,
                        xlo=g.XMIN + ci.min() * g.hx,
                        xhi=g.XMIN + (ci.max() + 1) * g.hx))
    return out

import pickle, os
CACHE = "sweep_cache.pkl"
if os.path.exists(CACHE):
    with open(CACHE, "rb") as f:
        records = pickle.load(f)
    print(f"loaded {len(records)} lambda values from {CACHE}")
else:
    records = {}
    for lam in LAMS:
        records[lam] = analyze(lam)
        descr = ", ".join(poincare_str(s['betti'])
                          + ('' if s['isolated'] else '*')
                          for s in records[lam])
        print(f"lambda={lam:+.2f}: {len(records[lam])} Morse sets  [{descr}]",
              flush=True)
    with open(CACHE, "wb") as f:
        pickle.dump(records, f)

# ------------------------------------------------- continuation by overlap
class Entity:
    _n = 0
    _named = set()
    def __init__(self):
        self.id = Entity._n; Entity._n += 1
        self.rows = []                     # (lam, xlo, xhi, betti)
        self.cells = set()
        self.name = None
    def christen(self):
        """Semantic name from the first row (side + stability)."""
        lam, xlo, xhi, b = self.rows[0]
        mid = 0.5 * (xlo + xhi)
        if b == (0, 0, 1):
            base = 'focus L' if mid < 0 else 'focus R'
        elif b == (0, 1, 0):
            base = 'saddle'
        elif b[0] >= 1:
            if xlo < -0.3 < 0.3 < xhi:
                base = 'merged'
            else:
                base = 'attr L' if mid < 0 else 'attr R'
        else:
            base = f'M{self.id}'
        while base in Entity._named:
            base += "′"
        Entity._named.add(base)
        self.name = base

entities, active = [], []                  # active: list of Entity
for lam in LAMS:
    sets_ = records[lam]
    used_e, used_s = set(), set()
    pairs = sorted(((len(e.cells & s['cells']), ei, si)
                    for ei, e in enumerate(active)
                    for si, s in enumerate(sets_)), reverse=True)
    assign = {}
    for ov, ei, si in pairs:
        if ov == 0 or ei in used_e or si in used_s:
            continue
        assign[si] = active[ei]
        used_e.add(ei); used_s.add(si)
    new_active = []
    for si, s in enumerate(sets_):
        e = assign.get(si)
        if e is None:
            e = Entity(); entities.append(e)
        betti = s['betti']
        if not s['isolated'] and e.rows:   # continuation: keep last clean index
            betti = e.rows[-1][3]
        e.rows.append((lam, s['xlo'], s['xhi'], betti))
        e.cells = s['cells']
        if e.name is None:
            e.christen()
        new_active.append(e)
    active = new_active
by_id = {e.id: e for e in entities}

# ---------------------------------------------------------------- barcode
# multiplicity-aware: an entity with beta_d = m carries m degree-d bars at
# once (the merged chunk holds TWO degree-1 generators), so bars open and
# close as the count rises and falls
dlam = LAMS[1] - LAMS[0]
bars = []                                  # (deg, birth, death, entity_id)
for e in entities:
    for d in range(3):
        open_bars = []                     # birth lambdas of currently open
        prev_lam = None
        for lam, _, _, b in e.rows:
            m = b[d]
            while len(open_bars) < m:
                open_bars.append(lam)
            while len(open_bars) > m:
                bars.append((d, open_bars.pop(), prev_lam, e.id))
            prev_lam = lam
        for b0 in open_bars:
            bars.append((d, b0, prev_lam, e.id))
bars.sort(key=lambda t: (t[0], t[1], t[3]))

nsets = [len(records[l]) for l in LAMS]
bifs = [0.5 * (LAMS[i] + LAMS[i + 1])
        for i in range(len(LAMS) - 1) if nsets[i] != nsets[i + 1]]

# ---------------------------------------------------------------- figure
fig, (axT, axB) = plt.subplots(2, 1, figsize=(12.5, 10), sharex=True,
                               gridspec_kw=dict(height_ratios=[1.35, 1],
                                                hspace=0.12))
# top: branch diagram
seen = set()
for lam in LAMS:
    for s in records[lam]:
        col, lab = TYPE_COLOR.get(s['betti'], ('0.5', 'other'))
        xlo, xhi = s['xlo'], s['xhi']
        if xhi - xlo < 0.05:
            m = 0.5 * (xlo + xhi); xlo, xhi = m - 0.025, m + 0.025
        axT.add_collection(LineCollection([[(lam, xlo), (lam, xhi)]],
                                          colors=col, linewidths=5, alpha=0.85,
                                          label=lab if lab not in seen else None))
        seen.add(lab)
for b in bifs:
    axT.axvline(b, color='0.6', ls='--', lw=1)
    axB.axvline(b, color='0.6', ls='--', lw=1)
axT.set_xlim(LAMS[0] - 0.015, LAMS[-1] + 0.055)
axT.set_ylim(g.XMIN, g.XMAX)
axT.set_ylabel("x-extent of Morse set")
axT.set_title("Branch diagram of the Morse sets over λ = H₀ "
              "(color = stability type from the Conley index)")
axT.legend(loc='upper left', fontsize=8.5, framealpha=0.95)
box = dict(fc='white', ec='0.7', alpha=0.9, pad=2.5)
axT.annotate("Hopf-like AR split:\npoint → loop + focus",
             (-0.265, -0.33), fontsize=9, ha='left', color='0.2', bbox=box)
axT.annotate("loops + saddle coalesce\n(grid cannot separate them\n"
             "near the collision at λ = 0)",
             (-0.055, 0.55), fontsize=9, ha='left', color='0.2', bbox=box)
axT.annotate("one loop around\nboth wells + saddle",
             (0.095, -0.4), fontsize=9, ha='left', color='0.2', bbox=box)

# bottom: barcode
for row, (d, b0, b1, eid) in enumerate(bars):
    axB.hlines(-row, b0, b1 + 0.6 * dlam, color=DEG_COLOR[d], lw=5)
    axB.annotate(DEG_LABEL[d], (b1 + 0.8 * dlam, -row), fontsize=8.5,
                 va='center', color=DEG_COLOR[d])
    axB.annotate(by_id[eid].name, (b0 - 0.4 * dlam, -row), fontsize=8,
                 va='center', ha='right', color='0.35')
for x, txt in [(bifs[0], "AR split\n(loops & foci born)"),
               (bifs[1], "coalescence\n(generators handed\nto the chunk)"),
               (bifs[2], "chunk resolves\n(loop + saddle\nseparate again)")]:
    axB.annotate(txt, (x, 3.3), fontsize=8, ha='center', va='top',
                 color='0.35', annotation_clip=False)
axB.set_ylim(-len(bars) + 0.2, 3.5)
axB.set_yticks([])
axB.set_xlabel("λ  =  target energy H₀")
axB.set_title("Conley–Morse persistence barcode — one bar per index "
              "generator, labeled by degree [β₀,β₁,"
              "β₂]; left label = the Morse set carrying it",
              fontsize=11)

fig.suptitle("Combinatorial bifurcations of the energy-regulated double-well "
             "family\n(branch diagram + Conley-Morse barcode; continuation by "
             "cell overlap — an illustration of, not the algorithm from, "
             "Dey–Lipiński–Soriano-Trigueros 2026)",
             fontsize=12.5)
fig.savefig("conley_barcode.png", dpi=140, bbox_inches='tight')
print(f"\nbars: {len(bars)}  bifurcation lambdas: {bifs}")
plt.show()
