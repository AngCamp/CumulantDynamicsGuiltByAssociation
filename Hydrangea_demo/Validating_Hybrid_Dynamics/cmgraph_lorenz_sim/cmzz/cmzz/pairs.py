"""Invariant sets of Morse-graph intervals, combinatorial index pairs, and
resolution coarsening of a Morse decomposition."""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.ndimage import binary_fill_holes


def reach(A, seeds, allowed=None):
    """Cells reachable from `seeds` (inclusive) along A (CSR rows -> cols)."""
    n = A.shape[0]
    vis = np.zeros(n, bool)
    vis[seeds] = True
    frontier = np.asarray(seeds, np.int64)
    while frontier.size:
        sub = A[frontier]
        nxt = np.unique(sub.indices)
        nxt = nxt[~vis[nxt]]
        if allowed is not None:
            nxt = nxt[allowed[nxt]]
        vis[nxt] = True
        frontier = nxt
    return vis


def invariant_set(frame, nodes):
    """Invariant set of a set of Morse nodes plus all connecting orbits between
    them (cells reachable from them that can also reach them)."""
    seeds = np.concatenate([frame.morse_cells[i] for i in nodes])
    fwd = reach(frame.A(), seeds)
    bwd = reach(frame.AT(), seeds)
    return fwd & bwd


def index_pair(frame, S_mask, rings=2, cover=None):
    """Combinatorial index pair (P, E) for the invariant set S.

    N = (S ∪ F(S)) grown by `rings` cubical rings;  P = cells reachable from S
    inside N;  E = P \\ S.  Then F(P) ∩ N ⊆ P, F(E) ∩ N ⊆ E (anything that maps
    back into S would itself be in S), and P \\ E = S.
    """
    g = frame.grid
    A = frame.A()
    S_idx = np.nonzero(S_mask)[0]
    img = np.zeros(g.n_cells, bool)
    img[np.unique(A[S_idx].indices)] = True
    N = g.neighbours_mask(S_mask | img, rings)
    P = reach(A, S_idx, allowed=N)
    E = P & ~S_mask
    # Covering pair only where it matters: the set has an exit (E non-empty)
    # and is clearly thinner than the side sets, so the plain pair would sit
    # inside the side Morse set and lose its exit in the intersection.
    if (cover is not None and E.any()
            and (cover & ~S_mask).sum() > 0.1 * max(1, cover.sum())):
        filled = binary_fill_holes(cover.reshape(tuple(g.size))).ravel()
        return _covering_pair(frame, S_mask, filled, rings)
    return P, E


def _covering_pair(frame, S_mask, cover, rings):
    r"""Index pair for S that also covers the cells `cover` (the neighbouring
    steps' Morse sets in a zigzag).  P = cells reachable from S ∪ cover inside N,
    E = cells of P that cannot reach S.  Then F(P) ∩ N ⊆ P, F(E) ∩ N ⊆ E, and
    P \ E = cells of P upstream of S, whose invariant part is S (other recurrence
    upstream of S inside the cover is absent along a one-to-one continuation).

    Used for middle spaces: a middle Morse set can be much thinner than the side
    Morse set containing it, and the plain pair (reach(S), reach(S) \ S) then sits
    inside the side Morse set, so its intersection with the side pair forgets the
    exit set and kills the class."""
    g = frame.grid
    A = frame.A()
    seeds = S_mask | cover
    idx = np.nonzero(seeds)[0]
    img = np.zeros(g.n_cells, bool)
    img[np.unique(A[idx].indices)] = True
    N = g.neighbours_mask(seeds | img, rings)
    P = reach(A, idx, allowed=N)
    up = reach(frame.AT(), np.nonzero(S_mask)[0], allowed=P)
    E = P & ~up
    return P, E


# ---------------------------------------------------------------- coarsening
def node_order_graph(frame):
    G = nx.DiGraph()
    G.add_nodes_from(range(frame.n_nodes))
    G.add_edges_from(frame.order)
    return nx.transitive_closure_dag(G) if nx.is_directed_acyclic_graph(G) else G


def coarsen(frame, touch_rings=1, trivial=()):
    """Absorb spurious Morse sets: a node with zero Conley index (`trivial`) that
    touches a comparable Morse set (within `touch_rings` cells) is merged into it
    (e.g. thin rings hugging an equilibrium, single self-loop cells).  Each
    trivial node joins exactly one neighbour (largest non-trivial first), so two
    genuine Morse sets are never fused.  Blocks are closed under convexity so
    each block is an interval of the order and the quotient is again a Morse
    decomposition.

    Returns (blocks, block_order) with blocks a list of sorted node lists."""
    G = node_order_graph(frame)
    g = frame.grid
    n = frame.n_nodes
    trivial = set(trivial)
    U = nx.Graph()
    U.add_nodes_from(range(n))
    for i in sorted(trivial):
        m = np.zeros(g.n_cells, bool)
        m[frame.morse_cells[i]] = True
        grown = g.neighbours_mask(m, touch_rings)
        cands = [j for j in range(n) if j != i and (G.has_edge(i, j) or G.has_edge(j, i))
                 and grown[frame.morse_cells[j]].any()]
        if cands:
            j = max(cands, key=lambda j: (j not in trivial, len(frame.morse_cells[j])))
            U.add_edge(i, j)
    blocks = [set(c) for c in nx.connected_components(U)]
    # convex closure + acyclicity, iterated to a fixed point
    changed = True
    while changed:
        changed = False
        for b in blocks:
            between = {m for m in range(n) if m not in b and
                       any(G.has_edge(i, m) for i in b) and any(G.has_edge(m, j) for j in b)}
            if between:
                b |= between
                changed = True
        # absorb blocks merged into others
        merged = []
        for b in blocks:
            for c in merged:
                if b & c:
                    c |= b
                    break
            else:
                merged.append(set(b))
        if len(merged) != len(blocks):
            changed = True
        blocks = merged
        Q = nx.DiGraph()
        Q.add_nodes_from(range(len(blocks)))
        where = {i: bi for bi, b in enumerate(blocks) for i in b}
        for (i, j) in frame.order:
            if where[i] != where[j]:
                Q.add_edge(where[i], where[j])
        cyc = list(nx.simple_cycles(Q))
        if cyc:
            new = set().union(*[blocks[c] for c in cyc[0]])
            blocks = [b for bi, b in enumerate(blocks) if bi not in cyc[0]] + [new]
            changed = True
    blocks = sorted((sorted(b) for b in blocks), key=lambda b: b[0])
    where = {i: bi for bi, b in enumerate(blocks) for i in b}
    order = sorted({(where[i], where[j]) for (i, j) in frame.order if where[i] != where[j]})
    return blocks, order
