"""Link per-step CMGs into a zigzag and compute their persistence.

    S_0  ←  M_{0|1}  →  S_1  ←  M_{1|2}  →  S_2 ...

S_k : CMG of step k (one cmgraph Morse graph)
M_k|k+1 : CMG of the cellwise intersection map F_k ∩ F_{k+1}  (analogue of V1 ∩ V2).
Because M ⊆ F_k and M ⊆ F_{k+1}, every middle Morse set lies inside a Morse set
on each side, so the inclusions run middle → sides exactly as in Dey-Mrozek-Slechta.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import networkx as nx

from .grid import CellGrid, TimePartition, step_enclosure, intersect_index_boxes
from .frame import CMGFrame, poincare_str
from .pairs import invariant_set, index_pair, coarsen
from .zigzag import Triangulator, zigzag_barcode


@dataclass
class Block:
    """A node of the (resolution-coarsened) CMG of one space."""
    space: int                 # position in S0, M01, S1, M12, ... sequence
    bid: int
    nodes: list                # cmgraph node ids merged in this block
    cells: np.ndarray          # Morse cells (union over nodes)
    S: np.ndarray = None       # invariant set mask (with internal connections)
    ranks: list = None         # our index-pair homology ranks
    cm_poincare: str = None    # cmgraph's Conley index (single-node blocks)
    cm_status: str = None
    track: int = -1
    cluster: int = -1
    level: int = 0             # height in the partial order (for drawing)

    @property
    def poincare(self):
        return poincare_str(self.ranks) if self.ranks is not None else "?"

    @property
    def trivial(self):
        return self.ranks is not None and sum(self.ranks) == 0


@dataclass
class Event:
    link: int                  # between step link and link+1
    t: float                   # boundary time
    kind: str                  # continuation / split / merge / birth / death
    before: list               # blocks in S_link
    after: list                # blocks in S_link+1
    morse_eq: str = ""
    mids: list = field(default_factory=list)


class CMGZigzag:
    def __init__(self, system, n_steps, grid_size, T=1.0, rings=2, touch_rings=1,
                 enclosure_kw=None, verbose=True):
        self.system, self.T = system, T
        self.part = TimePartition(system, n_steps, T)
        self.grid = CellGrid(system.bounds, grid_size)
        self.tri = Triangulator(self.grid)
        self.rings, self.touch_rings = rings, touch_rings
        self.enc_kw = enclosure_kw or {}
        self.verbose = verbose
        self.timings = {}
        self.stats = {}

    # ---------------------------------------------------------------- utils
    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _tick(self, key, t0):
        self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - t0

    @property
    def n_spaces(self):
        return 2 * self.part.n - 1

    def space_label(self, s):
        return f"S{s // 2}" if s % 2 == 0 else f"M{s // 2}|{s // 2 + 1}"

    def space_time(self, p, end=False):
        """Time coordinate of a space position p (half-integers allowed)."""
        ts = [st.t0 for st in self.part] + [self.part.steps[-1].t1]
        if float(p).is_integer() and int(p) % 2 == 0:
            k = int(p) // 2
            return ts[k + 1] if end else ts[k]
        return ts[int(np.floor(p / 2)) + 1]

    # ---------------------------------------------------------------- build
    def run(self):
        self.build_frames()
        self.build_blocks()
        self.link()
        self.persistence()
        return self

    def build_frames(self):
        g = self.grid
        self.encs, self.frames = [], []
        for st in self.part:
            t0 = time.perf_counter()
            ilo, ihi, est = step_enclosure(self.system, g, (st.lam0, st.lam1), st.tau, **self.enc_kw)
            self._tick("enclosure", t0)
            self.encs.append((ilo, ihi))
            self.stats.setdefault("escaped_cells", []).append(est["escaped_cells"])
        self.stats["mean_image_span"] = float(np.mean([(e[1] - e[0] + 1).prod(1).mean()
                                                       for e in self.encs]))
        for s in range(self.n_spaces):
            k = s // 2
            t0 = time.perf_counter()
            if s % 2 == 0:
                st = self.part.steps[k]
                fr = CMGFrame(g, *self.encs[k], kind="step", k=k, t=(st.t0, st.t1),
                              lam=(st.lam0, st.lam1))
            else:
                ilo, ihi, empty = intersect_index_boxes(self.encs[k], self.encs[k + 1])
                self.stats.setdefault("empty_middle_cells", []).append(int(empty.sum()))
                fr = CMGFrame(g, ilo, ihi, kind="mid", k=k, empty=empty,
                              t=self.part.steps[k].t1, lam=self.part.steps[k].lam1)
            self._tick("cmgraph_morse_graph", t0)
            self.timings["cmgraph_conley_only"] = (self.timings.get("cmgraph_conley_only", 0)
                                                   + fr.timings.get("conley_index", 0))
            self.frames.append(fr)
            self.log(f"  {self.space_label(s):>8}: {fr.n_nodes} Morse sets "
                     f"{[len(c) for c in fr.morse_cells]}")

    def build_blocks(self):
        self.blocks = []        # per space: list[Block]
        self.block_order = []
        for s, fr in enumerate(self.frames):
            # index of every raw Morse set first: zero index marks spurious sets
            raw_ranks = []
            for i in range(fr.n_nodes):
                t0 = time.perf_counter()
                S = np.zeros(self.grid.n_cells, bool)
                S[fr.morse_cells[i]] = True
                P, E = index_pair(fr, S, self.rings)
                self._tick("index_pairs", t0)
                t0 = time.perf_counter()
                bars, _ = zigzag_barcode(self.tri, [(P, E)])
                self._tick("pair_homology", t0)
                raw_ranks.append([sum(1 for x in bars if x[0] == q) for q in range(self.grid.dim + 1)])
            fr.raw_ranks = raw_ranks
            t0 = time.perf_counter()
            groups, order = coarsen(fr, self.touch_rings,
                                    trivial=[i for i, r in enumerate(raw_ranks) if sum(r) == 0])
            self._tick("coarsen", t0)
            bl = []
            for bi, nodes in enumerate(groups):
                cells = np.sort(np.concatenate([fr.morse_cells[i] for i in nodes]))
                b = Block(s, bi, nodes, cells)
                if len(nodes) == 1:
                    b.ranks = raw_ranks[nodes[0]]
                else:
                    t0 = time.perf_counter()
                    b.S = invariant_set(fr, nodes)
                    P, E = index_pair(fr, b.S, self.rings)
                    self._tick("index_pairs", t0)
                    t0 = time.perf_counter()
                    bars, _ = zigzag_barcode(self.tri, [(P, E)])
                    self._tick("pair_homology", t0)
                    b.ranks = [sum(1 for x in bars if x[0] == q) for q in range(self.grid.dim + 1)]
                if len(nodes) == 1 and fr.conley[nodes[0]]:
                    ci = fr.conley[nodes[0]]
                    b.cm_poincare = ci.get("poincare")
                    b.cm_status = ci["status"]
                bl.append(b)
            G = nx.DiGraph()
            G.add_nodes_from(range(len(bl)))
            G.add_edges_from(order)
            for v in reversed(list(nx.topological_sort(G))):   # attractors at level 0
                bl[v].level = 1 + max((bl[w].level for w in G.successors(v)), default=-1)
            self.blocks.append(bl)
            self.block_order.append(order)

    # ---------------------------------------------------------------- linking
    def _container(self, b_mid, side_blocks):
        best, best_n = None, 0
        for b in side_blocks:
            n = np.intersect1d(b_mid.cells, b.cells, assume_unique=True).size
            if n > best_n:
                best, best_n = b, n
        frac = best_n / max(1, len(b_mid.cells))
        return best, frac

    def link(self):
        C = nx.Graph()      # continuation graph over (space, bid)
        for s, bl in enumerate(self.blocks):
            for b in bl:
                C.add_node((s, b.bid))
        self.inclusions = []   # (mid space, mid bid, side space, side bid, fraction)
        fracs = []
        for s in range(1, self.n_spaces, 2):
            for b in self.blocks[s]:
                if b.trivial:          # zero-index (spurious) sets are drawn, not linked
                    continue
                for side in (s - 1, s + 1):
                    c, frac = self._container(b, [x for x in self.blocks[side] if not x.trivial])
                    fracs.append(frac)
                    if c is not None:
                        C.add_edge((s, b.bid), (side, c.bid))
                        self.inclusions.append((s, b.bid, side, c.bid, frac))
        self.stats["min_inclusion_fraction"] = float(min(fracs)) if fracs else 1.0
        self.C = C
        # clusters = connected components
        for ci, comp in enumerate(nx.connected_components(C)):
            for (s, bid) in comp:
                self.blocks[s][bid].cluster = ci
        # tracks = chains where continuation is one-to-one at every link
        self.events = []
        tid = 0
        prev_of = {}
        for k in range(self.part.n - 1):
            sL, sM, sR = 2 * k, 2 * k + 1, 2 * k + 2
            # bipartite structure across this link
            succ = {b.bid: set() for b in self.blocks[sL]}
            pred = {b.bid: set() for b in self.blocks[sR]}
            for m in self.blocks[sM]:
                L = [y for (x, y) in [e for e in C.edges((sM, m.bid))] if y[0] == sL]
                R = [y for (x, y) in [e for e in C.edges((sM, m.bid))] if y[0] == sR]
                for l in L:
                    for r in R:
                        succ[l[1]].add((r[1], m.bid))
                        pred[r[1]].add((l[1], m.bid))
            seen = set()
            for comp in nx.connected_components(nx.Graph(
                    [((sL, l), (sR, r)) for l, rs in succ.items() for (r, _) in rs])):
                Ls = sorted(x[1] for x in comp if x[0] == sL)
                Rs = sorted(x[1] for x in comp if x[0] == sR)
                kind = ("continuation" if len(Ls) == 1 and len(Rs) == 1 else
                        "split" if len(Ls) == 1 else "merge" if len(Rs) == 1 else "reorganisation")
                mids = {m for l in Ls for (_, m) in succ[l]}
                nontriv = [m for m in mids if not self.blocks[sM][m].trivial]
                ev = Event(k, self.part.steps[k].t1, kind, Ls, Rs)
                if kind == "continuation" and len(nontriv) > 1:
                    # both steps see one blob, but F_k ∩ F_{k+1} already separates it:
                    # the relevant middle CMG resolves the bifurcation at this boundary
                    ev.kind = "middle-only split"
                    ev.mids = sorted(nontriv)
                self.events.append(ev)
                seen |= set(comp)
            for b in self.blocks[sL]:
                if (sL, b.bid) not in seen and not b.trivial:
                    self.events.append(Event(k, self.part.steps[k].t1, "death", [b.bid], []))
            for b in self.blocks[sR]:
                if (sR, b.bid) not in seen and not b.trivial:
                    self.events.append(Event(k, self.part.steps[k].t1, "birth", [], [b.bid]))
        # assign tracks: follow 1-1 continuations through the mid block
        for s in range(0, self.n_spaces, 2):
            for b in self.blocks[s]:
                if b.track >= 0:
                    continue
                b.track = tid
                cur = b
                while cur.space + 2 < self.n_spaces:
                    k = cur.space // 2
                    ev = [e for e in self.events if e.link == k and e.kind == "continuation"
                          and e.before == [cur.bid]]
                    if not ev:
                        break
                    mids = [m for m in self.blocks[cur.space + 1]
                            if C.has_edge((cur.space + 1, m.bid), (cur.space, cur.bid))]
                    nxt = self.blocks[cur.space + 2][ev[0].after[0]]
                    for m in mids:
                        m.track = tid
                    nxt.track = tid
                    cur = nxt
                tid += 1
        starts = {}
        ends = {}
        for s in range(0, self.n_spaces, 2):
            for b in self.blocks[s]:
                starts.setdefault(b.track, s)
                ends[b.track] = s
        for s in range(1, self.n_spaces, 2):       # middles not on a chain
            for m in self.blocks[s]:
                if m.track >= 0:
                    continue
                R = [y for (_, y) in C.edges((s, m.bid)) if y[0] == s + 1]
                L = [y for (_, y) in C.edges((s, m.bid)) if y[0] == s - 1]
                r = self.blocks[s + 1][R[0][1]] if R else None
                l = self.blocks[s - 1][L[0][1]] if L else None
                taken = {x.track for x in self.blocks[s] if x.track >= 0}
                if r is not None and starts.get(r.track) == s + 1 and r.track not in taken:
                    m.track = r.track          # born at this boundary: prepend
                elif l is not None and ends.get(l.track) == s - 1 and l.track not in taken:
                    m.track = l.track          # dies after this boundary: append
                else:
                    m.track = tid
                    tid += 1
        self.n_tracks = tid
        # Morse equation at non-trivial events (Conley index bookkeeping)
        for e in self.events:
            if e.kind in ("split", "merge", "reorganisation", "middle-only split"):
                e.morse_eq = self._morse_equation(e)

    def _morse_equation(self, e):
        """Sum of the children's Conley polynomials minus the parent's must be
        (1+t)Q(t) with Q >= 0 (the children plus connections continue the parent)."""
        L = [self.blocks[2 * e.link][i] for i in e.before]
        R = [self.blocks[2 * e.link + 2][i] for i in e.after]
        if e.kind == "middle-only split":
            R = [self.blocks[2 * e.link + 1][i] for i in e.mids]
        few, many = (L, R) if len(L) <= len(R) else (R, L)
        D = self.grid.dim + 1
        p_many = np.sum([b.ranks for b in many], axis=0)
        p_few = np.sum([b.ranks for b in few], axis=0)
        diff = list(p_many - p_few)
        Q, r = [], diff[:]
        for q in range(D - 1):            # divide by (1+t)
            Q.append(r[q])
            r[q + 1] -= r[q]
        ok = r[-1] == 0 and all(c >= 0 for c in Q)
        return (f"Σ P(children) − P(parent) = {poincare_str([max(c,0) for c in diff]) if min(diff)>=0 else diff}"
                f" = (1+t)·({poincare_str(Q) if min(Q, default=0) >= 0 else Q})  "
                + ("✓ consistent" if ok else "✗ not of the form (1+t)Q, Q≥0"))

    # ---------------------------------------------------------------- persistence
    def _pair_at(self, s, bids, by_space=None):
        fr = self.frames[s]
        nodes = sorted(n for bid in bids for n in self.blocks[s][bid].nodes)
        S = invariant_set(fr, nodes)
        if s % 2 == 1 and by_space is not None:
            cover = np.zeros(self.grid.n_cells, bool)
            for side in (s - 1, s + 1):
                for bid in by_space.get(side, []):
                    cover[self.blocks[side][bid].cells] = True
            return index_pair(fr, S, self.rings, cover=cover)
        return index_pair(fr, S, self.rings)

    def persistence(self):
        """Zigzag persistence of index pairs along every track (individual Morse
        sets: CMG persistence of paper 1) and every cluster (the isolated
        invariant set formed by everything that continues together, the
        Conley-Morse barcode viewpoint of paper 2)."""
        self.bars = []      # dicts
        t0 = time.perf_counter()
        n_simp = 0
        groups = {}
        for s, bl in enumerate(self.blocks):
            for b in bl:
                if b.trivial:
                    continue
                groups.setdefault(("track", b.track), {}).setdefault(s, []).append(b.bid)
                groups.setdefault(("cluster", b.cluster), {}).setdefault(s, []).append(b.bid)
        done_sigs = set()
        for (kind, gid), by_space in sorted(groups.items()):
            span = sorted(by_space)
            sig = tuple((s, tuple(v)) for s, v in sorted(by_space.items()))
            if kind == "cluster" and sig in done_sigs:
                continue        # identical to a track: skip duplicate
            done_sigs.add(sig)
            s0, s1 = span[0], span[-1]
            spaces = []
            for s in range(s0, s1 + 1):
                spaces.append(self._pair_at(s, by_space[s], by_space) if s in by_space else None)
            bars, ns = zigzag_barcode(self.tri, spaces)
            n_simp = max(n_simp, ns)
            for (dim, a, b) in bars:
                if np.floor(b) < a:
                    # lives only on an auxiliary intersection term, never on a CMG space
                    self.stats["aux_only_bars"] = self.stats.get("aux_only_bars", 0) + 1
                    continue
                self.bars.append({"kind": kind, "id": gid, "dim": dim,
                                  "start": s0 + a, "end": s0 + b,
                                  "t_start": self.space_time(s0 + a),
                                  "t_end": self.space_time(s0 + b, end=True),
                                  "span": (s0, s1)})
        self._tick("zigzag_persistence", t0)
        self.stats["max_zigzag_simplices"] = n_simp
        self.label_bars()

    def label_bars(self):
        """Paper-2 style reading of each bar."""
        last = self.n_spaces - 1
        for bar in self.bars:
            a, b = bar["start"], bar["end"]
            s0, s1 = bar["span"]
            if a == 0 and b == last:
                lab = "persists over the whole second"
            elif bar["kind"] == "cluster" and (a, b) == (s0, s1):
                lab = "conserved through the bifurcation (continuation)"
            elif a > s0 or b < s1:
                lab = "index changes inside a continuation (bifurcation)"
            elif a > 0 and b < last:
                lab = "born and dies at bifurcations"
            elif a > 0:
                lab = "born at a bifurcation"
            else:
                lab = "dies at a bifurcation"
            bar["label"] = lab

    # ---------------------------------------------------------------- summaries
    def detection(self):
        """First link where the CMG changes in a way the Conley index sees."""
        for e in sorted(self.events, key=lambda e: e.link):
            if e.kind == "continuation":
                continue
            bl = ([self.blocks[2 * e.link][i] for i in e.before] +
                  [self.blocks[2 * e.link + 2][i] for i in e.after])
            if any(not b.trivial for b in bl):
                st = self.part.steps[e.link]
                return {"link": e.link, "t_detect": e.t, "kind": e.kind,
                        "bracket": (st.t0, st.t1), "lam_bracket": (st.lam0, st.lam1)}
        return None

    def to_dict(self):
        def f(x):
            return float(x)
        return {
            "system": self.system.name, "n_steps": self.part.n,
            "step_ms": 1000 * self.T / self.part.n,
            "grid": self.grid.size.tolist(),
            "steps": [{"k": st.k, "t": [f(st.t0), f(st.t1)], "lam": [f(st.lam0), f(st.lam1)]}
                      for st in self.part],
            "spaces": [{"label": self.space_label(s),
                        "raw_morse_sets": [int(len(c)) for c in self.frames[s].morse_cells],
                        "blocks": [{"bid": b.bid, "cells": int(len(b.cells)), "nodes": b.nodes,
                                    "conley": b.poincare, "ranks": b.ranks,
                                    "cmgraph_conley": b.cm_poincare, "cmgraph_status": b.cm_status,
                                    "track": b.track, "cluster": b.cluster, "level": b.level,
                                    "trivial": b.trivial} for b in self.blocks[s]],
                        "order": self.block_order[s]} for s in range(self.n_spaces)],
            "events": [{"link": e.link, "t": f(e.t), "kind": e.kind, "before": e.before,
                        "after": e.after, "mids": e.mids, "morse_equation": e.morse_eq} for e in self.events],
            "bars": [{k: (f(v) if isinstance(v, (float, np.floating)) else v)
                      for k, v in b.items()} for b in self.bars],
            "detection": ({k: (f(v) if np.isscalar(v) and not isinstance(v, str) else
                               ([f(u) for u in v] if isinstance(v, tuple) else v))
                           for k, v in self.detection().items()} if self.detection() else None),
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "stats": self.stats,
        }

    def report(self):
        lines = []
        for s, bl in enumerate(self.blocks):
            desc = ", ".join(f"{b.poincare}[{len(b.cells)}]" + ("·" if b.trivial else "")
                             for b in bl)
            lines.append(f"{self.space_label(s):>8}: {desc}")
        for e in self.events:
            if e.kind != "continuation":
                lines.append(f"  link {e.link}|{e.link + 1} t={e.t:.3f}: {e.kind} "
                             f"{e.before}->{e.after} {e.morse_eq}")
        return "\n".join(lines)
