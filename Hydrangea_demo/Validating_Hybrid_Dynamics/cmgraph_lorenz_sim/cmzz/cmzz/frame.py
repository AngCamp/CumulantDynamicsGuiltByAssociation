"""One discrete step -> one cmgraph Morse graph with Conley indices (a CMG)."""
from __future__ import annotations

import time

import numpy as np
import cmgraph as cm
from scipy.sparse.csgraph import connected_components

from .grid import CellGrid, cell_map_csr


def poincare_str(ranks):
    terms = []
    for k, r in enumerate(ranks):
        if r == 0:
            continue
        c = "" if (r == 1 and k > 0) else str(r)
        sup = str(k).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
        terms.append(c + ("" if k == 0 else ("t" if k == 1 else f"t{sup}")))
    return " + ".join(terms) if terms else "0"


class CMGFrame:
    """A Conley-Morse graph for one map on the shared grid.

    kind: 'step' (map of step k) or 'mid' (cellwise intersection of steps k, k+1).
    """

    def __init__(self, grid: CellGrid, ilo, ihi, kind="step", k=0, label="",
                 empty=None, t=None, lam=None, conley=True, threads=0,
                 conley_max_cells=1200):
        self.grid, self.ilo, self.ihi = grid, ilo, ihi
        self.kind, self.k, self.label = kind, k, label
        self.empty = empty if empty is not None else np.zeros(grid.n_cells, bool)
        self.t, self.lam = t, lam
        self.timings = {}
        self._A = None
        self.conley_max_cells = conley_max_cells
        self._compute(conley, threads)

    # ---------------------------------------------------------------- cmgraph
    def _compute(self, conley, threads):
        g = self.grid
        boxes_all = g.boxes_from_index(self.ilo, self.ihi)
        # empty image (middle only): send the image off-domain so the cell is transient
        if self.empty.any():
            off = g.bounds[:, 1] + 10 * g.width
            boxes_all[self.empty] = np.stack([off, off + g.width], -1)

        def boxmap(B):
            c = np.asarray(B).mean(axis=2)
            return boxes_all[g.point_to_flat(c)]

        t0 = time.perf_counter()
        self.model = cm.Model(bounds=g.bounds.tolist(), map=cm.BoxMap(boxmap, batched=True),
                              grid=cm.Grid(size=g.size.tolist()), threads=threads)
        self.mg = self.model.morse_graph()
        self.timings["morse_graph"] = time.perf_counter() - t0

        s = self.mg.summary()
        self.n_nodes = s["node_count"]
        self.order = [tuple(map(int, p)) for p in np.asarray(s["order"]).reshape(-1, 2)]
        self.morse_cells = []
        for i in range(self.n_nodes):
            nb = np.asarray(self.mg.node_boxes(i))
            self.morse_cells.append(np.sort(g.point_to_flat(nb.mean(axis=2))))
        self.label_array = np.full(g.n_cells, -1, np.int64)
        for i, c in enumerate(self.morse_cells):
            self.label_array[c] = i

        self.conley = [None] * self.n_nodes
        if conley:
            t0 = time.perf_counter()
            for i in range(self.n_nodes):
                self.conley[i] = self._conley(i)
            self.timings["conley_index"] = time.perf_counter() - t0

    def _conley(self, i):
        if len(self.morse_cells[i]) < 4:
            return {"valid": False, "status": "skipped (< 4 cells)"}
        if len(self.morse_cells[i]) > self.conley_max_cells:
            return {"valid": False, "status": f"skipped (> {self.conley_max_cells} cells)"}
        try:
            ci = self.mg.conley_index(node=i, want_homology=True)
        except cm.IndexPairInvalid:
            return {"valid": False, "status": "not isolated at this resolution"}
        except cm.CmgraphError as e:     # budget etc.
            return {"valid": False, "status": type(e).__name__}
        ranks = [h["rank"] for h in ci.get("homology", [])]
        reduced = [d["reduced_rcf"].shape[0] for d in ci["degrees"]]
        return {"valid": True, "status": "ok", "homology_ranks": ranks,
                "reduced_ranks": reduced,
                "f_star": [d["f_star"].tolist() for d in ci["degrees"]],
                "poincare": poincare_str(reduced)}

    # ---------------------------------------------------------------- helpers
    def A(self):
        if self._A is None:
            self._A = cell_map_csr(self.grid, self.ilo, self.ihi, self.empty)
        return self._A

    def AT(self):
        if getattr(self, "_AT", None) is None:
            self._AT = self.A().T.tocsr()
        return self._AT

    def release(self):
        """Drop the cmgraph engine and cached matrices (memory)."""
        self.model = self.mg = None
        self._A = self._AT = None

    def check_against_scc(self):
        """Recompute Morse sets from our own cell map and compare with cmgraph."""
        A = self.A()
        n_comp, lab = connected_components(A, directed=True, connection="strong")
        sizes = np.bincount(lab, minlength=n_comp)
        selfloop = np.asarray(A.diagonal()).astype(bool)
        rec = (sizes[lab] > 1) | selfloop
        ours = {frozenset(np.nonzero(lab == c)[0].tolist()) for c in np.unique(lab[rec])}
        theirs = {frozenset(c.tolist()) for c in self.morse_cells}
        return ours == theirs, len(ours), len(theirs)

    def node_name(self, i):
        tag = f"S{self.k}" if self.kind == "step" else f"M{self.k}|{self.k + 1}"
        return f"{tag}.{i}"

    def describe(self, i):
        ci = self.conley[i]
        p = ci["poincare"] if ci and ci["valid"] else "?"
        return {"node": i, "cells": int(len(self.morse_cells[i])), "conley": p,
                "status": ci["status"] if ci else "skipped"}
