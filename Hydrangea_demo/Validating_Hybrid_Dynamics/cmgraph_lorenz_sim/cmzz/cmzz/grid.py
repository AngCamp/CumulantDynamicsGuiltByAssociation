"""Uniform cubical grid, step partition, and box enclosures of one discrete step.

Every map in the pipeline (each step, and each zigzag middle) is stored as an
integer *index box* per cell: image(c) = all cells j with lo[c] <= j <= hi[c]
(per dimension).  The same integer boxes are handed to cmgraph, so cmgraph's
cell map and ours are identical by construction.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------- grid
class CellGrid:
    def __init__(self, bounds, size):
        self.bounds = np.asarray(bounds, float)
        self.dim = self.bounds.shape[0]
        self.size = np.array([size] * self.dim if np.isscalar(size) else size, dtype=np.int64)
        self.lo = self.bounds[:, 0]
        self.width = (self.bounds[:, 1] - self.bounds[:, 0]) / self.size
        self.n_cells = int(np.prod(self.size))
        # C-order strides for flat <-> multi index
        self.strides = np.array([int(np.prod(self.size[i + 1:])) for i in range(self.dim)], np.int64)

    def multi(self, flat):
        flat = np.asarray(flat, np.int64)
        return np.stack([(flat // s) % n for s, n in zip(self.strides, self.size)], axis=-1)

    def flat(self, multi):
        return (np.asarray(multi, np.int64) * self.strides).sum(-1)

    def cell_lo(self, flat=None):
        idx = self.multi(np.arange(self.n_cells) if flat is None else flat)
        return self.lo + idx * self.width

    def centers(self, flat=None):
        return self.cell_lo(flat) + 0.5 * self.width

    def point_to_flat(self, X):
        m = np.floor((np.asarray(X) - self.lo) / self.width).astype(np.int64)
        m = np.clip(m, 0, self.size - 1)
        return self.flat(m)

    def boxes_from_index(self, ilo, ihi):
        """Real boxes strictly inside the index range (0.25-cell inward margins) so
        that 'every cell the closed box meets' recovers exactly ilo..ihi."""
        lo = self.lo + (ilo + 0.25) * self.width
        hi = self.lo + (ihi + 0.75) * self.width
        return np.stack([lo, hi], axis=-1)

    def neighbours_mask(self, mask, rings=1):
        """Grow a boolean cell mask by `rings` cubical rings (Chebyshev adjacency)."""
        g = mask.reshape(tuple(self.size))
        for _ in range(rings):
            out = g.copy()
            for off in itertools.product((-1, 0, 1), repeat=self.dim):
                if not any(off):
                    continue
                src = [slice(max(0, -o), n - max(0, o)) for o, n in zip(off, self.size)]
                dst = [slice(max(0, o), n - max(0, -o)) for o, n in zip(off, self.size)]
                out[tuple(dst)] |= g[tuple(src)]
            g = out
        return g.ravel()


# ------------------------------------------------------------------ partition
@dataclass
class Step:
    k: int
    t0: float
    t1: float
    lam0: float
    lam1: float

    @property
    def tau(self):
        return self.t1 - self.t0


class TimePartition:
    """Cut [0, T] into n disjoint steps.  Step k is one discrete map
    phi_tau(.; lam) with tau = T/n, enclosed over every lam in [lam(t_k), lam(t_{k+1})]."""

    def __init__(self, system, n_steps: int, T: float = 1.0):
        self.system, self.n, self.T = system, n_steps, T
        ts = np.linspace(0.0, T, n_steps + 1)
        lams = system.param_at(ts, T)
        self.steps = [Step(k, ts[k], ts[k + 1], float(lams[k]), float(lams[k + 1]))
                      for k in range(n_steps)]
        self.boundaries = ts[1:-1]          # where consecutive steps are linked

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return self.n


# ------------------------------------------------------------------ enclosure
def step_enclosure(system, grid: CellGrid, lam_interval, tau, safety=2.0, floor=0.1,
                   n_x=3, n_lam=3, chunk=20000):
    """Sampled enclosure of phi_tau over each cell and over the parameter interval.

    Samples an n_x^d lattice in the cell and n_lam parameter values, takes the hull
    of the images, and pads it by `safety` x a second-difference (curvature) error
    estimate plus `floor` cells.  NON-RIGOROUS: it is an outer enclosure only when
    the padding dominates the sub-sample variation (not checked).

    Returns integer index boxes (ilo, ihi), each (n_cells, d), plus a dict of stats.
    """
    d = grid.dim
    t = np.linspace(0.0, 1.0, n_x)
    lattice = np.stack(np.meshgrid(*([t] * d), indexing="ij"), -1).reshape(-1, d)  # (S, d)
    lams = np.linspace(lam_interval[0], lam_interval[1], n_lam)
    ilo = np.empty((grid.n_cells, d), np.int64)
    ihi = np.empty((grid.n_cells, d), np.int64)
    escaped = 0
    max_pad = 0.0
    for a in range(0, grid.n_cells, chunk):
        idx = np.arange(a, min(a + chunk, grid.n_cells))
        lo = grid.cell_lo(idx)                                     # (m, d)
        P = lo[:, None, :] + lattice[None] * grid.width            # (m, S, d)
        X = np.broadcast_to(P[:, None], (len(idx), n_lam) + P.shape[1:])
        L = np.broadcast_to(lams[None, :, None], X.shape[:-1])
        Y = system.flow(np.array(X), L, tau)                       # (m, nl, S, d)
        ylo = Y.min(axis=(1, 2))
        yhi = Y.max(axis=(1, 2))
        # Padding.  The hull of the sampled lattice (which contains every corner and
        # both parameter endpoints) is exact for maps affine in (x, lam); the error
        # is governed by curvature, estimated by second differences along each
        # sampled axis, plus a floor of `floor` cells.
        Yg = Y.reshape((len(idx), n_lam) + (n_x,) * d + (d,))
        pad = np.zeros((len(idx), d))
        axes = [2 + j for j in range(d)] + ([1] if n_lam > 2 else [])
        for ax in axes:
            d2 = np.abs(np.diff(Yg, n=2, axis=ax))
            pad += d2.max(axis=tuple(range(1, 2 + d)))
        pad = safety * pad + floor * grid.width
        max_pad = max(max_pad, float((pad / grid.width).max()))
        ylo -= pad
        yhi += pad
        # every closed cell the closed box meets: ceil(u)-1 .. floor(v)
        u = (ylo - grid.lo) / grid.width
        v = (yhi - grid.lo) / grid.width
        jl = np.ceil(u).astype(np.int64) - 1
        jh = np.floor(v).astype(np.int64)
        escaped += int(((jl < 0) | (jh > grid.size - 1)).any(axis=1).sum())
        ilo[idx] = np.clip(jl, 0, grid.size - 1)
        ihi[idx] = np.clip(jh, 0, grid.size - 1)
    return ilo, ihi, {"escaped_cells": escaped, "max_pad_cells": max_pad}


def intersect_index_boxes(a, b):
    """Cellwise intersection of two maps (the zigzag middle, analogue of V1 ∩ V2)."""
    ilo = np.maximum(a[0], b[0])
    ihi = np.minimum(a[1], b[1])
    empty = (ilo > ihi).any(axis=1)
    return ilo, ihi, empty


def cell_map_csr(grid: CellGrid, ilo, ihi, empty=None):
    """Sparse adjacency A[c, j] = 1 iff j in image(c)."""
    n, d = ilo.shape
    span = np.maximum(ihi - ilo + 1, 0)
    if empty is not None:
        span[empty] = 0
    counts = span.prod(axis=1)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), np.int32)
    chunk = 1 << 16
    for a in range(0, n, chunk):          # chunked to bound memory
        b = min(n, a + chunk)
        cnt = counts[a:b]
        tot = int(cnt.sum())
        if tot == 0:
            continue
        rows = np.repeat(np.arange(a, b), cnt)
        pos = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        cols = np.zeros(tot, np.int64)
        for j in range(d - 1, -1, -1):
            sj = span[rows, j]
            cols += (ilo[rows, j] + pos % sj) * grid.strides[j]
            pos //= sj
        indices[indptr[a]:indptr[b]] = cols
    data = np.ones(len(indices), np.int8)
    return sp.csr_matrix((data, indices, indptr), shape=(n, n))
