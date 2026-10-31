"""Fast checks (pytest -q).  Small grids so the whole suite runs in ~1 minute."""
import numpy as np
import pytest

from cmzz.systems import Hopf2D
from cmzz.grid import CellGrid, TimePartition, step_enclosure, intersect_index_boxes, cell_map_csr
from cmzz.frame import CMGFrame
from cmzz.pairs import invariant_set, index_pair
from cmzz.zigzag import Triangulator, pair_homology, zigzag_barcode
import cmzz.zigzag as zz


@pytest.fixture(scope="module")
def hopf():
    s = Hopf2D()
    g = CellGrid(s.bounds, 160)
    part = TimePartition(s, 4)
    encs = [step_enclosure(s, g, (st.lam0, st.lam1), st.tau)[:2] for st in part]
    return s, g, part, encs


def test_enclosure_contains_true_images(hopf):
    s, g, part, encs = hopf
    rng = np.random.default_rng(0)
    for st, (ilo, ihi) in zip(part, encs):
        X = g.bounds[:, 0] + rng.random((4000, 2)) * (g.bounds[:, 1] - g.bounds[:, 0])
        lam = rng.uniform(st.lam0, st.lam1, 4000)
        Y = s.flow(X, lam, st.tau)
        c = g.point_to_flat(X)
        j = g.multi(g.point_to_flat(Y))
        inside = ((j >= ilo[c]) & (j <= ihi[c])).all(1)
        assert inside.mean() == 1.0


def test_csr_matches_cmgraph_scc(hopf):
    s, g, part, encs = hopf
    fr = CMGFrame(g, *encs[3], conley=False)
    same, ours, theirs = fr.check_against_scc()
    assert same, (ours, theirs)


def test_middle_is_contained_in_both_sides(hopf):
    s, g, part, encs = hopf
    ilo, ihi, empty = intersect_index_boxes(encs[2], encs[3])
    M = cell_map_csr(g, ilo, ihi, empty)
    for side in (encs[2], encs[3]):
        A = cell_map_csr(g, *side)
        assert (M.multiply(A) != M).nnz == 0       # every middle edge is a side edge


def test_index_pair_homology_matches_cmgraph(hopf):
    s, g, part, encs = hopf
    tri = Triangulator(g)
    fr = CMGFrame(g, *encs[3])        # λ ∈ [0.5, 1]: repeller Σ² and periodic attractor
    found = set()
    for i in range(fr.n_nodes):
        P, E = index_pair(fr, invariant_set(fr, [i]))
        h = pair_homology(tri, P, E)
        ci = fr.conley[i]
        if ci and ci["valid"]:
            assert h == ci["reduced_ranks"], (i, h, ci)
        found.add(tuple(h))
    assert (0, 0, 1) in found and (1, 1, 0) in found


def test_fzz_equals_dionysus():
    g = CellGrid([[0, 1], [0, 1]], 12)
    tri = Triangulator(g)
    m = lambda f: f(*np.meshgrid(np.arange(12), np.arange(12), indexing="ij")).ravel()
    disk = m(lambda i, j: (i - 6) ** 2 + (j - 6) ** 2 <= 20)
    ring = m(lambda i, j: ((i - 6) ** 2 + (j - 6) ** 2 <= 20) & ((i - 6) ** 2 + (j - 6) ** 2 >= 5))
    hole = disk & ~ring
    spaces = [(disk, None), (ring, None), (disk, ring), (disk, None)]
    res = {}
    for be in ("fzz", "dionysus"):
        zz.set_backend(be)
        res[be] = zigzag_barcode(tri, spaces)[0]
    zz.set_backend("fzz")
    assert res["fzz"] == res["dionysus"]
    assert pair_homology(tri, disk, ring) == [0, 0, 1]   # repeller: (disk, annulus)
    assert pair_homology(tri, disk, hole) == [0, 0, 0]   # (disk, sub-disk) is trivial
    assert pair_homology(tri, ring, None) == [1, 1, 0]
