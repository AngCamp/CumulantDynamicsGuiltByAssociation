# cmzz — zigzag persistence of Conley–Morse graphs on top of cmgraph

Implements, for a discrete-time model of a parameter-drifting flow, the two Dey papers:

* **CMG persistence** — Dey, Mrozek, Slechta, *Persistence of Conley-Morse Graphs in
  Combinatorial Dynamical Systems* (arXiv:2107.02115): a zigzag `G_k ← G_{k,k+1} → G_{k+1}`
  of Conley–Morse graphs, the middle being the CMG of the intersected dynamics, and the
  relative zigzag `(P_k,E_k) ⊇ (P_k,E_k)∩(P',E') ⊆ (P',E')` of index pairs along each
  continued Morse set.
* **Conley–Morse persistence barcode** — Dey, Lipiński, Soriano Trigueros (FoCM 2026,
  doi:10.1007/s10208-026-09766-6): bars that record when Conley-index generators are
  born, die or are conserved through a bifurcation, read off the zigzag module.

`cmgraph` does the per-step work (cell map → Morse graph → Conley index). This package
links the steps.

## Model

`[0, T]` is cut into `n` disjoint steps. **Step k is one discrete map**
`F_k ≈ φ_τ(·; λ)` with `τ = T/n`, enclosed over every parameter value
`λ ∈ [λ(t_k), λ(t_{k+1})]`. Consecutive steps are linked by the middle map
`F_k ∩ F_{k+1}` (cellwise intersection of images, the analogue of `V₁ ∩ V₂`). Since the
middle map's arrows are arrows of both sides, every middle Morse set sits inside a Morse
set on each side — the inclusions of the zigzag.

```
S0 ← M0|1 → S1 ← M1|2 → S2 ← ...      (S = step CMG, M = middle CMG)
```

## Classes

| class | role |
|---|---|
| `systems.DynamicalSystem` / `Hopf2D` / `Lorenz3D` / `ODESystem` | vector field `f(X, λ)`, schedule `λ(t)`, RK4 time-τ map |
| `grid.CellGrid`, `grid.TimePartition` | uniform cubical grid shared by all steps; the n steps |
| `grid.step_enclosure` | sampled enclosure of `φ_τ` over cell × parameter interval (non-rigorous, padded by a curvature estimate) |
| `frame.CMGFrame` | one cmgraph `Model` + `MorseGraph` per step or middle; Conley index per node |
| `pairs` | invariant sets of order intervals, combinatorial index pairs, absorption of zero-index (spurious) Morse sets |
| `zigzag` | `T(P) ∪ cone(T(E))` (Freudenthal triangulation) so `H(P,E)` = reduced homology; relative zigzag via **fzz** (Dey–Hou fast zigzag), Dionysus fallback |
| `pipeline.CMGZigzag` | builds frames → blocks → continuation graph → events (split/merge/birth/death/middle-only split, Morse-equation check) → tracks & clusters → barcodes |
| `viz` | portraits, CMG zigzag strip, barcodes, 4-vs-10 comparison, scaling plots |

## Run

```bash
pip install numpy scipy networkx matplotlib "cmgraph[viz]" dionysus
# optional, ~5x faster zigzag: build fzz (github.com/taohou01/fzz)
#   it needs boost/functional/hash.hpp; a 10-line shim is enough (see shim/ notes below)
python run_hopf.py 256 out 512      # 4 vs 10 steps at 256², plus 10 steps at 512²
python run_lorenz.py 32 out         # 3D smoke test
python run_scaling.py out           # timings vs grid / steps / dimension
pytest -q tests
```

fzz build without Boost: create `shim/boost/functional/hash.hpp` defining
`boost::hash_combine` with the usual `seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed<<6) + (seed>>2)`,
add `shim` and `pybind11.get_include()` to `include_dirs` in fzz's `setup.py`, then
`python setup.py build_ext --inplace` and put that folder on `PYTHONPATH`.

## Caveats

* Enclosures are **non-rigorous** (sampled hull + padding). Swap `step_enclosure` for an
  interval-arithmetic ODE enclosure to make results rigorous.
* Intersections of index pairs are not index pairs in general; as in the SoCG 2020 paper
  they are only used as the zigzag's middle terms.
* Zero-index Morse sets are treated as resolution artefacts: absorbed into a touching
  comparable Morse set, otherwise drawn grey and left out of linking.
