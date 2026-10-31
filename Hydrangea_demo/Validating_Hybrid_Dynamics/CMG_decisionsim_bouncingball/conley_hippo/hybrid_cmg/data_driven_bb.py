"""
Data-driven Conley-Morse graph of the bouncing ball from SIMULATED RUNS.

Instead of evaluating the hybrid suspension tau-map on every grid cell
(bouncing_ball_cmg.py), we:
  1. simulate N trajectories from random initial conditions in R,
     recording the state every tau = 0.5 units of SUSPENSION time
     (so reset traversals are sampled on the guard cylinder),
  2. reconstruct a multivalued map F_data from the observed pairs
     (cell(x_t) -> padded cells of x_{t+tau}),  Ulam-style,
  3. compute the Morse graph (Tarjan SCC) and Conley indices (Z/2
     homology of attracting blocks) of the reconstructed map,
  4. draw the resulting Conley-Morse graph next to the data and the
     Morse sets, and compare against the analytic-map computation.

Usage: python3 data_driven_bb.py [NX NY NP N_RUNS N_STEPS]
"""

import sys, random, importlib.util
from collections import defaultdict

NXa = sys.argv[1] if len(sys.argv) > 1 else "128"
NYa = sys.argv[2] if len(sys.argv) > 2 else "128"
NPa = sys.argv[3] if len(sys.argv) > 3 else "32"
N_RUNS  = int(sys.argv[4]) if len(sys.argv) > 4 else 200
N_STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 60

sys.argv = ["bb", NXa, NYa, NPa]
spec = importlib.util.spec_from_file_location(
    "bb", "/home/claude/hybrid_cmg/bouncing_ball_cmg.py")
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)

random.seed(7)

# ------------------------- 1. simulate runs ---------------------------------
runs = []                       # list of lists of suspension states
for _ in range(N_RUNS):
    st = ('B', random.uniform(bb.H_MIN, bb.H_MAX),
               random.uniform(bb.V_MIN, bb.V_MAX))
    traj = [st]
    for _ in range(N_STEPS):
        st = bb.advance(st[0], st[1], st[2], bb.TAU)
        traj.append(st)
        if st[0] == 'OUT':
            break
    runs.append(traj)

n_pairs = sum(len(t) - 1 for t in runs)
base_pts = [(a, b) for t in runs for (k, a, b) in t if k == 'B']
cyl_pts  = [(a, b) for t in runs for (k, a, b) in t if k == 'C']
print(f"simulated {N_RUNS} runs x <= {N_STEPS} steps: {n_pairs} pairs "
      f"({len(base_pts)} base states, {len(cyl_pts)} cylinder states)")

# ------------------- 2. reconstruct multivalued map -------------------------
# Smoothed Ulam / nearest-neighbor reconstruction: every cell inherits the
# transitions observed anywhere in its 3x3 in-chart neighborhood (source
# dilation = 1 cell), and each observed image is padded by 1 cell (target
# padding).  SRC_DILATE = 0 gives the raw Ulam map for comparison.
def reconstruct(src_dilate):
    F = [()] * bb.NCELLS
    tmp = defaultdict(set)
    covered = set()
    for t in runs:
        for a, b in zip(t, t[1:]):
            ca = bb.point_to_cell(*a)
            if ca == bb.OUT:
                continue
            cbs = bb.padded(bb.point_to_cell(*b))
            srcs = bb.padded(ca) if src_dilate else (ca,)
            for c in srcs:
                tmp[c].update(cbs)
            covered.add(ca)
    for c, s in tmp.items():
        F[c] = tuple(s)
    return F, covered, set(tmp)

F_raw, covered, _ = reconstruct(0)
comp0, nc0 = bb.tarjan_scc(F_raw)
mem0, rec0 = bb.recurrent_components(F_raw, comp0, nc0)
print(f"raw Ulam map:      {len(covered)} data cells, "
      f"{len(rec0)} Morse set(s)")

F, _, support = reconstruct(1)
print(f"smoothed (NN) map: {len(support)} cells defined "
      f"({100 * len(support) / (bb.NB + bb.NC):.1f}% of grid)")

# ------------------- 3. Morse graph + Conley indices ------------------------
comp, ncomp = bb.tarjan_scc(F)
members, rec = bb.recurrent_components(F, comp, ncomp)
print(f"SCCs: {ncomp}   recurrent (Morse sets): {len(rec)}")

cond = defaultdict(set)
for c in range(bb.NCELLS):
    for w in F[c]:
        if comp[c] != comp[w]:
            cond[comp[c]].add(comp[w])

def reaches(a, b):
    seen, st = {a}, [a]
    while st:
        x = st.pop()
        if x == b:
            return True
        for y in cond[x]:
            if y not in seen:
                seen.add(y); st.append(y)
    return False

reach_edges = [(a, b) for a in rec for b in rec if a != b and reaches(a, b)]
# Hasse diagram = transitive reduction
hasse = [(a, b) for (a, b) in reach_edges
         if not any((a, c) in reach_edges and (c, b) in reach_edges
                    for c in rec if c not in (a, b))]

info = {}
for cc in rec:
    M = set(members[cc])
    U, st = set(M), list(M)
    while st:
        x = st.pop()
        for y in F[x]:
            if y not in U:
                U.add(y); st.append(y)
    U.discard(bb.OUT)
    Vx = U - M
    (b0, b1, b2), nf, ne, nv = bb.betti_pair(U, Vx)
    nb  = sum(1 for c in M if c < bb.NB)
    ncy = sum(1 for c in M if bb.NB <= c < bb.NB + bb.NC)
    info[cc] = dict(index=(b0, b1, b2), size=len(M), nb=nb, ncy=ncy)
    bs = [c for c in M if c < bb.NB]
    loc = ""
    if bs:
        hs = [bb.H_MIN + (c // bb.NY) * bb.DH for c in bs]
        vs = [bb.V_MIN + (c % bb.NY) * bb.DV for c in bs]
        loc = (f"; base h in [{min(hs):.3f},{max(hs)+bb.DH:.3f}], "
               f"v in [{min(vs):.3f},{max(vs)+bb.DV:.3f}]")
    print(f"  Morse set [{cc}]: {len(M)} cells ({nb} base, {ncy} cyl), "
          f"Conley index {(b0, b1, b2)}{loc}")
print(f"Hasse edges: {hasse}")

# ------------------------------ 4. figure -----------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
(axA, axB), (axC, axD) = axes
colors = ["#d81b60", "#1e88e5", "#ffc107", "#004d40", "#8e24aa",
          "#e65100", "#00838f"]

# (A) simulated runs, base chart
for t in runs[:60]:
    xs = [a for (k, a, b) in t if k == 'B']
    ys = [b for (k, a, b) in t if k == 'B']
    axA.plot(xs, ys, '.', ms=1.2, alpha=0.35, color="#37474f")
axA.set(xlim=(bb.H_MIN, bb.H_MAX), ylim=(bb.V_MIN, bb.V_MAX),
        xlabel="h", ylabel="v",
        title=f"(a) simulated runs (base-chart samples, {N_RUNS} runs)")
axA.axhline(0, color="k", lw=0.4)

# (B) Morse sets, base chart, over data support
for c in support:
    if c < bb.NB:
        i, j = divmod(c, bb.NY)
        axB.add_patch(plt.Rectangle((bb.H_MIN + i * bb.DH, bb.V_MIN + j * bb.DV),
                                    bb.DH, bb.DV, color="#eceff1", lw=0))
for t_i, cc in enumerate(rec):
    col = colors[t_i % len(colors)]
    for c in members[cc]:
        if c < bb.NB:
            i, j = divmod(c, bb.NY)
            axB.add_patch(plt.Rectangle(
                (bb.H_MIN + i * bb.DH, bb.V_MIN + j * bb.DV),
                bb.DH, bb.DV, color=col, lw=0))
axB.set(xlim=(bb.H_MIN, 0.5), ylim=(-2, 2), xlabel="h", ylabel="v",
        title="(b) Morse sets, base chart (grey = data support), zoomed")
axB.axhline(0, color="k", lw=0.4)

# (C) Morse sets, cylinder chart
for c in support:
    if c >= bb.NB:
        m, k = divmod(c - bb.NB, bb.NP)
        axC.add_patch(plt.Rectangle((bb.V_MIN + m * bb.DV, k * bb.DS),
                                    bb.DV, bb.DS, color="#eceff1", lw=0))
for t_i, cc in enumerate(rec):
    col = colors[t_i % len(colors)]
    for c in members[cc]:
        if c >= bb.NB:
            m, k = divmod(c - bb.NB, bb.NP)
            axC.add_patch(plt.Rectangle((bb.V_MIN + m * bb.DV, k * bb.DS),
                                        bb.DV, bb.DS, color=col, lw=0))
axC.set(xlim=(bb.V_MIN, 0), ylim=(0, 1), xlabel="v_g (guard)",
        ylabel="s (phase)",
        title="(c) Morse sets, cylinder chart (grey = data support)")

# (D) Conley-Morse graph (Hasse diagram, nodes labeled by index)
#     depth = longest path from maxima
succ = {cc: [b for (a, b) in hasse if a == cc] for cc in rec}
pred = {cc: [a for (a, b) in hasse if b == cc] for cc in rec}
depth = {}
def get_depth(cc):
    if cc in depth:
        return depth[cc]
    depth[cc] = 0 if not pred[cc] else 1 + max(get_depth(p) for p in pred[cc])
    return depth[cc]
for cc in rec:
    get_depth(cc)
levels = defaultdict(list)
for cc in rec:
    levels[depth[cc]].append(cc)
pos = {}
maxd = max(levels) if levels else 0
for d, nodes in levels.items():
    for t_i, cc in enumerate(sorted(nodes)):
        pos[cc] = ((t_i + 1) / (len(nodes) + 1), maxd - d)
for (a, b) in hasse:
    axD.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle='-|>',
                                  mutation_scale=14, color="#546e7a",
                                  shrinkA=26, shrinkB=26, lw=1.1))
for t_i, cc in enumerate(rec):
    col = colors[t_i % len(colors)]
    x, y = pos[cc]
    trivial = info[cc]['index'] == (0, 0, 0)
    axD.scatter([x], [y], s=1500, color="#cfd8dc" if trivial else col,
                zorder=3, edgecolors="k", linewidths=0.8)
    axD.annotate(f"{info[cc]['index']}", (x, y), ha="center", va="center",
                 fontsize=9, zorder=4)
    axD.annotate(f"{info[cc]['size']} cells", (x, y - 0.33), ha="center",
                 va="top", fontsize=7, color="#37474f", zorder=4)
axD.set(xlim=(0, 1), ylim=(-0.6, maxd + 0.6),
        title="(d) Conley-Morse graph (grey = trivial index)")
axD.axis("off")

fig.suptitle(f"Bouncing ball: data-driven reconstruction from simulated runs "
             f"({bb.NX}x{bb.NY} base, {bb.NP} phases, tau={bb.TAU})",
             fontsize=13)
fig.tight_layout()
fig.savefig("/home/claude/hybrid_cmg/bouncing_ball_data_driven.png", dpi=140)
print("figure saved: bouncing_ball_data_driven.png")
