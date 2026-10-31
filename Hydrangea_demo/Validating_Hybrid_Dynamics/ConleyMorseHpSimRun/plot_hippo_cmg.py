"""
plot_hippo_cmg.py
-----------------
Figures for hippo_cmg.py.  Run:  python plot_hippo_cmg.py output/hippo_cmg_K5.pkl
Writes, with the suffix _K<K>:
  cmg_chart.png    spectrum + elbow, delay-PCA variance, 5D chart projections
  cmg_morse.png    Morse sets in the chart and the Conley-Morse graphs (full, detected guards + resets)
  cmg_guards.png   label-free guard detection vs the true switches
  cmg_lattice.png  lattice of attractors (downsets of the restricted Conley-Morse graph)
  cmg_sweep.png    which Morse sets persist across grid size n and step tau
"""
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = "output"
R = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/hippo_cmg_K5.pkl", "rb"))
SUF = f"_K{R['K']}"
Z, code, pos = R["Z"], R["code"], R["pos"]
G = R["grid"]
PAL = ["#1e88e5", "#d81b60", "#43a047", "#fb8c00", "#8e24aa", "#00897b", "#6d4c41", "#3949ab",
       "#c0ca33", "#e53935", "#5e35b1", "#039be5"]
STC = np.array(["#1e88e5", "#9e9e9e", "#d81b60"])          # WALK, STILL, SWR
PAIRS = [(0, 1), (0, 2), (1, 2), (3, 4)]


def centers(boxes):
    I = np.stack(np.unravel_index(np.asarray(boxes), (G["n"],) * Z.shape[1]), 1)
    return G["lo"] + (I + 0.5) / G["n"] * (G["hi"] - G["lo"])


def layered(nodes, edges):
    """Hasse layout: sources on top."""
    pred = {v: [a for a, b in edges if b == v] for v in nodes}
    depth = {}

    def dep(v):
        if v not in depth:
            depth[v] = 0 if not pred[v] else 1 + max(dep(p) for p in pred[v])
        return depth[v]
    for v in nodes:
        dep(v)
    lv = {}
    for v in nodes:
        lv.setdefault(depth[v], []).append(v)
    md = max(lv) if lv else 0
    return {v: ((i + 1) / (len(vs) + 1), md - d) for d, vs in lv.items() for i, v in enumerate(vs)}, md


def draw_graph(ax, nodes, edges, labels, colors, title, extra=()):
    p, md = layered(nodes, edges)
    for a, b in edges:
        ax.add_patch(FancyArrowPatch(p[a], p[b], arrowstyle="-|>", mutation_scale=13, color="#546e7a",
                                     shrinkA=26, shrinkB=26, lw=1.2))
    for a, b, lab in extra:
        if a in p and b in p and a != b:
            ax.add_patch(FancyArrowPatch(p[a], p[b], arrowstyle="-|>", mutation_scale=13, color="#d81b60",
                                         shrinkA=26, shrinkB=26, lw=1.3, ls="--",
                                         connectionstyle="arc3,rad=0.3"))
            ax.text((p[a][0] + p[b][0]) / 2, (p[a][1] + p[b][1]) / 2 + 0.15, lab, color="#d81b60",
                    fontsize=7, ha="center")
    for v in nodes:
        ax.scatter(*p[v], s=2600, color=colors[v], edgecolors="k", zorder=3, alpha=0.9)
        ax.annotate(labels[v], p[v], ha="center", va="center", fontsize=6.5, zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(-0.8, md + 0.8); ax.axis("off"); ax.set_title(title, fontsize=10)


def idx_str(i):
    return "skipped" if i is None else "(" + ",".join(map(str, i)) + ")"


# ------------------------------------------------------------------ chart
fig, ax = plt.subplots(2, 4, figsize=(19, 9))
k = np.arange(1, 31)
mp = (1 + np.sqrt(175 / 36000)) ** 2
ax[0, 0].plot(k, R["evals"][:30], "o-", ms=3)
ax[0, 0].plot([1, 30], [R["evals"][0], R["evals"][29]], color="0.6", lw=0.8)
ax[0, 0].axvline(R["K"], color="r", ls="--")
ax[0, 0].axhline(mp, color="0.4", ls=":", lw=1)
ax[0, 0].text(30, mp, "noise edge (Marchenko-Pastur)", ha="right", va="bottom", fontsize=7)
ax[0, 0].set(xlabel="PC", ylabel="eigenvalue (z-scored counts)", title=f"25 ms counts: decay elbow {R['K_elbow']}, using K = {R['K']}")
ax[0, 1].bar(np.arange(1, 16), R["dvar"][:15], color=["#1e88e5"] * 5 + ["#bdbdbd"] * 10)
ax[0, 1].set(xlabel="delay-PCA component", ylabel="fraction of variance",
             title=f"{R['params']['lags']} lags x K = {R['K']} dims -> 5D chart ({R['dvar'][:5].sum():.0%})")
sub = slice(None, None, 3)
for j, (a, b) in enumerate(PAIRS[:2]):
    ax[0, 2 + j].scatter(Z[sub, a], Z[sub, b], s=1, c=STC[code[sub]], alpha=0.5)
    ax[0, 2 + j].set(xlabel=f"Z{a + 1}", ylabel=f"Z{b + 1}", title="colour = state (WALK blue, STILL grey, SWR pink)")
for j, (a, b) in enumerate(PAIRS):
    walk = code == 0
    sc = ax[1, j].scatter(Z[walk][sub, a], Z[walk][sub, b], s=1, c=pos[walk][sub], cmap="hsv", vmin=0, vmax=4)
    ax[1, j].scatter(Z[~walk][sub, a], Z[~walk][sub, b], s=1, c="0.85", zorder=0)
    ax[1, j].set(xlabel=f"Z{a + 1}", ylabel=f"Z{b + 1}", title="WALK samples, colour = track position")
plt.colorbar(sc, ax=ax[1, 3], label="arc length s (m)")
fig.suptitle(f"5D delay chart of the CA1 simulation, K = {R['K']} (projections of a 5D cloud)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/cmg_chart{SUF}.png", dpi=115); plt.close(fig)

# ------------------------------------------------------------------ detected guards
ev = R["g_eval"]
fig, ax = plt.subplots(2, 1, figsize=(16, 6.5))
tb = np.arange(len(R["gscore"])) * R["params"]["bin_ms"] / 1000
for a_, (t0_, t1_) in zip(ax, [(0, tb[-1]), (120, 240)]):
    a_.plot(tb, R["gscore"], color="#37474f", lw=0.5, label="change score (spikes only)")
    a_.axhline(R["gthr"], color="r", ls="--", lw=0.8, label="threshold (median + 6 MAD)")
    for s_ in R["true_sw"] * R["params"]["bin_ms"] / 1000:
        a_.axvline(s_, color="#43a047", lw=0.8, alpha=0.6)
    a_.plot(R["guards"] * R["params"]["bin_ms"] / 1000, R["gscore"][R["guards"]], "v", color="#d81b60",
            ms=5, label="detected guards")
    a_.set_xlim(t0_, t1_); a_.set(xlabel="time (s)", ylabel="score")
ax[0].plot([], [], color="#43a047", label="true WALK/STILL switch (scoring only)")
ax[0].legend(fontsize=7, loc="upper right", ncol=4)
ax[0].set_title(f"label-free guard detection: {ev['n_guards']} guards; recall {ev['recall']:.2f}, "
                f"precision {ev['precision']:.2f} against {ev['n_true']} true switches (within 500 ms); "
                f"{ev['at_swr']:.0%} near SWR onsets", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/cmg_guards{SUF}.png", dpi=115); plt.close(fig)

# ------------------------------------------------------------------ Morse sets + graphs
fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(2, 4)
for r_i, routing in enumerate(("full", "detected")):
    M = R["main"][routing]
    show = sorted(M["rec"], key=lambda c: -M["desc"][c]["n_box"])[:12]
    col = {c: PAL[i % len(PAL)] for i, c in enumerate(show)}
    for j, (a, b) in enumerate(PAIRS[:2]):
        ax = fig.add_subplot(gs[r_i, j])
        ax.scatter(Z[::4, a], Z[::4, b], s=0.5, c="0.88", zorder=0)
        for c in show[::-1]:
            P = centers(M["members"][c])
            ax.scatter(P[:, a], P[:, b], s=5, marker="s", color=col[c], alpha=0.6,
                       label=f"{M['desc'][c]['label']} {idx_str(M['idx'].get(c, None))}" if j == 0 else None)
        ax.set(xlabel=f"Z{a + 1}", ylabel=f"Z{b + 1}")
        ax.set_title(f"{routing} map: Morse sets (box centres, projected)", fontsize=10)
        if j == 0:
            ax.legend(fontsize=6.5, markerscale=3, loc="best")
    nodes = M["restricted"]
    labs = {c: f"{M['desc'][c]['label'].replace(' (', chr(10) + '(')}\n{idx_str(M['idx'][c])}" for c in nodes}
    extra = []
    if routing == "detected" and M.get("resets"):
        extra = [(a, b, f"reset x{v}") for (a, b), v in M["resets"].items() if v >= 2]
    draw_graph(fig.add_subplot(gs[r_i, 2:]), nodes, M["hasse_restricted"], labs,
               {c: col.get(c, "#bdbdbd") for c in nodes},
               f"{routing} map: restricted Conley-Morse graph (nontrivial index, 6 slots = degrees 0..5)"
               + ("\npink dashed = detected guards as reset edges" if extra else ""), extra)
fig.suptitle(f"K = {R['K']}: 5D Ulam map, {R['params']['n']}^5 grid, step {R['params']['tau'] * R['params']['bin_ms']:.0f} ms, "
             f"edges seen >= {R['params']['mcount']} times",
             fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/cmg_morse{SUF}.png", dpi=110); plt.close(fig)

# ------------------------------------------------------------------ lattice
fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
for ax, routing in zip(axes, ("full", "detected")):
    M = R["main"][routing]
    L = M["lattice"]
    if not L:
        ax.axis("off"); ax.set_title(f"{routing}: lattice too large or empty"); continue
    short = {c: f"{M['desc'][c]['label'].split(' (')[0]}#{i}" for i, c in enumerate(M["restricted"])}
    ranks = {}
    for S in L:
        ranks.setdefault(len(S), []).append(S)
    pos_ = {S: ((i + 1) / (len(v) + 1), r) for r, v in ranks.items() for i, S in enumerate(v)}
    for S in L:
        for T in L:
            if len(T) == len(S) + 1 and S < T:
                ax.plot(*zip(pos_[S], pos_[T]), color="0.6", lw=0.8, zorder=0)
    for S in L:
        txt = "0" if not S else " v ".join(short[c] for c in sorted(S, key=str))
        ax.annotate(txt, pos_[S], ha="center", va="center", fontsize=6,
                    bbox=dict(boxstyle="round", fc="white", ec="0.4"))
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, max(ranks) + 0.6); ax.axis("off")
    ax.set_title(f"{routing} map: lattice of attractors = downsets of the restricted graph ({len(L)} elements)",
                 fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/cmg_lattice{SUF}.png", dpi=115); plt.close(fig)

# ------------------------------------------------------------------ sweep
keys = sorted({k[:3] for k in R["sweep"]})
cats = ["WALK ring", "WALK arc", "STILL", "mixed"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, routing in zip(axes, ("full", "detected")):
    Mx = np.zeros((len(cats), len(keys)))
    for j, kk in enumerate(keys):
        r = R["sweep"][(*kk, routing)]
        for c in r["rec"]:
            lab = r["desc"][c]["label"]
            for i, cat in enumerate(cats):
                if lab.startswith(cat):
                    Mx[i, j] += 1
    ax.imshow(Mx, cmap="Blues", aspect="auto")
    for i in range(len(cats)):
        for j in range(len(keys)):
            ax.text(j, i, int(Mx[i, j]), ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"n={n}\n{t * R['params']['bin_ms']:.0f} ms\n>={m}" for n, t, m in keys], fontsize=8)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    ax.set_title(f"{routing} map: Morse sets of each kind (grid n, step, min. edge count)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/cmg_sweep{SUF}.png", dpi=115); plt.close(fig)
print("figures written to", OUT)
