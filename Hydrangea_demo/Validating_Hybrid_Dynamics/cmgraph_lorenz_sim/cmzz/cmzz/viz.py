"""Figures: phase portraits per step, the CMG zigzag strip, barcodes, comparisons."""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import ListedColormap
import networkx as nx

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DEG = {0: SERIES[0], 1: SERIES[1], 2: SERIES[2], 3: SERIES[6]}   # homology degree colours
TRIVIAL = "#c9c8c2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
})


def track_colours(z):
    """Fixed-order hue per non-trivial track (by first appearance), grey otherwise."""
    order = []
    for bl in z.blocks:
        for b in bl:
            if not b.trivial and b.track not in order:
                order.append(b.track)
    cmap = {t: SERIES[(i + 3) % len(SERIES)] for i, t in enumerate(order)}   # 0-2 reserved for degrees
    return cmap


def bif_time(z):
    p = z.system.meta.get("bifurcation_param")
    a, b = z.system.param_range
    return None if p is None else (p - a) / (b - a) * z.T


# ------------------------------------------------------------------ portraits
def plot_portraits(z, path, dims=(0, 1), n_traj=6, max_cols=5):
    steps = list(range(0, z.n_spaces, 2))
    n = len(steps)
    cols = min(n, max_cols)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.85 * rows), squeeze=False)
    tc = track_colours(z)
    g = z.grid
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax, s in zip(axs.ravel(), steps):
        fr, st = z.frames[s], z.part.steps[s // 2]
        lab = np.full(g.n_cells, -1, int)
        colours = []
        for b in z.blocks[s]:
            lab[b.cells] = len(colours)
            colours.append(TRIVIAL if b.trivial else tc.get(b.track, TRIVIAL))
        img = lab.reshape(tuple(g.size))
        if g.dim > 2:            # project: any cell along the remaining axes
            other = tuple(i for i in range(g.dim) if i not in dims)
            img = img.max(axis=other)
        im = np.ma.masked_less(img.T.astype(float), 0)
        if colours:
            ax.imshow(im, origin="lower", interpolation="nearest",
                      extent=[*g.bounds[dims[0]], *g.bounds[dims[1]]],
                      cmap=ListedColormap(colours), vmin=-0.5, vmax=len(colours) - 0.5)
        lam = 0.5 * (st.lam0 + st.lam1)
        rng = np.random.default_rng(1)
        for _ in range(n_traj):
            x0 = g.bounds[:, 0] + rng.random(g.dim) * (g.bounds[:, 1] - g.bounds[:, 0])
            tr = z.system.trajectory(x0, lam, 6.0, 0.02)
            ax.plot(tr[:, dims[0]], tr[:, dims[1]], lw=0.5, color=MUTED, alpha=0.7)
        ax.set_xlim(*g.bounds[dims[0]])
        ax.set_ylim(*g.bounds[dims[1]])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(GRID)
        polys = ", ".join(b.poincare for b in z.blocks[s] if not b.trivial) or "—"
        ax.set_title(f"S{st.k}  t∈[{st.t0:.2f},{st.t1:.2f}]\n{z.system.param_name}∈"
                     f"[{st.lam0:.2f},{st.lam1:.2f}]   {polys}", fontsize=7.5, color=INK, loc="left")
    fig.suptitle(f"{z.system.name}: Morse sets per step ({z.part.n} steps, grid "
                 f"{'×'.join(map(str, g.size))}); colour = track, grey = zero index",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ CMG strip
def _hasse(n, order):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(order)
    try:
        return list(nx.transitive_reduction(G).edges())
    except Exception:
        return list(G.edges())


def plot_cmg_strip(z, path):
    """CMGs side by side: S0 ← M0|1 → S1 ← ... with red inclusion arrows
    (middle Morse set ⊆ side Morse set), as in Dey-Mrozek-Slechta Fig. 9."""
    tc = track_colours(z)
    W = max(8, 0.62 * z.n_spaces + 1.5)
    top_ = max([b.level for bl in z.blocks for b in bl] + [0])
    fig, ax = plt.subplots(figsize=(W, 2.2 + 0.9 * top_))
    pos = {}
    for s, bl in enumerate(z.blocks):
        lv = {}
        for b in bl:
            lv.setdefault(b.level, []).append(b)
        for level, items in lv.items():
            for j, b in enumerate(items):
                pos[(s, b.bid)] = (s + (j - (len(items) - 1) / 2) * 0.32, level)
    # inclusion arrows mid -> sides
    for (sm, bm, ss, bs, frac) in z.inclusions:
        (x0, y0), (x1, y1) = pos[(sm, bm)], pos[(ss, bs)]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=SERIES[7], lw=0.8, ls="--",
                                    shrinkA=9, shrinkB=9, alpha=0.8))
    # order edges inside each CMG
    for s, bl in enumerate(z.blocks):
        for (i, j) in _hasse(len(bl), z.block_order[s]):
            (x0, y0), (x1, y1) = pos[(s, i)], pos[(s, j)]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0, shrinkA=9, shrinkB=9))
    for s, bl in enumerate(z.blocks):
        for b in bl:
            x, y = pos[(s, b.bid)]
            c = TRIVIAL if b.trivial else tc.get(b.track, TRIVIAL)
            mid = s % 2 == 1
            ax.scatter([x], [y], s=150 if not mid else 95, color=c if not mid else SURFACE,
                       edgecolors=c, linewidths=2.0, zorder=3)
            ax.text(x, y - 0.3, b.poincare, ha="center", va="top", fontsize=6.5, color=INK2)
    for e in z.events:
        if e.kind not in ("continuation",):
            ax.axvline(2 * e.link + 1, color=SERIES[7], lw=0.6, alpha=0.25, zorder=0)
            ax.text(2 * e.link + 1, max([b.level for bl in z.blocks for b in bl] + [0]) + 0.55, e.kind, ha="center",
                    fontsize=7, color=SERIES[7])
    ax.set_xticks(range(z.n_spaces))
    ax.set_xticklabels([z.space_label(s) for s in range(z.n_spaces)], rotation=0, fontsize=7)
    ax.set_yticks([])
    top = max([b.level for bl in z.blocks for b in bl] + [0])
    ax.set_ylim(-0.8, top + 0.9)
    ax.set_xlim(-0.6, z.n_spaces - 0.4)
    ax.spines["left"].set_visible(False)
    ax.set_title("Conley–Morse graph zigzag: filled = step CMG, ring = middle CMG of F_k ∩ F_k+1\n"
                 "black arrow = flow order, red dashed = middle Morse set ⊆ side Morse set, label = Conley index",
                 fontsize=8, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ barcode
def draw_barcode(z, ax, title=None):
    tc = track_colours(z)
    rows = []
    keyorder = sorted({(b["kind"] != "cluster", b["id"]) for b in z.bars})
    for key in keyorder:
        bars = [b for b in z.bars if (b["kind"] != "cluster", b["id"]) == key]
        for b in sorted(bars, key=lambda b: (b["dim"], b["t_start"])):
            rows.append(b)
    for st in z.part:
        ax.axvline(st.t0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(z.T, color=GRID, lw=0.8, zorder=0)
    tb = bif_time(z)
    if tb is not None:
        ax.axvline(tb, color=INK, lw=1.0, ls=":", zorder=1)
        ax.text(tb, len(rows) - 0.3, " true bifurcation", fontsize=7, color=INK2, va="bottom")
    det = z.detection()
    if det:
        ax.axvline(det["t_detect"], color=SERIES[7], lw=1.2, zorder=1)
        ax.text(det["t_detect"], -0.9, f" detected t={det['t_detect']:.2f}", fontsize=7,
                color=INK2, va="top")
    labels = []
    for i, b in enumerate(rows):
        y = len(rows) - 1 - i
        c = DEG.get(b["dim"], INK2)
        if b["t_end"] - b["t_start"] < 1e-9:
            ax.scatter([b["t_start"]], [y], marker="D", s=24, color=c, zorder=3)
        else:
            ax.plot([b["t_start"], b["t_end"]], [y, y], color=c, lw=5, solid_capstyle="round", zorder=3)
        who = "cluster" if b["kind"] == "cluster" else f"track {b['id']}"
        labels.append(f"{who}  H{b['dim']}")
        ax.text(z.T + 0.01, y, b["label"], fontsize=6.5, color=INK2, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels[::-1], fontsize=7)
    for tick, b in zip(ax.get_yticklabels(), rows[::-1]):
        if b["kind"] != "cluster":
            col = tc.get(b["id"])
            if col:
                tick.set_color(INK)
    ax.set_xlim(-0.02, z.T + 0.02)
    ax.set_ylim(-1.4, len(rows) + 0.2)
    ax.set_xlabel("time (s)")
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    handles = [plt.Line2D([], [], color=DEG[k], lw=5, label=f"H{k}") for k in range(z.grid.dim + 1)]
    ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False, ncol=z.grid.dim + 1,
              bbox_to_anchor=(0, 1.0))


def plot_barcode(z, path):
    n = len(z.bars)
    fig, ax = plt.subplots(figsize=(9.5, 1.2 + 0.32 * n))
    draw_barcode(z, ax, f"Conley–Morse barcode: zigzag persistence of index pairs "
                        f"({z.part.n} steps of {1000 * z.T / z.part.n:.0f} ms)")
    fig.subplots_adjust(right=0.68, left=0.14)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(zs, path):
    heights = [1.2 + 0.32 * len(z.bars) for z in zs]
    fig, axs = plt.subplots(len(zs), 1, figsize=(9.5, sum(heights) + 0.6),
                            gridspec_kw={"height_ratios": heights}, squeeze=False)
    for ax, z in zip(axs[:, 0], zs):
        det = z.detection()
        tb = bif_time(z)
        txt = (f"{z.part.n} steps × {1000 * z.T / z.part.n:.0f} ms, grid {'×'.join(map(str, z.grid.size))}"
               + (f" — detected at t={det['t_detect']:.2f} s (error {det['t_detect'] - tb:+.2f} s)" if det and tb is not None
                  else " — no bifurcation detected"))
        draw_barcode(z, ax, txt)
    fig.subplots_adjust(right=0.68, left=0.14, hspace=0.55)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_scaling(rows, path):
    """rows: dicts with keys label, x (cells), stage timings."""
    stages = ["enclosure", "cmgraph_morse_graph", "index_pairs", "pair_homology", "zigzag_persistence"]
    names = ["enclosure (RK4)", "cmgraph Morse graphs", "index pairs", "pair homology", "zigzag (fzz)"]
    groups = list(dict.fromkeys(r["group"] for r in rows))
    fig, axs = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.4), squeeze=False)
    for ax, gname in zip(axs[0], groups):
        R = sorted([r for r in rows if r["group"] == gname], key=lambda r: r["cells"])
        x = [r["cells"] for r in R]
        for si, (st, nm) in enumerate(zip(stages, names)):
            y = [max(r["timings"].get(st, 0), 1e-3) for r in R]
            ax.plot(x, y, marker="o", ms=4, lw=2, color=SERIES[si], label=nm)
        ax.plot(x, [r["total"] for r in R], marker="s", ms=4, lw=2, color=INK, label="total")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("grid cells")
        ax.set_ylabel("seconds")
        ax.set_title(gname, fontsize=8.5, loc="left")
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)
