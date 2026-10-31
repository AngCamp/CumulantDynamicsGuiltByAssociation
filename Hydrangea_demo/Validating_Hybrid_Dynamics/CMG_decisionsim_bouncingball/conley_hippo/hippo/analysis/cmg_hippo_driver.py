#!/usr/bin/env python3
"""
============================================================================
HIPPOCAMPUS CONLEY-MORSE DRIVER
============================================================================
Run:   python3 cmg_hippo_driver.py
Needs: cmg_lib.py and hds/decode.py  (path set by CMG_PATH below)
       hippocampus.py, neurosim.py, hippocampus_config_v3.json  (SIM_DIR)

Stages (each prints what it is doing and writes ONE figure to figures/):

  STAGE 0  simulate (or load) the hippocampus model
  STAGE 1  Lipschitz step sweep  ->  fig1_lipschitz_steps.png
           PCA once at a 100 ms bin; the PC1 reconstruction is the
           trajectory; the time step is varied; the distribution of
           |d log L| per step is plotted. The elbow of the median curve
           chooses the time step s* used from here on (smallest step at or
           after the elbow).
  STAGE 2  chart and states         ->  fig2_chart_states.png
  STAGE 3  delay embedding (spacing s*, spanning 500 ms) and the Ulam
           matrix                    ->  fig3_ulam_walkthrough.png
  STAGE 4  Conley-Morse decoding     ->  fig4_cmg.png

Normalization throughout: each NEURON divided by its own total spike count,
then z-scored across time (hds.decode.normalize, mode="per_neuron").
============================================================================
"""
import os, sys, json, time, subprocess
import numpy as np

# --------------------------------------------------------------- settings
HERE      = os.path.dirname(os.path.abspath(__file__))
SIM_DIR   = os.path.dirname(HERE)                 # holds hippocampus.py
CMG_PATH  = "/home/claude/hybrid_cmg"             # holds cmg_lib.py, hds/
FIGDIR    = os.path.join(HERE, "figures")
CONFIG    = os.path.join(SIM_DIR, "hippocampus_config_v3.json")

BIN_SWEEP   = 100.0          # ms, bin used for the Lipschitz step sweep
FINE_MS     = 5.0            # ms, finest grid everything is built from
STEPS_MS    = [5, 10, 20, 25, 50, 100, 200, 400, 800]
DELAY_SPAN  = 500.0          # ms, how far back the delay embedding reaches

# The BIN is set by the signal, not by the Lipschitz sweep. A boxcar of width b
# attenuates a sinusoid of period T by |sinc(b/T)|; theta at THETA_HZ has
# T = 125 ms, so b = 25 ms keeps 94% of it, b = 100 ms keeps 23%, and b = 125 ms
# cancels it exactly. The Lipschitz sweep prefers large bins because they
# suppress Poisson noise, which is the opposite criterion; it is reported as a
# diagnostic and does NOT choose the bin.
THETA_HZ    = 8.0
BIN_MS      = 25.0           # = step = delay spacing
GRID_NX     = 40             # cells per chart axis
TAU_FACTOR  = 1              # reachability lag = TAU_FACTOR * s*
K_FIX       = 5              # PCs kept, held fixed (NOT chosen by the elbow)

sys.path.insert(0, CMG_PATH)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import cmg_lib as cl
from hds import decode as dc

dc.STEP_MS = FINE_MS
STATE_NAMES = ["WALK", "STILL", "SWR"]
SCOL = {"WALK": "#1e88e5", "STILL": "#9e9e9e", "SWR": "#d81b60"}
_t0 = time.time()

def log(msg):
    print(f"[{time.time()-_t0:6.1f}s] {msg}", flush=True)

# ============================================================== STAGE 0
def stage0_simulate():
    """Run hippocampus.py if its output is missing, then load the spikes."""
    cfg = json.load(open(CONFIG))
    out = os.path.join(SIM_DIR, cfg["output_dir"], "simulation.npz")
    if not os.path.exists(out):
        log(f"simulating {cfg['duration_s']:.0f} s -> {cfg['output_dir']} ...")
        subprocess.run([sys.executable, "hippocampus.py", os.path.basename(CONFIG)],
                       cwd=SIM_DIR, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    d = np.load(out)
    n = int(d["n_steps"])
    D = dict(S=np.unpackbits(d["spikes"], axis=0, count=n).astype(np.uint8),
             state=d["state"].astype(int), pos=d["pos"], speed=d["speed"],
             n=n, base=float(d["dt_ms"]) / 1000.0,
             pop_sizes=d["pop_sizes"], cfg=cfg)
    log(f"STAGE 0: {D['S'].shape[1]} neurons, {n/1000:.0f} s, "
        f"mean rate {D['S'].mean()/D['base']:.2f} Hz")
    for k, nm in enumerate(STATE_NAMES):
        print(f"            {nm:5s} {(D['state']==k).mean():6.2%}")
    ch = np.flatnonzero(np.diff(D["state"]) != 0)
    D["swr_on"] = np.array([c + 1 for c in ch
                            if D["state"][c] == 1 and D["state"][c + 1] == 2])
    print(f"            {len(D['swr_on'])} SWR events, "
          f"{len(ch)} state changes total")
    k = np.ones(21) / 21
    D["mua"] = np.convolve(D["S"].sum(1).astype(float), k, mode="same") \
        / D["base"] / D["S"].shape[1]
    return D

# ----------------------------------------------------------- PCA helpers
def pc_scores(D, bin_ms, step_ms, K=None):
    """bins -> per-neuron normalization -> correlation PCA -> K scores.

    K is FIXED at K_FIX (default 5). The variance elbow is still computed and
    returned, but only as a diagnostic: it is not used to choose K."""
    Xf, trf, ttf = dc.fine_bins([D["S"]], D["base"])
    Xw, tt, tr = dc.window_counts(Xf, trf, ttf, bin_ms, step_ms)
    Z = dc.normalize(Xw, mode="per_neuron")
    C = (Z.T @ Z) / len(Z)
    ev, vec = np.linalg.eigh(C.astype(np.float64))
    o = np.argsort(ev)[::-1]
    ev, vec = ev[o], vec[:, o]
    K_elbow = dc.elbow_k(ev)
    Kuse = int(K if K is not None else K_FIX)
    return Z @ vec[:, :Kuse], Kuse, ev, tt, K_elbow

def lipschitz(p, dt):
    """L = |second difference| / |first difference| / dt, elementwise."""
    p = np.atleast_2d(p.T).T
    Lp = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    Lv = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1) / dt ** 2
    return Lv / np.maximum(Lp[:-1], 1e-12)

def sinc_retained(bin_ms, freq_hz):
    """Fraction of a sinusoid's amplitude surviving a boxcar bin."""
    x = np.pi * bin_ms * freq_hz / 1000.0
    return 1.0 if x == 0 else abs(np.sin(x) / x)

def elbow_of(x_log, y):
    """Index of the point furthest from the chord of a monotone-ish curve."""
    x = (np.asarray(x_log, float) - min(x_log)) / (max(x_log) - min(x_log) + 1e-12)
    yy = (np.asarray(y, float) - min(y)) / (max(y) - min(y) + 1e-12)
    d = np.abs(x + yy - 1) / np.sqrt(2)
    return int(np.argmax(d))

# ============================================================== STAGE 1
def stage1_lipschitz_steps(D):
    """Distribution of the Lipschitz change vs time step, on PC1."""
    P, K, ev, tt, K_elbow = pc_scores(D, BIN_SWEEP, FINE_MS)
    pc1 = P[:, 0]                       # rank-1 reconstruction, up to a unit loading
    log(f"STAGE 1: PCA at {BIN_SWEEP:.0f} ms bin -> {len(pc1)} samples, K={K} fixed "
        f"(elbow would say {K_elbow}); Lipschitz computed on PC1")
    Ls, dLs, rows = {}, {}, []
    for s_ms in STEPS_MS:
        m = int(round(s_ms / FINE_MS))
        dt = s_ms / 1000.0
        p = pc1[::m]
        if len(p) < 50:
            continue
        L = lipschitz(p, dt)
        dL = np.abs(np.diff(np.log(np.maximum(L, 1e-12))))
        Ls[s_ms], dLs[s_ms] = L, dL
        rows.append(dict(s=s_ms, n=len(L), medL=float(np.median(L)),
                         noise=float(np.sqrt(3) / dt),
                         med_dL=float(np.median(dL)),
                         p99=float(np.percentile(dL, 99)),
                         iqr=float(np.subtract(*np.percentile(dL, [75, 25])))))
        print(f"            step {s_ms:4d} ms  median L {rows[-1]['medL']:9.2f} "
              f"(sqrt3/dt {rows[-1]['noise']:8.2f})  median |dlogL| "
              f"{rows[-1]['med_dL']:.3f}  p99 {rows[-1]['p99']:6.2f}")
    ss = np.array([r["s"] for r in rows])
    med = np.array([r["med_dL"] for r in rows])
    p99 = np.array([r["p99"] for r in rows])
    # The elbow goes on the UPPER TAIL, not the median. Undersampling at small
    # steps shows up as rare large jumps in log L, so the median is nearly flat
    # (it is dominated by ordinary smooth motion) and its chord elbow runs to
    # the far end of the axis -- which is how 800 ms got chosen. The p99 curve
    # is the one that actually falls as the step grows.
    i_el = elbow_of(np.log(ss), p99)
    s_star = int(ss[i_el])
    flat_med = (med.max() - med.min()) / max(med.mean(), 1e-9)
    flat_p99 = (p99.max() - p99.min()) / max(p99.mean(), 1e-9)
    log(f"STAGE 1: elbow on the p99 curve -> noise floor at step {s_star} ms "
        f"(p99 range {flat_p99:.0%} of its mean; median range only "
        f"{flat_med:.0%}, which is why the median has no usable elbow)")
    print(f"            theta ({THETA_HZ:.0f} Hz, T={1000/THETA_HZ:.0f} ms) amplitude "
          f"retained by a boxcar bin:")
    for b_ in (25, 50, 100, 125, 200):
        print(f"               bin {b_:4d} ms -> {sinc_retained(b_, THETA_HZ):5.1%}")
    ok = "at or above" if BIN_MS >= s_star else "BELOW"
    print(f"            -> step = {BIN_MS:.0f} ms: {ok} the {s_star} ms noise floor, "
          f"and keeps {sinc_retained(BIN_MS, THETA_HZ):.0%} of theta "
          f"(100 ms keeps {sinc_retained(100, THETA_HZ):.0%})")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    parts = ax.violinplot([dLs[s_] for s_ in ss], positions=np.arange(len(ss)),
                          showmedians=True, widths=.85)
    for b_ in parts["bodies"]:
        b_.set_facecolor("#1e88e5"); b_.set_alpha(.65)
    ax.plot(np.arange(len(ss)), p99, "o-", ms=4, lw=1.2, color="#37474f",
            label="p99 of |d log L|")
    ax.axvline(i_el, color="#546e7a", ls=":", lw=1.4,
               label=f"p99 elbow = noise floor at {s_star} ms")
    ax.axvline(list(ss).index(int(BIN_MS)), color="#d81b60", ls="--", lw=1.6,
               label=f"step used = {BIN_MS:.0f} ms "
                     f"({sinc_retained(BIN_MS, THETA_HZ):.0%} of theta kept)")
    ax.set_xticks(range(len(ss))); ax.set_xticklabels(ss)
    ax.set(xlabel="time step (ms)", ylabel="|d log L| per step",
           title=f"Change in the instantaneous Lipschitz constant vs time step\n"
                 f"(PC1 reconstruction; PCA computed once at a {BIN_SWEEP:.0f} ms bin)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig1_lipschitz_steps.png", dpi=130)
    plt.close(fig)
    log("STAGE 1: wrote fig1_lipschitz_steps.png")
    return s_star, rows

# ----------------------------------------------------------- chart build
def build_chart(D, bin_ms, step_ms, delay_spacing=None, span_ms=DELAY_SPAN,
                nx=GRID_NX):
    """PC scores, optionally delay-embedded, reduced to a 2D gridded chart."""
    P, K, ev, tt, K_elbow = pc_scores(D, bin_ms, step_ms)
    if delay_spacing is None:
        Z, tt2, m = P[:, :2], tt, 1
    else:
        ls = max(1, int(round(delay_spacing / step_ms)))
        m = int(round(span_ms / delay_spacing)) + 1
        E = cl.delay_embed(P, m, ls).reshape(-1, m * P.shape[1])
        tt2 = tt[(m - 1) * ls:][:len(E)]
        mu2, V2 = cl.pca_fit(E, d=2, max_n=40000)
        Z = cl.pca_apply(E, mu2, V2)
    g = cl.Grid2D(Z, nx, qlo=0.02, qhi=99.98)
    idx = np.clip((tt2 * 1000).astype(int), 0, D["n"] - 1)
    return dict(Z=Z, K=K, K_elbow=K_elbow, ev=ev, g=g, cz=g.cells(Z), tt=tt2,
                idx=idx, m_lags=m,
                state=D["state"][idx], pos=D["pos"][idx], speed=D["speed"][idx],
                bin_ms=bin_ms, step_ms=step_ms)

# ============================================================== STAGE 2
def stage2_chart_states(D, C, s_star):
    log("STAGE 2: chart and states")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1])
    axc = fig.add_subplot(gs[0, 0])
    for k, nm in enumerate(STATE_NAMES):
        m = C["state"] == k
        axc.scatter(C["Z"][m, 0], C["Z"][m, 1], s=3 if k < 2 else 13, color=SCOL[nm],
                    alpha=.45 if k < 2 else .95, label=f"{nm} ({m.mean():.1%})", zorder=k)
    axc.legend(fontsize=8, markerscale=2)
    axc.set(xlabel="delay-chart axis 1", ylabel="delay-chart axis 2",
            title="(a) chart coloured by TRUE state")
    axp = fig.add_subplot(gs[0, 1])
    mw = C["state"] == 0
    sc_ = axp.scatter(C["Z"][mw, 0], C["Z"][mw, 1], c=C["pos"][mw], s=3, cmap="hsv")
    plt.colorbar(sc_, ax=axp, label="ring position s (m)")
    axp.set(xlabel="delay-chart axis 1", ylabel="delay-chart axis 2",
            title="(b) WALK samples by track position\n(ring = place-field sequence)")
    axe = fig.add_subplot(gs[0, 2])
    pct = 100.0 * C["ev"] / C["ev"].sum()          # % variance per component
    axe.plot(range(1, 31), pct[:30], "o-", ms=4, color="#37474f")
    axe.plot(range(1, C["K"] + 1), pct[:C["K"]], "D", ms=9, color="#d81b60",
             ls="none", label=f"K={C['K']} kept (fixed)")
    axe.axvline(C["K_elbow"], color="#546e7a", ls=":", lw=1.2,
                label=f"elbow would say {C['K_elbow']}")
    axe.set(xlabel="component", ylabel="% variance explained",
            title=f"(c) PC var%  (top {C['K']} = {pct[:C['K']].sum():.1f}% total)")
    axe.legend(fontsize=8)
    tt_s = np.arange(D["n"]) / 1000.0
    axt = fig.add_subplot(gs[1, :])
    axt.plot(tt_s, D["state"], lw=.8, color="k")
    axt.set_yticks([0, 1, 2]); axt.set_yticklabels(STATE_NAMES)
    for s_ in D["swr_on"]:
        axt.axvline(s_ / 1000, color="#d81b60", lw=.5, alpha=.6)
    axt.set(ylabel="true state", xlim=(0, min(60, D["n"] / 1000)),
            title="(d) true state time series (red lines = SWR onsets, the guard)")
    axm = fig.add_subplot(gs[2, :], sharex=axt)
    axm.plot(tt_s, D["mua"], lw=.5, color="#37474f", label="all neurons")
    npyr = int(D["pop_sizes"][0])
    kk = np.ones(21) / 21
    pyr = np.convolve(D["S"][:, :npyr].sum(1).astype(float), kk, mode="same") \
        / D["base"] / npyr
    axm.plot(tt_s, pyr, lw=.5, color="#8e24aa", label="pyramidal only")
    axm.set(xlabel="time (s)", ylabel="MUA (Hz/neuron)", yscale="log",
            title="(e) population rate: SWRs are the sharp peaks during STILL")
    axm.legend(fontsize=8)
    fig.suptitle(f"Hippocampus chart and states  (bin {C['bin_ms']:.0f} ms, "
                 f"step {C['step_ms']:.0f} ms = s*)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig2_chart_states.png", dpi=125)
    plt.close(fig)
    log("STAGE 2: wrote fig2_chart_states.png")

# ============================================================== STAGE 3
def build_ulam(C, tau_ms, routing="none"):
    """Ulam transitions at lag tau; routing cuts transitions across guards."""
    ls = max(1, int(round(tau_ms / C["step_ms"])))
    i0 = np.arange(len(C["Z"]) - ls)
    a, b = C["state"][i0], C["state"][i0 + ls]
    if routing == "none":
        bad = np.zeros(len(i0), bool)
    elif routing == "swr":                 # cut transitions touching an SWR
        bad = (a == 2) | (b == 2)
    elif routing == "oracle":              # cut every cross-state transition
        bad = a != b
    src, dst = C["cz"][i0[~bad]], C["cz"][i0[~bad] + ls]
    ok = (src >= 0) & (dst >= 0)
    src, dst = src[ok], dst[ok]
    g = C["g"]
    A = cl.build_map(src, dst, g.nx, g.ny, src_dilate=0)
    A = cl.support_restrict(A, src, dst)
    return A, src, dst, ls, int(bad.sum())

def stage3_ulam(C, tau_ms):
    log(f"STAGE 3: delay chart has {C['m_lags']} lagged copies "
        f"(spacing {C['step_ms']:.0f} ms, span {(C['m_lags']-1)*C['step_ms']:.0f} ms); "
        f"Ulam matrix at lag {tau_ms:.0f} ms")
    A, src, dst, ls, _ = build_ulam(C, tau_ms, "none")
    g = C["g"]
    c_star = np.bincount(src).argmax()
    land = dst[src == c_star]
    cells_land, n_land = np.unique(land, return_counts=True)
    img = A[c_star].indices
    print(f"            A: {A.shape[0]}x{A.shape[1]}, {A.nnz} nonzeros "
          f"({A.nnz/g.N**2:.2e} density); busiest cell {c_star} has "
          f"{len(land)} transitions into {len(cells_land)} cells")
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.6))
    ax[0].scatter(C["Z"][::3, 0], C["Z"][::3, 1], s=1.5, color="0.85")
    inc = np.flatnonzero(C["cz"] == c_star)
    ax[0].scatter(C["Z"][inc, 0], C["Z"][inc, 1], s=8, color="#1e88e5",
                  label=f"{len(inc)} samples in cell {c_star}")
    fut = inc[inc < len(C["Z"]) - ls] + ls
    ax[0].scatter(C["Z"][fut, 0], C["Z"][fut, 1], s=8, color="#d81b60",
                  label=f"same samples {tau_ms:.0f} ms later")
    for a_ in inc[:60]:
        if a_ + ls < len(C["Z"]):
            ax[0].annotate("", xy=C["Z"][a_ + ls], xytext=C["Z"][a_],
                           arrowprops=dict(arrowstyle="->", color="0.45", lw=0.4))
    ax[0].legend(fontsize=8)
    ax[0].set(xlabel="chart axis 1", ylabel="chart axis 2",
              title="step 4: one source cell and where its samples go")
    ci, cj = divmod(c_star, g.ny)
    li, lj = np.divmod(cells_land, g.ny)
    lo_i, hi_i = max(min(ci, li.min()) - 2, 0), max(ci, li.max()) + 3
    lo_j, hi_j = max(min(cj, lj.min()) - 2, 0), max(cj, lj.max()) + 3
    cnts = np.zeros(g.N); cnts[cells_land] = n_land
    im = ax[1].imshow(cnts.reshape(g.nx, g.ny)[lo_i:hi_i, lo_j:hi_j].T, origin="lower",
                      cmap="Blues", extent=(lo_i-.5, hi_i-.5, lo_j-.5, hi_j-.5))
    ax[1].scatter([ci], [cj], marker="s", s=140, facecolor="none",
                  edgecolor="#1e88e5", lw=2, label="source cell")
    plt.colorbar(im, ax=ax[1], label="count of landings"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"Ulam COUNT row for cell {c_star}: {len(cells_land)} landing "
                    f"cells,\n{len(land)} transitions")
    padm = np.zeros(g.N); padm[img] = 1
    ax[2].imshow(padm.reshape(g.nx, g.ny)[lo_i:hi_i, lo_j:hi_j].T, origin="lower",
                 cmap="Greys", vmin=0, vmax=1.6,
                 extent=(lo_i-.5, hi_i-.5, lo_j-.5, hi_j-.5))
    ax[2].scatter([ci], [cj], marker="s", s=140, facecolor="none",
                  edgecolor="#1e88e5", lw=2)
    ax[2].set_title(f"boolean row of A: F(cell {c_star})\n"
                    f"{len(cells_land)} landing + padding = {len(img)} cells")
    for a_ in ax[1:]:
        a_.set_xlabel("chart axis 1 (cell index)")
        a_.set_ylabel("chart axis 2 (cell index)")
    fig.suptitle(f"How the Ulam matrix is built  (bin {C['bin_ms']:.0f} ms, "
                 f"step {C['step_ms']:.0f} ms, {C['m_lags']} lags to "
                 f"{(C['m_lags']-1)*C['step_ms']:.0f} ms, lag {tau_ms:.0f} ms, "
                 f"{g.nx}x{g.ny} grid; A has {A.nnz} nonzeros of {g.N**2:,})")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig3_ulam_walkthrough.png", dpi=120)
    plt.close(fig)
    log("STAGE 3: wrote fig3_ulam_walkthrough.png")

# ============================================================== STAGE 4
def morse_sets(C, an, min_cells=2, min_samp=5):
    out = []
    for cc in an["rec"]:
        cells = an["members"][cc]
        if len(cells) < min_cells:
            continue
        m = np.isin(C["cz"], cells)
        if m.sum() < min_samp:
            continue
        comp = np.array([(C["state"][m] == k).mean() for k in range(3)])
        out.append(dict(id=cc, n=len(cells), nsamp=int(m.sum()), comp=comp, mask=m,
                        idx=an["index"].get(cc),
                        label=STATE_NAMES[int(comp.argmax())],
                        purity=float(comp.max())))
    out.sort(key=lambda x: -x["n"])
    return out

def draw_graph(ax, sets, hasse, title):
    ids = [s["id"] for s in sets]
    pred = {i: [a for a, b in hasse if b == i and a in ids] for i in ids}
    depth = {}
    def dep(i):
        if i not in depth:
            depth[i] = 0 if not pred[i] else 1 + max(dep(p) for p in pred[i])
        return depth[i]
    for i in ids: dep(i)
    lv = {}
    for i in ids: lv.setdefault(depth[i], []).append(i)
    md = max(lv) if lv else 0
    pos = {}
    for dd, ns in lv.items():
        for t, i in enumerate(ns):
            pos[i] = ((t + 1) / (len(ns) + 1), md - dd)
    for a, b in hasse:
        if a in pos and b in pos:
            ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>",
                                         mutation_scale=12, color="#546e7a",
                                         shrinkA=25, shrinkB=25, lw=1.0))
    for s in sets:
        ax.scatter(*pos[s["id"]], s=1500, color=SCOL[s["label"]], edgecolors="k", zorder=3)
        ax.annotate(f"{s['label']}\n{s['idx']}\n{s['purity']:.0%}", pos[s["id"]],
                    ha="center", va="center", fontsize=7, zorder=4)
    ax.set(xlim=(0, 1), ylim=(-.7, md + .7)); ax.axis("off")
    ax.set_title(title, fontsize=9.5)

def stage4_cmg(C, tau_ms):
    log(f"STAGE 4: Conley-Morse decoding at lag {tau_ms:.0f} ms")
    ROUT = ["none", "swr", "oracle"]
    TITLE = {"none": "no routing", "swr": "SWR guards routed",
             "oracle": "all state transitions routed"}
    results = {}
    for rt in ROUT:
        A, src, dst, ls, ncut = build_ulam(C, tau_ms, rt)
        an = cl.analyze(A, C["g"].ny, "", index=True, verbose=False)
        sets = morse_sets(C, an)
        results[rt] = dict(A=A, an=an, sets=sets, ncut=ncut)
        got = sorted({s["label"] for s in sets if s["purity"] >= 0.5})
        print(f"            [{rt:6s}] cut {ncut:5d}; {len(an['rec'])} recurrent comps, "
              f"{len(sets)} Morse sets; states recovered: {got}")
        for s in sets[:5]:
            comp = " ".join(f"{STATE_NAMES[k][0]}{s['comp'][k]:.2f}" for k in range(3))
            print(f"                {s['n']:4d} cells {s['nsamp']:6d} samp  "
                  f"idx {s['idx']}  {comp}  -> {s['label']} ({s['purity']:.0%})")
    Cc = C["g"].centers()
    fig, axes = plt.subplots(3, len(ROUT), figsize=(6 * len(ROUT), 12),
                             gridspec_kw=dict(height_ratios=[1.4, 1, 1]))
    for j, rt in enumerate(ROUT):
        R = results[rt]
        ax = axes[0, j]
        ax.scatter(C["Z"][::3, 0], C["Z"][::3, 1], s=1, color="0.9")
        for s in R["sets"][:8]:
            cells = np.array(R["an"]["members"][s["id"]])
            ax.scatter(Cc[cells, 0], Cc[cells, 1], s=10, marker="s",
                       color=SCOL[s["label"]], alpha=.85)
        ax.set(xlabel="chart axis 1", ylabel="chart axis 2")
        ax.set_title(f"{TITLE[rt]}\n{len(R['sets'])} Morse sets", fontsize=10)
        draw_graph(axes[1, j], R["sets"][:8], R["an"]["hasse"], "Conley-Morse graph")
        axl = axes[2, j]
        axl.plot(C["tt"], C["state"], lw=.7, color="k")
        for si, s in enumerate(R["sets"][:8]):
            m = s["mask"]
            axl.scatter(C["tt"][m], np.full(m.sum(), -0.55 - 0.3 * si), s=.8,
                        color=SCOL[s["label"]])
        axl.set_yticks([0, 1, 2]); axl.set_yticklabels(STATE_NAMES, fontsize=7)
        t1 = C["tt"][0]
        axl.set(xlabel="time (s)", xlim=(t1, t1 + 40), ylim=(-.55 - .3 * 8, 2.4))
        axl.set_title("true state (line) vs Morse set membership (dots)", fontsize=9)
    fig.suptitle(f"Hippocampus Conley-Morse graph  (bin {C['bin_ms']:.0f} ms, "
                 f"step {C['step_ms']:.0f} ms, {C['m_lags']} lags to "
                 f"{(C['m_lags']-1)*C['step_ms']:.0f} ms, lag {tau_ms:.0f} ms)",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig4_cmg.png", dpi=120)
    plt.close(fig)
    log("STAGE 4: wrote fig4_cmg.png")
    return results

# ================================================================== main
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    D = stage0_simulate()
    s_star, _ = stage1_lipschitz_steps(D)      # diagnostic only
    # the chart: bin = step = delay spacing = BIN_MS, chosen to preserve theta
    C = build_chart(D, bin_ms=BIN_MS, step_ms=BIN_MS,
                    delay_spacing=BIN_MS, span_ms=DELAY_SPAN)
    log(f"chart: {len(C['Z'])} samples, K={C['K']} fixed (elbow would say "
        f"{C['K_elbow']}), {C['m_lags']} lagged copies -> {C['m_lags']*C['K']}-dim "
        f"embedding, {len(np.unique(C['cz'][C['cz']>=0]))}/{C['g'].N} cells occupied")
    stage2_chart_states(D, C, BIN_MS)
    tau = TAU_FACTOR * BIN_MS
    stage3_ulam(C, tau)
    stage4_cmg(C, tau)
    log(f"done. figures in {FIGDIR}")

if __name__ == "__main__":
    main()
