"""
Conley-Morse decoding of the hippocampus simulation.

Ground truth (simulator state trace):
    WALK   58.0%   locomotion on a square track, theta, place fields
    STILL  40.7%   quiescence
    SWR     1.3%   24 events, ~88 ms, always STILL -> SWR -> STILL
                   = the guard / reset of this system

Two charts are compared:
  pc12    bin -> normalize -> correlation PCA -> (PC1, PC2)            [as asked]
  delay   same PC scores, then a delay embedding with m=7 copies at
          50 ms spacing (6 lagged copies, out to 300 ms), then a second
          PCA to 2D.  Fine bin + many lags, instead of one coarse bin.

Routing of the combinatorial map (which transitions are kept):
  none    everything. The behavioural state machine cycles, so this should be
          one recurrent class.
  oracle  only transitions whose two ends are in the same TRUE state
  swr     only transitions with no SWR end (true SWR times)
  det     same, but SWR times detected from the MUA
Cut transitions are counted as RESET EDGES between the Morse sets they join.
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, "/home/claude/hybrid_cmg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import cmg_lib as cl
from hds import decode as dc

dc.STEP_MS = 5.0                      # fine-bin grid; windows/steps are multiples
OUT = "/home/claude/hippo/"
t0 = time.time()
def T(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

STATE_NAMES = ["WALK", "STILL", "SWR"]
SCOL = {"WALK": "#1e88e5", "STILL": "#9e9e9e", "SWR": "#d81b60"}

B, SSTEP = 25, 25                     # 25 ms bins, 25 ms step
M_LAGS, LAG_SPACING = 7, 50           # 6 lagged copies at 50 ms -> out to 300 ms
TAU = 50                              # reachability lag (ms), swept later
NX = 40

# ------------------------------------------------------------------ load
d = np.load(OUT + "output/simulation.npz")
n_steps = int(d["n_steps"])
S = np.unpackbits(d["spikes"], axis=0, count=n_steps).astype(np.uint8)
state_ms = d["state"].astype(int)
speed_ms, pos_ms = d["speed"], d["pos"]
base = float(d["dt_ms"]) / 1000.0
pop_sizes = d["pop_sizes"]
print(f"{S.shape[1]} neurons, {n_steps} ms, mean rate {S.mean()/base:.2f} Hz")

def transitions(a=None, b=None):
    ch = np.flatnonzero(np.diff(state_ms) != 0)
    return np.array([c + 1 for c in ch
                     if (a is None or state_ms[c] == a)
                     and (b is None or state_ms[c + 1] == b)], int)
swr_on, swr_off = transitions(1, 2), transitions(2, 1)
beh_tr = np.r_[transitions(0, 1), transitions(1, 0)]
print(f"guards: {len(swr_on)} SWR onsets, {len(beh_tr)} WALK<->STILL transitions")

# ------------------------------------------------------------ MUA + guard
k = np.ones(21) / 21
mua_s = np.convolve(S.sum(1).astype(float), k, mode="same") / base / S.shape[1]

def detect_swr(thresh_sd, min_gap_ms=150):
    base_mu, base_sd = mua_s[state_ms == 1].mean(), mua_s[state_ms == 1].std()
    th = base_mu + thresh_sd * base_sd
    on = np.flatnonzero(np.diff(np.r_[0, (mua_s > th).astype(int)]) == 1)
    ev = []
    for o in on:
        if not ev or o - ev[-1] > min_gap_ms:
            ev.append(o)
    return np.array(ev, int), th, mua_s > th

def match(det, truth, tol_ms=60):
    if len(det) == 0:
        return 0.0, 0.0
    hit = sum(1 for t in truth if np.abs(det - t).min() <= tol_ms)
    used = sum(1 for t_ in det if np.abs(truth - t_).min() <= tol_ms)
    return used / len(det), hit / max(len(truth), 1)

sds = np.arange(2.0, 14.1, 0.5)
pr, rc = zip(*[match(detect_swr(s_)[0], swr_on) for s_ in sds])
pr, rc = np.array(pr), np.array(rc)
f1 = 2 * pr * rc / (pr + rc + 1e-9)
best = int(np.argmax(f1))
det_ev, det_th, det_above = detect_swr(sds[best])
print(f"SWR detection: mean+{sds[best]:.1f}sd of STILL MUA -> {len(det_ev)} events, "
      f"precision {pr[best]:.2f}, recall {rc[best]:.2f}")
# detected state trace: 2 where the MUA is above threshold near a detected onset
det_state_ms = np.where(det_above, 2, state_ms * 0 + 1)
det_state_ms[state_ms == 0] = 0        # WALK/STILL split is not the detector's job

# ------------------------------------------------------------ the charts
def pc_scores(b_ms, s_ms):
    Xf, trf, ttf = dc.fine_bins([S], base)
    Xw, tt, tr = dc.window_counts(Xf, trf, ttf, b_ms, s_ms)
    ev, vec, Z0 = dc.corr_svd(Xw)
    K = dc.elbow_k(ev)
    return Z0 @ vec[:, :max(K, 2)], K, ev, tt

def make_chart(kind, b_ms=B, s_ms=SSTEP, nx=NX, m=M_LAGS, spacing=LAG_SPACING):
    sc, K, ev, tt = pc_scores(b_ms, s_ms)
    if kind == "pc12":
        Z, tt2 = sc[:, :2], tt
    else:
        ls = max(1, int(round(spacing / s_ms)))
        D = cl.delay_embed(sc, m, ls).reshape(-1, m * sc.shape[1])
        tt2 = tt[(m - 1) * ls:][:len(D)]
        mu2, V2 = cl.pca_fit(D, d=2, max_n=40000)
        Z = cl.pca_apply(D, mu2, V2)
    g = cl.Grid2D(Z, nx, qlo=0.02, qhi=99.98)
    idx = np.clip((tt2 * 1000).astype(int), 0, n_steps - 1)
    return dict(kind=kind, Z=Z, K=K, ev=ev, g=g, cz=g.cells(Z), tt=tt2, idx=idx,
                state=state_ms[idx], det=det_state_ms[idx], pos=pos_ms[idx],
                speed=speed_ms[idx], s_ms=s_ms, b_ms=b_ms)

# ------------------------------------------------------- map construction
def build_cmg(C, tau_ms, routing="none", index=True):
    ls = max(1, int(round(tau_ms / C["s_ms"])))
    i0 = np.arange(len(C["Z"]) - ls)
    a, b_ = C["state"][i0], C["state"][i0 + ls]
    if routing == "none":
        bad = np.zeros(len(i0), bool)
    elif routing == "oracle":
        bad = a != b_
    elif routing == "swr":
        bad = (a == 2) | (b_ == 2)
    elif routing == "det":
        da, db = C["det"][i0], C["det"][i0 + ls]
        bad = (da == 2) | (db == 2)
    src, dst = C["cz"][i0[~bad]], C["cz"][i0[~bad] + ls]
    ok = (src >= 0) & (dst >= 0)
    g = C["g"]
    A = cl.build_map(src[ok], dst[ok], g.nx, g.ny, src_dilate=0)
    A = cl.support_restrict(A, src[ok], dst[ok])
    an = cl.analyze(A, g.ny, "", index=index, verbose=False)
    return dict(A=A, an=an, ls=ls, i0=i0, bad=bad, ncut=int(bad.sum()))

def label_sets(C, an, min_cells=2, min_samp=5):
    out = []
    for cc in an["rec"]:
        cells = an["members"][cc]
        if len(cells) < min_cells:
            continue
        m = np.isin(C["cz"], cells)
        if m.sum() < min_samp:
            continue
        comp = np.array([(C["state"][m] == kk).mean() for kk in range(3)])
        out.append(dict(id=cc, n=len(cells), nsamp=int(m.sum()), comp=comp,
                        idx=an["index"].get(cc), mask=m,
                        label=STATE_NAMES[int(comp.argmax())], purity=float(comp.max())))
    out.sort(key=lambda x: -x["n"])
    return out

def reset_edges(C, R, sets):
    """Cut transitions, attributed to the Morse sets of their two ends."""
    lab = {}
    for s_ in sets:
        for c_ in R["an"]["members"][s_["id"]]:
            lab[int(c_)] = s_["label"]
    i0, bad = R["i0"], R["bad"]
    out = {}
    for i in np.flatnonzero(bad):
        a_ = lab.get(int(C["cz"][i]))
        b_ = lab.get(int(C["cz"][i + R["ls"]]))
        if a_ and b_ and a_ != b_:
            out[(a_, b_)] = out.get((a_, b_), 0) + 1
    return out

# =============================================== build both primary charts
charts = {kind: make_chart(kind) for kind in ("pc12", "delay")}
for kind, C in charts.items():
    occ = len(np.unique(C["cz"][C["cz"] >= 0]))
    print(f"chart {kind}: {len(C['Z'])} samples, K={C['K']}, occupied {occ}/{C['g'].N}"
          + ("" if kind == "pc12" else
             f"  (delay: {M_LAGS} copies x {LAG_SPACING} ms = 0-{(M_LAGS-1)*LAG_SPACING} ms)"))
T("charts")

# ===================================== FIG 1: Ulam walkthrough on PC1-PC2
C0 = charts["pc12"]
R0 = build_cmg(C0, TAU, "none", index=False)
g = C0["g"]; Cc = g.centers(); ls = R0["ls"]
i0 = np.arange(len(C0["Z"]) - ls)
src_all, dst_all = C0["cz"][i0], C0["cz"][i0 + ls]
ok = (src_all >= 0) & (dst_all >= 0)
src_all, dst_all = src_all[ok], dst_all[ok]
c_star = np.bincount(src_all).argmax()
land = dst_all[src_all == c_star]
cells_land, n_land = np.unique(land, return_counts=True)
img = R0["A"][c_star].indices
fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.6))
ax[0].scatter(C0["Z"][::3, 0], C0["Z"][::3, 1], s=1.5, color="0.85")
inc = np.flatnonzero(C0["cz"] == c_star)
ax[0].scatter(C0["Z"][inc, 0], C0["Z"][inc, 1], s=8, color="#1e88e5",
              label=f"{len(inc)} samples in cell {c_star}")
fut = inc[inc < len(C0["Z"]) - ls] + ls
ax[0].scatter(C0["Z"][fut, 0], C0["Z"][fut, 1], s=8, color="#d81b60",
              label=f"same samples {TAU} ms later")
for a_ in inc[:60]:
    if a_ + ls < len(C0["Z"]):
        ax[0].annotate("", xy=C0["Z"][a_ + ls], xytext=C0["Z"][a_],
                       arrowprops=dict(arrowstyle="->", color="0.45", lw=0.4))
ax[0].legend(fontsize=8)
ax[0].set(xlabel="PC1", ylabel="PC2", title="step 4: one source cell and where its samples go")
ci, cj = divmod(c_star, g.ny)
li, lj = np.divmod(cells_land, g.ny)
lo_i, hi_i = max(min(ci, li.min()) - 2, 0), max(ci, li.max()) + 3
lo_j, hi_j = max(min(cj, lj.min()) - 2, 0), max(cj, lj.max()) + 3
cnts = np.zeros(g.N); cnts[cells_land] = n_land
im = ax[1].imshow(cnts.reshape(g.nx, g.ny)[lo_i:hi_i, lo_j:hi_j].T, origin="lower",
                  cmap="Blues", extent=(lo_i-.5, hi_i-.5, lo_j-.5, hi_j-.5))
ax[1].scatter([ci], [cj], marker="s", s=140, facecolor="none", edgecolor="#1e88e5", lw=2,
              label="source cell")
plt.colorbar(im, ax=ax[1], label="count of landings"); ax[1].legend(fontsize=8)
ax[1].set_title(f"Ulam COUNT row for cell {c_star}: {len(cells_land)} landing cells,\n"
                f"{len(land)} transitions")
padm = np.zeros(g.N); padm[img] = 1
ax[2].imshow(padm.reshape(g.nx, g.ny)[lo_i:hi_i, lo_j:hi_j].T, origin="lower", cmap="Greys",
             vmin=0, vmax=1.6, extent=(lo_i-.5, hi_i-.5, lo_j-.5, hi_j-.5))
ax[2].scatter([ci], [cj], marker="s", s=140, facecolor="none", edgecolor="#1e88e5", lw=2)
ax[2].set_title(f"boolean row of A: F(cell {c_star})\n"
                f"{len(cells_land)} landing + padding = {len(img)} cells, counts dropped")
for a in ax[1:]:
    a.set_xlabel("PC1 cell index"); a.set_ylabel("PC2 cell index")
fig.suptitle(f"Hippocampus: the Ulam matrix on the (PC1, PC2) chart  "
             f"(b={B} ms, step={SSTEP} ms, lag={TAU} ms, K={C0['K']}, {NX}x{NX} grid; "
             f"A has {R0['A'].nnz} nonzeros of {g.N**2:,})")
fig.tight_layout(); fig.savefig(OUT + "hippo_ulam_walkthrough.png", dpi=120); plt.close(fig)
T("fig1")

# ============================================ FIG 2: charts, states, MUA
fig = plt.figure(figsize=(18, 10.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1.35, 1, 1])
for j, kind in enumerate(("pc12", "delay")):
    C = charts[kind]
    axc = fig.add_subplot(gs[0, j])
    for kk, nm in enumerate(STATE_NAMES):
        m = C["state"] == kk
        axc.scatter(C["Z"][m, 0], C["Z"][m, 1], s=3 if kk < 2 else 14, color=SCOL[nm],
                    alpha=.45 if kk < 2 else .95, label=f"{nm} ({m.mean():.1%})", zorder=kk)
    axc.legend(fontsize=8, markerscale=2)
    ttl = "(PC1, PC2)" if kind == "pc12" else f"delay chart ({M_LAGS} copies, 0-300 ms)"
    axc.set(xlabel="axis 1", ylabel="axis 2", title=f"({'ab'[j]}) {ttl}, coloured by TRUE state")
axp = fig.add_subplot(gs[0, 2])
C = charts["delay"]; mw = C["state"] == 0
sc_ = axp.scatter(C["Z"][mw, 0], C["Z"][mw, 1], c=C["pos"][mw], s=3, cmap="hsv")
plt.colorbar(sc_, ax=axp, label="track position s (m)")
axp.set(xlabel="axis 1", ylabel="axis 2", title="(c) delay chart, WALK samples by position")
axe = fig.add_subplot(gs[0, 3])
for kind, col in (("pc12", "#37474f"), ("delay", "#d81b60")):
    axe.plot(range(1, 31), charts[kind]["ev"][:30], "o-", ms=3, color=col,
             label=f"{kind} (K={charts[kind]['K']})")
axe.set(xlabel="component", ylabel="eigenvalue", yscale="log",
        title="(d) correlation spectrum"); axe.legend(fontsize=8)
axt = fig.add_subplot(gs[1, :])
tt_s = np.arange(n_steps) / 1000.0
axt.plot(tt_s, state_ms, lw=.8, color="k")
axt.set_yticks([0, 1, 2]); axt.set_yticklabels(STATE_NAMES)
for s_ in swr_on: axt.axvline(s_ / 1000, color="#d81b60", lw=.6, alpha=.7)
axt.set(ylabel="true state", title="(e) true state (red = SWR onsets, the guard)")
axm = fig.add_subplot(gs[2, :], sharex=axt)
axm.plot(tt_s, mua_s, lw=.4, color="#37474f")
axm.axhline(det_th, color="#d81b60", ls="--", lw=.8, label=f"guard threshold ({sds[best]:.1f} SD)")
axm.set(xlabel="time (s)", ylabel="MUA (Hz/neuron)", yscale="log",
        title=f"(f) population rate; detection P={pr[best]:.2f} R={rc[best]:.2f}")
axm.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "hippo_chart_states.png", dpi=120); plt.close(fig)
T("fig2")

# ====================================================== CMG under routings
ROUTINGS = ["none", "oracle", "swr", "det"]
TITLES = {"none": "no routing", "oracle": "oracle routing\n(within-state only)",
          "swr": "SWR guards routed\n(true times)", "det": "SWR guards DETECTED\n(from MUA)"}
res = {}
for kind, C in charts.items():
    for rt in ROUTINGS:
        R = build_cmg(C, TAU, rt, index=True)
        sets = label_sets(C, R["an"])
        R["sets"] = sets
        R["resets"] = reset_edges(C, R, sets)
        res[(kind, rt)] = R
        got = sorted({s_["label"] for s_ in sets if s_["purity"] >= 0.5})
        print(f"\n[{kind} / {rt}] cut {R['ncut']}; {len(R['an']['rec'])} rec comps, "
              f"{len(sets)} sets; states as sets: {got}")
        for s_ in sets[:6]:
            comp = " ".join(f"{STATE_NAMES[kk][0]}{s_['comp'][kk]:.2f}" for kk in range(3))
            print(f"    {s_['n']:4d} cells {s_['nsamp']:6d} samp  idx {s_['idx']}  "
                  f"{comp}  -> {s_['label']} ({s_['purity']:.0%})")
        if R["resets"]:
            print("     reset edges:", dict(R["resets"]))
T("CMG")

# ========================================== FIG 3: Morse sets + CMG + time
def draw_graph(ax, sets, hasse, resets, title):
    ids = [s_["id"] for s_ in sets]
    pred = {i: [a for a, b_ in hasse if b_ == i and a in ids] for i in ids}
    depth = {}
    def dep(i):
        if i not in depth:
            depth[i] = 0 if not pred[i] else 1 + max(dep(p) for p in pred[i])
        return depth[i]
    for i in ids: dep(i)
    lv = {}
    for i in ids: lv.setdefault(depth[i], []).append(i)
    md = max(lv) if lv else 0
    pos_ = {}
    for dd, ns in lv.items():
        for t_, i in enumerate(ns):
            pos_[i] = ((t_ + 1) / (len(ns) + 1), md - dd)
    for a, b_ in hasse:
        if a in pos_ and b_ in pos_:
            ax.add_patch(FancyArrowPatch(pos_[a], pos_[b_], arrowstyle="-|>", mutation_scale=12,
                                         color="#546e7a", shrinkA=25, shrinkB=25, lw=1.0))
    byl = {}
    for s_ in sets: byl.setdefault(s_["label"], s_["id"])
    for (a_, b_), cnt in (resets or {}).items():
        if a_ in byl and b_ in byl and byl[a_] in pos_ and byl[b_] in pos_:
            ax.add_patch(FancyArrowPatch(pos_[byl[a_]], pos_[byl[b_]], arrowstyle="-|>",
                                         mutation_scale=12, color="#d81b60", lw=1.3, ls="--",
                                         shrinkA=25, shrinkB=25,
                                         connectionstyle="arc3,rad=0.35"))
    for s_ in sets:
        ax.scatter(*pos_[s_["id"]], s=1500, color=SCOL[s_["label"]], edgecolors="k", zorder=3)
        ax.annotate(f"{s_['label']}\n{s_['idx']}\n{s_['purity']:.0%}", pos_[s_["id"]],
                    ha="center", va="center", fontsize=7, zorder=4)
    ax.set(xlim=(0, 1), ylim=(-.7, md + .7)); ax.axis("off"); ax.set_title(title, fontsize=9.5)

for kind in ("pc12", "delay"):
    C = charts[kind]; Cc = C["g"].centers()
    fig, axes = plt.subplots(3, 4, figsize=(19, 12),
                             gridspec_kw=dict(height_ratios=[1.4, 1, 1]))
    for j, rt in enumerate(ROUTINGS):
        R = res[(kind, rt)]
        ax = axes[0, j]
        ax.scatter(C["Z"][::3, 0], C["Z"][::3, 1], s=1, color="0.9")
        for s_ in R["sets"][:8]:
            cells = np.array(R["an"]["members"][s_["id"]])
            ax.scatter(Cc[cells, 0], Cc[cells, 1], s=10, marker="s",
                       color=SCOL[s_["label"]], alpha=.85)
        ax.set(xlabel="axis 1", ylabel="axis 2")
        ax.set_title(f"{TITLES[rt]}\n{len(R['sets'])} Morse sets", fontsize=9.5)
        draw_graph(axes[1, j], R["sets"][:8], R["an"]["hasse"], R["resets"],
                   "Conley-Morse graph (dashed = reset)")
        axl = axes[2, j]
        axl.plot(C["tt"], C["state"], lw=.7, color="k")
        for si, s_ in enumerate(R["sets"][:8]):
            m = s_["mask"]
            axl.scatter(C["tt"][m], np.full(m.sum(), -0.55 - 0.3 * si), s=.8,
                        color=SCOL[s_["label"]])
        axl.set_yticks([0, 1, 2]); axl.set_yticklabels(STATE_NAMES, fontsize=7)
        axl.set(xlabel="time (s)", xlim=(50, 75), ylim=(-0.55 - 0.3 * 8, 2.4))
        axl.set_title("true state (line) vs Morse set membership (dots)", fontsize=9)
    ttl = "(PC1, PC2) chart" if kind == "pc12" else \
          f"delay chart ({M_LAGS} copies at {LAG_SPACING} ms, out to {(M_LAGS-1)*LAG_SPACING} ms)"
    fig.suptitle(f"Hippocampus Conley-Morse decoding - {ttl}  "
                 f"(b={B} ms, step={SSTEP} ms, lag={TAU} ms)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT + f"hippo_cmg_{kind}.png", dpi=115); plt.close(fig)
T("fig3")

# ================================================= FIG 4: guard assessment
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ax[0].plot(sds, pr, "o-", color="#1e88e5", label="precision")
ax[0].plot(sds, rc, "s-", color="#d81b60", label="recall")
ax[0].axvline(sds[best], color="k", ls=":", lw=.8)
ax[0].set(xlabel="threshold (SDs above STILL mean MUA)", ylabel="vs true SWR onsets",
          title=f"(a) SWR guard detection\nbest F1 at {sds[best]:.1f} SD: "
                f"P={pr[best]:.2f} R={rc[best]:.2f}")
ax[0].legend(fontsize=8)
W = 400
seg = np.array([mua_s[o-W:o+W] for o in swr_on if o-W >= 0 and o+W < n_steps])
ax[1].plot(np.arange(-W, W)/1000., seg.T, lw=.3, color="0.75")
ax[1].plot(np.arange(-W, W)/1000., seg.mean(0), lw=2, color="#d81b60",
           label=f"mean of {len(seg)} SWRs")
ax[1].axhline(det_th, color="k", ls="--", lw=.8, label="threshold")
ax[1].axvline(0, color="k", lw=.8)
ax[1].set(xlabel="time from true SWR onset (s)", ylabel="MUA (Hz/neuron)", yscale="log",
          title="(b) the guard in the population rate"); ax[1].legend(fontsize=8)
C = charts["delay"]; lsj = max(1, int(round(TAU / C["s_ms"])))
disp = np.linalg.norm(C["Z"][lsj:] - C["Z"][:-lsj], axis=1)
scale = np.linalg.norm(C["Z"].max(0) - C["Z"].min(0))
near = np.zeros(len(disp), bool)
for o in swr_on / 1000.0:
    near |= np.abs(C["tt"][:-lsj] - o) < 0.06
ax[2].hist(disp[~near]/scale, bins=60, color="0.75", density=True, label="ordinary steps")
ax[2].hist(disp[near]/scale, bins=25, color="#d81b60", density=True, alpha=.8,
           label="steps at SWR onset")
ax[2].set(xlabel=f"chart displacement per {TAU} ms / chart diameter", ylabel="density",
          yscale="log", title="(c) SWR onset is a jump in the chart")
ax[2].legend(fontsize=8)
fig.suptitle("The guard: STILL -> SWR is fast, large and detectable", fontsize=12)
fig.tight_layout(); fig.savefig(OUT + "hippo_guard.png", dpi=120); plt.close(fig)
T("fig4")

# ================================================================= sweep
B_SET = [25, 50, 100, 200]
TAU_SET = [25, 50, 100, 200, 300]
rows = []
for b_ in B_SET:
    s_ = max(dc.STEP_MS, b_ // 2)
    for kind in ("pc12", "delay"):
        Cb = make_chart(kind, b_, int(s_))
        for tau_ in TAU_SET:
            if tau_ < s_: continue
            for rt in ROUTINGS:
                R = build_cmg(Cb, tau_, rt, index=True)
                sets = label_sets(Cb, R["an"])
                got = sorted({x["label"] for x in sets if x["purity"] >= 0.5})
                rows.append(dict(b=b_, tau=tau_, kind=kind, routing=rt, K=Cb["K"],
                                 nsets=len(sets), got=got, n=len(got)))
    T(f"sweep b={b_}")

fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
for i, kind in enumerate(("pc12", "delay")):
    for j, rt in enumerate(ROUTINGS):
        M = np.full((len(B_SET), len(TAU_SET)), np.nan)
        for r in rows:
            if r["kind"] != kind or r["routing"] != rt: continue
            M[B_SET.index(r["b"]), TAU_SET.index(r["tau"])] = r["n"]
        axes[i, j].imshow(M, cmap="RdYlGn", vmin=0, vmax=3, aspect="auto")
        for a_ in range(len(B_SET)):
            for b2 in range(len(TAU_SET)):
                r = next((r for r in rows if r["kind"] == kind and r["routing"] == rt
                          and r["b"] == B_SET[a_] and r["tau"] == TAU_SET[b2]), None)
                if r is None:
                    axes[i, j].add_patch(plt.Rectangle((b2-.5, a_-.5), 1, 1, color="#cfd8dc"))
                else:
                    axes[i, j].text(b2, a_, "".join(x[0] for x in r["got"]) + f"\n{r['nsets']}",
                                    ha="center", va="center", fontsize=7)
        axes[i, j].set_xticks(range(len(TAU_SET))); axes[i, j].set_xticklabels(TAU_SET)
        axes[i, j].set_yticks(range(len(B_SET))); axes[i, j].set_yticklabels(B_SET)
        axes[i, j].set(xlabel="lag tau (ms)", ylabel="bin b (ms)")
        axes[i, j].set_title(f"{kind}: {TITLES[rt]}".replace("\n", " "), fontsize=9)
fig.suptitle("States recovered as distinct Morse sets (W=WALK, S=STILL, R=SWR; "
             "letters = which; number = total sets)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT + "hippo_sweep.png", dpi=115); plt.close(fig)
json.dump(rows, open(OUT + "hippo_sweep.json", "w"), indent=1, default=str)
print("\nsettings recovering all three states:")
for kind in ("pc12", "delay"):
    for rt in ROUTINGS:
        ok = [(r["b"], r["tau"]) for r in rows if r["kind"] == kind
              and r["routing"] == rt and r["n"] == 3]
        print(f"  {kind:5s} {rt:6s}: {len(ok):2d}  {ok}")
T("done")
