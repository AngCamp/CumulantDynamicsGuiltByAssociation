"""
All analyses on the cached hybrid nested-rings simulation (hds_sim.pkl).

FIG 1 hds_elbow.png        eigen-spectrum per bin size (light->dark), elbow K,
                           participation ratio vs bin size
FIG 2 hds_wedge.png        ||x ^ v|| over the elbow-K PCs (20 ms bins), keyed
                           to the switches: columns guard entry / jump /
                           return / guard exit; rows hits / failures; lines =
                           attractor of origin (inner vs outer ring)
FIG 3 hds_poprate.png      same layout, population firing rate
FIG 4 hds_quotients.png    user's quotient figure aligned to the jump (guard
                           firing): Lv, Lp, real vs Poisson control, 5 settings
FIG 5 hds_lipschitz.png    window x step grid: Lv/Lp per regime vs ground
                           truth and noise-only sqrt(3)/dt; jump-step ratios
FIG 6 hds_guard_pcs.png    guard visualization: each PC's score aligned to
                           the jump + each PC's own quotient; step-ratio of
                           every PC across settings (which PCs detect it)
FIG 7 hds_procrustes.png   Procrustes residual to the next WINDOW SIZE, one
                           curve per k = 1..10, + pairwise heat map at elbow k
"""
import sys, time, pickle, warnings
sys.path.insert(0, "/home/claude/hybrid_cmg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hds import decode as dc, neurons as nr

warnings.filterwarnings("ignore")
OUT = "/home/claude/hybrid_cmg/"
D = pickle.load(open(OUT + "hds_sim.pkl", "rb"))
rec, segs = D['rec'], D['segs']
base = rec['base']
mode, guard, ring = rec['mode'], rec['guard'], D['ring']
Ltrue = D['Ltrue']
r_in, r_bar, r_out = D['rings']
G = D['meta']['G']
n_tr = len(segs)
t0 = time.time()

def T(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# ------------------------------------------------------------- events
def merged_visits(gap_s=0.1):
    gap = int(gap_s / base)
    out = []
    jumps = {}
    rets = {}
    for (i, b, k, _) in rec['events']:
        (jumps if k == 'jump' else rets).setdefault(i, []).append(b)
    for tr in range(n_tr):
        g = np.r_[0, guard[tr], 0]
        on = list(np.flatnonzero(np.diff(g) == 1))
        off = list(np.flatnonzero(np.diff(g) == -1) - 1)
        # merge short gaps
        vs = []
        for a, b in zip(on, off):
            if vs and a - vs[-1][1] <= gap:
                vs[-1][1] = b
            else:
                vs.append([a, b])
        J = np.array(sorted(jumps.get(tr, [])))
        R = np.array(sorted(rets.get(tr, [])))
        for a, b in vs:
            if mode[tr, a] != 0:
                continue
            jj = J[(J >= a) & (J <= b + 2)] if len(J) else []
            hit = len(jj) > 0
            jb = int(jj[0]) if hit else None
            rb = None
            if hit and len(R):
                rr = R[R > jb]
                rb = int(rr[0]) if len(rr) else None
            out.append(dict(tr=tr, entry=a, exit=b, hit=hit, jump=jb,
                            ret=rb, ring=int(ring[tr, a])))
    return out

visits = merged_visits()
for r, nm in [(1, 'outer'), (0, 'inner')]:
    vv = [v for v in visits if v['ring'] == r]
    print(f"merged guard visits {nm}: {len(vv)}, hit rate "
          f"{np.mean([v['hit'] for v in vv]):.2f}")
bt = lambda b: (b + 0.5) * base                          # base bin -> seconds

# fine bins (10 ms) and Poisson control
Xf, trf, ttf = dc.fine_bins(segs, base)
Xsur, _, _ = dc.fine_bins(nr.poisson_control(segs, base, 100.0, seed=0), base)
# ground-truth labels at fine resolution (mode/ring at bin centre)
fstep = int(round(dc.STEP_MS / 1000 / base))
def lab_at(arr, tr, tt):
    b = np.clip((tt / base).astype(int), 0, arr.shape[1] - 1)
    return arr[tr, b]
T("fine bins + control ready")

# ============================================================ FIG 1: elbow
BINS = [10, 20, 30, 50, 80, 120, 200, 400]
cm = plt.cm.Blues(np.linspace(0.3, 1.0, len(BINS)))
spec, Ks, PR, PR30 = [], [], [], []
for w in BINS:
    Xw, tt, tr = dc.window_counts(Xf, trf, ttf, w, w)
    ev, vec, Z = dc.corr_svd(Xw)
    spec.append(ev[:30]); Ks.append(dc.elbow_k(ev))
    PR.append(dc.participation_ratio(ev)); PR30.append(dc.participation_ratio(ev, 30))
    del Z
T(f"elbow: K per bin {dict(zip(BINS, Ks))}")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for i, w in enumerate(BINS):
    ax[0].plot(range(1, 31), spec[i], '-o', ms=3, color=cm[i], label=f"{w} ms (K={Ks[i]})")
    ax[0].plot(Ks[i], spec[i][Ks[i]-1], 'D', ms=8, color=cm[i], mec='k')
ax[0].set(xlabel="component", ylabel="eigenvalue (correlation matrix)",
          title="spectra by bin size (diamond = elbow K)", yscale='log')
ax[0].legend(fontsize=7, ncol=2)
for i, w in enumerate(BINS):
    ax[1].plot(w, PR[i], 'o', ms=9, color=cm[i], mec='k')
    ax[2].plot(w, PR30[i], 'o', ms=9, color=cm[i], mec='k')
ax[1].plot(BINS, PR, '-', color='0.6', zorder=0)
ax[2].plot(BINS, PR30, '-', color='0.6', zorder=0)
ax2 = ax[2].twinx(); ax2.plot(BINS, Ks, 's--', color='#d95f0e'); ax2.set_ylabel("elbow K", color='#d95f0e')
ax[1].set(xscale='log', xlabel="bin (ms)", ylabel="participation ratio (full spectrum)",
          title="participation ratio, all 1200 components")
ax[2].set(xscale='log', xlabel="bin (ms)", ylabel="participation ratio (top 30)",
          title="participation ratio, top 30 + elbow K")
fig.suptitle("Dimensionality vs bin size (normalized: /pop total, z-scored; correlation-matrix PCA)")
fig.tight_layout(); fig.savefig(OUT + "hds_elbow.png", dpi=125); plt.close(fig)

# ======================================== FIG 2/3: switch-keyed wedge + rate
W20 = 20
Xw, tt20, tr20 = dc.window_counts(Xf, trf, ttf, W20, W20)
x20, K20, _ = dc.scores(Xw)
dt20 = W20 / 1000
xc = x20 - x20.mean(0)
v20 = np.full_like(xc, np.nan)
same = np.r_[tr20[1:] == tr20[:-1], False]
v20[:-1] = (xc[1:] - xc[:-1]) / dt20
v20[~same] = np.nan
nx2, nv2 = (xc ** 2).sum(1), (v20 ** 2).sum(1)
wedge = np.sqrt(np.clip(nx2 * nv2 - (xc * v20).sum(1) ** 2, 0, None))
poprate = Xw.sum(1) / dt20 / Xw.shape[1]                 # mean Hz per neuron
T(f"20 ms scores: K={K20}")

EV_COLS = [("guard entry", 'entry'), ("jump (switch)", 'jump'),
           ("return (gain reset)", 'ret'), ("guard exit", 'exit')]
WIN = (-0.6, 1.2)

def keyed_fig(sig, ylab, fname, title):
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.5), sharex=True)
    for row, hit in enumerate([True, False]):
        for col, (nm, key) in enumerate(EV_COLS):
            ax = axes[row, col]
            for r, rn, c in [(0, 'inner', 'tab:blue'), (1, 'outer', 'tab:red')]:
                vv = [v for v in visits if v['hit'] == hit and v['ring'] == r
                      and v[key] is not None]
                if not vv:
                    continue
                et = np.array([bt(v[key]) for v in vv])
                etr = np.array([v['tr'] for v in vv])
                off, M = dc.aligned(sig, tt20, tr20, et, etr, dt20, WIN)
                mu, se = dc.nanmean_sem(M)
                ax.plot(off, mu, color=c, lw=1.2, label=f"{rn} origin (n={len(vv)})")
                ax.fill_between(off, mu - se, mu + se, color=c, alpha=0.2, lw=0)
            ax.axvline(0, color='0.5', lw=0.8)
            if row == 0:
                ax.set_title(nm, fontsize=10)
            if col == 0:
                ax.set_ylabel(("hits (guard fired)" if hit else "failures (no jump)") + "\n" + ylab)
            ax.legend(fontsize=7, loc='upper left')
            if row == 1:
                ax.set_xlabel("time from event (s)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(); fig.savefig(OUT + fname, dpi=125); plt.close(fig)

keyed_fig(wedge, "||x ^ v||", "hds_wedge.png",
          f"hybrid nested rings: ||x ^ v|| over all {K20} elbow PCs (20 ms bins), keyed to the switches")
keyed_fig(poprate, "pop. rate (Hz/neuron)", "hds_poprate.png",
          "hybrid nested rings: population firing rate, keyed to the switches")
T("wedge + rate figures")

# =========================================== FIG 4: user's quotient figure
SETTINGS = [(10, 10), (20, 20), (50, 50), (100, 100), (100, 10)]
JWIN = (-1.0, 1.5)
hits = [v for v in visits if v['hit']]
jt = np.array([bt(v['jump']) for v in hits]); jtr = np.array([v['tr'] for v in hits])
rows = []
fig, axes = plt.subplots(2, len(SETTINGS), figsize=(3.3 * len(SETTINGS), 5.5), sharex=True)
cache_scores = {}
for col, (w, s) in enumerate(SETTINGS):
    dt_s = s / 1000
    label = f"{w} ms bins" if w == s else f"{w} ms window / {s} ms step"
    for kind, X, ls in (("real", Xf, "-"), ("Poisson control", Xsur, "--")):
        Xw_, tt_, tr_ = dc.window_counts(X, trf, ttf, w, s)
        Lp, Lv, K, xs = dc.quotients(Xw_, tr_, dt_s)
        if kind == "real":
            cache_scores[(w, s)] = (xs, tt_, tr_, K)
        for row, (nm, sig) in enumerate((("|dv|/dt", Lv), ("|dp|/dt", Lp))):
            off, M = dc.aligned(sig, tt_, tr_, jt, jtr, dt_s, JWIN)
            med = dc.nanmed(M)
            b0 = np.nanmedian(med[off < 0])
            axes[row, col].plot(off, med / b0, ls, lw=1.2,
                                color='k' if kind == 'real' else 'tab:orange', label=kind)
            if row == 0:
                step = np.nanmedian(med[(off > 0.05) & (off < 0.5)]) / b0
        rows.append(dict(setting=label, data=kind, K=K,
                         LvLp=np.nanmedian(Lv) / np.nanmedian(Lp),
                         noise=np.sqrt(3) / dt_s if w == s else np.nan, step=step))
    axes[0, col].set_title(label, fontsize=9)
    for row in range(2):
        axes[row, col].axvline(0, color='0.5', lw=0.8)
        axes[row, col].axhline(1, color='0.8', lw=0.8)
axes[0, 0].set_ylabel("|dv|/dt / pre-jump median")
axes[1, 0].set_ylabel("|dp|/dt / pre-jump median")
for ax in axes[-1]:
    ax.set_xlabel("time from guard jump (s)")
axes[0, 0].legend(fontsize=8)
fig.suptitle("hybrid nested rings: jump-aligned quotients, real vs Poisson control (guard hits)")
fig.tight_layout(); fig.savefig(OUT + "hds_quotients.png", dpi=125); plt.close(fig)
print("\nquotient table:")
for r in rows:
    print(f"  {r['setting']:24s} {r['data']:16s} K={r['K']:2d}  Lv/Lp={r['LvLp']:7.1f}  "
          f"noise-only={r['noise']:7.1f}  jump step={r['step']:.2f}")
T("quotient figure")

# ============================================= FIG 5: window x step grid
WS = [10, 20, 50, 100, 200]
SS = [10, 20, 50, 100]
REG = [('base', 'outer'), ('base', 'inner'), ('high', 'outer'), ('high', 'inner')]
truth = {}
for mm, mn in [(0, 'base'), (1, 'high')]:
    for r, rn in [(1, 'outer'), (0, 'inner')]:
        sel = (mode == mm) & (ring == r)
        truth[(mn, rn)] = np.median(Ltrue[sel])
grid = {k: np.full((len(WS), len(SS)), np.nan) for k in REG + ['step_real', 'step_ctrl', 'K']}
for i, w in enumerate(WS):
    for j, s in enumerate(SS):
        if s > w:
            continue
        dt_s = s / 1000
        for kind, X in (("real", Xf), ("ctrl", Xsur)):
            Xw_, tt_, tr_ = dc.window_counts(X, trf, ttf, w, s)
            Lp, Lv, K, _ = dc.quotients(Xw_, tr_, dt_s)
            off, M = dc.aligned(Lv, tt_, tr_, jt, jtr, dt_s, JWIN)
            med = dc.nanmed(M); b0 = np.nanmedian(med[off < 0])
            grid['step_' + kind][i, j] = np.nanmedian(med[(off > 0.05) & (off < 0.5)]) / b0
            if kind == "real":
                grid['K'][i, j] = K
                mo = lab_at(mode, tr_, tt_); ri = lab_at(ring, tr_, tt_)
                for (mn, rn) in REG:
                    sel = (mo == (0 if mn == 'base' else 1)) & (ri == (1 if rn == 'outer' else 0))
                    grid[(mn, rn)][i, j] = np.nanmedian(Lv[sel]) / np.nanmedian(Lp[sel])
        T(f"grid w={w} s={s}")
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for p, (mn, rn) in enumerate(REG):
    ax = axes.flat[p]
    for j, s in enumerate(SS):
        ok = ~np.isnan(grid[(mn, rn)][:, j])
        ax.plot(np.array(WS)[ok], grid[(mn, rn)][ok, j], 'o-', label=f"step {s} ms")
    ax.axhline(truth[(mn, rn)], color='k', lw=2, label=f"true L = {truth[(mn, rn)]:.1f}")
    for j, s in enumerate(SS):
        ax.plot(WS, [np.sqrt(3) / (s / 1000)] * len(WS), ':', color=f"C{j}", lw=1)
    ax.set(xscale='log', yscale='log', xlabel="window (ms)", ylabel="median Lv / Lp",
           title=f"{mn} regime, {rn} ring\n(dotted: noise-only sqrt(3)/step)")
    ax.legend(fontsize=7)
for p, (key, nm) in enumerate([('step_real', 'jump step Lv post/pre: real'),
                               ('step_ctrl', 'jump step: Poisson control'),
                               ('K', 'elbow K')]):
    ax = axes.flat[4 + p]
    im = ax.imshow(grid[key], cmap='viridis', aspect='auto')
    for a in range(len(WS)):
        for b in range(len(SS)):
            if not np.isnan(grid[key][a, b]):
                ax.text(b, a, f"{grid[key][a, b]:.2f}" if key != 'K' else f"{int(grid[key][a, b])}",
                        ha='center', va='center', color='w', fontsize=9)
    ax.set_xticks(range(len(SS))); ax.set_xticklabels(SS)
    ax.set_yticks(range(len(WS))); ax.set_yticklabels(WS)
    ax.set(xlabel="step (ms)", ylabel="window (ms)", title=nm)
    fig.colorbar(im, ax=ax)
ax = axes.flat[7]
ratio = grid['step_real'] / grid['step_ctrl']
im = ax.imshow(ratio, cmap='RdBu_r', vmin=0.5, vmax=1.5, aspect='auto')
for a in range(len(WS)):
    for b in range(len(SS)):
        if not np.isnan(ratio[a, b]):
            ax.text(b, a, f"{ratio[a, b]:.2f}", ha='center', va='center', fontsize=9)
ax.set_xticks(range(len(SS))); ax.set_xticklabels(SS)
ax.set_yticks(range(len(WS))); ax.set_yticklabels(WS)
ax.set(xlabel="step (ms)", ylabel="window (ms)", title="real / control jump step\n(>1: switch detected beyond Poisson)")
fig.colorbar(im, ax=ax)
fig.suptitle(f"Lipschitz estimates (Lv/Lp) vs window and step, against ground truth (gain G={G:g})")
fig.tight_layout(); fig.savefig(OUT + "hds_lipschitz.png", dpi=120); plt.close(fig)
print("\ntrue L:", {f"{a}/{b}": round(v, 1) for (a, b), v in truth.items()})

# ====================================== FIG 6: guard seen by individual PCs
xs, ttq, trq, Kq = cache_scores[(20, 20)]
dtq = 0.02
fig = plt.figure(figsize=(18, 2.2 * Kq + 3))
gs = fig.add_gridspec(Kq + 1, 4, height_ratios=[1] * Kq + [1.6])
inner_h = [v for v in hits if v['ring'] == 0]; outer_h = [v for v in hits if v['ring'] == 1]
for p in range(Kq):
    sc_ = xs[:, p]
    Lp_i = np.full(len(sc_), np.nan); Lp_i[1:] = np.abs(np.diff(sc_)) / dtq
    Lv_i = np.full(len(sc_), np.nan); Lv_i[1:-1] = np.abs(sc_[2:] - 2 * sc_[1:-1] + sc_[:-2]) / dtq ** 2
    sp_ = np.r_[False, trq[1:] == trq[:-1]]; sn_ = np.r_[trq[:-1] == trq[1:], False]
    Lp_i[~sp_] = np.nan; Lv_i[~(sp_ & sn_)] = np.nan
    for col, (sig, nm) in enumerate([(sc_, f"PC{p+1} score"), (Lp_i, f"PC{p+1} |dp|/dt"),
                                     (Lv_i, f"PC{p+1} |dv|/dt")]):
        ax = fig.add_subplot(gs[p, col])
        for vv, c, rn in [(inner_h, 'tab:blue', 'inner->outer'), (outer_h, 'tab:red', 'outer->inner')]:
            et = np.array([bt(v['jump']) for v in vv]); etr = np.array([v['tr'] for v in vv])
            off, M = dc.aligned(sig, ttq, trq, et, etr, dtq, (-0.5, 0.8))
            if col == 0:
                for k_ in range(min(12, len(M))):
                    ax.plot(off, M[k_], color=c, lw=0.3, alpha=0.4)
                mu, se = dc.nanmean_sem(M)
            else:
                mu = dc.nanmed(M); b0 = np.nanmedian(mu[off < 0]); mu = mu / b0; se = 0 * mu
            ax.plot(off, mu, color=c, lw=1.5, label=f"{rn} (n={len(vv)})")
            ax.fill_between(off, mu - se, mu + se, color=c, alpha=0.2, lw=0)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_ylabel(nm + ("" if col == 0 else "\n/ pre median"), fontsize=8)
        if p == 0:
            ax.set_title(["PC score at guard jumps (faint: single events)",
                          "per-PC position quotient", "per-PC velocity quotient"][col], fontsize=10)
            ax.legend(fontsize=7)
        if p == Kq - 1:
            ax.set_xlabel("time from guard jump (s)")
# column 4: per-PC detection ratio vs setting
ax = fig.add_subplot(gs[:Kq, 3])
det = np.full((Kq, len(SETTINGS)), np.nan)
for j, (w, s) in enumerate(SETTINGS):
    xs_, tt_, tr_, K_ = cache_scores[(w, s)]
    dt_ = s / 1000
    for p in range(min(Kq, K_)):
        sc_ = xs_[:, p]
        Lp_i = np.full(len(sc_), np.nan); Lp_i[1:] = np.abs(np.diff(sc_)) / dt_
        Lp_i[~np.r_[False, tr_[1:] == tr_[:-1]]] = np.nan
        off, M = dc.aligned(Lp_i, tt_, tr_, jt, jtr, dt_, (-1.0, 1.0))
        med = dc.nanmed(M)
        det[p, j] = np.nanmedian(med[(off >= 0) & (off < 0.15)]) / np.nanmedian(med[off < -0.1])
im = ax.imshow(det, cmap='magma', aspect='auto')
for a in range(det.shape[0]):
    for b in range(det.shape[1]):
        if not np.isnan(det[a, b]):
            ax.text(b, a, f"{det[a, b]:.2f}", ha='center', va='center',
                    color='w' if det[a, b] < np.nanmax(det) * 0.7 else 'k', fontsize=9)
ax.set_xticks(range(len(SETTINGS))); ax.set_xticklabels([f"{w}/{s}" for w, s in SETTINGS], rotation=30)
ax.set_yticks(range(Kq)); ax.set_yticklabels([f"PC{p+1}" for p in range(Kq)])
ax.set(xlabel="window/step (ms)", title="guard detection per PC\n|dp|/dt in first 150 ms / pre-jump")
fig.colorbar(im, ax=ax)
# bottom row: example trial, PC traces with guard/jump/regime shading
axE = fig.add_subplot(gs[Kq, :])
ex = max(hits, key=lambda v: v['jump'])['tr'] if hits else 0
m = trq == ex
tq = ttq[m]
for p in range(Kq):
    axE.plot(tq, xs[m, p] / (np.std(xs[:, p]) + 1e-9) - 3 * p, lw=0.8, label=f"PC{p+1}")
gb = guard[ex]; mb = mode[ex]; tb = (np.arange(len(gb)) + 0.5) * base
axE.fill_between(tb, -3 * Kq, 3, where=gb > 0, color='#ffd54f', alpha=0.35, lw=0, label='guard active')
axE.fill_between(tb, -3 * Kq, 3, where=mb > 0, color='#ef9a9a', alpha=0.35, lw=0, label='gain-up regime')
for v in hits:
    if v['tr'] == ex:
        axE.axvline(bt(v['jump']), color='k', lw=1)
axE.set_xlim(tq[0] + 2, tq[0] + 12)
axE.set(xlabel="time in trial (s)", ylabel="PC score (z, offset)",
        title=f"example trial {ex}: PC traces, guard (yellow), gain-up regime (red), jumps (black)")
axE.legend(fontsize=7, ncol=Kq + 2, loc='lower left')
fig.suptitle(f"Guard visualization: jumps between attractors as seen by each of the K={Kq} PCs (20 ms bins)",
             fontsize=13)
fig.tight_layout(); fig.savefig(OUT + "hds_guard_pcs.png", dpi=115); plt.close(fig)
T("guard PC figure")

# ======================================================= FIG 7: Procrustes
WINS = [20, 30, 50, 70, 110, 170, 260, 400]
K_MODEL = 10
half = max(WINS) // 2 // int(dc.STEP_MS)
centers = {}
for ti in np.unique(trf):
    nfin = (trf == ti).sum()
    centers[ti] = np.arange(half, nfin - half)
var_ = []
ks_low = None
for w in WINS:
    Xw_, tt_, tr_ = dc.window_counts(Xf, trf, ttf, w, dc.STEP_MS, centers=centers)
    ev, vec, Z = dc.corr_svd(Xw_)
    if ks_low is None:
        ks_low = dc.elbow_k(ev)
    sc_ = Z @ vec[:, :K_MODEL]
    var_.append(sc_ / sc_.std(0))
    del Z
    T(f"procrustes window {w}")
n_w = len(WINS)
proc_next = np.full((K_MODEL, n_w - 1), np.nan)
for k in range(1, K_MODEL + 1):
    for li in range(n_w - 1):
        proc_next[k - 1, li] = dc.procrustes_resid(var_[li][:, :k], var_[li + 1][:, :k])
kk = max(ks_low, 1)
proc_mat = np.zeros((n_w, n_w))
for a in range(n_w):
    for b in range(n_w):
        if a != b:
            proc_mat[a, b] = dc.procrustes_resid(var_[a][:, :kk], var_[b][:, :kk])
fig, (axP, axH) = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw=dict(width_ratios=[1.4, 1]))
ccols = plt.cm.YlOrRd(np.linspace(0.35, 0.95, K_MODEL))
mid = np.sqrt(np.array(WINS[:-1]) * np.array(WINS[1:]))
for k in range(K_MODEL):
    axP.plot(mid, proc_next[k], '-o', ms=3.5, color=ccols[k], lw=1.6, label=f"k={k+1}")
axP.set_xscale('log')
axP.set_xlabel("window size (ms, log)  —  'timescale'")
axP.set_ylabel("Procrustes residual to next window\n(rotation + stretch)")
axP.set_title("Rotation of the k-component PCA embedding\ndown to the next window size")
axP.legend(fontsize=7, ncol=2, loc='upper left')
im = axH.imshow(proc_mat, cmap='magma_r', origin='lower')
axH.set_xticks(range(n_w)); axH.set_xticklabels(WINS, fontsize=8)
axH.set_yticks(range(n_w)); axH.set_yticklabels(WINS, fontsize=8)
axH.set_xlabel("to window (ms)"); axH.set_ylabel("from window (ms)")
axH.set_title(f"pairwise Procrustes distance\n(k = elbow at {WINS[0]} ms = {kk} components)")
fig.colorbar(im, ax=axH, label="residual")
fig.suptitle("PCA across window sizes (common 10 ms step & time stamps) — hybrid nested rings", fontsize=13, y=1.03)
fig.savefig(OUT + "hds_procrustes.png", dpi=125, bbox_inches='tight'); plt.close(fig)
print("\nProcrustes next-window residuals (rows k=1..10):")
print(np.round(proc_next, 3))
T("done")
