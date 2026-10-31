"""
Conley-Morse graphs: ground truth from the true flow, and decoded from spikes.

PART 1  GROUND TRUTH (analytic field on the true latent plane)
  GT-flow    base vector field only. Expected: source at the origin,
             inner ring (attractor), repelling barrier circle, outer ring.
  GT-hybrid  state = plane x {base, high}. Edges: base flow; guard cells
             (guard neurons above threshold) also jump to the other ring in
             the high copy; high cells flow at gain G and may return to base
             at any time (exponential dwell). Jump edges are what make the
             two rings one hybrid recurrent class.

PART 2  DECODED from spikes, two simulated scenarios with the same neurons:
  flow     hazard = 0 (no guard firing)
  hybrid   hazard = 3/s (the analyses you already have)
  pipeline: window counts (b, s) -> normalize -> correlation PCA, elbow K
            -> delay embedding (m = 3, lag ell) -> 2D chart -> Ulam map on a
            64x64 grid -> Morse sets (SCC) -> Conley index
  hybrid routing: transitions whose spike windows straddle a guard event are
  removed from the flow map; each event becomes a RESET EDGE between the
  Morse sets before and after it.
     oracle   true jump times
     detected events from the latent velocity quotient |dv| at scale b
              (median + 6 MAD), scored against the true jumps

Data support only contains the attractors (trajectories never sit on the
source or the barrier), so the decodable part of the ground truth is:
  flow   : two separate Morse sets (inner, outer), each (1,1,0), no edges
  hybrid : the same two flow Morse sets + reset edges in BOTH directions
"""
import sys, time, pickle, warnings
sys.path.insert(0, "/home/claude/hybrid_cmg")
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import cmg_lib as cl
from hds import scenarios as sc, decode as dc, dynamics as dy

warnings.filterwarnings("ignore")
OUT = "/home/claude/hybrid_cmg/"
t0 = time.time()
def T(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

system, readout, x0, meta = sc.nested_rings_hybrid(n_trials=24)
rings = meta['rings']
G = meta['G']
R_BAR = rings.r_bar

# ============================================================ helpers
def draw_graph(ax, nodes, edges, labels, colors, title, extra_edges=()):
    """nodes: list of ids; edges: Hasse edges (a above b); extra_edges:
    (a, b, label) drawn dashed (reset edges)."""
    pred = {n: [a for a, b in edges if b == n] for n in nodes}
    depth = {}
    def dep(n):
        if n not in depth:
            depth[n] = 0 if not pred[n] else 1 + max(dep(p) for p in pred[n])
        return depth[n]
    for n in nodes: dep(n)
    lv = {}
    for n in nodes: lv.setdefault(depth[n], []).append(n)
    md = max(lv) if lv else 0
    pos = {}
    for d_, ns in lv.items():
        for i, n in enumerate(ns):
            pos[n] = ((i + 1) / (len(ns) + 1), md - d_)
    for a, b in edges:
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle='-|>', mutation_scale=14,
                                     color='#546e7a', shrinkA=24, shrinkB=24, lw=1.2))
    for a, b, lab in extra_edges:
        if a in pos and b in pos:
            ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle='-|>', mutation_scale=14,
                                         color='#d81b60', shrinkA=24, shrinkB=24, lw=1.4,
                                         ls='--', connectionstyle="arc3,rad=0.35"))
            mx = (pos[a][0] + pos[b][0]) / 2 + (0.08 if pos[a][0] < pos[b][0] else -0.08)
            ax.text(mx, (pos[a][1] + pos[b][1]) / 2 + 0.25, lab, color='#d81b60',
                    fontsize=7, ha='center')
    for n in nodes:
        ax.scatter(*pos[n], s=1500, color=colors[n], edgecolors='k', zorder=3)
        ax.annotate(labels[n], pos[n], ha='center', va='center', fontsize=7.5, zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(-0.7, md + 0.7); ax.axis('off'); ax.set_title(title, fontsize=10)

# ============================================================ PART 1: GT
NG = 150
LIM = 1.3
TAU = 1.0     # long enough that radial contraction (2.7/s) beats 1-cell padding

def cell_samples_plane(nx, lim, k=3):
    i, j = np.divmod(np.arange(nx * nx), nx)
    h = 2 * lim / nx
    f = (np.arange(k) + 0.5) / k
    fx, fy = np.meshgrid(f, f)
    xs = -lim + (i[:, None] + fx.ravel()[None]) * h
    ys = -lim + (j[:, None] + fy.ravel()[None]) * h
    return xs, ys, h

def flow(P, gain, tau=TAU, dt=0.002):
    X = np.column_stack([P, np.zeros(len(P))])
    f = lambda Y: system.field(Y) * gain
    for _ in range(int(round(tau / dt))):
        k1 = f(X); k2 = f(X + dt / 2 * k1); k3 = f(X + dt / 2 * k2); k4 = f(X + dt * k3)
        X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X[:, :2]

def to_cell(P, nx, lim):
    h = 2 * lim / nx
    i = np.floor((P[:, 0] + lim) / h).astype(int)
    j = np.floor((P[:, 1] + lim) / h).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < nx)
    return np.where(ok, i * nx + j, -1)

xs, ys, h = cell_samples_plane(NG, LIM)
ncell = NG * NG
S = np.column_stack([xs.ravel(), ys.ravel()])
src_cell = np.repeat(np.arange(ncell), xs.shape[1])

def edges_from(P_img, src, offset_src=0, offset_dst=0):
    c = to_cell(P_img, NG, LIM)
    ok = c >= 0
    return src[ok] + offset_src, c[ok] + offset_dst

def padded_matrix(rows, cols, n_total, nx_copies):
    """pad targets by one cell within the same copy."""
    R, C = [], []
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            copy, cc = np.divmod(cols, ncell)
            ci, cj = np.divmod(cc, NG)
            ni, nj = ci + du, cj + dv
            ok = (ni >= 0) & (ni < NG) & (nj >= 0) & (nj < NG)
            R.append(rows[ok]); C.append(copy[ok] * ncell + ni[ok] * NG + nj[ok])
    R = np.concatenate(R); C = np.concatenate(C)
    A = sp.csr_matrix((np.ones(len(R), bool), (R, C)), shape=(n_total, n_total))
    A.sum_duplicates()
    return A

# --- GT-flow
img1 = flow(S, 1.0)
r1, c1 = edges_from(img1, src_cell)
A_gtf = padded_matrix(r1, c1, ncell, 1)
an_gtf = cl.analyze(A_gtf, NG, "GT-flow", index=True, verbose=True)
T("GT flow")

# --- GT-hybrid: two copies (0 base, 1 high)
grp, theta = meta['guard_group'], meta['guard_theta']
cent = np.column_stack([-LIM + (np.divmod(np.arange(ncell), NG)[0] + .5) * h,
                        -LIM + (np.divmod(np.arange(ncell), NG)[1] + .5) * h])
lam_c = readout.expected_rates(np.column_stack([cent, np.zeros(ncell)]), np.zeros(ncell),
                               np.zeros(ncell, int), None, grp).mean(1)
guard_cell = lam_c > theta
swap = sc.ring_swap(rings.r_in, rings.r_out, rings.r_bar)
rows, cols = [r1], [c1]                                    # base flow
gs = guard_cell[src_cell]
Sj = swap(np.column_stack([S[gs], np.zeros(gs.sum())]), None)[:, :2]
for P in (Sj, flow(Sj, G)):                               # jump (+ flow) -> high copy
    r, c = edges_from(P, src_cell[gs], 0, ncell); rows.append(r); cols.append(c)
imgG = flow(S, G)
r, c = edges_from(imgG, src_cell, ncell, ncell); rows.append(r); cols.append(c)   # high flow
for P in (S, img1, imgG):                                  # return to base at any time
    r, c = edges_from(P, src_cell, ncell, 0); rows.append(r); cols.append(c)
A_gth = padded_matrix(np.concatenate(rows), np.concatenate(cols), 2 * ncell, 2)
comp_h, rec_h, mem_h, hasse_h, _ = cl.morse_graph(A_gth)
T(f"GT hybrid: {len(rec_h)} Morse sets")

def describe_gt(mem, cells):
    cs = np.array(cells)
    copy, cc = np.divmod(cs, ncell)
    P = cent[cc]
    rho = np.linalg.norm(P, axis=1)
    return copy, rho

def gt_name(rho):
    m = np.median(rho)
    if m < 0.25: return "origin source"
    if abs(m - rings.r_in) < 0.06: return "inner ring"
    if abs(m - rings.r_bar) < 0.04: return "barrier circle"
    if abs(m - rings.r_out) < 0.06: return "outer ring"
    return f"set rho~{m:.2f}"

print("\nGT-flow Morse sets:")
gtf_info = {}
for cc_ in an_gtf['rec']:
    _, rho = describe_gt(None, an_gtf['members'][cc_])
    nm = gt_name(rho)
    gtf_info[cc_] = nm
    print(f"   {nm:16s} {len(an_gtf['members'][cc_]):5d} cells  index {an_gtf['index'].get(cc_)}")
print("   Hasse edges:", [(gtf_info[a], gtf_info[b]) for a, b in an_gtf['hasse']])
print("GT-hybrid Morse sets:")
gth_info = {}
for cc_ in rec_h:
    copy, rho = describe_gt(None, mem_h[cc_])
    parts = []
    for k, kn in ((0, 'base'), (1, 'high')):
        rk = rho[copy == k]
        for nm, c0, wdt in (('inner', rings.r_in, 0.06), ('barrier', rings.r_bar, 0.035),
                            ('outer', rings.r_out, 0.06)):
            n_ = int((np.abs(rk - c0) < wdt).sum())
            if n_ >= 10:
                parts.append(f"{nm}/{kn}")
    gth_info[cc_] = " + ".join(parts) if parts else "other"
    print(f"   [{len(mem_h[cc_]):5d} cells] {gth_info[cc_]}")
print("   Hasse edges:", [(gth_info[a], gth_info[b]) for a, b in hasse_h])

# --- figure: ground truth
fig = plt.figure(figsize=(17, 9.5))
gsf = fig.add_gridspec(2, 3)
pal = ["#d81b60", "#1e88e5", "#ffc107", "#004d40", "#8e24aa", "#e65100", "#00838f"]
ax = fig.add_subplot(gsf[0, 0])
th = np.linspace(0, 2 * np.pi, 200)
for i, cc_ in enumerate(an_gtf['rec']):
    cs = np.array(an_gtf['members'][cc_])
    ax.scatter(cent[cs, 0], cent[cs, 1], s=6, marker='s', color=pal[i % 7],
               label=f"{gtf_info[cc_]} {an_gtf['index'].get(cc_)}")
ax.set_aspect('equal'); ax.legend(fontsize=7, loc='upper right')
ax.set_title("GT-flow: Morse sets of the true vector field\n(true latent plane, 150x150 grid, tau = 1 s)")
# streamlines
gx, gy = np.meshgrid(np.linspace(-LIM, LIM, 30), np.linspace(-LIM, LIM, 30))
V = system.field(np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]))
ax.streamplot(gx, gy, V[:, 0].reshape(gx.shape), V[:, 1].reshape(gx.shape), color='0.75',
              density=0.9, linewidth=0.5, arrowsize=0.6)
ax2 = fig.add_subplot(gsf[0, 1])
cols_f = {cc_: pal[i % 7] for i, cc_ in enumerate(an_gtf['rec'])}
draw_graph(ax2, an_gtf['rec'], an_gtf['hasse'],
           {cc_: f"{gtf_info[cc_]}\n{an_gtf['index'].get(cc_)}" for cc_ in an_gtf['rec']},
           cols_f, "GT-flow Conley-Morse graph")
for k, nm in enumerate(["base", "high (gain x%g)" % G]):
    ax = fig.add_subplot(gsf[1, k])
    for i, cc_ in enumerate(rec_h):
        cs = np.array(mem_h[cc_]); copy, cc2 = np.divmod(cs, ncell)
        sel = copy == k
        ax.scatter(cent[cc2[sel], 0], cent[cc2[sel], 1], s=6, marker='s', color=pal[i % 7])
    gc_ = np.flatnonzero(guard_cell)
    if k == 0:
        ax.scatter(cent[gc_, 0], cent[gc_, 1], s=4, marker='s', color='#ffd54f', alpha=0.35,
                   label='guard region (guard neurons > threshold)')
        ax.legend(fontsize=7, loc='upper right')
    ax.set_aspect('equal')
    ax.set_title(f"GT-hybrid, {nm} copy of the plane\n(colors = hybrid Morse sets)")
ax = fig.add_subplot(gsf[1, 2])
cols_h = {cc_: pal[i % 7] for i, cc_ in enumerate(rec_h)}
draw_graph(ax, rec_h, hasse_h, {cc_: gth_info[cc_].replace(" + ", "\n+ ") for cc_ in rec_h},
           cols_h, "GT-hybrid Conley-Morse graph\n(jumps join the rings into one recurrent class)")
ax = fig.add_subplot(gsf[0, 2]); ax.axis('off')
ax.text(0, 0.95, "Decodable part of the ground truth\n(data only visits the attractors):", fontsize=11,
        va='top', weight='bold')
ax.text(0, 0.7, "flow scenario:\n  inner ring (1,1,0)   outer ring (1,1,0)\n  no reachability between them\n\n"
        "hybrid scenario:\n  same two FLOW Morse sets, each (1,1,0)\n"
        "  + reset edges  outer -> inner  and  inner -> outer\n"
        "  (together: one hybrid recurrent class)", fontsize=10, va='top', family='monospace')
fig.suptitle("Ground truth Conley-Morse graphs from the true underlying flow", fontsize=13)
fig.tight_layout(); fig.savefig(OUT + "cmg_ground_truth.png", dpi=120); plt.close(fig)
T("GT figure")

# ============================================================ PART 2: data
def simulate(hazard, seed):
    sysx, ro, x0_, meta_ = sc.nested_rings_hybrid(n_trials=24, hazard=hazard)
    rec = sysx.simulate(x0_, T=30.0, readout=ro, dt=0.001, base=0.005, burn=3.0, seed=seed)
    segs = ro.spikes(rec, sysx, seed=2)
    return rec, segs

D = pickle.load(open(OUT + "hds_sim.pkl", "rb"))
data = {'hybrid': (D['rec'], D['segs'])}
data['flow'] = simulate(0.0, seed=0)
T("flow scenario simulated (hazard = 0)")

M_EMB, NXC = 3, 64

def prep(rec, segs):
    base = rec['base']
    Xf, trf, ttf = dc.fine_bins(segs, base)
    X = rec['X']
    ring = (np.sqrt(X[..., 0] ** 2 + X[..., 1] ** 2) > R_BAR).astype(int)
    jumps = [(i, (b + 0.5) * base) for (i, b, k, _) in rec['events'] if k == 'jump']
    returns = [(i, (b + 0.5) * base) for (i, b, k, _) in rec['events'] if k == 'return']
    return dict(Xf=Xf, trf=trf, ttf=ttf, ring=ring, mode=rec['mode'], base=base, jumps=jumps,
                returns=returns)

P = {k: prep(*v) for k, v in data.items()}
T("fine bins ready")

def detect_events(L, tr, tt, b, s):
    """|dv| at scale b (second difference over b/s steps), within trial;
    events = peaks above median + 6 MAD, >= 300 ms apart."""
    nh = max(1, int(round(b / s)))
    sc_ = np.full(len(L), np.nan)
    sc_[nh:-nh] = np.linalg.norm(L[2 * nh:] - 2 * L[nh:-nh] + L[:-2 * nh], axis=1)
    okt = np.zeros(len(L), bool)
    okt[nh:-nh] = (tr[2 * nh:] == tr[:-2 * nh])
    sc_[~okt] = np.nan
    med = np.nanmedian(sc_); mad = np.nanmedian(np.abs(sc_ - med)) + 1e-12
    th_ = med + 6 * mad
    ev = []
    gap = 0.3
    for ti in np.unique(tr):
        m = np.flatnonzero((tr == ti) & (sc_ > th_))
        last = -1e9
        for i in m:
            if tt[i] - last > gap:
                j = i + np.nanargmax(sc_[i:i + max(2, nh)]) if i + 1 < len(sc_) else i
                if tr[j] != ti: j = i
                ev.append((ti, tt[j])); last = tt[j]
    return ev

def match_events(det, truth, tol):
    det_by = {}
    for ti, t in det: det_by.setdefault(ti, []).append(t)
    hit = 0; used = 0
    for ti, t in truth:
        dd = np.abs(np.array(det_by.get(ti, [])) - t)
        if len(dd) and dd.min() <= tol: hit += 1
    for ti, t in det:
        tt_ = np.array([t2 for (i2, t2) in truth if i2 == ti])
        if len(tt_) and np.abs(tt_ - t).min() <= tol: used += 1
    rec_ = hit / max(len(truth), 1); prec = used / max(len(det), 1)
    return prec, rec_

def delay_states(Pd, b, s, ell):
    """Full delay-embedded latent states (m x K dims), times, trials."""
    Xw, tt, tr = dc.window_counts(Pd['Xf'], Pd['trf'], Pd['ttf'], b, s)
    Lat, K, _ = dc.scores(Xw)
    Ls = int(round(ell / s))
    Ds, tts, trs = [], [], []
    for ti in np.unique(tr):
        m = np.flatnonzero(tr == ti)
        if len(m) <= (M_EMB - 1) * Ls + Ls: continue
        Dm = cl.delay_embed(Lat[m], M_EMB, Ls).reshape(-1, M_EMB * Lat.shape[1])
        Ds.append(Dm); tts.append(tt[m][(M_EMB - 1) * Ls:][:len(Dm)])
        trs.append(np.full(len(Dm), ti))
    return np.vstack(Ds), np.concatenate(tts), np.concatenate(trs)


def decode_cmg(Pd, b, s, ell, routing=None, keep=False):
    """routing: None | 'oracle' | 'detected'."""
    Xw, tt, tr = dc.window_counts(Pd['Xf'], Pd['trf'], Pd['ttf'], b, s)
    Lat, K, _ = dc.scores(Xw)
    Ls = int(round(ell / s))
    # events
    ev, pr_rc = [], (np.nan, np.nan)
    ev_reset = None                    # events counted as reset edges
    if routing == 'oracle':
        # BOTH hybrid events are routed: the jump (state teleport, field gain
        # up) and the return (field gain back down). Windowing turns the gain
        # change itself into a chart discontinuity, so the return is a guard.
        ev = Pd['jumps'] + Pd['returns']
        ev_reset = Pd['jumps']
    elif routing == 'detected':
        ev = detect_events(Lat, tr, tt, b, s)
        if Pd['jumps']:
            pr_rc = match_events(ev, Pd['jumps'], tol=b / 1000 / 2 + 0.06)
    elif routing == 'lipschitz':
        Dl, ttl, trl = delay_states(Pd, b, s, ell)
        ev, _, _ = LIP_GUARD.detect(Dl, trl, ttl, s / 1000)
        if Pd['jumps']:
            pr_rc = match_events(ev, Pd['jumps'],
                                 tol=b / 2000 + LIP_GUARD.window_s / 2 + 0.06)
    # delay embedding per trial
    Ds, tts, trs, idxs = [], [], [], []
    for ti in np.unique(tr):
        m = np.flatnonzero(tr == ti)
        if len(m) <= (M_EMB - 1) * Ls + Ls: continue
        Dm = cl.delay_embed(Lat[m], M_EMB, Ls).reshape(-1, M_EMB * Lat.shape[1])
        Ds.append(Dm); tts.append(tt[m][(M_EMB - 1) * Ls:][:len(Dm)])
        trs.append(np.full(len(Dm), ti))
    Dall = np.vstack(Ds); tts = np.concatenate(tts); trs = np.concatenate(trs)
    mu2, V2 = cl.pca_fit(Dall, d=2, max_n=40000)
    Z = cl.pca_apply(Dall, mu2, V2)
    g = cl.Grid2D(Z, NXC)
    cz = g.cells(Z)
    # pairs (i, i+Ls) within trial
    i0 = np.arange(len(Z) - Ls)
    same = trs[i0] == trs[i0 + Ls]
    i0 = i0[same]; i1 = i0 + Ls
    span_lo = tts[i0] - b / 2000 - (M_EMB - 1) * ell / 1000
    span_hi = tts[i1] + b / 2000
    clean = np.ones(len(i0), bool)
    ev_by = {}
    for ti, t in ev: ev_by.setdefault(ti, []).append(t)
    for ti, ts in ev_by.items():
        ts = np.array(ts)
        m = np.flatnonzero(trs[i0] == ti)
        if not len(m): continue
        hitm = ((span_lo[m, None] <= ts[None]) & (span_hi[m, None] >= ts[None])).any(1)
        clean[m[hitm]] = False
    A = cl.build_map(cz[i0[clean]], cz[i1[clean]], g.nx, g.ny, src_dilate=0)
    A = cl.support_restrict(A, cz[i0[clean]], cz[i1[clean]])
    an = cl.analyze(A, g.ny, f"{b}/{s}/{ell}", index=True, verbose=False)
    ring_s = Pd['ring'][trs, np.clip((tts / Pd['base']).astype(int), 0, Pd['ring'].shape[1] - 1)]
    mode_s = Pd['mode'][trs, np.clip((tts / Pd['base']).astype(int), 0, Pd['mode'].shape[1] - 1)]
    sets = []
    for cc_ in an['rec']:
        idx = an['index'].get(cc_)
        if idx is None or idx == (0, 0, 0): continue
        msk = np.isin(cz, an['members'][cc_])
        if msk.sum() < 20: continue
        sets.append(dict(id=cc_, idx=idx, fin=(ring_s[msk] == 0).mean(),
                         hfrac=(mode_s[msk] > 0).mean(), n=len(an['members'][cc_])))
    sets.sort(key=lambda x: -x['n'])
    inner = [x for x in sets if x['fin'] > 0.8]; outer = [x for x in sets if x['fin'] < 0.2]
    mixed = [x for x in sets if 0.2 <= x['fin'] <= 0.8]
    sep = bool(inner and outer and not mixed)
    bi = [x for x in inner if x['hfrac'] < 0.5]; bo = [x for x in outer if x['hfrac'] < 0.5]
    idx_ok = sep and bool(bi) and bool(bo) and bi[0]['idx'] == (1, 1, 0) and \
        bo[0]['idx'] == (1, 1, 0)
    # reset edges between flow Morse sets
    resets = {('outer', 'inner'): 0, ('inner', 'outer'): 0, ('same', 'same'): 0}
    if ev and sep:
        C = g.centers()
        lab_cells = {}
        for x_ in sets:
            for c_ in an['members'][x_['id']]:
                lab_cells[c_] = 'inner' if x_['fin'] > 0.8 else 'outer' if x_['fin'] < 0.2 else 'mixed'
        lc = np.array(list(lab_cells)); ll = np.array([lab_cells[c_] for c_ in lc])
        def nearest(zpt):
            return ll[np.argmin(((C[lc] - zpt) ** 2).sum(1))]
        for ti, t in (ev if ev_reset is None else ev_reset):
            m = np.flatnonzero(trs == ti)
            pre = m[(tts[m] + b / 2000) < t]
            post = m[(tts[m] - b / 2000 - (M_EMB - 1) * ell / 1000) > t]
            if not len(pre) or not len(post): continue
            a_, b_ = nearest(Z[pre[-1]]), nearest(Z[post[0]])
            if a_ == b_: resets[('same', 'same')] += 1
            elif (a_, b_) in resets: resets[(a_, b_)] += 1
    out = dict(K=K, sep=sep, idx_ok=idx_ok, sets=sets, mixed=bool(mixed), resets=resets,
               prec=pr_rc[0], rec=pr_rc[1], n_ev=len(ev))
    if keep:
        out.update(Z=Z, cz=cz, an=an, g=g, ring_s=ring_s, mode_s=mode_s, ev=ev, trs=trs, tts=tts)
    return out

def score(o, scenario):
    if scenario == 'flow':
        return 2 if o['idx_ok'] else 1 if o['sep'] else 0
    r = o['resets']
    both = r[('outer', 'inner')] >= 5 and r[('inner', 'outer')] >= 5
    if o['idx_ok'] and both: return 2
    if o['sep'] and both: return 1.5
    if o['sep']: return 1
    return 0

B_SET = [20, 30, 60, 120, 240]
S_SET = [10, 30]
L_SET = [10, 30, 90, 180]
runs = [('flow', None), ('hybrid', None), ('hybrid', 'oracle'), ('hybrid', 'detected')]

# Lipschitz-bound guard detector on the delay embedding. OFF for routing:
# set enabled=True to add a ('hybrid', 'lipschitz') routing run to the sweep.
LIP_GUARD = dc.LipschitzGuard(window_s=0.1, max_rate=8.0, L_max=None,
                              min_gap_s=0.3, enabled=False)
if LIP_GUARD.enabled:
    runs.append(('hybrid', 'lipschitz'))
results = {}
for scen, routing in runs:
    for b in B_SET:
        for s in S_SET:
            for ell in L_SET:
                if ell < s or ell % s: continue
                o = decode_cmg(P[scen], b, s, ell, routing)
                results[(scen, routing, b, s, ell)] = o
                r = o['resets']
                print(f"  {scen:6s} {str(routing):8s} b={b:3d} s={s:2d} l={ell:3d}  K={o['K']}  "
                      f"sep={o['sep']!s:5} idx_ok={o['idx_ok']!s:5} "
                      f"resets o->i {r[('outer','inner')]:3d} i->o {r[('inner','outer')]:3d} "
                      f"same {r[('same','same')]:3d}  "
                      f"det P/R {o['prec']:.2f}/{o['rec']:.2f} n_ev={o['n_ev']}  "
                      f"score {score(o, scen)}", flush=True)
    T(f"sweep {scen}/{routing} done")
pickle.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
            open(OUT + "cmg_sweep.pkl", "wb"))

# --- false positives of the detector in the flow scenario
fp = [results[('flow', None, b, s, l)] for (sc_, ro_, b, s, l) in results if sc_ == 'flow']

# ============================================================ figures
fig, axes = plt.subplots(len(runs), 2, figsize=(11, 4.0 * len(runs)))
for r_i, (scen, routing) in enumerate(runs):
    for c_i, s in enumerate(S_SET):
        ax = axes[r_i, c_i]
        M = np.full((len(B_SET), len(L_SET)), np.nan)
        for i, b in enumerate(B_SET):
            for j, ell in enumerate(L_SET):
                o = results.get((scen, routing, b, s, ell))
                if o is not None: M[i, j] = score(o, scen)
        ax.imshow(M, cmap='RdYlGn', vmin=0, vmax=2, aspect='auto')
        for i in range(len(B_SET)):
            for j in range(len(L_SET)):
                o = results.get((scen, routing, B_SET[i], s, L_SET[j]))
                if o is None:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color='#cfd8dc'))
                    continue
                txt = f"K={o['K']}"
                if scen == 'hybrid' and routing:
                    rr = o['resets']
                    txt += f"\n{rr[('outer','inner')]}/{rr[('inner','outer')]}"
                ax.text(j, i, txt, ha='center', va='center', fontsize=7)
        ax.set_xticks(range(len(L_SET))); ax.set_xticklabels(L_SET)
        ax.set_yticks(range(len(B_SET))); ax.set_yticklabels(B_SET)
        ax.set(xlabel="lag ell (ms)", ylabel="bin b (ms)")
        name = f"{scen}" + (f", {routing} routing" if routing else ", no routing")
        ax.set_title(f"{name}, step {s} ms", fontsize=10)
fig.suptitle("Decoded Conley-Morse graph vs ground truth\n"
             "flow: green = both rings separate + (1,1,0) each | yellow = separated | red = merged\n"
             "hybrid: green = flow sets (1,1,0) + reset edges both ways (numbers: outer->inner / inner->outer)",
             fontsize=11)
fig.tight_layout(); fig.savefig(OUT + "cmg_sweep.png", dpi=115); plt.close(fig)
T("sweep figure")

def best(scen, routing):
    """Top score; ties broken by the weaker reset direction (hybrid), then
    the smallest bin, then the smallest lag."""
    cand = [k for k in results if k[0] == scen and k[1] == routing]
    top = max(score(results[k], scen) for k in cand)
    ks = [k for k in cand if score(results[k], scen) == top]
    def minres(k):
        r = results[k]['resets']
        return min(r[('outer', 'inner')], r[('inner', 'outer')])
    ks.sort(key=lambda k: (-minres(k), k[2], k[4], k[3]))
    return ks[0], top

fig = plt.figure(figsize=(18, 11))
gsx = fig.add_gridspec(2, 4)
for r_i, (scen, routing) in enumerate([('flow', None), ('hybrid', 'oracle')]):
    k, top = best(scen, routing)
    o = decode_cmg(P[scen], k[2], k[3], k[4], routing, keep=True)
    Z, cz, an, g = o['Z'], o['cz'], o['an'], o['g']
    ax = fig.add_subplot(gsx[r_i, 0])
    sub = slice(0, len(Z), 4)
    ax.scatter(Z[sub, 0], Z[sub, 1], c=o['ring_s'][sub], cmap='coolwarm', s=1, vmin=-0.3, vmax=1.3)
    hi = o['mode_s'] > 0
    ax.scatter(Z[hi][::4, 0], Z[hi][::4, 1], s=1, color='k', alpha=0.25, label='high-gain (jump) state')
    ax.legend(fontsize=7, markerscale=6)
    ax.set_title(f"{scen}: chart, b={k[2]} s={k[3]} l={k[4]} ms\n(red outer, blue inner, black = jump state)")
    ax = fig.add_subplot(gsx[r_i, 1])
    C = g.centers()
    nodes, labs, cols, idmap = [], {}, {}, {}
    for i, x_ in enumerate(sorted(o['sets'], key=lambda x: -x['n'])):
        cs = np.array(an['members'][x_['id']])
        nm = 'inner' if x_['fin'] > 0.8 else 'outer' if x_['fin'] < 0.2 else 'mixed'
        md = 'high' if x_.get('hfrac', 0) >= 0.5 else 'base'
        ax.scatter(C[cs, 0], C[cs, 1], s=6, marker='s', color=pal[i % 7],
                   label=f"{nm}/{md} {x_['idx']}")
        nodes.append(x_['id']); labs[x_['id']] = f"{nm}/{md}\n{x_['idx']}"; cols[x_['id']] = pal[i % 7]
        if md == 'base' and nm not in idmap:
            idmap[nm] = x_['id']
    if routing:
        for (ti, t) in o['ev'][:400:3]:
            m = np.flatnonzero(o['trs'] == ti)
            pre = m[(o['tts'][m] + k[2] / 2000) < t]
            post = m[(o['tts'][m] - k[2] / 2000 - (M_EMB - 1) * k[4] / 1000) > t]
            if len(pre) and len(post):
                ax.annotate('', xy=Z[post[0]], xytext=Z[pre[-1]],
                            arrowprops=dict(arrowstyle='->', color='#d81b60', lw=0.4, alpha=0.5))
    ax.legend(fontsize=7)
    ax.set_title("decoded flow Morse sets" + (" + reset edges (pink)" if routing else ""))
    ax = fig.add_subplot(gsx[r_i, 2])
    extra = []
    if routing and 'inner' in idmap and 'outer' in idmap:
        rr = o['resets']
        extra = [(idmap['outer'], idmap['inner'], f"reset x{rr[('outer','inner')]}"),
                 (idmap['inner'], idmap['outer'], f"reset x{rr[('inner','outer')]}")]
    draw_graph(ax, nodes, [(a, b) for a, b in an['hasse'] if a in nodes and b in nodes], labs, cols,
               f"decoded Conley-Morse graph ({scen})", extra)
    ax = fig.add_subplot(gsx[r_i, 3]); ax.axis('off')
    if scen == 'flow':
        txt = ("ground truth (decodable part):\n  inner (1,1,0), outer (1,1,0),\n  no edges\n\n"
               f"decoded: score {top}/2\n  K = {o['K']}")
    else:
        rr = o['resets']
        txt = ("ground truth (decodable part):\n  inner (1,1,0), outer (1,1,0)\n  + resets both ways\n\n"
               f"decoded: score {top}/2, K = {o['K']}\n  resets outer->inner {rr[('outer','inner')]}\n"
               f"  resets inner->outer {rr[('inner','outer')]}\n  (true jumps: {len(P['hybrid']['jumps'])})")
    ax.text(0, 0.9, txt, fontsize=11, va='top', family='monospace')
fig.suptitle("Best decoded Conley-Morse graphs (top score; ties: most complete resets, then smallest bin)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT + "cmg_decoded_best.png", dpi=115); plt.close(fig)
T("best-decode figure; done")

# ============================================ Lipschitz guard: validation only
# (LIP_GUARD.enabled is False: it is NOT used for routing above; here we only
#  check that it detects the true jumps and stays quiet without them)
VAL_SETTINGS = [(60, 10, 30), (120, 30, 90), (240, 30, 90)]
WINDOWS = [0.1, 0.2]
RATES = np.geomspace(1, 200, 14)
val = {}
for (b, s, ell) in VAL_SETTINGS:
    Dh, tth, trh = delay_states(P['hybrid'], b, s, ell)
    Df, ttf_, trf_ = delay_states(P['flow'], b, s, ell)
    minutes_f = len(Df) * s / 1000 / 60
    for wdw in WINDOWS:
        g_ = dc.LipschitzGuard(window_s=wdw, max_rate=1.0, enabled=False)
        Lh, rh = g_.trace(Dh, trh, s / 1000)
        Lf, rf = g_.trace(Df, trf_, s / 1000)
        pr, rc, fp = [], [], []
        for mr in RATES:
            g_.max_rate = mr
            evh, _, _ = g_.detect(Dh, trh, tth, s / 1000)
            evf, _, _ = g_.detect(Df, trf_, ttf_, s / 1000)
            p_, r_ = match_events(evh, P['hybrid']['jumps'], tol=b / 2000 + wdw / 2 + 0.06)
            pr.append(p_); rc.append(r_); fp.append(len(evf) / minutes_f)
        val[(b, s, ell, wdw)] = dict(pr=np.array(pr), rc=np.array(rc), fp=np.array(fp),
                                     Lh=Lh, rh=rh, tth=tth, trh=trh)
        k_ = int(np.nanargmax(2 * np.array(pr) * np.array(rc) / (np.array(pr) + np.array(rc) + 1e-9)))
        print(f"  Lipschitz guard b={b} s={s} l={ell} window={wdw*1e3:.0f} ms: best max_rate "
              f"{RATES[k_]:.1f}/s -> precision {pr[k_]:.2f} recall {rc[k_]:.2f}, "
              f"false alarms in flow scenario {fp[k_]:.1f}/min", flush=True)
T("Lipschitz guard validation")

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for c_i, (b, s, ell) in enumerate(VAL_SETTINGS):
    ax = axes[0, c_i]
    for wdw, ls in zip(WINDOWS, ['-', '--']):
        v = val[(b, s, ell, wdw)]
        ax.plot(RATES, v['pr'], ls, color='#1e88e5', label=f"precision, W={wdw*1e3:.0f} ms")
        ax.plot(RATES, v['rc'], ls, color='#d81b60', label=f"recall, W={wdw*1e3:.0f} ms")
    ax2 = ax.twinx()
    for wdw, ls in zip(WINDOWS, ['-', '--']):
        ax2.plot(RATES, val[(b, s, ell, wdw)]['fp'], ls, color='0.5')
    ax2.set_ylabel("false alarms / min (flow scenario, grey)")
    ax.axvline(LIP_GUARD.max_rate, color='k', lw=0.8, ls=':')
    ax.set(xscale='log', xlabel="max_rate bound on |d log L / dt| (1/s)", ylabel="vs true jumps",
           title=f"b={b} s={s} l={ell} ms  (dotted: configured bound)")
    ax.legend(fontsize=7, loc='center left')
    # example trace
    ax = axes[1, c_i]
    v = val[(b, s, ell, WINDOWS[0])]
    ti = P['hybrid']['jumps'][len(P['hybrid']['jumps']) // 2][0]
    m = v['trh'] == ti
    t_ = v['tth'][m]
    ax.plot(t_, v['rh'][m], color='#37474f', lw=0.8, label='|d log L / dt|')
    ax.axhline(LIP_GUARD.max_rate, color='r', ls='--', lw=0.8, label='configured bound')
    for (tj, tjt) in P['hybrid']['jumps']:
        if tj == ti: ax.axvline(tjt, color='#d81b60', lw=1)
    ax3 = ax.twinx(); ax3.plot(t_, v['Lh'][m], color='#43a047', lw=0.8); ax3.set_ylabel("L hat", color='#43a047')
    ax.set_xlim(t_[0] + 2, t_[0] + 12); ax.set_yscale('log')
    ax.set(xlabel="time in trial (s)", title=f"trial {ti}: rate of change of L hat; pink = true jumps")
    ax.legend(fontsize=7, loc='upper left')
fig.suptitle("Lipschitz-bound guard detector on the delay embedding (validated, NOT enabled for routing)",
             fontsize=13)
fig.tight_layout(); fig.savefig(OUT + "cmg_lipschitz_guard.png", dpi=115); plt.close(fig)
T("all done")
