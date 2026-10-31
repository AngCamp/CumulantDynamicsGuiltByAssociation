"""
Sim A: three neural populations whose rates trace a planar limit cycle
(outer ring, radius 1, slow CCW) containing a THIN counter-rotating band
(inner ring, radius 0.72, fast CW) separated by an unstable circle at 0.86.

Radial field:  rho' = -a (rho-0.72)(rho-0.86)(rho-1)   (a=18)
Angular field: theta' = +w_out outside the barrier, -w_in inside (reverse!)
Observable:    y = scalar random readout of 3 softplus population rates.

Question: which window/lag strategies recover the inner reverse band as a
separate Morse set with index (1,1,0), and which smear it away?
"""

import sys
sys.path.insert(0, "/home/claude/hybrid_cmg")
import numpy as np
import cmg_lib as cl

rng = np.random.default_rng(42)

# ------------------------------ simulate -------------------------------------
DT      = 0.002          # integration step (s)
REC     = 5              # record every REC steps -> sample period 10 ms
T_TRAJ  = 40.0           # s per trajectory
N_TRAJ  = 60
W_OUT   = 3.0            # outer angular speed (rad/s), period ~2.1 s
W_IN    = -12.0          # inner band REVERSED and fast, period ~0.52 s
A_RAD   = 70.0           # strong radial field: barrier not noise-crossable
R_IN, R_BAR, R_OUT = 0.72, 0.86, 1.0
SIG     = 0.012          # state noise

M3, _ = np.linalg.qr(rng.normal(size=(3, 3)))   # well-conditioned mixing
M3 *= 0.9
b3 = np.array([0.5, 0.6, 0.4])
w_read = rng.normal(size=3)
w_read /= np.linalg.norm(w_read)

BURN = 3.0               # s discarded per trajectory (kills transients)

def simulate(n_traj, t_traj):
    steps = int(t_traj / DT)
    burn = int(BURN / DT)
    ys, rhos = [], []
    for tr in range(n_traj):
        # start half the trajectories near each ring
        rho = R_OUT + rng.normal(0, .05) if tr % 2 == 0 else R_IN + rng.normal(0, .03)
        th  = rng.uniform(0, 2 * np.pi)
        z   = rng.normal(0, .2)
        rec_y, rec_r, rec_u = [], [], []
        for s in range(steps):
            drho = -A_RAD * (rho - R_IN) * (rho - R_BAR) * (rho - R_OUT)
            w    = W_OUT if rho > R_BAR else W_IN
            rho += drho * DT + SIG * np.sqrt(DT) * rng.normal()
            th  += w * DT + SIG * np.sqrt(DT) * rng.normal()
            z   += -3.0 * z * DT + SIG * np.sqrt(DT) * rng.normal()
            if s >= burn and s % REC == 0:
                x = np.array([rho * np.cos(th), rho * np.sin(th), z])
                u = np.log1p(np.exp(M3 @ x + b3))         # softplus rates
                rec_y.append(w_read @ u)
                rec_u.append(u)
                rec_r.append(rho)
        ys.append(np.array(rec_y))
        us.append(np.array(rec_u))
        rhos.append(np.array(rec_r))
    return ys, us, rhos

import os
CACHE = "/home/claude/hybrid_cmg/sim_a_data.npz"
if os.path.exists(CACHE):
    print("loading cached simulation ...")
    d = np.load(CACHE)
    y, U, rho, n_per = d['y'], d['U'], d['rho'], int(d['n_per'])
else:
    print("simulating ...")
    us = []
    ys, us, rhos = simulate(N_TRAJ, T_TRAJ)
    y   = np.concatenate(ys)             # scalar observable, 10 ms samples
    U   = np.concatenate(us)             # 3 population rates (true latents)
    rho = np.concatenate(rhos)           # ground truth radius label
    n_per = len(ys[0])
    np.savez_compressed(CACHE, y=y, U=U, rho=rho, n_per=n_per)
print(f"  {N_TRAJ} traj x {n_per} samples = {len(y)} total (10 ms sampling)")

# NOTE: trajectories are concatenated; transitions across the seam are junk.
# Mask them out by marking the last k samples of each trajectory invalid.
def seam_mask(lag):
    bad = np.zeros(len(y), bool)
    for t in range(1, N_TRAJ + 1):
        bad[t * n_per - lag: t * n_per] = True
    return bad

# ------------------------- strategies on the embedding ------------------------
DSTEP   = 12         # fine delay step: 120 ms (inner period 520 ms -> ~1/4)
DSTEP_C = 48         # coarse delay step: 480 ms (outer period 2.1 s -> ~1/4)
K       = 12         # coarse lag & window: 120 ms
M       = 3
NX      = 72

# ---- chart selection: pick the lag tuple that keeps the rings disjoint ------
# (empirical injectivity check -- the FNN step of embedding practice)
def ring_overlap(lags, w=1):
    yS = cl.boxcar(y, w)
    D = cl.delay_embed_lags(yS, lags)
    mu, V = cl.pca_fit(D, d=2, seed=0)
    Z = cl.pca_apply(D, mu, V)
    r = rho[max(lags):][:len(Z)]
    gt = cl.Grid2D(Z, 48)
    c = gt.cells(Z)
    inner = np.bincount(np.clip(c, 0, gt.N - 1),
                        weights=((r < R_BAR) & (c >= 0)), minlength=gt.N)
    outer = np.bincount(np.clip(c, 0, gt.N - 1),
                        weights=((r >= R_BAR) & (c >= 0)), minlength=gt.N)
    both = ((inner > 2) & (outer > 2)).sum()
    used = ((inner > 2) | (outer > 2)).sum()
    return both / max(used, 1)

candidates = [
    (0, 12, 24), (0, 24, 48), (0, 36, 72), (0, 48, 96),
    (0, 12, 36), (0, 12, 48), (0, 12, 72),
    (0, 12, 24, 48), (0, 12, 36, 72), (0, 12, 48, 96),
]
print("\nscalar-delay chart selection (fraction of cells where rings overlap)")
print("-- documenting the failure of a single-scale scalar delay chart:")
scores = {lg: ring_overlap(lg) for lg in candidates}
for lg, s in sorted(scores.items(), key=lambda kv: kv[1]):
    print(f"   lags {lg}: overlap {s:.3f}")
LAGS_F = min(scores, key=scores.get)
print(f"best scalar-delay chart lags: {LAGS_F} "
      f"(overlap {scores[LAGS_F]:.3f} -- still bridges the rings)")
print("-> switching chart to population-rate PCA (the latents of the real "
      "pipeline);\n   delay/lag structure enters through the transition maps.")

# run_strategies works on one contiguous series; seams add noise edges but
# are < 0.2% of pairs. We handle them by passing the seam-masked map builder
# through a thin wrapper: simplest is to run per-trajectory and merge pairs.
# For clarity we rebuild the harness inline, per strategy, with seam masks.

def chart_rates(Uv, w, dseed=0):
    US = cl.boxcar(Uv, w)
    mu, V = cl.pca_fit(US, d=2, seed=dseed)
    return cl.pca_apply(US, mu, V), (mu, V)

def masked_pairs(cells, lag):
    sm = seam_mask(lag)[:len(cells)]
    src, dst = cells[:-lag], cells[lag:]
    good = ~sm[:-lag]
    return src[good], dst[good]

print("\nbuilding charts and maps ...")
Zf, basis_f = chart_rates(U, 1)
rho_f = rho[:len(Zf)]
g  = cl.Grid2D(Zf, NX)
cf = g.cells(Zf)

Zc, _ = chart_rates(U, K, 1)
gc  = cl.Grid2D(Zc, NX)
cc  = gc.cells(Zc)
rho_c = rho[:len(Zc)]

yc_on_f = cl.pca_apply(cl.boxcar(U, K), *basis_f)
ccf = g.cells(yc_on_f)

# also keep the failed scalar-delay chart for the figure
Zdelay = None
yS = cl.boxcar(y, 1)
Dd = cl.delay_embed_lags(yS, LAGS_F)
mud, Vd = cl.pca_fit(Dd, d=2, seed=0)
Zdelay = cl.pca_apply(Dd, mud, Vd)
rho_delay = rho[max(LAGS_F):][:len(Zdelay)]

def mk(cells, lag, grid):
    s, dd = masked_pairs(cells, lag)
    A = cl.build_map(s, dd, grid.nx, grid.ny, src_dilate=0)
    return cl.support_restrict(A, s, dd)

A1 = mk(cf, 1, g)
Ak = cl.support_restrict(cl.compose(A1, K), *masked_pairs(cf, 1))
A3 = mk(cc, K, gc)
A4 = mk(cc, 1, gc)
A5 = mk(cf, K, g)
A6 = cl.intersect(Ak, mk(ccf, K, g))

strategies = {
    'fine-1step'     : (A1, g,  cf,  rho_f),
    f'composed^{K}'  : (Ak, g,  cf,  rho_f),
    'coarse-matched' : (A3, gc, cc,  rho_c),
    'bigwin-finelag' : (A4, gc, cc,  rho_c),
    'finewin-biglag' : (A5, g,  cf,  rho_f),
    'intersect'      : (A6, g,  cf,  rho_f),
}

# ----------------------------- evaluation ------------------------------------
def ring_purity(cells_of_set, cells_data, rho_data):
    """(mean rho, fraction of samples from the inner ring) in the set."""
    cs = set(int(c) for c in cells_of_set)
    mask = np.isin(cells_data, list(cs))
    if not mask.any():
        return np.nan, np.nan
    r = rho_data[mask]
    return r.mean(), (r < R_BAR).mean()

print("\n================ results ================")
results = {}
for name, (A, grid, cdata, rdata) in strategies.items():
    an = cl.analyze(A, grid.ny, name, index=True, verbose=False)
    sets = []
    for ccid in an['rec']:
        mr, fin = ring_purity(an['members'][ccid], cdata, rdata)
        sets.append((ccid, len(an['members'][ccid]), an['index'][ccid],
                     mr, fin))
    sets.sort(key=lambda s: -s[1])
    nontriv = [s for s in sets if s[2] != (0, 0, 0) and not np.isnan(s[3])]
    inner = [s for s in nontriv if s[4] > 0.8]        # >80% inner samples
    outer = [s for s in nontriv if s[4] < 0.2]
    mixed = [s for s in nontriv if 0.2 <= s[4] <= 0.8]
    recovered = ("BOTH rings separate" if inner and outer and not mixed else
                 "MERGED (one mixed set)" if mixed else
                 "outer only" if outer else
                 "inner only" if inner else "none")
    results[name] = (an, sets, recovered)
    print(f"\n[{name}]  ->  {recovered}")
    for ccid, size, idx, mr, fin in sets[:6]:
        if np.isnan(mr):
            print(f"   set {ccid}: {size:5d} cells  index {idx}  (no data)")
        else:
            lab = ("inner" if fin > 0.8 else "outer" if fin < 0.2 else "MIXED")
            print(f"   set {ccid}: {size:5d} cells  index {idx}  "
                  f"rho={mr:.3f}  [{lab}]")

# ------------------------------- figure --------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(17, 8))
ax0 = axes[0, 0]
subD = slice(0, len(Zdelay), 7)
sc = ax0.scatter(Zdelay[subD, 0], Zdelay[subD, 1], c=rho_delay[subD], s=1.2,
                 cmap="coolwarm", vmin=0.6, vmax=1.05)
plt.colorbar(sc, ax=ax0, label="true rho")
ax0.set_title(f"FAILED chart: scalar delay {LAGS_F}\n"
              "rings fold across each other (non-injective)")

ax1 = axes[1, 0]
sub = slice(0, len(Zf), 7)
sc = ax1.scatter(Zf[sub, 0], Zf[sub, 1], c=rho_f[sub], s=1.2,
                 cmap="coolwarm", vmin=0.6, vmax=1.05)
plt.colorbar(sc, ax=ax1, label="true rho")
ax1.set_title("WORKING chart: population-rate PCA\n(red outer CCW, "
              "blue inner CW band)")

panels = ['fine-1step', f'composed^{K}', 'coarse-matched',
          'bigwin-finelag', 'finewin-biglag', 'intersect']
pal = ["#d81b60", "#1e88e5", "#ffc107", "#004d40", "#8e24aa", "#e65100"]
for p_i, name in enumerate(panels):
    ax = axes.flat[1 + p_i if p_i < 3 else 4 + (p_i - 3)]
    an, sets, recovered = results[name]
    grid = strategies[name][1]
    C = grid.centers()
    shown = 0
    for s_i, (ccid, size, idx, mr, fin) in enumerate(sets):
        if idx == (0, 0, 0) and size < 20:
            continue
        cells = np.array(an['members'][ccid])
        ax.scatter(C[cells, 0], C[cells, 1], s=4,
                   color=pal[s_i % len(pal)],
                   label=f"{idx}, rho~{mr:.2f}" if not np.isnan(mr) else f"{idx}")
        shown += 1
        if shown >= 5:
            break
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"{name}\n-> {recovered}", fontsize=10)

fig.suptitle("Sim A: thin reverse-flow band - which strategies resolve it? "
             f"(dstep={DSTEP*10} ms, coarse lag/window={K*10} ms)", fontsize=13)
fig.tight_layout()
fig.savefig("/home/claude/hybrid_cmg/sim_a_results.png", dpi=130)
print("\nfigure saved: sim_a_results.png")
