"""
Sim B: one hour of a mouse in a 1m x 1m arena with hippocampal-style
place coding and sharp-wave-ripple (SWR) replay.

Behavior (switching system):
  WALK  : OU velocity with a home bias toward (0.5, 0.5); Exp(8 s) duration
  STILL : v = 0; Exp(6 s) duration; SWR events at 0.25 Hz during stillness
SWR content (each burst ~100 ms, 8x time compression, high rate gain):
  20%  random-noise trajectory
  20%  replay of the trajectory of the previous walk (forward)
  60%  replay of a recent INWARD (home-directed) trajectory segment
Neurons: 64 place cells on an 8x8 lattice, Mexican-hat tuning
  (sigma_exc = 0.10 m, sigma_inh = 0.25 m, rectified), lognormal peak rates
  (median ~8 Hz), baseline 0.3 Hz, Poisson spikes in 5 ms bins.

Analyses:
  1. SWR detection from population rate (the empirical guard: MUA threshold)
     -> precision/recall vs ground truth
  2. CMG strategies on the rate-PCA chart:
       coarse (w=200 ms, lag=200 ms) -> spatial map, home-base Morse set
       fine   (w=10 ms,  lag=10 ms)  -> does replay structure appear?
  3. Replay-edge recovery: long-range (>0.25 m) transitions, SWR-conditioned,
     at fine vs coarse lag.
"""

import sys, os
sys.path.insert(0, "/home/claude/hybrid_cmg")
import numpy as np
import cmg_lib as cl

rng = np.random.default_rng(3)

DT      = 0.005                  # 5 ms bins
T_TOTAL = 3600.0                 # one hour
N_BINS  = int(T_TOTAL / DT)      # 720,000
HOME    = np.array([0.5, 0.5])
N_SIDE  = 8
N_CELLS = N_SIDE * N_SIDE

CACHE = "/home/claude/hybrid_cmg/sim_b_data.npz"

# ------------------------------ behavior -------------------------------------
def simulate_behavior():
    pos = np.zeros((N_BINS, 2))
    vel = np.zeros(2)
    state = np.zeros(N_BINS, np.int8)          # 0 walk, 1 still, 2 SWR
    virt = np.zeros((N_BINS, 2))               # virtual (replayed) position
    p = np.array([0.3, 0.4])
    t = 0
    walk_hist = []                             # list of past walk segments
    swr_truth = []                             # (start_bin, end_bin, kind)
    while t < N_BINS:
        walking = rng.random() < 0.5 if t == 0 else (state[t - 1] != 0)
        # alternate walk / still
        if walking:                            # previous was still -> walk
            dur = int(rng.exponential(8.0) / DT) + 40
            seg = []
            for s in range(min(dur, N_BINS - t)):
                vel += (-vel / 1.0 - 1.5 * (p - HOME) * 0.4) * DT \
                       + 0.25 * np.sqrt(DT) * rng.normal(size=2)
                p = p + vel * DT
                for d_ in range(2):            # reflect at walls
                    if p[d_] < 0.02:
                        p[d_] = 0.04 - p[d_]; vel[d_] = abs(vel[d_])
                    if p[d_] > 0.98:
                        p[d_] = 1.96 - p[d_]; vel[d_] = -abs(vel[d_])
                pos[t] = p; virt[t] = p; state[t] = 0
                seg.append(p.copy())
                t += 1
                if t >= N_BINS:
                    break
            walk_hist.append(np.array(seg))
            if len(walk_hist) > 6:
                walk_hist.pop(0)
        else:                                  # still period
            dur = int(rng.exponential(6.0) / DT) + 40
            end = min(t + dur, N_BINS)
            swr_starts = []
            tt = t
            while tt < end:                    # Poisson SWRs at 0.25 Hz
                gap = int(rng.exponential(1 / 0.25) / DT)
                tt += gap
                if tt + 24 < end:
                    swr_starts.append(tt)
                tt += 24
            for s in range(t, end):
                pos[s] = p; virt[s] = p; state[s] = 1
            t = end
            for s0 in swr_starts:
                L = int(rng.uniform(0.08, 0.14) / DT)      # 16-28 bins
                u = rng.random()
                if u < 0.2 or not walk_hist:               # random noise
                    kind = 'noise'
                    path = rng.uniform(0.05, 0.95, size=(L, 2))
                elif u < 0.4:                              # replay last walk
                    kind = 'lastwalk'
                    W = walk_hist[-1]
                    idx = np.linspace(0, len(W) - 1,
                                      min(L, len(W)) * 8)[:L * 8:8]
                    path = W[idx.astype(int) % len(W)]
                else:                                      # inward replay
                    kind = 'inward'
                    W = max(walk_hist[-3:], key=len)
                    d_home = np.linalg.norm(W - HOME, axis=1)
                    far = int(np.argmax(d_home))
                    seg_in = W[far:] if len(W) - far > 8 else W
                    idx = np.linspace(0, len(seg_in) - 1, L)
                    path = seg_in[idx.astype(int)]
                Lp = min(len(path), end - s0)
                virt[s0:s0 + Lp] = path[:Lp]
                state[s0:s0 + Lp] = 2
                swr_truth.append((s0, s0 + Lp, kind))
    return pos, virt, state, swr_truth

# ------------------------------ neurons --------------------------------------
def make_rates(virt, state):
    gx, gy = np.meshgrid(np.linspace(.1, .9, N_SIDE),
                         np.linspace(.1, .9, N_SIDE))
    centers = np.stack([gx.ravel(), gy.ravel()], 1)
    peak = rng.lognormal(np.log(8.0), 0.5, N_CELLS)        # Hz, lognormal
    SIG_E, SIG_I = 0.10, 0.25
    lam = np.empty((N_BINS, N_CELLS), np.float32)
    CH = 40000
    for a in range(0, N_BINS, CH):
        b = min(a + CH, N_BINS)
        d2 = ((virt[a:b, None, :] - centers[None]) ** 2).sum(-1)
        mex = np.exp(-d2 / (2 * SIG_E ** 2)) - 0.45 * np.exp(-d2 / (2 * SIG_I ** 2))
        lam[a:b] = np.maximum(mex, 0) * peak + 0.3
    gain = np.where(state == 2, 6.0, 1.0)[:, None]         # SWR rate gain
    lam *= gain
    spk = rng.poisson(lam * DT).astype(np.float32)
    return spk, lam, centers

if os.path.exists(CACHE):
    print("loading cached hour ...")
    d = np.load(CACHE)
    pos, virt, state = d['pos'], d['virt'], d['state']
    spk, centers = d['spk'], d['centers']
    swr_truth = d['swr_truth']
else:
    print("simulating one hour of behavior ...")
    pos, virt, state, swr_truth = simulate_behavior()
    print(f"  SWR events: {len(swr_truth)}  "
          f"(kinds: {[k for *_, k in swr_truth[:8]]}...)")
    print("generating spikes ...")
    spk, lam, centers = make_rates(virt, state)
    swr_truth = np.array([(a, b, {'noise': 0, 'lastwalk': 1, 'inward': 2}[k])
                          for a, b, k in swr_truth])
    np.savez_compressed(CACHE, pos=pos, virt=virt, state=state, spk=spk,
                        centers=centers, swr_truth=swr_truth)
print(f"  bins: {N_BINS}, spikes: {spk.sum():.0f}, "
      f"mean rate {spk.sum() / N_CELLS / T_TOTAL:.2f} Hz/cell, "
      f"SWR events: {len(swr_truth)}")

# ---------------- 1. SWR detection: the empirical guard ----------------------
mua = spk.sum(1)
mua_s = cl.boxcar(mua, 4)                       # 20 ms smoothing
mu0, sd0 = mua_s.mean(), mua_s.std()
TH = mu0 + 3.5 * sd0
det = mua_s > TH
# event extraction
dd = np.diff(det.astype(np.int8))
starts = np.flatnonzero(dd == 1)
ends = np.flatnonzero(dd == -1)
if len(ends) and len(starts) and ends[0] < starts[0]:
    ends = ends[1:]
L = min(len(starts), len(ends))
det_ev = [(s, e) for s, e in zip(starts[:L], ends[:L]) if e - s >= 3]

true_mask = state == 2
hit = sum(1 for s, e in det_ev if true_mask[s:e].any())
prec = hit / max(len(det_ev), 1)
cov = sum(1 for a, b, k in swr_truth
          if det[int(a):int(b)].any()) / max(len(swr_truth), 1)
print(f"\nSWR detection (MUA > mean+3.5sd): {len(det_ev)} events, "
      f"precision {prec:.2f}, recall {cov:.2f}")

# ---------------- 2. CMG strategies on the rate-PCA chart --------------------
NXB = 28                             # coarse grid (dense occupancy per cell)
NXB_F = 48                           # fine grid
W_COARSE, LAG_COARSE = 40, 40        # 200 ms
W_FINE, LAG_FINE = 2, 2             # 10 ms

def chartB(w, dseed=0):
    S = cl.boxcar(spk, w) / DT       # smoothed rates (Hz)
    mu, V = cl.pca_fit(S, d=2, seed=dseed, max_n=30000)
    return cl.pca_apply(S, mu, V)

print("\nbuilding charts ...")
Zc = chartB(W_COARSE)
Zf = chartB(W_FINE, 1)
gc = cl.Grid2D(Zc, NXB)
gf = cl.Grid2D(Zf, NXB_F)
cc, cf = gc.cells(Zc), gf.cells(Zf)

def mkB(cells, lag, grid, exclude=None):
    src, dst = cells[:-lag], cells[lag:]
    if exclude is not None:
        good = ~(exclude[:-lag] | exclude[lag:])
        src, dst = src[good], dst[good]
    A = cl.build_map(src, dst, grid.nx, grid.ny, src_dilate=0)
    return cl.support_restrict(A, src, dst)

print("coarse CMG (w=200 ms, lag=200 ms):")
Ac = mkB(cc, LAG_COARSE, gc)
anc = cl.analyze(Ac, gc.ny, "coarse", index=True, verbose=True)

print("fine CMG (w=10 ms, lag=10 ms):")
Af = mkB(cf, LAG_FINE, gf)
anf = cl.analyze(Af, gf.ny, "fine", index=False, verbose=True)

# locate the coarse Morse sets in ARENA coordinates via decoded position
def arena_location(an, cells_data, n_top=3):
    out = []
    for ccid in sorted(an['rec'], key=lambda c: -len(an['members'][c]))[:n_top]:
        cs = set(int(x) for x in an['members'][ccid])
        mask = np.isin(cells_data, list(cs))
        if mask.sum():
            p = pos[:len(cells_data)][mask]
            out.append((ccid, len(cs), p.mean(0), mask.mean()))
    return out

print("\ncoarse Morse sets in arena coordinates:")
for ccid, size, ctr, occ in arena_location(anc, cc):
    print(f"   set {ccid}: {size} cells, arena center ({ctr[0]:.2f},"
          f"{ctr[1]:.2f}), occupancy {occ:.1%}  "
          f"{'<-- HOME' if np.linalg.norm(ctr - HOME) < 0.15 else ''}")

# ---------------- 3. replay-edge recovery ------------------------------------
# long-range transitions in ARENA space at fine lag, inside vs outside SWRs
def long_range_fraction(lag, mask_src):
    p0 = virt[:-lag][mask_src[:-lag]]
    p1 = virt[lag:][mask_src[:-lag]]
    d = np.linalg.norm(p1 - p0, axis=1)
    return (d > 0.25).mean(), len(d)

swr_m = state == 2
walk_m = state == 0
fr_swr, n1 = long_range_fraction(LAG_FINE, swr_m)
fr_walk, n2 = long_range_fraction(LAG_FINE, walk_m)
fr_coarse, n3 = long_range_fraction(LAG_COARSE, walk_m | swr_m)
print(f"\nlong-range (>0.25 m) transition fraction:")
print(f"   fine lag  (10 ms), inside SWRs : {fr_swr:.3f}   (n={n1})")
print(f"   fine lag  (10 ms), during walk : {fr_walk:.4f}  (n={n2})")
print(f"   coarse lag (200 ms), everywhere: {fr_coarse:.3f} (n={n3})")

# and in the fine NEURAL chart: edges conditioned on SWR vs not
def neural_jump_fraction(lag, mask_src):
    z0 = Zf[:-lag][mask_src[:-lag]]
    z1 = Zf[lag:][mask_src[:-lag]]
    d = np.linalg.norm(z1 - z0, axis=1)
    scale = np.linalg.norm(gf.hi - gf.lo)
    return (d > 0.15 * scale).mean()

print(f"   neural-chart big-jump fraction, fine lag: "
      f"SWR {neural_jump_fraction(LAG_FINE, swr_m):.3f} vs "
      f"walk {neural_jump_fraction(LAG_FINE, walk_m):.4f}")

# ------------------------------- figure --------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

ax = axes[0, 0]
w_idx = state == 0
ax.plot(pos[w_idx, 0][::40], pos[w_idx, 1][::40], '.', ms=0.6,
        color="#90a4ae", alpha=.5)
ax.plot(*HOME, 'r*', ms=16)
ax.set(xlim=(0, 1), ylim=(0, 1), title="(a) hour of walking (grey), "
                                       "home (star)")

ax = axes[0, 1]
t0 = int(swr_truth[4][0]) - 400
sl = slice(max(t0, 0), t0 + 1200)
tt = np.arange(sl.start, sl.stop) * DT
ax.plot(tt, mua_s[sl], lw=0.7, color="#37474f")
ax.axhline(TH, color="r", ls="--", lw=0.8, label="guard threshold")
for a, b, k in swr_truth:
    if sl.start < a < sl.stop:
        ax.axvspan(a * DT, b * DT, color="#ffd54f", alpha=.5)
ax.legend(fontsize=8)
ax.set(title=f"(b) MUA + SWR truth (yellow); "
             f"detection P={prec:.2f} R={cov:.2f}", xlabel="time (s)")

ax = axes[0, 2]
ex = next((e for e in swr_truth if e[2] == 2), swr_truth[0])
a, b, k = int(ex[0]), int(ex[1]), int(ex[2])
ax.plot(virt[a - 600:a, 0], virt[a - 600:a, 1], '-', lw=0.8,
        color="#90a4ae", label="recent walk")
ax.plot(virt[a:b, 0], virt[a:b, 1], 'o-', ms=3, lw=1.6, color="#d81b60",
        label="SWR replay (compressed)")
ax.plot(*HOME, 'r*', ms=14)
ax.legend(fontsize=8)
ax.set(xlim=(0, 1), ylim=(0, 1), title="(c) example inward replay event")

ax = axes[1, 0]
C = gc.centers()
pal = ["#d81b60", "#1e88e5", "#ffc107", "#004d40", "#8e24aa"]
for i, ccid in enumerate(sorted(anc['rec'],
                                key=lambda c: -len(anc['members'][c]))[:5]):
    cells = np.array(anc['members'][ccid])
    ax.scatter(C[cells, 0], C[cells, 1], s=8, color=pal[i % 5],
               label=f"{anc['index'].get(ccid)}")
ax.legend(fontsize=7)
ax.set(title="(d) coarse CMG Morse sets (neural chart)\n"
             "big set = spatial map manifold")

ax = axes[1, 1]
sub = slice(0, len(Zf), 29)
cols = np.where(state[:len(Zf)][sub] == 2, "#d81b60", "#b0bec5")
ax.scatter(Zf[sub, 0], Zf[sub, 1], s=1.5, c=cols)
ax.set(title="(e) fine chart: SWR bins (pink) leave the\n"
             "behavioral manifold (grey) - the guard set")

ax = axes[1, 2]
bars = [fr_walk, fr_swr, fr_coarse]
ax.bar(range(3), bars, color=["#90a4ae", "#d81b60", "#546e7a"])
ax.set_xticks(range(3))
ax.set_xticklabels(["fine lag\nwalk", "fine lag\nSWR", "coarse lag\nall"],
                   fontsize=8)
ax.set_title("(f) long-range (>0.25 m) arena transitions\n"
             "replay reachability exists ONLY at fine lag in SWRs")

fig.suptitle("Sim B: 1 h mouse, place cells, SWR replay -- the empirical "
             "guard and scale-dependent reachability", fontsize=13)
fig.tight_layout()
fig.savefig("/home/claude/hybrid_cmg/sim_b_results.png", dpi=130)
print("\nfigure saved: sim_b_results.png")
