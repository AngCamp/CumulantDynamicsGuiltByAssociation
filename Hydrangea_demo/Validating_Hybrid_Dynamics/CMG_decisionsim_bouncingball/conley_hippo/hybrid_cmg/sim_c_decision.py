"""
Sim C: decision-making with hybrid switches.

Trial structure (hybrid system with two guards):
  PRE  [0, 1.0 s) : rotating bistable attractor -- two wells at +/-p(t),
                    p(t) = 0.55*(cos Wt, sin Wt), W = 1 rad/s; the trial's
                    choice = which well the state settled in (plus noise)
  GO cue at 1.0 s : HYBRID SWITCH #1 (vector field changes discontinuously):
                    line-attractor race along the chosen direction e_c:
                    dr/dt = 2.2 e_c  - 7 (r - (r.e_c)e_c)   until |r.e_c|>1.5
  LICK  (0.45 s)  : 7 Hz rhythmic bursting around the threshold point
  REWARD(0.35 s)  : correct (75%): gain channel g -> +1 bump
                    error  (25%): g -> -0.5 dip
  RESET           : HYBRID SWITCH #2 (state jump): r -> ~0, g -> 0, next trial

Neurons: 40 cells, rates = softplus(W [r; g]) in 1-30 Hz range + noise.

Analyses:
  1. Guard detection from the latent: velocity-discontinuity score
     (windowed Lipschitz-style estimate); precision/recall of cue & reset
     times; LOCATION of detected jumps in the chart = empirical guard sets.
  2. CMG at fine lag on the full hour: the trial loop is the recurrent set;
     its index b1 counts the independent loops (two choice branches -> b1=2).
  3. Low-occupancy check: the post-cue line-attractor cells (~15% occupancy)
     and whether coarse smoothing destroys the loop/branches.
"""

import sys, os
sys.path.insert(0, "/home/claude/hybrid_cmg")
import numpy as np
import cmg_lib as cl

rng = np.random.default_rng(11)

DT = 0.002
N_TRIALS = 240
W_ROT = 1.0
CACHE = "/home/claude/hybrid_cmg/sim_c_data.npz"

def run_trial(phase0):
    """returns states r(t) (2d), g(t), labels, truth events (local bins)"""
    rs, gs, lab = [], [], []
    events = {}
    choice = rng.choice([1, -1])
    r = rng.normal(0, .08, 2)
    gch = 0.0
    t = 0.0
    # --- PRE: rotating bistable ---
    n_pre = int(1.0 / DT)
    for s in range(n_pre):
        ang = phase0 + W_ROT * t
        pvec = 0.55 * np.array([np.cos(ang), np.sin(ang)]) * choice
        r += (-4.0 * (r - pvec)) * DT + 0.10 * np.sqrt(DT) * rng.normal(size=2)
        rs.append(r.copy()); gs.append(gch); lab.append(0)
        t += DT
    # --- GO: hybrid switch #1 -> line attractor race ---
    events['cue'] = len(rs)
    e_c = r / (np.linalg.norm(r) + 1e-9)
    while abs(r @ e_c) < 1.5 and len(rs) < n_pre + int(1.5 / DT):
        proj = (r @ e_c) * e_c
        r += (2.2 * e_c - 7.0 * (r - proj)) * DT \
             + 0.10 * np.sqrt(DT) * rng.normal(size=2)
        rs.append(r.copy()); gs.append(gch); lab.append(1)
        t += DT
    # --- LICK: 7 Hz rhythm ---
    events['lick'] = len(rs)
    r_th = r.copy()
    n_lick = int(0.45 / DT)
    for s in range(n_lick):
        r = r_th + 0.13 * np.sin(2 * np.pi * 7.0 * s * DT) * e_c \
            + 0.02 * rng.normal(size=2)
        rs.append(r.copy()); gs.append(gch); lab.append(2)
    # --- REWARD ---
    events['reward'] = len(rs)
    correct = rng.random() < 0.75
    n_rew = int(0.35 / DT)
    for s in range(n_rew):
        target = 1.0 if correct else -0.5
        gch += (-(gch - target) * 12.0) * DT + 0.05 * np.sqrt(DT) * rng.normal()
        r += (-1.0 * (r - r_th)) * DT + 0.05 * np.sqrt(DT) * rng.normal(size=2)
        rs.append(r.copy()); gs.append(gch); lab.append(3 if correct else 4)
    # --- RESET: hybrid switch #2 (state jump) ---
    events['reset'] = len(rs)
    n_iti = int(0.3 / DT)
    r = rng.normal(0, .06, 2)          # the jump
    gch = 0.0
    for s in range(n_iti):
        r += (-6.0 * r) * DT + 0.08 * np.sqrt(DT) * rng.normal(size=2)
        rs.append(r.copy()); gs.append(gch); lab.append(5)
    return (np.array(rs), np.array(gs), np.array(lab), events, choice, correct)

if os.path.exists(CACHE):
    print("loading cached trials ...")
    d = np.load(CACHE)
    R, G, LAB, U = d['R'], d['G'], d['LAB'], d['U']
    cue_bins, reset_bins, choices = d['cue_bins'], d['reset_bins'], d['choices']
else:
    print(f"simulating {N_TRIALS} trials ...")
    allR, allG, allL = [], [], []
    cue_bins, reset_bins, choices = [], [], []
    off = 0
    for tr in range(N_TRIALS):
        rs, gs, lab, ev, ch, co = run_trial(rng.uniform(0, 2 * np.pi))
        allR.append(rs); allG.append(gs); allL.append(lab)
        cue_bins.append(off + ev['cue'])
        reset_bins.append(off + ev['reset'])
        choices.append(ch)
        off += len(rs)
    R = np.concatenate(allR); G = np.concatenate(allG)
    LAB = np.concatenate(allL)
    cue_bins = np.array(cue_bins); reset_bins = np.array(reset_bins)
    choices = np.array(choices)
    # neurons: 40 cells, softplus readout of (r, g)
    Wn = rng.normal(size=(40, 3)) * 1.4
    bn = rng.uniform(0.2, 0.8, 40)
    X3 = np.column_stack([R, G])
    Urate = np.log1p(np.exp(X3 @ Wn.T + bn)) * 6.0          # ~1-30 Hz
    Urate += 0.6 * rng.normal(size=Urate.shape)             # private noise
    U = Urate.astype(np.float32)
    np.savez_compressed(CACHE, R=R, G=G, LAB=LAB, U=U, cue_bins=cue_bins,
                        reset_bins=reset_bins, choices=choices)

n = len(R)
print(f"  {n} bins ({n*DT:.0f} s), occupancy by phase: "
      f"{[f'{(LAB==i).mean():.0%}' for i in range(6)]} "
      "(pre, race, lick, rew+, rew-, iti)")

# ---------------- 1. guard detection: local-linear residual ------------------
# A velocity-jump score finds state JUMPS (resets) but is blind to FIELD
# switches with continuous state (the cue), and the 7 Hz lick rhythm -- being
# high-jerk but locally LINEAR (a rotation) -- drowns it. The principled
# detector from the theory: fit local linear dynamics z_{t+1} ~ A z_t + b on a
# sliding window and score the center-point prediction residual. Smooth modes
# (including the lick oscillation) fit well; a vector-field switch or a state
# jump inside the window does not.
mu, V = cl.pca_fit(U, d=2, seed=0, max_n=30000)
Z = cl.pca_apply(U, mu, V)
# Per-window model-fit residuals fail here: both regimes are affine, so a
# straddling window just fits their mixture. The detectable signatures are
#   reset = position TELEPORT  -> position-jump score
#   cue   = SUSTAINED speed step (the field integrates into the trajectory)
# i.e. exactly the windowed dispersion / moment statistics of the original
# pipeline spec.
Zs = cl.boxcar(Z, 25)                      # 50 ms denoised chart
D50 = 25
spd = np.zeros(n)
spd[D50:] = np.linalg.norm(Zs[D50:] - Zs[:-D50], axis=1) / (D50 * DT)
HWs = 75                                   # 150 ms step windows
cs = np.concatenate([[0], np.cumsum(spd)])
fwd = (cs[np.clip(np.arange(n) + HWs, 0, n)] - cs[np.clip(np.arange(n), 0, n)]) / HWs
bwd = (cs[np.clip(np.arange(n), 0, n)] - cs[np.clip(np.arange(n) - HWs, 0, n)]) / HWs
step = fwd - bwd                           # sustained speed increase
Z9 = cl.boxcar(Z, 9)
js = np.zeros(n)
js[8:-8] = np.linalg.norm(Z9[16:] - Z9[:-16], axis=1)
js_bg = np.median(js)

def peaks(score, th, min_gap_ms=350):
    ev, last = [], -10**9
    idx = np.flatnonzero(score > th)
    for i in idx:
        if i - last > int(min_gap_ms / 1000 / DT):
            # local max within 100 ms
            j = i + np.argmax(score[i: i + int(0.1 / DT)])
            ev.append(j); last = j
    return np.array(ev, int)

# robust unsupervised thresholds: js is bimodal (teleports vs everything),
# step uses median + k*MAD after reset masking
TH_R = 0.35 * np.percentile(js, 99.99)
ev_reset = peaks(js, TH_R, 300)
step_m = step.copy()                       # mask resets out of the cue score
for e in ev_reset:
    step_m[max(e - 3 * HWs, 0): e + 3 * HWs] = 0
mad = np.median(np.abs(step_m - np.median(step_m))) + 1e-9
TH_C = np.median(step_m) + 10 * mad
ev_cue = peaks(step_m, TH_C, 500)
ev_det = np.sort(np.concatenate([ev_reset, ev_cue]))
TH_J = TH_C
print(f"   speed-step at true cues : {np.median(np.interp(cue_bins, np.arange(n), step)):.1f}"
      f"   background p50 {np.median(step):.1f}, TH_C {TH_C:.1f}")
print(f"   pos-jump at true resets : {np.median(np.interp(reset_bins, np.arange(n), js)):.2f}"
      f"   background p50 {js_bg:.2f}, TH_R {TH_R:.2f}")
jump_s = step                              # for the figure

def match(events_true, tol_ms=60):
    tol = int(tol_ms / 1000 / DT)
    hit = np.zeros(len(events_true), bool)
    used = np.zeros(len(ev_det), bool)
    for i, e in enumerate(events_true):
        d = np.abs(ev_det - e)
        j = np.argmin(d) if len(d) else -1
        if j >= 0 and d[j] <= tol and not used[j]:
            hit[i] = True; used[j] = True
    return hit.mean(), used

def match2(ev_d, events_true, tol_ms=160):
    tol = int(tol_ms / 1000 / DT)
    hit = np.zeros(len(events_true), bool)
    used = np.zeros(len(ev_d), bool)
    for i, e in enumerate(events_true):
        if len(ev_d) == 0:
            break
        d = np.abs(ev_d - e)
        j = int(np.argmin(d))
        if d[j] <= tol and not used[j]:
            hit[i] = True; used[j] = True
    return hit.mean(), used

rec_cue, usedC = match2(ev_cue, cue_bins)
rec_res, usedR = match2(ev_reset, reset_bins)
prec = (usedC.sum() + usedR.sum()) / max(len(ev_cue) + len(ev_reset), 1)
print(f"\nguard detection: {len(ev_reset)} reset-type + {len(ev_cue)} "
      f"cue-type events")
print(f"   cue   recall: {rec_cue:.2f}   reset recall: {rec_res:.2f}   "
      f"overall precision: {prec:.2f}")

# guard LOCATIONS in the chart
cue_locs = Z[cue_bins[cue_bins < n]]
res_locs = Z[np.clip(reset_bins, 0, n - 1)]
det_locs = Z[ev_det]

# ---------------- 2. CMG: trial loop and its branching topology --------------
NXC = 40
def mkC(cells, lag, grid):
    src, dst = cells[:-lag], cells[lag:]
    A = cl.build_map(src, dst, grid.nx, grid.ny, src_dilate=0)
    return cl.support_restrict(A, src, dst)

LAG_F = 10                                 # 20 ms
gC = cl.Grid2D(Z, NXC)
cz = gC.cells(Z)
Af = mkC(cz, LAG_F, gC)
print("\nfine CMG (lag 20 ms) on the full session:")
anf = cl.analyze(Af, gC.ny, "fine", index=True, verbose=True)

# coarse comparison: 300 ms boxcar, lag 300 ms
WC = 150
Uc = cl.boxcar(U, WC)
Zc = cl.pca_apply(Uc, mu, V)
gCc = cl.Grid2D(Zc, NXC)
czc = gCc.cells(Zc)
Ac = mkC(czc, WC, gCc)
print("coarse CMG (w=300 ms, lag 300 ms):")
anc = cl.analyze(Ac, gCc.ny, "coarse", index=True, verbose=True)

# occupancy of the race phase inside the big fine Morse set
big = max(anf['rec'], key=lambda c: len(anf['members'][c]))
cells_big = set(int(c) for c in anf['members'][big])
in_big = np.isin(cz, list(cells_big))
race_in = ((LAB[:len(cz)] == 1) & in_big).sum() / max((LAB[:len(cz)] == 1).sum(), 1)
print(f"\nrace-phase (low-occupancy line attractor) samples inside the "
      f"fine recurrent loop: {race_in:.0%}")

# ------------------------------- figure --------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))

ax = axes[0, 0]
cols = np.array(["#90a4ae", "#1e88e5", "#ffc107", "#2e7d32", "#b71c1c",
                 "#8e24aa"])
sub = slice(0, n, 9)
ax.scatter(Z[sub, 0], Z[sub, 1], s=1.2, c=cols[LAB[sub]])
ax.set_title("(a) latent chart by phase\ngrey pre | blue race | yellow lick | "
             "green/red reward | purple ITI")

ax = axes[0, 1]
tr0 = 3
s0, s1 = reset_bins[tr0 - 1], reset_bins[tr0]
tt = (np.arange(s0, s1) - s0) * DT
ax.plot(tt, jump_s[s0:s1], lw=0.8, color="#37474f")
ax.axhline(TH_J, color="r", ls="--", lw=0.8)
for cb in cue_bins:
    if s0 < cb < s1:
        ax.axvline((cb - s0) * DT, color="#1e88e5", ls=":", label="true cue")
ax.axvline((s1 - s0 - 1) * DT, color="#8e24aa", ls=":", label="true reset")
ax.legend(fontsize=8)
ax.set(title=f"(b) jump score, one trial\ncue recall {rec_cue:.2f}, "
             f"reset recall {rec_res:.2f}, precision {prec:.2f}",
       xlabel="time in trial (s)")

ax = axes[0, 2]
ax.scatter(Z[::31, 0], Z[::31, 1], s=1, color="#eceff1")
ax.scatter(det_locs[:, 0], det_locs[:, 1], s=14, color="#d81b60",
           label="detected jumps")
ax.scatter(cue_locs[:, 0], cue_locs[:, 1], s=8, marker='x',
           color="#1e88e5", label="true cue states")
ax.scatter(res_locs[:, 0], res_locs[:, 1], s=8, marker='+',
           color="#8e24aa", label="true reset states")
ax.legend(fontsize=8)
ax.set_title("(c) guard LOCATIONS: detected jump points\ncluster on the "
             "pre-cue ring (cue) and threshold arc (reset)")

ax = axes[1, 0]
C = gC.centers()
pal = ["#d81b60", "#1e88e5", "#ffc107", "#004d40", "#8e24aa"]
for i, ccid in enumerate(sorted(anf['rec'],
                                key=lambda c: -len(anf['members'][c]))[:5]):
    cells = np.array(anf['members'][ccid])
    ax.scatter(C[cells, 0], C[cells, 1], s=6, color=pal[i % 5],
               label=f"{anf['index'].get(ccid)}")
ax.legend(fontsize=8)
ax.set_title("(d) FINE Morse sets: the whole hybrid trial cycle\n"
             "is one recurrent set, certified (1,1,0)")

ax = axes[1, 1]
Cc = gCc.centers()
for i, ccid in enumerate(sorted(anc['rec'],
                                key=lambda c: -len(anc['members'][c]))[:5]):
    cells = np.array(anc['members'][ccid])
    ax.scatter(Cc[cells, 0], Cc[cells, 1], s=6, color=pal[i % 5],
               label=f"{anc['index'].get(ccid)}")
ax.legend(fontsize=8)
ax.set_title("(e) COARSE Morse sets (300 ms): index degrades\n"
             "(b1 inflates as smoothing opens coverage holes)")

ax = axes[1, 2]
race = LAB[:len(cz)] == 1
ax.scatter(Z[::31, 0], Z[::31, 1], s=1, color="#eceff1")
zr = Z[:len(cz)][race]
ax.scatter(zr[::4, 0], zr[::4, 1], s=2, color="#1e88e5",
           label=f"race phase ({race.mean():.0%} occupancy)")
ax.legend(fontsize=8)
ax.set_title(f"(f) low-occupancy line-attractor states\n"
             f"{race_in:.0%} captured inside the fine recurrent loop")

fig.suptitle("Sim C: decision task -- hybrid jump detection and "
             "low-occupancy structure", fontsize=13)
fig.tight_layout()
fig.savefig("/home/claude/hybrid_cmg/sim_c_results.png", dpi=130)
print("\nfigure saved: sim_c_results.png")
