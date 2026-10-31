"""
Choosing the bin size from the stability of the instantaneous Lipschitz constant.

For each (bin b, step s):
    PC scores p(t)  (correlation PCA of the binned, normalized counts)
    Lp(t) = |p(t+1) - p(t)| / dt                     speed
    Lv(t) = |p(t+1) - 2p(t) + p(t-1)| / dt^2         acceleration
    L(t)  = Lv(t) / Lp(t)                            instantaneous Lipschitz (1/s)
    dL(t) = |log L(t+1) - log L(t)|                  jaggedness, per step

Three references tell us whether L is signal or Poisson noise:
  1. Poisson control  - each neuron resampled from its own 100 ms-smoothed rate,
     so slow rate structure is kept and all fast structure is destroyed.
     Where real and control dL agree, L is noise.
  2. Split-half reliability - PCA run separately on two random halves of the
     neurons; corr(L_A, L_B) is 0 when L is noise and rises when it is signal.
  3. The noise-only value of Lv/Lp for white noise is sqrt(3)/dt; L sitting on
     that line means the trajectory is indistinguishable from noise.

The usable scale is where dL stops shrinking (the distribution stabilizes),
the real/control separation is large, and split-half reliability is high.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/claude/hybrid_cmg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import cmg_lib as cl
from hds import decode as dc

dc.STEP_MS = 5.0
OUT = "/home/claude/hippo/"
t0 = time.time()
rng = np.random.default_rng(0)

d = np.load(OUT + "output/simulation.npz")
n_steps = int(d["n_steps"])
S = np.unpackbits(d["spikes"], axis=0, count=n_steps).astype(np.uint8)
state_ms = d["state"].astype(int)
N = S.shape[1]
print(f"{N} neurons, {n_steps/1000:.0f} s, mean rate {S.mean()/0.001:.2f} Hz")

# Poisson control at 1 ms: keep each neuron's 100 ms-smoothed rate, destroy the rest
rate = gaussian_filter1d(S.astype(np.float32), 100.0, axis=0, mode="nearest")
S_ctrl = (rng.random(S.shape) < np.clip(rate, 0, 1)).astype(np.uint8)
print(f"control mean rate {S_ctrl.mean()/0.001:.2f} Hz")

def pc_traj(Sx, b, s, cols=None, norm="per_neuron"):
    """binned -> normalized -> correlation PCA -> elbow-K scores."""
    Xf, trf, ttf = dc.fine_bins([Sx if cols is None else Sx[:, cols]], 0.001)
    Xw, tt, tr = dc.window_counts(Xf, trf, ttf, b, s)
    Z = dc.normalize(Xw, mode=norm)
    C = (Z.T @ Z) / len(Z)
    ev, vec = np.linalg.eigh(C.astype(np.float64))
    o = np.argsort(ev)[::-1]; ev, vec = ev[o], vec[:, o]
    K = dc.elbow_k(ev)
    return Z @ vec[:, :max(K, 2)], K, ev, tt

def lipschitz(p, dt):
    Lp = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    Lv = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1) / dt ** 2
    L = Lv / np.maximum(Lp[:-1], 1e-12)
    return Lp, Lv, L

def dlog(x):
    x = np.maximum(np.asarray(x, float), 1e-12)
    return np.abs(np.diff(np.log(x)))

BINS = [10, 20, 40, 80, 160, 320, 640]
half = rng.permutation(N)
hA, hB = np.sort(half[:N//2]), np.sort(half[N//2:])

rows = []
dists = {}
for b in BINS:
    s = b                                   # non-overlapping
    dt = s / 1000.0
    p, K, ev, tt = pc_traj(S, b, s)
    pc, Kc, _, _ = pc_traj(S_ctrl, b, s)
    pA, _, _, _ = pc_traj(S, b, s, cols=hA)
    pB, _, _, _ = pc_traj(S, b, s, cols=hB)
    Lp, Lv, L = lipschitz(p, dt)
    _, _, Lc = lipschitz(pc, dt)
    _, _, LA = lipschitz(pA, dt)
    _, _, LB = lipschitz(pB, dt)
    dL, dLc = dlog(L), dlog(Lc)
    m = min(len(LA), len(LB))
    rel = np.corrcoef(np.log(np.maximum(LA[:m], 1e-12)),
                      np.log(np.maximum(LB[:m], 1e-12)))[0, 1]
    noise_line = np.sqrt(3) / dt
    rows.append(dict(b=b, s=s, K=K, Kc=Kc,
                     medL=float(np.median(L)), medLc=float(np.median(Lc)),
                     noise=float(noise_line),
                     L_over_noise=float(np.median(L) / noise_line),
                     med_dL=float(np.median(dL)), med_dLc=float(np.median(dLc)),
                     iqr_dL=float(np.subtract(*np.percentile(dL, [75, 25]))),
                     ratio=float(np.median(dLc) / max(np.median(dL), 1e-9)),
                     split_half_r=float(rel), n=len(L)))
    dists[b] = (dL, dLc)
    print(f"b={b:4d} ms  K={K:2d}  med L={np.median(L):8.1f}  "
          f"(noise sqrt3/dt={noise_line:7.1f}, ratio {np.median(L)/noise_line:.2f})  "
          f"med |dlogL| real={np.median(dL):.3f} ctrl={np.median(dLc):.3f}  "
          f"split-half r={rel:+.3f}", flush=True)

# -------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 3, figsize=(17.5, 9))
bb = np.array([r["b"] for r in rows])

a = ax[0, 0]
a.plot(bb, [r["med_dL"] for r in rows], "o-", color="#1e88e5", label="real")
a.plot(bb, [r["med_dLc"] for r in rows], "s--", color="#ff8f00", label="Poisson control")
a.set(xscale="log", xlabel="bin = step (ms)", ylabel="median |d log L| per step",
      title="(a) jaggedness of the instantaneous Lipschitz constant")
a.legend(fontsize=8)

a = ax[0, 1]
parts = a.violinplot([np.log10(np.maximum(dists[b][0], 1e-6)) for b in BINS],
                     positions=np.arange(len(BINS)), showmedians=True, widths=.8)
for p_ in parts["bodies"]:
    p_.set_facecolor("#1e88e5"); p_.set_alpha(.6)
parts2 = a.violinplot([np.log10(np.maximum(dists[b][1], 1e-6)) for b in BINS],
                      positions=np.arange(len(BINS)) + .0, showmedians=True, widths=.5)
for p_ in parts2["bodies"]:
    p_.set_facecolor("#ff8f00"); p_.set_alpha(.45)
a.set_xticks(range(len(BINS))); a.set_xticklabels(BINS)
a.set(xlabel="bin = step (ms)", ylabel="log10 |d log L|",
      title="(b) full distribution (blue real, orange control)\nthey separate once the bin beats the noise")

a = ax[0, 2]
a.plot(bb, [r["ratio"] for r in rows], "o-", color="#2e7d32")
a.axhline(1.0, color="k", ls="--", lw=.8, label="real = control (pure noise)")
a.set(xscale="log", xlabel="bin = step (ms)",
      ylabel="control / real median |d log L|",
      title="(c) separation from the Poisson control"); a.legend(fontsize=8)

a = ax[1, 0]
a.plot(bb, [r["split_half_r"] for r in rows], "o-", color="#8e24aa")
a.axhline(0, color="k", lw=.8)
a.set(xscale="log", xlabel="bin = step (ms)", ylabel="corr(log L) between neuron halves",
      title="(d) split-half reliability of L(t)")

a = ax[1, 1]
a.plot(bb, [r["medL"] for r in rows], "o-", color="#1e88e5", label="median L, real")
a.plot(bb, [r["medLc"] for r in rows], "s--", color="#ff8f00", label="control")
a.plot(bb, [r["noise"] for r in rows], ":", color="k", label=r"white noise $\sqrt{3}/dt$")
a.set(xscale="log", yscale="log", xlabel="bin = step (ms)", ylabel="median L (1/s)",
      title="(e) is L anything but noise?"); a.legend(fontsize=8)

a = ax[1, 2]
a.plot(bb, [r["K"] for r in rows], "o-", color="#1e88e5", label="real")
a.plot(bb, [r["Kc"] for r in rows], "s--", color="#ff8f00", label="Poisson control")
a.set(xscale="log", xlabel="bin = step (ms)", ylabel="elbow K",
      title="(f) elbow dimensionality\n(control K > 1 means the elbow is fitting noise)")
a.legend(fontsize=8)

fig.suptitle("Choosing the bin: where does the instantaneous Lipschitz constant "
             "stop being Poisson noise?", fontsize=13)
fig.tight_layout(); fig.savefig(OUT + "hippo_binscale.png", dpi=120); plt.close(fig)

import json
json.dump(rows, open(OUT + "hippo_binscale.json", "w"), indent=1)
print(f"\n[{time.time()-t0:.0f}s] saved hippo_binscale.png")
best = [r for r in rows if r["ratio"] > 1.3 and r["split_half_r"] > 0.3]
print("bins where L is signal (control/real > 1.3 and split-half r > 0.3):",
      [r["b"] for r in best] or "none")
