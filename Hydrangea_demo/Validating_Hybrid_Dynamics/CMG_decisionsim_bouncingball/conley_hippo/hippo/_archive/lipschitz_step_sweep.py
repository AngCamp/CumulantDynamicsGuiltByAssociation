"""
Distribution of the change in the instantaneous Lipschitz constant, as a
function of the TIME STEP, with the bin held fixed.

  PCA is computed ONCE, on 100 ms bins stepped every 5 ms (per-neuron
  normalization: each neuron divided by its own total spikes, then z-scored).
  The elbow-K scores p(t) are the trajectory.

  For each time step s (a multiple of 5 ms) the trajectory is subsampled every
  s/5 samples and, with dt = s,

      Lp = |p(t+1) - p(t)| / dt                  speed
      Lv = |p(t+1) - 2p(t) + p(t-1)| / dt^2      acceleration
      L  = Lv / Lp                               instantaneous Lipschitz (1/s)
      dlogL = |log L(t+1) - log L(t)|            its change, per step

  Only the step varies, so any change in the distribution comes from the
  sampling interval and not from the amount of smoothing in the bin.

  Reference line: for an uncorrelated (white) trajectory, L = sqrt(3)/dt.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/claude/hybrid_cmg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmg_lib as cl
from hds import decode as dc

dc.STEP_MS = 5.0
OUT = "/home/claude/hippo/"
BIN_MS = 100.0
STEPS = [5, 10, 20, 25, 50, 100, 200, 400, 800]
STATE_NAMES = ["WALK", "STILL", "SWR"]
t0 = time.time()

d = np.load(OUT + "output_v2/simulation.npz")
n_steps = int(d["n_steps"])
S = np.unpackbits(d["spikes"], axis=0, count=n_steps).astype(np.uint8)
state_ms = d["state"].astype(int)
print(f"{S.shape[1]} neurons, {n_steps/1000:.0f} s, mean rate {S.mean()/0.001:.2f} Hz")
for k, nm in enumerate(STATE_NAMES):
    print(f"   {nm}: {(state_ms==k).mean():.2%}")

# ---------------------------------------------------------------- PCA once
Xf, trf, ttf = dc.fine_bins([S], 0.001)
Xw, tt, tr = dc.window_counts(Xf, trf, ttf, BIN_MS, dc.STEP_MS)
Z = dc.normalize(Xw)                      # per-neuron total, then z-score
C = (Z.T @ Z) / len(Z)
ev, vec = np.linalg.eigh(C.astype(np.float64))
o = np.argsort(ev)[::-1]; ev, vec = ev[o], vec[:, o]
K = dc.elbow_k(ev)
P = Z @ vec[:, :max(K, 2)]
idx = np.clip((tt * 1000).astype(int), 0, n_steps - 1)
y = state_ms[idx]
print(f"PCA: bin {BIN_MS:.0f} ms, step {dc.STEP_MS:.0f} ms -> {P.shape[0]} samples, "
      f"elbow K = {K}, top eigenvalues {np.round(ev[:5],2)}")

def lipschitz(p, dt):
    Lp = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    Lv = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1) / dt ** 2
    return Lv / np.maximum(Lp[:-1], 1e-12)

rows, Ls, dLs = [], {}, {}
for s_ms in STEPS:
    m = int(round(s_ms / dc.STEP_MS))
    dt = s_ms / 1000.0
    p = P[::m]
    if len(p) < 50:
        continue
    L = lipschitz(p, dt)
    dL = np.abs(np.diff(np.log(np.maximum(L, 1e-12))))
    Ls[s_ms], dLs[s_ms] = L, dL
    noise = np.sqrt(3) / dt
    rows.append(dict(s=s_ms, n=len(L), medL=float(np.median(L)), noise=float(noise),
                     ratio=float(np.median(L) / noise),
                     med_dL=float(np.median(dL)),
                     iqr_dL=float(np.subtract(*np.percentile(dL, [75, 25]))),
                     p90_dL=float(np.percentile(dL, 90))))
    print(f"  step {s_ms:4d} ms  n={len(L):6d}  median L {np.median(L):8.2f}  "
          f"(sqrt3/dt = {noise:8.2f}, ratio {np.median(L)/noise:.3f})   "
          f"median |dlogL| {np.median(dL):.3f}  IQR {rows[-1]['iqr_dL']:.3f}")

ss = np.array([r["s"] for r in rows])

# -------------------------------------------------------------- the figure
fig = plt.figure(figsize=(17.5, 9.5))
gs = fig.add_gridspec(2, 3)

ax = fig.add_subplot(gs[0, 0])
data = [dLs[s_] for s_ in ss]
parts = ax.violinplot(data, positions=np.arange(len(ss)), showmedians=True, widths=.85)
for b_ in parts["bodies"]:
    b_.set_facecolor("#1e88e5"); b_.set_alpha(.65)
ax.set_xticks(range(len(ss))); ax.set_xticklabels(ss)
ax.set(xlabel="time step (ms)", ylabel="|d log L| per step",
       title=f"(a) distribution of the Lipschitz change\n(bin fixed at {BIN_MS:.0f} ms)")

ax = fig.add_subplot(gs[0, 1])
ax.plot(ss, [r["med_dL"] for r in rows], "o-", color="#1e88e5", label="median")
ax.fill_between(ss, [np.percentile(dLs[s_], 25) for s_ in ss],
                [np.percentile(dLs[s_], 75) for s_ in ss], color="#1e88e5", alpha=.2,
                label="IQR")
ax.plot(ss, [r["p90_dL"] for r in rows], "--", color="#1e88e5", alpha=.7, label="90th pct")
ax.set(xscale="log", xlabel="time step (ms)", ylabel="|d log L| per step",
       title="(b) median and spread of the change")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2])
ax.plot(ss, [r["medL"] for r in rows], "o-", color="#2e7d32", label="median L (data)")
ax.plot(ss, [r["noise"] for r in rows], ":", color="k", label=r"white noise $\sqrt{3}/dt$")
ax.set(xscale="log", yscale="log", xlabel="time step (ms)", ylabel="L (1/s)",
       title="(c) the Lipschitz constant itself")
ax.legend(fontsize=8)
ax2 = ax.twinx()
ax2.plot(ss, [r["ratio"] for r in rows], "s--", color="#d81b60", ms=4)
ax2.axhline(1.0, color="#d81b60", lw=.6, ls=":")
ax2.set_ylabel("L / white-noise value", color="#d81b60")

ax = fig.add_subplot(gs[1, 0])
for s_, col in zip([s for s in (10, 50, 200, 800) if s in Ls],
                   ["#90caf9", "#1e88e5", "#1565c0", "#0d47a1"]):
    v = np.log10(np.maximum(Ls[s_], 1e-12))
    ax.hist(v, bins=70, histtype="step", density=True, color=col, label=f"step {s_} ms")
ax.set(xlabel="log10 L (1/s)", ylabel="density",
       title="(d) distribution of L at several steps")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1:])
s_show = 50 if 50 in Ls else ss[0]
m = int(round(s_show / dc.STEP_MS))
tt_s = tt[::m][1:-1][:len(Ls[s_show])]
ax.plot(tt_s, Ls[s_show], lw=.5, color="#37474f", label=f"L(t), step {s_show} ms")
ax.axhline(np.sqrt(3) / (s_show / 1000.0), color="k", ls=":", lw=.8,
           label=r"$\sqrt{3}/dt$")
ys = y[::m][1:-1][:len(Ls[s_show])]
for k, nm in enumerate(STATE_NAMES):
    ax.fill_between(tt_s, 0, 1, where=(ys == k), transform=ax.get_xaxis_transform(),
                    color=["#1e88e5", "#9e9e9e", "#d81b60"][k], alpha=.13, lw=0)
ax.set(yscale="log", xlabel="time (s)", ylabel="L (1/s)", xlim=(20, 90),
       title="(e) L(t) with the true state shaded "
             "(blue WALK, grey STILL, red SWR)")
ax.legend(fontsize=8, loc="upper right")

fig.suptitle(f"Change in the instantaneous Lipschitz constant vs time step  "
             f"(PCA computed once at a {BIN_MS:.0f} ms bin, K={K}; "
             f"pyramidal rate raised, inhibition reduced)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT + "hippo_lipschitz_steps.png", dpi=125)
print(f"\n[{time.time()-t0:.0f}s] saved hippo_lipschitz_steps.png")
