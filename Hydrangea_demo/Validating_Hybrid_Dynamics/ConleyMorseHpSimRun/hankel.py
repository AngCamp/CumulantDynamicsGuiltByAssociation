"""
hankel.py
---------
Hankel-matrix (delay-PCA / SSA / HAVOK) analysis of the CA1 simulation, aimed at
fast states such as SWRs.

  1. spikes (output/simulation.npz) -> 25 ms bin counts, z-scored
  2. covariance PCA of the counts -> top-K PC scores P(t)    (K = eigenvalue-decay elbow)
  3. Hankel matrix: row t = (P(t), P(t-1), ..., P(t-L+1)), one 25 ms chunk per lag
  4. SVD of the centred Hankel matrix: H = U S V^T
       component i: time series v_i(t) = U[:, i] S[i]; lag pattern V[i] (L lags x K PCs)
  5. every component is scored after the fact against the simulator's states
       (scoring only, never used to build anything):
       SWR effect = (mean v_i during SWR - mean v_i during stillness) / sd during stillness
       timescale  = time for the autocorrelation of v_i to fall below 1/e
  6. HAVOK (Brunton et al. 2017): regress dv/dt of components 1..r-1 on 1..r; the
     last one, v_r, acts as the forcing term that should spike at intermittent fast events

Run:  python hankel.py [L] [r]       defaults L = 20 lags (500 ms), r = 15
Writes output/hankel_L<L>.png and prints the components most related to SWRs.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neurosim import binned_counts

OUT, BIN_MS = "output", 25.0
L = int(sys.argv[1]) if len(sys.argv) > 1 else 20
R_HAVOK = int(sys.argv[2]) if len(sys.argv) > 2 else 15
N_COMP = 40                                        # components scored

# ------------------------------------------------------------------ 25 ms PCA
sim = np.load(f"{OUT}/simulation.npz")
n_steps, dt_ms = int(sim["n_steps"]), float(sim["dt_ms"])
S = np.unpackbits(sim["spikes"], axis=0, count=n_steps).astype(bool)
w = int(BIN_MS / dt_ms)
X = binned_counts(S, dt_ms, BIN_MS)
nb = len(X)
st = sim["state"][: nb * w].reshape(nb, w)
code = np.array([np.bincount(r, minlength=3).argmax() for r in st])     # 0 WALK, 1 STILL, 2 SWR
Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
ev, E = np.linalg.eigh(np.cov(Xz.T))
ev, E = ev[::-1], E[:, ::-1]
k = np.arange(1, 31)
K = int(k[np.argmax(ev[0] + (ev[29] - ev[0]) / 29 * (k - 1) - ev[:30])])
P = Xz @ E[:, :K]

# ------------------------------------------------------------------ Hankel SVD
n = nb - L + 1
H = np.stack([P[L - 1 - j: L - 1 - j + n] for j in range(L)], 1).reshape(n, L * K)   # row t ends at bin t+L-1
H = H - H.mean(0)
U, s, Vt = np.linalg.svd(H, full_matrices=False)
N_COMP = min(N_COMP, H.shape[1])
V = U[:, :N_COMP] * s[:N_COMP]                       # component time series
c = code[L - 1:]                                     # state at the most recent bin of each row

# ------------------------------------------------------------------ score components (after the fact)
swr, still = c == 2, c == 1
sd = V[still].std(0) + 1e-12
effect = (V[swr].mean(0) - V[still].mean(0)) / sd
walk_eff = (V[c == 0].mean(0) - V[still].mean(0)) / sd


def acf_time(x):
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * len(x))
    a = np.fft.irfft(f * np.conj(f))[: len(x)]
    a /= a[0]
    below = np.flatnonzero(a < np.exp(-1))
    return (below[0] if len(below) else len(x)) * BIN_MS


tscale = np.array([acf_time(V[:, i]) for i in range(N_COMP)])
order = np.argsort(-np.abs(effect))
print(f"{nb} bins, K = {K} PCs, L = {L} lags ({L * BIN_MS:.0f} ms), Hankel {H.shape}")
print("components most related to SWRs (|effect| = shift during SWR in stillness-sd units):")
for i in order[:8]:
    print(f"  comp {i + 1:2d}: SWR effect {effect[i]:+5.2f}   WALK effect {walk_eff[i]:+5.2f}   "
          f"autocorr time {tscale[i]:6.0f} ms   variance {s[i] ** 2 / (s ** 2).sum():.3%}")

# ------------------------------------------------------------------ HAVOK forcing
r = R_HAVOK
Vr = U[:, :r] * s[:r]
dV = (Vr[2:, : r - 1] - Vr[:-2, : r - 1]) / (2 * BIN_MS / 1000)
Xi, *_ = np.linalg.lstsq(Vr[1:-1], dV, rcond=None)
forcing = Vr[:, r - 1]
f_eff = (forcing[swr].mean() - forcing[still].mean()) / (forcing[still].std() + 1e-12)
f_abs = (np.abs(forcing[swr]).mean() - np.abs(forcing[still]).mean()) / (np.abs(forcing[still]).std() + 1e-12)
print(f"HAVOK r = {r}: forcing v_{r} SWR effect {f_eff:+.2f} (signed), {f_abs:+.2f} (magnitude)")

# ------------------------------------------------------------------ event-triggered averages
on = np.flatnonzero(np.diff(swr.astype(int)) == 1) + 1
win = np.arange(-20, 21)
on = on[(on + win[0] >= 0) & (on + win[-1] < n)]


def eta(x):
    return np.array([x[on + d].mean() for d in win]) / (x[still].std() + 1e-12)


# ------------------------------------------------------------------ figure
best = order[0]
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(3, 3)
ax = fig.add_subplot(gs[0, 0])
ax.semilogy(np.arange(1, min(60, len(s)) + 1), s[:60] ** 2 / (s ** 2).sum(), "o-", ms=3)
ax.set(xlabel="Hankel component", ylabel="fraction of variance", title=f"singular spectrum (K = {K} PCs x {L} lags)")
ax = fig.add_subplot(gs[0, 1])
sc = ax.scatter(np.arange(1, N_COMP + 1), effect, c=tscale, cmap="viridis_r", s=30, norm=matplotlib.colors.LogNorm())
plt.colorbar(sc, ax=ax, label="autocorrelation time (ms)")
ax.axhline(0, color="0.5", lw=0.6)
ax.set(xlabel="Hankel component (variance order)", ylabel="SWR effect (stillness sd)",
       title="which components respond during SWRs")
ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(Vt[best].reshape(L, K).T, aspect="auto", cmap="RdBu_r",
               extent=[-0.5 * BIN_MS, (L - 0.5) * BIN_MS, K + 0.5, 0.5])
plt.colorbar(im, ax=ax)
ax.set(xlabel="lag (ms back)", ylabel="PC", title=f"lag pattern of component {best + 1} (strongest SWR response)")
ax = fig.add_subplot(gs[1, :])
t0 = on[len(on) // 2] - 200
sl = slice(max(0, t0), t0 + 400)
tt = (np.arange(n)[sl] + L - 1) * BIN_MS / 1000
ax.plot(tt, V[sl, best] / sd[best], lw=0.8, label=f"component {best + 1}")
ax.plot(tt, forcing[sl] / forcing[still].std(), lw=0.8, alpha=0.8, label=f"HAVOK forcing v_{r}")
for i in np.flatnonzero(swr[sl]):
    ax.axvspan(tt[i] - BIN_MS / 2000, tt[i] + BIN_MS / 2000, color="#d81b60", alpha=0.25, lw=0)
walk_sl = c[sl] == 0
ax.fill_between(tt, ax.get_ylim()[0], ax.get_ylim()[1], where=walk_sl, color="#1e88e5", alpha=0.08, lw=0)
ax.set(xlabel="time (s)", ylabel="stillness sd",
       title="10 s of data: pink = true SWR bins, blue = WALK (scoring only)")
ax.legend(fontsize=8)
ax = fig.add_subplot(gs[2, 0])
for i in order[:4]:
    ax.plot(win * BIN_MS, eta(V[:, i]), label=f"comp {i + 1} ({tscale[i]:.0f} ms)")
ax.plot(win * BIN_MS, eta(forcing), "k--", label=f"HAVOK v_{r}")
ax.axvline(0, color="#d81b60", lw=0.8)
ax.set(xlabel="time from SWR onset (ms)", ylabel="mean (stillness sd)",
       title=f"SWR-triggered average ({len(on)} onsets)")
ax.legend(fontsize=7)
ax = fig.add_subplot(gs[2, 1])
ax.scatter(walk_eff, effect, c=tscale, cmap="viridis_r", norm=matplotlib.colors.LogNorm(), s=25)
for i in order[:4]:
    ax.annotate(str(i + 1), (walk_eff[i], effect[i]), fontsize=8)
ax.axhline(0, color="0.6", lw=0.5); ax.axvline(0, color="0.6", lw=0.5)
ax.set(xlabel="WALK effect", ylabel="SWR effect", title="components separate WALK and SWR differently")
ax = fig.add_subplot(gs[2, 2])
ax.hist(V[still, best] / sd[best], bins=60, alpha=0.6, density=True, label="STILL")
ax.hist(V[swr, best] / sd[best], bins=60, alpha=0.6, density=True, label="SWR")
ax.hist(V[c == 0, best] / sd[best], bins=60, alpha=0.4, density=True, label="WALK")
ax.set(xlabel=f"component {best + 1} (stillness sd)", title="distribution by state")
ax.legend(fontsize=8)
fig.suptitle("Hankel SVD of the 25 ms latents: searching for the fast SWR state", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/hankel_L{L}.png", dpi=110); plt.close(fig)
print(f"wrote {OUT}/hankel_L{L}.png")
