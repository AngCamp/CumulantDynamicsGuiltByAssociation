"""
Sim A (validation): planted ground-truth latent -> Poisson spiking neurons
-> the decoding pipeline, swept over bin b, step s, lag ell.

Latent: outer ring rho=1 (CCW, 3 rad/s, period 2.1 s) nesting a thin
counter-rotating inner band rho=0.72 (CW, 12 rad/s, period 0.52 s),
separated by an unstable circle at 0.86. Plus a weak OU third dimension.

Neurons: N=200, lambda_i = r0_i * exp(g * w_i . x + eta(t)),
  r0 ~ lognormal(median 7 Hz), g = 1.7, N = 1200, w_i unit vectors in 3D,
  eta = shared fast OU gain noise (tau 5 ms, sd 0.15).
  Poisson counts at 5 ms base resolution.
Noise floor calibrated by the knee test (target ~30 ms).

Score per setting (ground truth = true rho of each chart sample):
  - inner set: nontrivial-index Morse set with >80% inner-ring samples
  - outer set: same, >80% outer-ring samples
  - both index (1,1,0)?
  - reverse flow: inner and outer chart circulation have OPPOSITE sign
"""

import sys, os, time, itertools
sys.path.insert(0, "/home/claude/hybrid_cmg")
import numpy as np
import cmg_lib as cl
import pipeline as pl

rng = np.random.default_rng(7)
CACHE = "/home/claude/hybrid_cmg/simA_spk.npz"

DT = 0.001           # latent integration
BASE = 0.005         # spike-count base resolution
N_TRAJ, T_TRAJ, BURN = 40, 30.0, 3.0
W_OUT, W_IN = 3.0, -12.0
A_RAD = 70.0
R_IN, R_BAR, R_OUT = 0.72, 0.86, 1.0
SIG = 0.012
N_NEUR, GAIN = 1200, 1.7   # calibrated: noise-floor knee ~30 ms


def simulate():
    steps = int(T_TRAJ / DT)
    burn = int(BURN / DT)
    sub = int(BASE / DT)
    rho = np.where(np.arange(N_TRAJ) % 2 == 0,
                   R_OUT + rng.normal(0, .03, N_TRAJ),
                   R_IN + rng.normal(0, .02, N_TRAJ))
    th = rng.uniform(0, 2 * np.pi, N_TRAJ)
    z = rng.normal(0, .1, N_TRAJ)
    eta = np.zeros(N_TRAJ)
    n_out = (steps - burn) // sub
    X = np.zeros((N_TRAJ, n_out, 3))
    ETA = np.zeros((N_TRAJ, n_out))
    RHO = np.zeros((N_TRAJ, n_out))
    acc = np.zeros((N_TRAJ, 3)); acc_e = np.zeros(N_TRAJ); acc_r = np.zeros(N_TRAJ)
    k = 0
    sq = np.sqrt(DT)
    for s in range(steps):
        drho = -A_RAD * (rho - R_IN) * (rho - R_BAR) * (rho - R_OUT)
        w = np.where(rho > R_BAR, W_OUT, W_IN)
        rho = rho + drho * DT + SIG * sq * rng.normal(size=N_TRAJ)
        th = th + w * DT + SIG * sq * rng.normal(size=N_TRAJ)
        z = z - 3.0 * z * DT + 0.05 * sq * rng.normal(size=N_TRAJ)
        eta = eta - eta / 0.005 * DT + 0.15 * np.sqrt(2 / 0.005) * sq \
            * rng.normal(size=N_TRAJ)
        if s >= burn:
            acc += np.stack([rho * np.cos(th), rho * np.sin(th), z], 1)
            acc_e += eta; acc_r += rho
            if (s - burn + 1) % sub == 0 and k < n_out:
                X[:, k] = acc / sub; ETA[:, k] = acc_e / sub
                RHO[:, k] = acc_r / sub
                acc[:] = 0; acc_e[:] = 0; acc_r[:] = 0
                k += 1
    return X, ETA, RHO


def spikes(X, ETA, n_neur=N_NEUR, gain=GAIN, seed=1):
    r = np.random.default_rng(seed)
    W = r.normal(size=(n_neur, 3))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    r0 = r.lognormal(np.log(7.0), 0.6, n_neur)
    segs = []
    for tr in range(X.shape[0]):
        lam = r0 * np.exp(gain * X[tr] @ W.T + ETA[tr][:, None])
        segs.append(r.poisson(lam * BASE).astype(np.uint8))
    return segs, r0


if os.path.exists(CACHE):
    d = np.load(CACHE)
    X, ETA, RHO = d['X'][:24], d['ETA'][:24], d['RHO'][:24]
else:
    t0 = time.time()
    print("simulating latent ...")
    X, ETA, RHO = simulate()
    np.savez_compressed(CACHE, X=X, ETA=ETA, RHO=RHO)
    print(f"  done {time.time()-t0:.0f}s")
segs, r0 = spikes(X, ETA)
allc = np.concatenate(segs)
rates = allc.mean(0) / BASE
print(f"{X.shape[0]} trajectories x {T_TRAJ-BURN:.0f} s, {N_NEUR} neurons; "
      f"rate median {np.median(rates):.1f} Hz, "
      f"5-95% [{np.percentile(rates,5):.1f}, {np.percentile(rates,95):.1f}] Hz")
N_TRAJ = X.shape[0]
labels = [RHO[i] for i in range(N_TRAJ)]

# ------------------------- noise-floor calibration ---------------------------
# speed of the PCA latent vs bin size (s=b): noise regime ~ b^-1.5, signal
# regime flat; knee = noise floor. Also corr(speed, population rate).
print("\nnoise-floor calibration (latent speed vs bin, step = bin):")
knee_b, knee_v, knee_c = [], [], []
for b in [0.005, 0.010, 0.020, 0.030, 0.050, 0.080, 0.120, 0.200]:
    Rraw = [pl.bin_segment(sg, BASE, b, b)[0] for sg in segs]
    Rs = pl.normalize(Rraw)
    Ra = np.concatenate(Rs)
    mu, V = cl.pca_fit(Ra, d=3, max_n=40000)
    sp, pr = [], []
    for k, R in enumerate(Rs):
        L = cl.pca_apply(R, mu, V)
        sp.append(np.linalg.norm(np.diff(L, axis=0), axis=1) / b)
        pr.append(Rraw[k].sum(1)[1:])
    sp = np.concatenate(sp); pr = np.concatenate(pr)
    c = np.corrcoef(sp, pr)[0, 1]
    knee_b.append(b); knee_v.append(np.median(sp)); knee_c.append(c)
    print(f"   b={b*1e3:5.0f} ms  median latent speed {np.median(sp):9.1f}  "
          f"corr(speed, pop rate) {c:+.2f}")
lb, lv = np.log(knee_b), np.log(knee_v)
# knee: where local slope crosses -0.75 (midway between -1.5 and 0)
slopes = np.diff(lv) / np.diff(lb)
mid_b = np.exp((lb[1:] + lb[:-1]) / 2)
ki = np.argmax(slopes > -0.75) if (slopes > -0.75).any() else len(slopes) - 1
print(f"   local slopes: {np.round(slopes, 2)}")
print(f"   -> estimated noise floor (knee): ~{mid_b[ki]*1e3:.0f} ms")

# --------------------------------- sweep -------------------------------------
B_SET = [0.010, 0.030, 0.060, 0.120, 0.240]
S_SET = [0.010, 0.030]
L_SET = [0.010, 0.030, 0.090, 0.180]
rows = []
print("\nsweep (b, s, ell) -> recovery:")
for b, s, ell in itertools.product(B_SET, S_SET, L_SET):
    if ell < s or abs(ell / s - round(ell / s)) > 1e-9:
        continue
    t0 = time.time()
    out = pl.decode(segs, BASE, b, s, ell, labels=labels, nx=64)
    an, lab = out['an'], out['labels']
    found = []
    for cc in an['rec']:
        idx = an['index'].get(cc)
        if idx is None or idx == (0, 0, 0):
            continue
        m = np.isin(out['cells'], an['members'][cc])
        if m.sum() < 20:
            continue
        fin = (lab[m] < R_BAR).mean()
        rot = pl.rotation_sense(out['Zs'], out['cells'], an['members'][cc],
                                out['L_steps'])
        found.append((cc, idx, fin, rot, len(an['members'][cc])))
    inner = [f for f in found if f[2] > 0.8]
    outer = [f for f in found if f[2] < 0.2]
    mixed = [f for f in found if 0.2 <= f[2] <= 0.8]
    sep = bool(inner and outer and not mixed)
    idx_ok = sep and inner[0][1] == (1, 1, 0) and outer[0][1] == (1, 1, 0)
    rev = sep and np.sign(inner[0][3]) == -np.sign(outer[0][3]) \
        and inner[0][3] != 0
    status = ("SEPARATE+REVERSED" if sep and rev else
              "separate" if sep else
              "MERGED" if mixed else
              "outer only" if outer else "inner only" if inner else "none")
    rows.append(dict(b=b, s=s, ell=ell, sep=sep, idx_ok=idx_ok, rev=rev,
                     status=status,
                     idx=[f[1] for f in found][:3], nset=len(an['rec'])))
    print(f"   b={b*1e3:4.0f} s={s*1e3:3.0f} ell={ell*1e3:4.0f} ms  "
          f"{status:18s} idx_ok={idx_ok!s:5}  sets={len(an['rec'])} "
          f"nontriv={[(f[1], round(f[2],2)) for f in found][:3]}  "
          f"({time.time()-t0:.1f}s)")

np.save("/home/claude/hybrid_cmg/simA_sweep.npy", rows, allow_pickle=True)

# --------------------------------- figure ------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 9.5))
ax = fig.add_subplot(2, 3, 1)
ax.loglog(np.array(knee_b) * 1e3, knee_v, 'o-', color="#37474f")
ref = np.array(knee_b) * 1e3
ax.loglog(ref, knee_v[0] * (ref / ref[0]) ** -1.5, ':', color="#b71c1c",
          label="noise slope -3/2")
ax.axvline(mid_b[ki] * 1e3, color="#1e88e5", ls="--",
           label=f"knee ~{mid_b[ki]*1e3:.0f} ms")
ax.set(xlabel="bin b (ms)", ylabel="median latent speed",
       title="(a) noise-floor calibration")
ax.legend(fontsize=8)
ax2 = ax.twinx()
ax2.plot(np.array(knee_b) * 1e3, knee_c, 's--', color="#ffb300", ms=4)
ax2.set_ylabel("corr(speed, pop rate)", color="#ffb300")

for p, s in enumerate(S_SET):
    ax = fig.add_subplot(2, 3, 2 + p)
    M = np.full((len(B_SET), len(L_SET)), np.nan)
    for r in rows:
        if r['s'] != s:
            continue
        i, j = B_SET.index(r['b']), L_SET.index(r['ell'])
        M[i, j] = 2 if (r['sep'] and r['rev'] and r['idx_ok']) else \
            1.5 if (r['sep'] and r['rev']) else 1 if r['sep'] else 0
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(L_SET)))
    ax.set_xticklabels([f"{l*1e3:.0f}" for l in L_SET])
    ax.set_yticks(range(len(B_SET)))
    ax.set_yticklabels([f"{b*1e3:.0f}" for b in B_SET])
    ax.set(xlabel="lag ell (ms)", ylabel="bin b (ms)",
           title=f"({'bc'[p]}) step s = {s*1e3:.0f} ms\n"
                 "green: both rings + reversal + (1,1,0)x2\n"
                 "yellow: separated | red: merged/lost | grey: invalid")
    for i in range(len(B_SET)):
        for j in range(len(L_SET)):
            if np.isnan(M[i, j]):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           color="#cfd8dc"))

# example charts: best and worst
def show(ax, b, s, ell, title):
    out = pl.decode(segs, BASE, b, s, ell, labels=labels, nx=64)
    Z, lab = out['Z'], out['labels']
    sub = slice(0, len(Z), 5)
    ax.scatter(Z[sub, 0], Z[sub, 1], c=lab[sub], s=1, cmap="coolwarm",
               vmin=0.6, vmax=1.05)
    ax.set_title(title, fontsize=10)

good = [r for r in rows if r['sep'] and r['rev'] and r['idx_ok']]
bad = [r for r in rows if not r['sep']]
if good:
    r = good[len(good) // 2]
    show(fig.add_subplot(2, 3, 4), r['b'], r['s'], r['ell'],
         f"(d) recovered: b={r['b']*1e3:.0f} s={r['s']*1e3:.0f} "
         f"ell={r['ell']*1e3:.0f} ms\n(color = true rho)")
if bad:
    r = bad[0]
    show(fig.add_subplot(2, 3, 5), r['b'], r['s'], r['ell'],
         f"(e) failed: b={r['b']*1e3:.0f} s={r['s']*1e3:.0f} "
         f"ell={r['ell']*1e3:.0f} ms -> {r['status']}")
if len(bad) > 1:
    r = bad[-1]
    show(fig.add_subplot(2, 3, 6), r['b'], r['s'], r['ell'],
         f"(f) failed: b={r['b']*1e3:.0f} s={r['s']*1e3:.0f} "
         f"ell={r['ell']*1e3:.0f} ms -> {r['status']}")

fig.suptitle("Sim A through the decoding pipeline: Poisson spikes -> bin/step "
             "-> PCA -> delay embedding -> CMG", fontsize=13)
fig.tight_layout()
fig.savefig("/home/claude/hybrid_cmg/simA_pipeline_sweep.png", dpi=125)
print("\nfigure saved: simA_pipeline_sweep.png")
