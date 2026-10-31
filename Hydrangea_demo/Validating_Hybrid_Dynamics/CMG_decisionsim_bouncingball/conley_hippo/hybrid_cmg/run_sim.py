"""Simulate the hybrid nested-rings scenario once and cache everything."""
import sys, time, pickle; sys.path.insert(0, ".")
import numpy as np
from hds import scenarios as sc, dynamics as dy

t0 = time.time()
system, readout, x0, meta = sc.nested_rings_hybrid(n_trials=24)
rec = system.simulate(x0, T=30.0, readout=readout, dt=0.001, base=0.005, burn=3.0, seed=0)
segs = readout.spikes(rec, system, seed=2)
print(f"simulated in {time.time()-t0:.0f}s; guard group {len(meta['guard_group'])} neurons, theta {meta['guard_theta']:.1f} Hz")
ev = rec['events']
kinds = {}
for e in ev: kinds[e[2]] = kinds.get(e[2], 0) + 1
print("events:", kinds)
X = rec['X']; mode = rec['mode']; guard = rec['guard']
ring = (np.sqrt(X[...,0]**2 + X[...,1]**2) > meta['rings'].r_bar).astype(int)
print(f"occupancy: high-gain {np.mean(mode==1):.1%}, guard {np.mean(guard==1):.1%}, outer ring {np.mean(ring==1):.1%}")
# guard visits: hits vs failures by origin ring
visits = []
for tr in range(X.shape[0]):
    g = np.r_[0, guard[tr], 0]
    on = np.flatnonzero(np.diff(g) == 1); off = np.flatnonzero(np.diff(g) == -1)
    jumps = sorted(b for (i, b, k, _) in ev if i == tr and k == 'jump')
    for a, b in zip(on, off):
        if mode[tr, a] != 0: continue
        hit = any(a <= j <= b for j in jumps)
        visits.append((tr, a, b, hit, ring[tr, a]))
v = np.array([(h, r) for *_, h, r in visits])
for r, nm in [(1, 'outer'), (0, 'inner')]:
    m = v[:, 1] == r
    print(f"  {nm}: {m.sum()} visits, hit rate {v[m,0].mean():.2f}")
# true Lipschitz per regime/ring
Xf = X.reshape(-1, 3); mf = mode.reshape(-1).astype(int); rf = ring.reshape(-1)
L = system.lipschitz_true(Xf, mf)
for mm, mn in [(0, 'base'), (1, 'high')]:
    for r, rn in [(1, 'outer'), (0, 'inner')]:
        sel = (mf == mm) & (rf == r)
        if sel.sum(): print(f"  true L [{mn},{rn}]: median {np.median(L[sel]):.1f} /s  (n={sel.sum()})")
allc = np.concatenate(segs); rates = allc.mean(0) / rec['base']
print(f"rates median {np.median(rates):.1f} Hz, 5-95% {np.percentile(rates,[5,95]).round(1)}")
pickle.dump(dict(rec=rec, segs=segs, visits=visits, Ltrue=L.reshape(X.shape[:2]),
                 ring=ring, meta={k: v for k, v in meta.items() if k != 'rings'},
                 rings=(meta['rings'].r_in, meta['rings'].r_bar, meta['rings'].r_out)),
            open("hds_sim.pkl", "wb"))
print("cached hds_sim.pkl")

# integrity check: every change of ring must be explained by a guard jump
chg = 0; unexplained = 0
for tr in range(X.shape[0]):
    ch = np.flatnonzero(np.diff(ring[tr]) != 0)
    J = np.array([b for (i, b, k, _) in ev if i == tr and k == 'jump'])
    for c in ch:
        chg += 1
        if not len(J) or np.abs(J - c).min() > 4:
            unexplained += 1
print(f"integrity: {chg} ring changes, {unexplained} not explained by a guard jump")
