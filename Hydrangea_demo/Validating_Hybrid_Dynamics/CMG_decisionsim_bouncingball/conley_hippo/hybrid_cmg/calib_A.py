import sys; sys.path.insert(0, ".")
import numpy as np, cmg_lib as cl, pipeline as pl
d = np.load("simA_spk.npz"); X, ETA = d['X'][:20], d['ETA'][:20]
BASE = 0.005
def knee(n, med, gain, seed=1):
    r = np.random.default_rng(seed)
    W = r.normal(size=(n, 3)); W /= np.linalg.norm(W, axis=1, keepdims=True)
    r0 = r.lognormal(np.log(med), 0.6, n)
    segs = [r.poisson(r0*np.exp(gain*X[t]@W.T + ETA[t][:,None])*BASE).astype(np.uint8)
            for t in range(X.shape[0])]
    rates = np.concatenate(segs).mean(0)/BASE
    bs = [0.005,0.01,0.02,0.03,0.05,0.08,0.12,0.2]; v=[]
    for b in bs:
        Rs = [pl.bin_segment(s, BASE, b, b)[0] for s in segs]
        mu, V = cl.pca_fit(np.concatenate(Rs), d=3, max_n=30000)
        sp = np.concatenate([np.linalg.norm(np.diff(cl.pca_apply(R,mu,V),axis=0),axis=1)/b for R in Rs])
        v.append(np.median(sp))
    lb, lv = np.log(bs), np.log(v)
    sl = np.diff(lv)/np.diff(lb); mid = np.exp((lb[1:]+lb[:-1])/2)
    ki = np.argmax(sl > -0.75) if (sl > -0.75).any() else len(sl)-1
    return mid[ki]*1e3, np.round(sl,2), np.percentile(rates,[5,50,95])
for n, med, g in [(200,4,1.45),(800,6,1.6),(800,7,1.7),(1200,7,1.7),(1500,8,1.6)]:
    k, sl, rr = knee(n, med, g)
    print(f"N={n:5d} med={med} gain={g}: knee ~{k:.0f} ms  rates5/50/95={np.round(rr,1)}  slopes={sl}", flush=True)
