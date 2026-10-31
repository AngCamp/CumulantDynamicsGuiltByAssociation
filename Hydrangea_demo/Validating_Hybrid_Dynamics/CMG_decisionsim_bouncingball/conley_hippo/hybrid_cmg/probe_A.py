import sys; sys.path.insert(0, ".")
import numpy as np, importlib
import cmg_lib as cl, pipeline as pl
src = open("simA_spiking.py").read().split("# ------------------------- noise-floor calibration")[0]
exec(src)
def run(b, s, ell, nx, pad):
    orig = cl.build_map
    cl.build_map = lambda a, c, x, y, src_dilate=0, tgt_pad=1: orig(a, c, x, y, src_dilate=0, tgt_pad=pad)
    out = pl.decode(segs, BASE, b, s, ell, labels=labels, nx=nx)
    cl.build_map = orig
    an, lab = out['an'], out['labels']
    res = []
    for cc in an['rec']:
        idx = an['index'].get(cc)
        if idx is None or idx == (0,0,0): continue
        m = np.isin(out['cells'], an['members'][cc])
        if m.sum() < 20: continue
        res.append((idx, round((lab[m] < R_BAR).mean(), 2)))
    return res
for (b, s, ell) in [(0.06, 0.01, 0.03), (0.03, 0.01, 0.03), (0.01, 0.01, 0.01)]:
    for nx, pad in [(64, 1), (64, 0), (96, 0), (128, 0)]:
        print(f"b={b*1e3:.0f} s={s*1e3:.0f} ell={ell*1e3:.0f}  nx={nx} pad={pad}: {run(b,s,ell,nx,pad)}", flush=True)
