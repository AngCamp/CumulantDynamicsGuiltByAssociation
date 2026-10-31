import sys; sys.path.insert(0, ".")
src = open("run_cmg.py").read()
part1 = src.split("# ============================================================ PART 2")[0]
part2 = src.split("# ============================================================ PART 2")[1].split("B_SET = [20")[0]
# skip GT compute for speed: define minimal context
pre = part1.split("# ============================================================ PART 1: GT")[0]
exec(compile(pre, "pre", "exec"))
exec(compile("#" + part2, "p2", "exec"))
import numpy as np
for scen, routing in [('flow', None), ('hybrid', 'oracle')]:
    o = decode_cmg(P[scen], 120, 10, 30, routing, keep=True)
    an = o['an']
    print(f"\n{scen}/{routing}: {len(an['rec'])} recurrent comps; nontrivial sets:")
    for cc_ in sorted(an['rec'], key=lambda c: -len(an['members'][c]))[:6]:
        msk = np.isin(o['cz'], an['members'][cc_])
        print(f"   {len(an['members'][cc_]):5d} cells idx {an['index'].get(cc_)} "
              f"inner frac {(o['ring_s'][msk]==0).mean():.2f}  high-mode frac {(o['mode_s'][msk]>0).mean():.2f}  n_samp {msk.sum()}")
    # chart overlap of rings irrespective of map
    g = o['g']; cz = o['cz']
    ci = np.bincount(cz[(cz>=0)&(o['ring_s']==0)], minlength=g.N)
    co = np.bincount(cz[(cz>=0)&(o['ring_s']==1)], minlength=g.N)
    print(f"   chart cells with both rings (>2 samples each): {((ci>2)&(co>2)).sum()}  inner-only {((ci>2)&(co<=2)).sum()} outer-only {((co>2)&(ci<=2)).sum()}")
    hm = o['mode_s'] > 0
    ch = np.bincount(cz[(cz>=0)&hm], minlength=g.N)
    print(f"   high-mode samples: {hm.mean():.2%}; cells containing high-mode samples {(ch>2).sum()}")

o = decode_cmg(P['hybrid'], 120, 10, 30, 'oracle', keep=True)
cz, g = o['cz'], o['g']
combo = o['ring_s'] * 2 + (o['mode_s'] > 0)        # 0 inner/base 1 inner/high 2 outer/base 3 outer/high
names = ['inner/base', 'inner/high', 'outer/base', 'outer/high']
cnt = np.zeros((4, g.N))
for k in range(4):
    cnt[k] = np.bincount(cz[(cz >= 0) & (combo == k)], minlength=g.N)
both = (cnt[[0, 1]].sum(0) > 2) & (cnt[[2, 3]].sum(0) > 2)
print("\noverlap cells:", both.sum(), " composition of samples in overlap cells:")
for k in range(4):
    print(f"   {names[k]:11s} {cnt[k, both].sum():.0f}")
# chart radius by combo
c0 = o['Z'].mean(0)
rad = np.linalg.norm(o['Z'] - c0, axis=1)
for k in range(4):
    m = combo == k
    print(f"   chart radius {names[k]:11s} median {np.median(rad[m]):.2f}  IQR {np.percentile(rad[m],25):.2f}-{np.percentile(rad[m],75):.2f}")
