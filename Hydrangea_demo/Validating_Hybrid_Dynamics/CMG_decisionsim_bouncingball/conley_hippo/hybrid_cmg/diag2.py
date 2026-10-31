import sys; sys.path.insert(0, "/home/claude/hybrid_cmg")
src = open("/home/claude/hybrid_cmg/run_cmg.py").read()
part1 = src.split("# ============================================================ PART 2")[0]
part2 = src.split("# ============================================================ PART 2")[1].split("B_SET = [20")[0]
pre = part1.split("# ============================================================ PART 1: GT")[0]
exec(compile(pre, "pre", "exec")); exec(compile("#" + part2, "p2", "exec"))
for (b, s, l) in [(60, 10, 90), (120, 10, 30), (120, 30, 90), (240, 10, 30), (30, 10, 90)]:
    o = decode_cmg(P['hybrid'], b, s, l, 'oracle')
    print(f"b={b} s={s} l={l}: sep={o['sep']} idx_ok={o['idx_ok']} resets={o['resets']}")
    for x in o['sets'][:6]:
        print(f"     n={x['n']:4d} idx {x['idx']} inner-frac {x['fin']:.2f} high-frac {x['hfrac']:.2f}")

# count clean pairs that cross rings (ground truth labels at both ends)
import numpy as np
b, s, l = 120, 10, 30
Pd = P['hybrid']
Xw, tt, tr = dc.window_counts(Pd['Xf'], Pd['trf'], Pd['ttf'], b, s)
Ls = int(round(l / s))
# true ring label at every 5 ms base bin -> per trial transitions of ring
ring = Pd['ring']
cross_t = []
for ti in range(ring.shape[0]):
    ch = np.flatnonzero(np.diff(ring[ti]) != 0)
    cross_t += [(ti, (c + 1) * Pd['base']) for c in ch]
jt = set((ti, round(t, 3)) for ti, t in Pd['jumps'])
near_jump = 0
for ti, t in cross_t:
    if any(abs(t - t2) < 0.02 for (i2, t2) in Pd['jumps'] if i2 == ti): near_jump += 1
print(f"\ntrue ring changes: {len(cross_t)}; within 20 ms of a jump: {near_jump}; "
      f"NOT explained by jumps: {len(cross_t) - near_jump}")
mode = Pd['mode']
# examine unexplained changes: radius trace around them
X = data['hybrid'][0]['X']
un = [(ti, t) for ti, t in cross_t if not any(abs(t - t2) < 0.02 for (i2, t2) in Pd['jumps'] if i2 == ti)]
for ti, t in un[:5]:
    bb = int(t / Pd['base'])
    r = np.sqrt(X[ti, bb-4:bb+5, 0]**2 + X[ti, bb-4:bb+5, 1]**2)
    print(f"   trial {ti} t={t:.3f}s mode {mode[ti, bb-2:bb+3]} rho {np.round(r, 3)}")
