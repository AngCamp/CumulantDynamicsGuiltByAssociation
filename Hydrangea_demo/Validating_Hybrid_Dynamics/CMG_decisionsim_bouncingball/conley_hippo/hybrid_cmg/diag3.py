import sys; sys.path.insert(0, "/home/claude/hybrid_cmg")
src = open("/home/claude/hybrid_cmg/run_cmg.py").read()
part1 = src.split("# ============================================================ PART 2")[0]
part2 = src.split("# ============================================================ PART 2")[1].split("B_SET = [20")[0]
pre = part1.split("# ============================================================ PART 1: GT")[0]
exec(compile(pre, "pre", "exec")); exec(compile("#" + part2, "p2", "exec"))
for (b, s, l) in [(120, 30, 90), (240, 30, 90)]:
    o = decode_cmg(P['hybrid'], b, s, l, 'oracle')
    print(f"b={b} s={s} l={l}:")
    for x in o['sets'][:8]:
        print(f"   n={x['n']:4d} idx {x['idx']} inner-frac {x['fin']:.2f} high-frac {x['hfrac']:.2f}")
