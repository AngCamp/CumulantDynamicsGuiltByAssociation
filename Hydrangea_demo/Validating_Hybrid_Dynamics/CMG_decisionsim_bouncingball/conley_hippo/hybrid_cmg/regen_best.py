import sys, pickle; sys.path.insert(0, "/home/claude/hybrid_cmg")
src = open("/home/claude/hybrid_cmg/run_cmg.py").read()
part1 = src.split("# ============================================================ PART 2")[0]
part2 = src.split("# ============================================================ PART 2")[1]
pre = part1.split("# ============================================================ PART 1: GT")[0]
defs = part2.split("B_SET = [20")[0]
fig_block = "def best(" + part2.split("def best(")[1].split("# ============================================ Lipschitz guard")[0]
exec(compile(pre, "pre", "exec")); exec(compile("#" + defs, "defs", "exec"))
B_SET = [20, 30, 60, 120, 240]; S_SET = [10, 30]; L_SET = [10, 30, 90, 180]
pal = ["#d81b60", "#1e88e5", "#ffc107", "#004d40", "#8e24aa", "#e65100", "#00838f"]
results = pickle.load(open(OUT + "cmg_sweep.pkl", "rb"))
exec(compile(fig_block, "fig", "exec"))
print("best:", best('flow', None), best('hybrid', 'oracle'))
