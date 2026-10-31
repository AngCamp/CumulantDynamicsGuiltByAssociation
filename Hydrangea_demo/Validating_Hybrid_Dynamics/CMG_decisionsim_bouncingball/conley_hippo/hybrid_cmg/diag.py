import importlib.util, sys
sys.argv = ["x","128","128","32"]
spec = importlib.util.spec_from_file_location("bb", "bouncing_ball_cmg.py")
bb = importlib.util.module_from_spec(spec)
import types
# prevent main() from running
src = open("bouncing_ball_cmg.py").read().replace('if __name__ == "__main__":\n    main()', '')
exec(compile(src, "bb", "exec"), bb.__dict__)
F = bb.build_map()
comp, ncomp = bb.tarjan_scc(F)
members, rec = bb.recurrent_components(F, comp, ncomp)
cc = rec[0]; M = members[cc]
def desc(c):
    if c == bb.OUT: return "OUT"
    if c < bb.NB:
        i,j = divmod(c, bb.NY)
        return f"B(h={bb.H_MIN+(i+0.5)*bb.DH:.3f},v={bb.V_MIN+(j+0.5)*bb.DV:.3f})"
    m,k = divmod(c-bb.NB, bb.NP)
    return f"C(vg={bb.V_MIN+(m+0.5)*bb.DV:.3f},s={(k+0.5)*bb.DS:.3f})"
# highest-|v| base cell in the Morse set
hi = max((c for c in M if c < bb.NB), key=lambda c: abs(bb.V_MIN+(c%bb.NY+0.5)*bb.DV))
# low-energy member: cylinder cell near vg=0
lo = next(c for c in M if c >= bb.NB and (c-bb.NB)//bb.NP == bb.NG-1)
print("hi:", desc(hi), " lo:", desc(lo))
# BFS lo -> hi within full graph, print path
from collections import deque
def path(src, dst):
    prev = {src: None}; q = deque([src])
    while q:
        x = q.popleft()
        if x == dst: break
        for y in F[x]:
            if y not in prev:
                prev[y] = x; q.append(y)
    if dst not in prev: return None
    p = []; x = dst
    while x is not None: p.append(x); x = prev[x]
    return p[::-1]
p = path(lo, hi)
print("path lo->hi length:", None if p is None else len(p))
if p:
    for c in p: print("   ", desc(c))
