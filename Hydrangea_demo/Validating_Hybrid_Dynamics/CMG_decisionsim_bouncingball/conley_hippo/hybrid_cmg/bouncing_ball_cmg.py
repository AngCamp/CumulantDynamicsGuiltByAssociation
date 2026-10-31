"""
Conley-Morse graph of the bouncing ball as a HYBRID system, computed on the
hybrid suspension (Kalies-Rivas-Wehbe, arXiv:2610.15123, Sec. 6.1 setup).

System:  (h', v') = (v, -g),  guard G = {h=0, v<=0},  reset r(0,v) = (0,-c v)
Params (paper):  g = 9.81, c = 0.8, tau = 0.5, R = [0,2] x [-5,5]
Paper's answer:  ONE Morse set, Conley index dims (1,1,0)  ~ H_*(S^1)

Suspension: base chart = R minus G (flow), cylinder chart = (G cap R) x [0,1]
with (g,0) ~ g  and  (g,1) ~ r(g).  Each reset costs 1 unit of suspension
time, which regularizes Zeno (at most one completed reset per tau=0.5 step).

The multivalued map is built heuristically by sampling cell corners/centers
(same spirit as the paper: "constructed heuristically via sampling"),
with one-cell in-chart padding.  Morse graph via Tarjan SCC; Conley index of
the minimal Morse set via Z/2 cellular homology of its attracting block
(relative homology of the pair (N,L), L = exit part; L is expected empty
for an attractor, giving the absolute homology of a block ~ S^1).
"""

import sys, time
from collections import defaultdict

# ----------------------------- parameters ----------------------------------
G_ACC = 9.81
C_RES = 0.8
TAU   = 0.5

H_MIN, H_MAX = 0.0, 2.0
V_MIN, V_MAX = -5.0, 5.0

NX = int(sys.argv[1]) if len(sys.argv) > 1 else 256   # base cells in h
NY = int(sys.argv[2]) if len(sys.argv) > 2 else 256   # base cells in v
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 64    # phase intervals in s

DH = (H_MAX - H_MIN) / NX
DV = (V_MAX - V_MIN) / NY
DS = 1.0 / NP

NG = NY // 2            # guard rows: v in [-5, 0]  ->  rows 0 .. NG-1
VZERO_ROW = NG          # vertex index m with v_m = 0

def vcoord(j):  return V_MIN + j * DV     # vertex v-coordinate, j in [0..NY]

# ----------------------------- cell ids ------------------------------------
# base cell (i,j):      id = i*NY + j                      (i in [0,NX), j in [0,NY))
# cyl  cell (m,k):      id = NB + m*NP + k                 (m in [0,NG), k in [0,NP))
# OUT:                  id = NB + NC
NB  = NX * NY
NC  = NG * NP
OUT = NB + NC
NCELLS = NB + NC + 1

def base_id(i, j): return i * NY + j
def cyl_id(m, k):  return NB + m * NP + k

# ------------------------- hybrid suspension tau-map ------------------------
def advance(kind, a, b, T):
    """kind='B': base point (h,v)=(a,b);  kind='C': cylinder point (vg,s)=(a,b).
    Advance suspension time T.  Returns (kind, a, b) or ('OUT',0,0)."""
    h, v, vg, s = (a, b, 0.0, 0.0) if kind == 'B' else (0.0, 0.0, a, b)
    guard_tol = 1e-12
    for _ in range(64):
        if T <= 1e-15:
            break
        if kind == 'B':
            if h <= guard_tol and v <= 0.0:          # on the guard -> cylinder
                kind, vg, s = 'C', v, 0.0
                continue
            # time to impact h=0 (downward crossing)
            disc = v * v + 2.0 * G_ACC * h
            sigma = (v + disc ** 0.5) / G_ACC
            if sigma >= T:                            # stays in flight
                h = h + v * T - 0.5 * G_ACC * T * T
                v = v - G_ACC * T
                T = 0.0
            else:                                     # hits the guard
                T -= sigma
                kind, vg, s = 'C', -(disc ** 0.5), 0.0
        else:  # cylinder
            if vg >= -1e-9:                           # suspended periodic orbit
                s = (s + T) % 1.0                     # (0,1) ~ r(0,0)=(0,0)
                vg, T = 0.0, 0.0
            elif (1.0 - s) > T:                       # stays on the cylinder
                s += T
                T = 0.0
            else:                                     # reaches s=1: (g,1)~r(g)
                T -= (1.0 - s)
                kind, h, v = 'B', 0.0, -C_RES * vg    # v > 0: leaves the guard
    if kind == 'B':
        if h < -1e-9 or h > H_MAX + 1e-9 or v < V_MIN - 1e-9 or v > V_MAX + 1e-9:
            return ('OUT', 0.0, 0.0)
        return ('B', min(max(h, 0.0), H_MAX), min(max(v, V_MIN), V_MAX))
    return ('C', min(max(vg, V_MIN), 0.0), s % 1.0)

def point_to_cell(kind, a, b):
    if kind == 'OUT':
        return OUT
    if kind == 'B':
        i = min(max(int((a - H_MIN) / DH), 0), NX - 1)
        j = min(max(int((b - V_MIN) / DV), 0), NY - 1)
        return base_id(i, j)
    m = min(max(int((a - V_MIN) / DV), 0), NG - 1)
    k = min(max(int(b / DS), 0), NP - 1)
    return cyl_id(m, k)

# ------------------------ multivalued map by sampling ------------------------
FR = (0.0, 0.5, 1.0)   # corners, edge midpoints, center (3x3 per cell)

def cell_samples(cid):
    if cid < NB:
        i, j = divmod(cid, NY)
        h0, v0 = H_MIN + i * DH, V_MIN + j * DV
        return [('B', h0 + fa * DH, v0 + fb * DV) for fa in FR for fb in FR]
    m, k = divmod(cid - NB, NP)
    vg0, s0 = V_MIN + m * DV, k * DS
    return [('C', vg0 + fa * DV, min(s0 + fb * DS, 1.0)) for fa in FR for fb in FR]

def padded(cid):
    """one-cell in-chart padding around a target cell"""
    if cid == OUT:
        return (OUT,)
    out = []
    if cid < NB:
        i, j = divmod(cid, NY)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < NX and 0 <= jj < NY:
                    out.append(base_id(ii, jj))
    else:
        m, k = divmod(cid - NB, NP)
        for dm in (-1, 0, 1):
            for dk in (-1, 0, 1):
                mm, kk = m + dm, k + dk
                if 0 <= mm < NG and 0 <= kk < NP:
                    out.append(cyl_id(mm, kk))
    return out

def build_map():
    F = [None] * NCELLS
    F[OUT] = ()
    t0 = time.time()
    for cid in range(NB + NC):
        targets = set()
        for (kind, a, b) in cell_samples(cid):
            targets.update(padded(point_to_cell(*advance(kind, a, b, TAU))))
        F[cid] = tuple(targets)
        if cid % 20000 == 0:
            print(f"  map {cid}/{NB+NC}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  map built in {time.time()-t0:.1f}s")
    return F

# ----------------------------- Tarjan SCC -----------------------------------
def tarjan_scc(F):
    n = NCELLS
    index = [-1] * n; low = [0] * n; on = [False] * n
    comp = [-1] * n; stack = []; ncomp = 0; counter = 0
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = counter; counter += 1
                stack.append(v); on[v] = True
            recurse = False
            succ = F[v]
            for ii in range(pi, len(succ)):
                w = succ[ii]
                if index[w] == -1:
                    work[-1] = (v, ii + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                elif on[w]:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                while True:
                    w = stack.pop(); on[w] = False; comp[w] = ncomp
                    if w == v:
                        break
                ncomp += 1
            work.pop()
            if work:
                u, _ = work[-1]
                low[u] = min(low[u], low[v])
    return comp, ncomp

def recurrent_components(F, comp, ncomp):
    members = defaultdict(list)
    for c in range(NCELLS):
        members[comp[c]].append(c)
    rec = []
    for cc, cells in members.items():
        cs = set(cells)
        if any(w in cs for c in cells for w in F[c]):   # internal edge
            rec.append(cc)
    return members, rec

# ---------------------- Z/2 cellular homology of a block ---------------------
def cell_edges(cid):
    """Edges of a 2-cell as frozensets of canonical vertex ids, with the
    suspension identifications baked in.  Degenerate (collapsed) edges dropped.
      base vertex  ('v', i, j)                 i in [0..NX], j in [0..NY]
      cyl  vertex  ('v', 0, m)   at s=0   (bottom glues to the guard line h=0)
      cyl  vertex  snapped        at s=1   (top glues to h=0 at v = -c * v_m)
    """
    def cylv(m, k):
        if k == 0:
            return ('v', 0, m)                                   # (g,0) ~ g
        if k == NP:
            jj = round((-C_RES * vcoord(m) - V_MIN) / DV)         # (g,1) ~ r(g)
            return ('v', 0, min(max(jj, 0), NY))
        return ('cv', m, k)
    if cid < NB:
        i, j = divmod(cid, NY)
        vs = [('v', i, j), ('v', i + 1, j), ('v', i + 1, j + 1), ('v', i, j + 1)]
    else:
        m, k = divmod(cid - NB, NP)
        vs = [cylv(m, k), cylv(m + 1, k), cylv(m + 1, k + 1), cylv(m, k + 1)]
    es = []
    for t in range(4):
        a, b = vs[t], vs[(t + 1) % 4]
        if a != b:
            es.append(frozenset((a, b)))
    return es

def gf2_rank(rows):
    pivots = {}
    r = 0
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                r += 1
                break
    return r

def betti_pair(N_cells, L_cells):
    """dims of H_*(|N|, |L|; Z/2) for forward-invariant families L <= N.

    Scalable version: rank of d1 via union-find (graph rank), rank of d2 via
    free-edge peeling (collapse every 2-cell that has an edge of face-degree 1,
    which is a row with a private pivot), then GF(2) elimination only on the
    small leftover core."""
    def faces(cells):
        E, V = set(), set()
        for c in cells:
            for e in cell_edges(c):
                E.add(e)
                V.update(e)
        return E, V
    EN, VN = faces(N_cells)
    EL, VL = faces(L_cells)
    F2 = [c for c in N_cells if c not in L_cells]         # relative 2-cells
    E  = list(EN - EL)
    V  = list(VN - VL)
    eidx = {e: t for t, e in enumerate(E)}
    vidx = {v: t for t, v in enumerate(V)}

    # ---- rank d1 (edges -> vertices) via union-find --------------------------
    # incidence-matrix rank; an edge with one endpoint in L has boundary = a
    # single vertex, handled as a union with a virtual L-basepoint
    parent = list(range(len(V) + 1))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    BASE = len(V)
    r1 = 0
    for e in E:
        vs = [vidx[v] for v in e if v in vidx]
        if not vs:
            continue
        a = vs[0]
        b = vs[1] if len(vs) == 2 else BASE
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            r1 += 1

    # ---- rank d2 (2-cells -> edges) via peeling + core elimination ----------
    cell_rel_edges = {}
    edge_faces = defaultdict(list)
    for c in F2:
        es = [eidx[e] for e in cell_edges(c) if e in eidx]
        cell_rel_edges[c] = es
        for t in es:
            edge_faces[t].append(c)
    alive = set(F2)
    deg = {t: len(fs) for t, fs in edge_faces.items()}
    from collections import deque
    q = deque(t for t, d in deg.items() if d == 1)
    used_edge = set()
    r2 = 0
    while q:
        t = q.popleft()
        if deg.get(t, 0) != 1 or t in used_edge:
            continue
        c = next((f for f in edge_faces[t] if f in alive), None)
        if c is None:
            continue
        alive.discard(c)
        used_edge.add(t)
        r2 += 1
        for t2 in cell_rel_edges[c]:
            if t2 != t:
                deg[t2] -= 1
                if deg[t2] == 1:
                    q.append(t2)
    # leftover core (closed-surface candidates): small, eliminate over GF(2)
    core_rows = []
    for c in alive:
        bits = 0
        for t in cell_rel_edges[c]:
            if t not in used_edge:
                bits ^= (1 << t)
        core_rows.append(bits)
    r2 += gf2_rank(core_rows)

    b0 = len(V) - r1
    b1 = len(E) - r1 - r2
    b2 = len(F2) - r2
    return (b0, b1, b2), len(F2), len(E), len(V)

# --------------------------------- main --------------------------------------
def main():
    print(f"grid: base {NX}x{NY}, cylinder {NG}x{NP}  "
          f"({NB} base + {NC} cyl cells), tau={TAU}")
    F = build_map()

    comp, ncomp = tarjan_scc(F)
    members, rec = recurrent_components(F, comp, ncomp)
    print(f"SCCs: {ncomp}   recurrent (Morse sets): {len(rec)}")
    for cc in rec:
        cells = members[cc]
        nb = sum(1 for c in cells if c < NB)
        ncy = sum(1 for c in cells if NB <= c < NB + NC)
        bs = [c for c in cells if c < NB]
        if bs:
            hs = [H_MIN + (c // NY) * DH for c in bs]
            vs = [V_MIN + (c % NY) * DV for c in bs]
            loc = (f"base h in [{min(hs):.3f},{max(hs)+DH:.3f}], "
                   f"v in [{min(vs):.3f},{max(vs)+DV:.3f}]")
        else:
            loc = "no base cells"
        print(f"  Morse set [{cc}]: {len(cells)} cells ({nb} base, {ncy} cyl); {loc}")

    # Morse graph order (reachability between recurrent comps on condensation)
    cond = defaultdict(set)
    for c in range(NCELLS):
        for w in F[c]:
            if comp[c] != comp[w]:
                cond[comp[c]].add(comp[w])
    def reaches(a, b):
        seen, st = {a}, [a]
        while st:
            x = st.pop()
            if x == b:
                return True
            for y in cond[x]:
                if y not in seen:
                    seen.add(y); st.append(y)
        return False
    recset = set(rec)
    edges = [(a, b) for a in rec for b in rec if a != b and reaches(a, b)]
    print(f"Morse graph reachability edges among recurrent comps: {edges}")

    # ---- Conley index of each Morse set -------------------------------------
    for cc in rec:
        below = [b for (a, b) in edges if a == cc]
        tag = "minimal" if not below else f"-> {below}"
        # U = forward closure of M, V = U \ M  (both forward invariant)
        M = set(members[cc])
        U, st = set(M), list(M)
        while st:
            x = st.pop()
            for y in F[x]:
                if y not in U:
                    U.add(y); st.append(y)
        out_reached = OUT in U
        U.discard(OUT)
        Vx = U - M
        (b0, b1, b2), nf, ne, nv = betti_pair(U, Vx)
        print(f"  Conley index of Morse set [{cc}] ({tag}): "
              f"dims (CH0,CH1,CH2) = ({b0},{b1},{b2})   "
              f"[N: {len(U)} cells, L: {len(Vx)} cells, rel complex "
              f"{nf}F/{ne}E/{nv}V, OUT reached: {out_reached}]")

    # ---- figure -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
        colors = ["#d81b60", "#1e88e5", "#ffc107", "#004d40", "#8e24aa"]
        for t, cc in enumerate(rec):
            col = colors[t % len(colors)]
            bs = [c for c in members[cc] if c < NB]
            cy = [c for c in members[cc] if NB <= c < NB + NC]
            for c in bs:
                i, j = divmod(c, NY)
                ax1.add_patch(plt.Rectangle((H_MIN + i * DH, V_MIN + j * DV),
                                            DH, DV, color=col, lw=0))
            for c in cy:
                m, k = divmod(c - NB, NP)
                ax2.add_patch(plt.Rectangle((V_MIN + m * DV, k * DS),
                                            DV, DS, color=col, lw=0))
        ax1.set(xlim=(H_MIN, H_MAX), ylim=(V_MIN, V_MAX),
                xlabel="h", ylabel="v", title="Morse set, base chart")
        ax2.set(xlim=(V_MIN, 0), ylim=(0, 1),
                xlabel="v_g (guard)", ylabel="s (phase)",
                title="Morse set, cylinder chart")
        ax1.axhline(0, color="k", lw=0.4); ax1.axvline(0, color="k", lw=0.4)
        fig.suptitle(f"Bouncing ball hybrid Conley-Morse computation "
                     f"(tau={TAU}, {NX}x{NY} base, {NP} phases)")
        fig.tight_layout()
        fig.savefig("bouncing_ball_morse.png", dpi=140)
        print("figure saved: bouncing_ball_morse.png")
    except Exception as e:
        print("plot skipped:", e)

if __name__ == "__main__":
    main()
