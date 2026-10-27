# %% ----------------------------------------------------- imports & params
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SEED = 11
rng_master = np.random.default_rng(SEED)

PARAMS = dict(
    # populations ------------------------------------------------------
    N_E=80, N_I=20, N_A=40, N_B=40, N_BASKET=6,      # S2 basket count
    # single-neuron ----------------------------------------------------
    tau_E=20e-3, tau_I=10e-3, tau_basket=5e-3,       # membrane time consts
    rmax_E=60.0, rmax_I=100.0, rmax_basket=140.0,    # F-I ceilings (Hz)
    theta=12.0, k_E=2.5, k_I=2.5,                    # F-I threshold/slope
    # weights below solve the sigmoid fixed-point equations for the target
    # operating points: baseline (E=3, I=8 Hz) and decided (win=35, lose=1,
    # I=15 Hz) -- see derivation in the repo conversation
    # connectivity (pool-level mean weights; divided by N_pre per pool,
    # then jittered per synapse by lognormal(sigma_w)) -----------------
    w_EE_self=0.42, w_EE_cross=0.05,                 # recurrent excitation
    w_EI=0.085,                                      # E -> I
    w_IE=-0.865,                                     # I -> E (shared pool)
    w_II=-0.10,
    sigma_w=0.25,                                    # synaptic heterogeneity
    # S2: basket cells
    w_IE_basket=-0.35, w_EI_basket=0.11, w_II_basket=-0.14,
    w_IE_reg_S2=-0.53,                               # basket+reg ~ S1 total
    # S3: selective inhibition (within vs cross subnetwork)
    sel_in_EI=1.7, sel_cross_EI=0.3,                 # E->I selectivity
    sel_in_IE=0.68, sel_cross_IE=0.32,               # I->E (sums to 1: matched)
    # inputs -----------------------------------------------------------
    I_bg_E=9.94, I_bg_I=6.2, I_stim=3.6, I_coh=1.2,              # background & stimulus
    sigma_noise=0.5, tau_noise=20e-3, sigma_pool=0.12,
    sigma_ev=0.5, tau_ev=150e-3,      # slow evidence-stream fluctuations
    # burst + reset ----------------------------------------------------
    G_burst=1.55, burst_rise=15e-3, burst_hold=60e-3, burst_fall=40e-3,
    w_adapt=0.9, tau_adapt=120e-3,                   # adaptation -> reset
    # timeline ---------------------------------------------------------
    dt=1e-3, t_fix=0.35, t_dec=1.0, t_rec=1.6, t_end=1.8,
    # task -------------------------------------------------------------
    coherences=(0.6, 0.4, 0.2, 0.1),                 # |p_B - p_A|
    trials_per_cond=10,                              # x 2 directions x 4 = 80
    # Conley-Morse-from-data -------------------------------------------
    cm_grid=45, cm_lag_ms=300, cm_bloat=0, cm_soft_norm=5.0,      # Hz in soft normalizer
)

SCENARIOS = ["S1 Wang baseline", "S2 Basket control", "S3 Selective I"]
POOL_COLORS = dict(EA='#1f77b4', EB='#2ca02c', I='#d62728',
                   BASKET='#9467bd', IA='#e377c2', IB='#8c564b')

# %% ----------------------------------------------------- CM utilities
# (compact data-driven versions of the machinery from conley_morse_notebook)
def cm_cell_of(P, xmin, xmax, ymin, ymax, n):
    i = np.floor((P[:, 0] - xmin) / (xmax - xmin) * n).astype(int)
    j = np.floor((P[:, 1] - ymin) / (ymax - ymin) * n).astype(int)
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    return np.where(ok, j * n + i, -1)

def cm_decompose(edges, n_cells):
    """Recurrent SCCs + transitively reduced Morse graph from edge list."""
    A = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                   shape=(n_cells, n_cells)).tocsr()
    _, scc = connected_components(A, directed=True, connection='strong')
    size = np.bincount(scc)
    selfl = np.zeros(len(size), bool)
    selfl[scc[edges[edges[:, 0] == edges[:, 1], 0]]] = True
    rec = (size > 1) | selfl

    def cells(s): return np.flatnonzero(scc == s)
    morse = [cells(s) for s in np.flatnonzero(rec)]
    # merge 8-adjacent recurrent SCCs (coarsening is always legitimate)
    def merge(sets, n):
        pts = [set(zip((c % n).tolist(), (c // n).tolist())) for c in sets]
        parent = list(range(len(sets)))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for a in range(len(sets)):
            dil = {(i + di, j + dj) for (i, j) in pts[a]
                   for di in (-1, 0, 1) for dj in (-1, 0, 1)}
            for b in range(a + 1, len(sets)):
                if dil & pts[b]:
                    parent[find(a)] = find(b)
        groups = {}
        for k in range(len(sets)):
            groups.setdefault(find(k), []).append(k)
        return [np.unique(np.concatenate([sets[k] for k in g]))
                for g in groups.values()]
    morse = merge(morse, int(np.sqrt(n_cells)))
    morse = sorted([m for m in morse if len(m) >= 2],
                   key=lambda c: -len(c))
    cell_morse = np.full(n_cells, -1)
    for k, cs in enumerate(morse):
        cell_morse[cs] = k
    # reachability order between Morse sets, transitively reduced
    cond = {}
    for u, v in edges:
        if scc[u] != scc[v]:
            cond.setdefault(scc[u], set()).add(scc[v])
    scc_of_set = [set(np.unique(scc[cs])) for cs in morse]
    def reach_sets(k):
        seen = set(scc_of_set[k]); stack = list(seen); out = set()
        while stack:
            for t in cond.get(stack.pop(), ()):
                if t not in seen:
                    seen.add(t); stack.append(t)
                    for k2, ss in enumerate(scc_of_set):
                        if k2 != k and t in ss:
                            out.add(k2)
        return out
    R = {k: reach_sets(k) for k in range(len(morse))}
    gedges = [(a, b) for a in R for b in R[a]
              if not any(b in R[m] for m in R[a] if m != b)]
    return morse, cell_morse, gedges

def draw_cm_graph(ax, edges, occ, names=None, colors=None):
    n = len(occ)
    occ = np.asarray(occ, float)
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:max(n, 1)]
    if names is None:
        names = [f"M{k}" for k in range(n)]
    depth = {k: 0 for k in range(n)}
    for _ in range(n):
        for a, b in edges:
            depth[b] = max(depth[b], depth[a] + 1)
    maxd = max(depth.values()) if n else 0
    layers = {}
    for k in range(n):
        layers.setdefault(depth[k], []).append(k)
    pos = {}
    for d in sorted(layers):
        for q, k in enumerate(sorted(layers[d])):
            pos[k] = ((q + 1) / (len(layers[d]) + 1),
                      0.85 - 0.7 * d / max(maxd, 1))
    radii = {k: 0.06 + 0.22 * np.sqrt(occ[k]) for k in range(n)}
    scale = 1.0
    for a in range(n):
        for b in range(a + 1, n):
            d_ = np.linalg.norm(np.array(pos[a]) - np.array(pos[b]))
            scale = min(scale, max(d_ - 0.07, 0.02) / (radii[a] + radii[b]))
    radii = {k: r * scale for k, r in radii.items()}
    ax.set_aspect('equal'); ax.set_axis_off()
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 1)
    for a, b in edges:
        pa, pb = np.array(pos[a]), np.array(pos[b])
        u = pb - pa; L = np.linalg.norm(u)
        if L < 1e-9: continue
        u /= L
        ax.add_patch(FancyArrowPatch(pa + u * (radii[a] + .008),
                                     pb - u * (radii[b] + .010),
                                     arrowstyle='-|>', mutation_scale=12,
                                     color='0.3', lw=1.3, shrinkA=0,
                                     shrinkB=0, zorder=2))
    for k in range(n):
        ax.add_patch(Circle(pos[k], radii[k], color=colors[k], zorder=3))
        ax.text(*pos[k], f"{names[k]}\n{occ[k]*100:.0f}%", ha='center',
                va='center', fontsize=7.5, color='white',
                fontweight='bold', zorder=4)
    return pos

# %% ----------------------------------------------------- fixed identities
P = PARAMS
N = P['N_E'] + P['N_I']
pool = np.array(['EA'] * P['N_A'] + ['EB'] * P['N_B'] + ['I'] * P['N_I'])
is_E = np.arange(N) < P['N_E']
I_idx = np.flatnonzero(~is_E)
basket_idx = I_idx[:P['N_BASKET']]              # fixed basket identities
regI_idx = I_idx[P['N_BASKET']:]
# S3 subnetworks: built from the NON-basket I cells (7 + 7), so basket
# cells stay non-selective in every scenario, per the waveform story.
# (The original spec said 10+10; this deviation keeps basket identity.)
selA_idx = regI_idx[:len(regI_idx) // 2]
selB_idx = regI_idx[len(regI_idx) // 2:]

# per-neuron parameter jitter (identities fixed by rng_master)
gain_jit = rng_master.lognormal(0, 0.12, N)
theta_i = P['theta'] + rng_master.normal(0, 0.5, N)

# waveform-PCA descriptor space (fixed): PC1 ~ excitatory, PC2 ~ inhibitory
wf = np.zeros((N, 2))
wf[is_E, 0] = rng_master.normal(1.1, 0.45, is_E.sum())
wf[is_E, 1] = rng_master.normal(0.1, 0.45, is_E.sum())
wf[regI_idx, 0] = rng_master.normal(0.25, 0.45, len(regI_idx))   # overlaps E
wf[regI_idx, 1] = rng_master.normal(1.0, 0.55, len(regI_idx))    # overlaps E
wf[basket_idx, 0] = rng_master.normal(-0.1, 0.35, len(basket_idx))
wf[basket_idx, 1] = rng_master.normal(2.4, 0.22, len(basket_idx))  # separable

# %% ----------------------------------------------------- connectivity
def build_W(scenario):
    """Pool-structured weight matrix with per-synapse lognormal jitter."""
    rg = np.random.default_rng(SEED + 100)      # same jitter all scenarios
    W = np.zeros((N, N))
    A = pool == 'EA'; B = pool == 'EB'; I = ~is_E

    def block(post, pre, w):
        if w == 0: return
        n_pre = pre.sum()
        jit = rg.lognormal(0, P['sigma_w'], (post.sum(), n_pre))
        W[np.ix_(np.flatnonzero(post), np.flatnonzero(pre))] += \
            (w / n_pre) * jit

    block(A, A, P['w_EE_self']); block(B, B, P['w_EE_self'])
    block(A, B, P['w_EE_cross']); block(B, A, P['w_EE_cross'])

    if scenario == 0:                           # S1: one shared I pool
        block(I, A, P['w_EI']); block(I, B, P['w_EI'])
        block(A, I, P['w_IE']); block(B, I, P['w_IE'])
        block(I, I, P['w_II'])
    elif scenario == 1:                         # S2: basket + regular I
        bk = np.isin(np.arange(N), basket_idx)
        rI = np.isin(np.arange(N), regI_idx)
        for Ipop, wei, wie, wii in ((bk, P['w_EI_basket'],
                                     P['w_IE_basket'], P['w_II_basket']),
                                    (rI, P['w_EI'], P['w_IE_reg_S2'],
                                     P['w_II'])):
            block(Ipop, A, wei); block(Ipop, B, wei)
            block(A, Ipop, wie); block(B, Ipop, wie)
            block(Ipop, I, wii)
    else:                                       # S3: selective subnetworks
        IA = np.isin(np.arange(N), selA_idx)
        IB = np.isin(np.arange(N), selB_idx)
        bk = np.isin(np.arange(N), basket_idx)
        wIE_sel = P['w_IE'] * 0.78              # selective share of I->E ...
        wIE_bk = P['w_IE'] * 0.22               # ... basket share (total ~ S1)
        block(IA, A, P['w_EI'] * P['sel_in_EI'])
        block(IA, B, P['w_EI'] * P['sel_cross_EI'])
        block(IB, B, P['w_EI'] * P['sel_in_EI'])
        block(IB, A, P['w_EI'] * P['sel_cross_EI'])
        block(bk, A, P['w_EI']); block(bk, B, P['w_EI'])
        block(A, IA, wIE_sel * P['sel_in_IE'])
        block(A, IB, wIE_sel * P['sel_cross_IE'])
        block(B, IB, wIE_sel * P['sel_in_IE'])
        block(B, IA, wIE_sel * P['sel_cross_IE'])
        block(A, bk, wIE_bk); block(B, bk, wIE_bk)
        block(IA, I, P['w_II']); block(IB, I, P['w_II'])
        block(bk, I, P['w_II'])
    return W

def neuron_consts(scenario):
    tau = np.where(is_E, P['tau_E'], P['tau_I']).astype(float)
    rmax = np.where(is_E, P['rmax_E'], P['rmax_I']).astype(float)
    kk = np.where(is_E, P['k_E'], P['k_I']).astype(float)
    Ibg = np.where(is_E, P['I_bg_E'], P['I_bg_I']).astype(float)
    if scenario in (1, 2):                      # basket kinetics & drive
        tau[basket_idx] = P['tau_basket']
        rmax[basket_idx] = P['rmax_basket']
        Ibg[basket_idx] = P['I_bg_I'] + 0.5     # high baseline rate
    return tau, rmax, kk, Ibg

# %% ----------------------------------------------------- simulation
def gain_profile(tt):
    """Recurrent gain G(t): 1, then burst pulse after the decision period."""
    t0 = P['t_fix'] + P['t_dec']
    up = np.clip((tt - t0) / P['burst_rise'], 0, 1)
    dn = np.clip((tt - t0 - P['burst_rise'] - P['burst_hold'])
                 / P['burst_fall'], 0, 1)
    return 1 + (P['G_burst'] - 1) * (up - dn)

def simulate_trial(W, consts, coh, rng):
    tau, rmax, kk, Ibg = consts
    dt = P['dt']
    T = int(round(P['t_end'] / dt))
    tt = np.arange(T) * dt
    G = gain_profile(tt)
    stim_on = (tt >= P['t_fix']) & (tt < P['t_fix'] + P['t_dec'])
    I_A = P['I_stim'] + P['I_coh'] * coh
    I_B = P['I_stim'] - P['I_coh'] * coh

    W_pos = np.clip(W, 0, None); W_neg = np.clip(W, None, 0)
    r = np.zeros(N) + 1.0
    a = np.zeros(N)                             # adaptation
    eta = np.zeros(N)
    ev = np.zeros(2)                            # slow evidence noise (A, B)
    al_e = dt / P['tau_ev']
    rates = np.empty((T, N), np.float32)
    al_n = dt / P['tau_noise']
    for t in range(T):
        eta += -al_n * eta + P['sigma_noise'] * np.sqrt(2 * al_n) \
            * rng.standard_normal(N)
        shared = P['sigma_pool'] * rng.standard_normal(3)
        I_ext = Ibg + eta
        I_ext[pool == 'EA'] += shared[0]
        I_ext[pool == 'EB'] += shared[1]
        I_ext[~is_E] += shared[2]
        ev += -al_e * ev + P['sigma_ev'] * np.sqrt(2 * al_e) \
            * rng.standard_normal(2)
        if stim_on[t]:
            I_ext[pool == 'EA'] += I_A + ev[0]
            I_ext[pool == 'EB'] += I_B + ev[1]
        x = G[t] * (W_pos @ r) + (W_neg @ r) + I_ext - a
        phi = rmax / (1 + np.exp(-np.clip((x - theta_i) * gain_jit / kk,
                                          -30, 30)))
        r += dt / tau * (-r + phi)
        a += dt / P['tau_adapt'] * (-a + P['w_adapt'] * r * (G[t] > 1.02))
        rates[t] = r
    spikes = rng.random(rates.shape) < rates * dt
    return rates, spikes

def run_scenario(scenario):
    rng = np.random.default_rng(SEED + 1000 * (scenario + 1))
    W = build_W(scenario)
    consts = neuron_consts(scenario)
    cohs = []
    for c in P['coherences']:
        cohs += [+c] * P['trials_per_cond'] + [-c] * P['trials_per_cond']
    cohs = np.array(cohs)
    rng.shuffle(cohs)
    T = int(round(P['t_end'] / P['dt']))
    R = np.empty((len(cohs), T, N), np.float32)
    S = np.zeros((len(cohs), T, N), bool)
    for tr, c in enumerate(cohs):
        R[tr], S[tr] = simulate_trial(W, consts, c, rng)
    t_dec_end = int(round((P['t_fix'] + P['t_dec']) / P['dt'])) - 1
    rA = R[:, t_dec_end, pool == 'EA'].mean(1)
    rB = R[:, t_dec_end, pool == 'EB'].mean(1)
    choice = np.where(rA > rB, +1, -1)          # +1 = A
    return dict(W=W, cohs=cohs, R=R, S=S, choice=choice,
                correct=(choice == np.sign(cohs)) | (cohs == 0))

# %% ----------------------------------------------------- analyses
def soft_norm(R):
    """(trials, T, N) -> per-neuron soft-normalized, mean-centered."""
    mx = R.reshape(-1, N).max(0)
    Z = R / (mx + P['cm_soft_norm'])
    return Z - Z.reshape(-1, N).mean(0)

def pca_latents(R, n_rec, d=3, fit_slice=None):
    """PCA on the recorded window; optionally FIT on a sub-epoch only
    (e.g. the decision period, so the burst's huge variance does not
    swallow the decision plane) while projecting the whole window."""
    Z = soft_norm(R)[:, :n_rec]
    Xf = (Z[:, fit_slice] if fit_slice else Z).reshape(-1, N)
    mu = Xf.mean(0)
    _, _, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
    L = (Z.reshape(-1, N) - mu) @ Vt[:d].T
    return L.reshape(Z.shape[0], n_rec, d), Vt[:d]

def selectivity_auc(res, n_rec):
    """Per-neuron |ROC AUC - 0.5|*2 signed toward A, over late decision."""
    t0 = int(round((P['t_fix'] + 0.6) / P['dt']))
    t1 = int(round((P['t_fix'] + P['t_dec']) / P['dt']))
    m = res['R'][:, t0:t1].mean(1)              # (trials, N)
    A = m[res['choice'] > 0]; B = m[res['choice'] < 0]
    auc = np.empty(N)
    for i in range(N):
        auc[i] = (A[:, i][:, None] > B[:, i][None, :]).mean() \
            + 0.5 * (A[:, i][:, None] == B[:, i][None, :]).mean()
    return 2 * (auc - 0.5)                      # in [-1, 1], + = prefers A

def conley_morse_from_data(L2, t_cut, grid=None, lag_ms=None, bloat=None):
    """L2: (trials, T_rec, 2) latents. The multivalued map is built from
    t < t_cut only (the gain surge makes the flow nonautonomous, and the
    task protocol itself is a loop: baseline -> attractor -> burst -> decay
    back toward baseline would be ONE genuine recurrent cycle). Membership,
    occupancy and density still cover the whole recorded window."""
    n = P['cm_grid'] if grid is None else grid
    lag_ms_ = P['cm_lag_ms'] if lag_ms is None else lag_ms
    bloat_ = P['cm_bloat'] if bloat is None else bloat
    lag = int(round(lag_ms_ * 1e-3 / P['dt']))
    allpts = L2.reshape(-1, 2)
    pad = 0.02 * (allpts.max(0) - allpts.min(0))
    x0, y0 = allpts.min(0) - pad; x1, y1 = allpts.max(0) + pad
    E = []
    for tr in range(L2.shape[0]):
        c = cm_cell_of(L2[tr, :t_cut], x0, x1, y0, y1, n)
        src, dst = c[:-lag], c[lag:]
        ok = (src >= 0) & (dst >= 0)
        si, di = src[ok], dst[ok]
        bl = bloat_                             # chain epsilon >= cell size
        for a in range(-bl, bl + 1):
            for b in range(-bl, bl + 1):
                dd = di + a + b * n
                good = ((di % n + a >= 0) & (di % n + a < n)
                        & (di // n + b >= 0) & (di // n + b < n))
                E.append(np.stack([si[good], dd[good]], 1))
    edges = np.unique(np.concatenate(E), axis=0)
    morse, cell_morse, gedges = cm_decompose(edges, n * n)
    traj_cells = np.stack([cm_cell_of(L2[tr], x0, x1, y0, y1, n)
                           for tr in range(L2.shape[0])])
    trace = np.where(traj_cells >= 0, cell_morse[traj_cells], -1)
    occ = np.array([(trace == k).mean() for k in range(len(morse))])
    density = np.bincount(traj_cells[traj_cells >= 0].ravel(),
                          minlength=n * n).reshape(n, n).astype(float)
    return dict(morse=morse, cell_morse=cell_morse, gedges=gedges,
                trace=trace, occ=occ, density=density, n=n,
                box=(x0, x1, y0, y1))

