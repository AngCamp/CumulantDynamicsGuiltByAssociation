# =============================================================================
# cca_conley_morse.py -- multiview (time-lagged) CCA on the S1 decision
# circuit, with redundancy-index component selection and a Conley-Morse
# consistency check between the low-lag and high-lag embeddings.
#
# VIEWS: V_l = X(t + l) for 8 log-spaced lags l in [20, 400] ms -- each lag
# is one "view" (block) of the soft-normalized population rates over the
# decision epoch, restricted to the time window common to all lags.
#
# MULTIVIEW CCA (MAXVAR / Carroll's generalized CCA): whiten each view by
# its own PCA, horizontally concatenate the whitened blocks, SVD -> the left
# singular vectors are the CONSENSUS variates; each view's canonical variate
# is its block's projection, and rho_i(l) = corr(consensus_i, variate_i(l))
# is the canonical correlation of component i at lag l.
#
# REDUNDANCY INDEX (per component i, per lag l), as specified:
#     RI_i(l) = SharedVar(V_l, variate_i(l)) * rho_i(l)^2
# where SharedVar = mean squared loading (correlation) of the ORIGINAL
# view-l channels on their own i-th variate -- i.e. variance explained
# WITHIN THAT LAG'S BLOCK, not in the concatenated matrix.
#
# COMPONENT SELECTION: elbow (max perpendicular distance to the chord of the
# sorted spectrum) of RI at the LOWEST lag -> k_low; same at the HIGHEST lag
# -> k_high. The Conley-Morse decomposition is retraced in both embeddings
# and mismatches are reported at the end.
#
# OUTPUT FIGURES
#   cca_procrustes.png   (a) Orthogonal-Procrustes residual "down to the
#                        next lag" for the k-component trajectory embedding,
#                        one curve per k = 1..K  (the sketch you drew);
#                        (b) full lag x lag Procrustes heat map at k = k_low
#   cca_variance.png     variance explained vs lag: whole CCA model at that
#                        lag; RI-elbowed subset at that lag; plus the bar
#                        plot comparing three elbow policies (global RI,
#                        block/timescale-weighted RI, per-lag RI)
#   cca_conley_mismatch.png  Morse sets + membership in the k_low vs k_high
#                        embeddings, matched by trajectory overlap, with a
#                        per-time-step mismatch strip
#
# Caveats stated once: with an empirical map we certify RECURRENCE (Morse
# sets + graph), not index pairs, so "comparable Conley indices" here means
# matched Morse-set counts / types / graph position. Shared-variance sums
# over components are approximate (canonical variates of one view are not
# exactly orthogonal in channel space).
# =============================================================================
exec(open('_sim_base.py').read())        # simulation + CM machinery (S1 etc.)

import numpy.linalg as npl

# %% -------------------------------------------------- simulate S1 + views
LAGS_MS = np.unique(np.round(np.logspace(np.log10(20), np.log10(400), 8))
                    .astype(int))
K_MODEL = 10                              # components kept in the MCCA model
PCA_RANK = 20                             # per-view whitening rank

print("simulating S1 ...", flush=True)
res = run_scenario(0)
n_rec = int(round(P['t_rec'] / P['dt']))
t_cut = int(round((P['t_fix'] + P['t_dec']) / P['dt']))
Z = soft_norm(res['R'])[:, :n_rec]        # (trials, T, N)
n_tr = Z.shape[0]

lag_steps = (LAGS_MS * 1e-3 / P['dt']).round().astype(int)
max_lag = lag_steps.max()
T_common = t_cut - max_lag                # window where every lag is defined
views = []                                # V_l: (n_tr*T_common, N), centered
for ls in lag_steps:
    V = Z[:, ls:ls + T_common].reshape(-1, N)
    views.append(V - V.mean(0))
n_samp = views[0].shape[0]
print(f"views: {len(views)} lags {list(LAGS_MS)} ms, "
      f"{n_samp} samples x {N} channels")

# %% -------------------------------------------------- multiview CCA
whiteners, Zw = [], []
for V in views:
    U_, S_, Vt_ = npl.svd(V, full_matrices=False)
    r = min(PCA_RANK, (S_ > 1e-8 * S_[0]).sum())
    W = Vt_[:r].T / S_[:r] * np.sqrt(n_samp)    # V @ W is white
    whiteners.append(W)
    Zw.append(V @ W)
C = np.concatenate(Zw, axis=1)
Uc, Sc, Vtc = npl.svd(C, full_matrices=False)
consensus = Uc[:, :K_MODEL] * np.sqrt(n_samp)   # unit-variance consensus

# per-view canonical variates + stats
blocks = np.cumsum([0] + [z.shape[1] for z in Zw])
variates, rho, shared = [], np.zeros((K_MODEL, len(views))), \
    np.zeros((K_MODEL, len(views)))
for li, V in enumerate(views):
    Bv = Vtc[:K_MODEL, blocks[li]:blocks[li + 1]].T     # whitened weights
    xv = Zw[li] @ Bv                                    # (n_samp, K)
    xv = xv / xv.std(0)
    variates.append(xv)
    for i in range(K_MODEL):
        rho[i, li] = np.corrcoef(consensus[:, i], xv[:, i])[0, 1]
        loads = (V * xv[:, i:i + 1]).mean(0) / V.std(0)  # corr per channel
        shared[i, li] = np.mean(loads ** 2)
RI = shared * rho ** 2                                  # (K, n_lags)

def elbow(vals, floor=1e-12):
    """Components kept = position of the largest LOG-gap in the descending
    spectrum (robust where the chord rule degenerates on convex spectra:
    its endpoint terms tie at 0 up to float error)."""
    v = np.sort(np.maximum(np.asarray(vals, float), floor))[::-1]
    gaps = np.log(v[:-1] / v[1:])
    return int(np.argmax(gaps) + 1)

order_low = np.argsort(-RI[:, 0])
order_high = np.argsort(-RI[:, -1])
k_low = elbow(RI[:, 0])
k_high = elbow(RI[:, -1])
print(f"RI elbow at lag {LAGS_MS[0]} ms: k_low = {k_low}  "
      f"(components {order_low[:k_low]})")
print(f"RI elbow at lag {LAGS_MS[-1]} ms: k_high = {k_high} "
      f"(components {order_high[:k_high]})")

# %% -------------------------------------------------- Procrustes distances
def procrustes_resid(Xc, Yc):
    """Normalized orthogonal-Procrustes residual ||X R - Y||_F / ||Y||_F."""
    U_, S_, Vt_ = npl.svd(Xc.T @ Yc, full_matrices=False)
    num = (Xc ** 2).sum() + (Yc ** 2).sum() - 2 * S_.sum()
    return np.sqrt(max(num, 0.0)) / npl.norm(Yc)

n_lag = len(views)
proc_next = np.full((K_MODEL, n_lag - 1), np.nan)   # k x (consecutive pairs)
for k in range(1, K_MODEL + 1):
    for li in range(n_lag - 1):
        proc_next[k - 1, li] = procrustes_resid(variates[li][:, :k],
                                                variates[li + 1][:, :k])
proc_mat = np.zeros((n_lag, n_lag))                 # full heat map at k_low
kk = max(k_low, 1)
for a in range(n_lag):
    for b in range(n_lag):
        if a != b:
            proc_mat[a, b] = procrustes_resid(variates[a][:, :kk],
                                              variates[b][:, :kk])

# %% -------------------------------------------------- variance + policies
ve_full = shared.sum(0)                             # whole model, per lag
ve_perlag, k_perlag = np.zeros(n_lag), np.zeros(n_lag, int)
for li in range(n_lag):
    kl = elbow(RI[:, li])
    k_perlag[li] = kl
    top = np.argsort(-RI[:, li])[:kl]
    ve_perlag[li] = shared[top, li].sum()

# policy scores
score_global = RI.mean(1)                           # plain average over lags
blockvar = np.array([v.var(0).sum() for v in views])
w_block = blockvar / blockvar.sum()
score_block = RI @ w_block                          # block-variance weighted
kg, kb = elbow(score_global), elbow(score_block)
sel_g = np.argsort(-score_global)[:kg]
sel_b = np.argsort(-score_block)[:kb]
ve_g = shared[sel_g].sum(0)                         # per lag
ve_b = shared[sel_b].sum(0)
print(f"elbow policies: global k={kg}, block-weighted k={kb}, "
      f"per-lag k={list(k_perlag)}")

# %% -------------------------------------------------- figures 1 + 2
fig, (axP, axH) = plt.subplots(1, 2, figsize=(13.5, 5),
                               gridspec_kw=dict(width_ratios=[1.4, 1]))
ccols = plt.cm.YlOrRd(np.linspace(0.35, 0.95, K_MODEL))
mid = np.sqrt(LAGS_MS[:-1] * LAGS_MS[1:])           # geometric midpoints
for k in range(K_MODEL):
    axP.plot(mid, proc_next[k], '-o', ms=3.5, color=ccols[k], lw=1.6,
             label=f"k={k+1}")
axP.set_xscale('log')
axP.set_xlabel("lag (ms, log)  —  'timescale'")
axP.set_ylabel("Procrustes residual to next lag\n(rotation + stretch)")
axP.set_title("Rotation of the k-component CCA embedding\n"
              "down to the next timescale grid (your sketch, for CCA)")
axP.legend(fontsize=7, ncol=2, loc='upper left')
im = axH.imshow(proc_mat, cmap='magma_r', origin='lower')
axH.set_xticks(range(n_lag)); axH.set_xticklabels(LAGS_MS, fontsize=8)
axH.set_yticks(range(n_lag)); axH.set_yticklabels(LAGS_MS, fontsize=8)
axH.set_xlabel("to lag (ms)"); axH.set_ylabel("from lag (ms)")
axH.set_title(f"pairwise Procrustes distance\n(k = k_low = {kk} components)")
fig.colorbar(im, ax=axH, label="residual")
fig.suptitle("Multiview CCA across lags — S1 Wang baseline",
             fontsize=13, y=1.04)
fig.savefig("cca_procrustes.png", dpi=130, bbox_inches='tight')
plt.show()

fig, (axV, axB) = plt.subplots(1, 2, figsize=(13.5, 4.8))
axV.plot(LAGS_MS, ve_full * 100, '-o', color='0.25', lw=2,
         label=f"whole model ({K_MODEL} comps)")
axV.plot(LAGS_MS, ve_perlag * 100, '-s', color='#d95f0e', lw=2,
         label="RI-elbowed at that lag")
for li in range(n_lag):
    axV.annotate(f"k={k_perlag[li]}", (LAGS_MS[li], ve_perlag[li] * 100),
                 (0, -14), textcoords='offset points', fontsize=7.5,
                 ha='center', color='#d95f0e')
axV.set_xscale('log')
axV.set_xlabel("lag (ms, log)")
axV.set_ylabel("% variance of that lag's block explained")
axV.set_title("Variance explained vs lag:\nfull CCA model vs RI elbow")
axV.legend(fontsize=9)
axV.annotate("note: variates are not orthogonal in channel space, so the\n"
             "full-model sum of squared loadings can exceed 100%",
             (0.02, 0.02), xycoords='axes fraction', fontsize=7.5,
             color='0.45')
wbar = 0.26
xs = np.arange(n_lag)
axB.bar(xs - wbar, ve_g * 100, wbar, color='#4c72b0',
        label=f"global RI elbow (k={kg})")
axB.bar(xs, ve_b * 100, wbar, color='#0e7b8a',
        label=f"block-weighted RI elbow (k={kb})")
axB.bar(xs + wbar, ve_perlag * 100, wbar, color='#d95f0e',
        label="per-lag RI elbow (k varies)")
axB.set_xticks(xs); axB.set_xticklabels(LAGS_MS, fontsize=8)
axB.set_xlabel("lag (ms)")
axB.set_ylabel("% block variance explained by kept comps")
axB.set_title("Elbow policies compared\n(global vs timescale/block-weighted "
              "vs per-lag redundancy)")
axB.legend(fontsize=8)
fig.savefig("cca_variance.png", dpi=130, bbox_inches='tight')
plt.show()

# %% ----------------------- Conley-Morse retrace: k_low vs k_high embedding
def embed_full(order, k):
    """Project the FULL recorded window onto the view-0 canonical directions
    of the KEPT components (that lag's RI ranking). k >= 2 -> top-2 of the
    kept set span the CM plane; k == 1 -> an HONEST 1-D embedding (second
    coordinate identically 0), so the retrace sees only what the selection
    kept. Returns (tr, n_rec, 2)."""
    idx = order[:min(max(k, 1), 2)]
    W0 = whiteners[0] @ Vtc[:K_MODEL, blocks[0]:blocks[1]].T   # N x K
    Zfull = Z.reshape(-1, N) - views[0].mean(0)
    L = Zfull @ W0[:, idx]
    L = L / L.std(0)
    if L.shape[1] == 1:
        L = np.concatenate([L, np.zeros_like(L)], axis=1)
    return L.reshape(n_tr, n_rec, 2)

def cm_from_data(L2, t_cut):
    """conley_morse_from_data with a degenerate-axis guard, so a 1-D
    embedding (constant second coordinate) bins into a single row of
    cells instead of dividing by a zero span."""
    L2 = L2.copy()
    span = L2[:, :, 1].max() - L2[:, :, 1].min()
    if span < 1e-9:
        L2[:, :, 1] += 1e-6 * np.random.default_rng(0).standard_normal(
            L2.shape[:2])                 # breaks the zero span; one cell row
    return conley_morse_from_data(L2, t_cut)

L_low = embed_full(order_low, k_low)
L_high = embed_full(order_high, k_high)
cm_low = cm_from_data(L_low, t_cut)
cm_high = cm_from_data(L_high, t_cut)

# match Morse sets between embeddings by shared trajectory time-points
def match_sets(cmA, cmB):
    nA, nB = len(cmA['morse']), len(cmB['morse'])
    ov = np.zeros((nA, nB))
    for a in range(nA):
        for b in range(nB):
            ov[a, b] = ((cmA['trace'] == a) & (cmB['trace'] == b)).sum()
    pairs = {}
    used_b = set()
    for a in np.argsort(-ov.max(1)):
        b = int(np.argmax(ov[a]))
        if ov[a, b] > 0 and b not in used_b:
            pairs[a] = b
            used_b.add(b)
    return pairs, ov

pairs, ov = match_sets(cm_low, cm_high)
remap = np.full(len(cm_high['morse']), -2)
for a, b in pairs.items():
    remap[b] = a
tr_high_m = np.where(cm_high['trace'] >= 0,
                     remap[np.clip(cm_high['trace'], 0, None)], -1)
both = (cm_low['trace'] >= 0) & (cm_high['trace'] >= 0)
mismatch = both & (cm_low['trace'] != tr_high_m)
mm_rate = mismatch.sum() / max(both.sum(), 1)
print(f"Morse sets: k_low embedding {len(cm_low['morse'])}, "
      f"k_high embedding {len(cm_high['morse'])}; matched pairs {pairs}")
print(f"membership mismatch where both assign a set: {mm_rate*100:.1f}%")

fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5),
                         gridspec_kw=dict(width_ratios=[1, 1, 1.1],
                                          height_ratios=[1, 0.55],
                                          hspace=0.38, wspace=0.3))
for col, (cmx, Lx, k_, lagname, order) in enumerate(
        [(cm_low, L_low, k_low, f"lowest lag ({LAGS_MS[0]} ms)", order_low),
         (cm_high, L_high, k_high, f"highest lag ({LAGS_MS[-1]} ms)",
          order_high)]):
    ax = axes[0, col]
    n_ = cmx['n']; bx = cmx['box']
    xe = np.linspace(bx[0], bx[1], n_ + 1)
    ye = np.linspace(bx[2], bx[3], n_ + 1)
    cols = plt.cm.tab10(np.linspace(0, 1, 10))[:max(len(cmx['morse']), 1)]
    overlay = np.ma.masked_where(cmx['cell_morse'].reshape(n_, n_) < 0,
                                 cmx['cell_morse'].reshape(n_, n_))
    ax.pcolormesh(xe, ye, overlay, cmap=mcolors.ListedColormap(cols),
                  vmin=-0.5, vmax=len(cmx['morse']) - 0.5, alpha=0.5)
    lut = np.vstack([[0.55, 0.55, 0.55, 1.0], cols])
    for tr in range(0, n_tr, 2):
        Lr = Lx[tr][::2]
        mem = cmx['trace'][tr][::2]
        segs = np.stack([Lr[:-1], Lr[1:]], axis=1)
        cc = lut[mem[:-1] + 1].copy(); cc[:, 3] = 0.5
        ax.add_collection(LineCollection(segs, colors=cc, lw=0.4))
    dim = "1-D" if k_ == 1 else "2-D"
    comps = [int(c) for c in order[:min(max(k_, 1), 2)]]
    ax.set_title(f"elbowed at {lagname}:\nk = {k_} ({dim}), "
                 f"{len(cmx['morse'])} Morse sets (comps {comps})",
                 fontsize=9.5)
    ax.set_xlabel("CC-a"); ax.set_ylabel("CC-b")
    if k_ == 1:                           # 1-D embedding: y is only jitter
        ax.set_yticks([])
        ax.set_ylabel("(1-D: no second axis)")
    draw_cm_graph(axes[1, col], cmx['gedges'], cmx['occ'], colors=cols)
    axes[1, col].set_title("Morse graph", fontsize=10)

axO = axes[0, 2]
imO = axO.imshow(ov / max(ov.sum(), 1), cmap='Blues', aspect='auto')
axO.set_xticks(range(len(cm_high['morse'])))
axO.set_xticklabels([f"hi-M{b}" for b in range(len(cm_high['morse']))])
axO.set_yticks(range(len(cm_low['morse'])))
axO.set_yticklabels([f"lo-M{a}" for a in range(len(cm_low['morse']))])
for a in range(ov.shape[0]):
    for b in range(ov.shape[1]):
        axO.annotate(f"{ov[a,b]/max(ov.sum(),1)*100:.0f}%", (b, a),
                     ha='center', va='center', fontsize=9,
                     color='k' if ov[a, b] < 0.5 * ov.max() else 'w')
axO.set_title("set-matching overlap\n(shared trajectory time-points)",
              fontsize=10)
fig.colorbar(imO, ax=axO, shrink=0.8)

axM = axes[1, 2]
tr_show = np.argsort(res['cohs'])[::8]
strip = np.where(mismatch[tr_show], 2.0,
                 np.where(both[tr_show], 1.0, 0.0))
axM.imshow(strip, aspect='auto', interpolation='nearest',
           cmap=mcolors.ListedColormap(['0.92', '#9ecae1', '#d62728']),
           vmin=-0.5, vmax=2.5)
axM.set_yticks([])
axM.set_xlabel("time step")
axM.set_title(f"membership MISMATCH (red) — {mm_rate*100:.1f}% of "
              "co-assigned points", fontsize=10)
fig.suptitle("Conley-Morse retrace in the low-lag vs high-lag CCA "
             "embeddings — where the decompositions disagree\n"
             "(empirical map: Morse sets certify recurrence; 'index' "
             "comparison = matched set counts / graph position)",
             fontsize=12.5)
fig.savefig("cca_conley_mismatch.png", dpi=130, bbox_inches='tight')
plt.show()
print("saved cca_procrustes.png, cca_variance.png, cca_conley_mismatch.png")

# %% -------------------- rotation+stretch visualization (consecutive lags)
# "Show the bigger lag's top-2 CC embedding, then the next lag down ROTATED
# onto it" -- the geometric content of the Procrustes distance. Each panel:
# grey cloud = embedding at the bigger lag; colored cloud = embedding at the
# next smaller lag after the optimal orthogonal alignment R (rotation, plus
# reflection if det R = -1); arrows = where R sends the smaller lag's basis,
# scaled by the per-axis std ratio (the residual stretch R cannot absorb).
pairs_desc = [(n_lag - 1 - i, n_lag - 2 - i) for i in range(4)]  # big -> next
fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
sub = slice(0, n_samp, 23)                    # subsample for legibility
for ax, (bi, si) in zip(axes.ravel(), pairs_desc):
    Xb = variates[bi][:, :2]                  # bigger lag (base frame)
    Xs = variates[si][:, :2]                  # next lag down
    U_, S_, Vt_ = npl.svd(Xs.T @ Xb, full_matrices=False)
    Rm = U_ @ Vt_                             # optimal Xs -> Xb alignment
    Xs_rot = Xs @ Rm
    resid = npl.norm(Xs_rot - Xb) / npl.norm(Xb)
    theta = np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))
    refl = npl.det(Rm) < 0
    ax.scatter(*Xb[sub].T, s=3, color='0.55', alpha=0.35,
               label=f"lag {LAGS_MS[bi]} ms (base)")
    ax.scatter(*Xs_rot[sub].T, s=3, color='#d95f0e', alpha=0.35,
               label=f"lag {LAGS_MS[si]} ms, rotated")
    tr0 = int(np.argmax(res['cohs']))         # one easy trial's path, both
    seg = slice(tr0 * T_common, (tr0 + 1) * T_common)
    ax.plot(*Xb[seg].T, color='0.2', lw=1.1)
    ax.plot(*Xs_rot[seg].T, color='#a03b00', lw=1.1)
    # per-axis residual after alignment (variates are unit-std, so the
    # mismatch is SHAPE change, not marginal scale)
    rax = npl.norm(Xs_rot - Xb, axis=0) / npl.norm(Xb, axis=0)
    for j, col in enumerate(('#1f77b4', '#2ca02c')):
        v = Rm.T[j] * 1.25                    # unit-std frame: fixed length
        ax.annotate('', xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=2.2))
        ax.annotate(f"e{j+1}: resid {rax[j]:.2f}", v * 1.12, fontsize=8,
                    color=col, ha='center')
    ax.set_title(f"lag {LAGS_MS[bi]} → {LAGS_MS[si]} ms:   "
                 f"θ = {theta:+.1f}°"
                 + ("  (reflected)" if refl else "")
                 + f",  residual = {resid:.3f}", fontsize=10)
    ax.set_xlabel("CC1 (base frame)"); ax.set_ylabel("CC2 (base frame)")
    ax.legend(fontsize=7.5, loc='lower right', markerscale=3)
    ax.set_aspect('equal')
fig.suptitle("What the Procrustes distance measures: the next timescale's "
             "top-2 CC embedding,\nrotated onto the bigger lag's frame "
             "(arrows = rotated basis, scaled by the residual stretch)",
             fontsize=12.5)
fig.savefig("cca_rotation.png", dpi=130, bbox_inches='tight')
plt.show()
print("saved cca_rotation.png")
