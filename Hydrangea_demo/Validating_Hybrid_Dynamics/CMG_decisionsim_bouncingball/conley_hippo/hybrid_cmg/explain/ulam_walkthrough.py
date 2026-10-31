"""Walk the Ulam-matrix construction in run_cmg.decode_cmg on the real
cached data, printing shapes at every step and drawing one cell's image."""
import sys, pickle, numpy as np
sys.path.insert(0, "/home/claude/hybrid_cmg")
import cmg_lib as cl
from hds import decode as dc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = pickle.load(open("/home/claude/hybrid_cmg/hds_sim.pkl", "rb"))
rec, segs = D['rec'], D['segs']
b, s, ell, m, NX = 120, 10, 90, 3, 64          # best flow setting

# 1. bin counts, normalize, correlation PCA, keep elbow K
Xf, trf, ttf = dc.fine_bins(segs, rec['base'])
Xw, tt, tr = dc.window_counts(Xf, trf, ttf, b, s)
Lat, K, ev = dc.scores(Xw)
print(f"1. PC scores Lat: {Lat.shape}  (samples x K={K}), step {s} ms")

# 2. delay embed with lag ell: Ls steps
Ls = ell // s
Ds, trs = [], []
for ti in np.unique(tr):
    idx = np.flatnonzero(tr == ti)
    Dm = cl.delay_embed(Lat[idx], m, Ls).reshape(-1, m * K)
    Ds.append(Dm); trs.append(np.full(len(Dm), ti))
Dall = np.vstack(Ds); trs = np.concatenate(trs)
print(f"2. delay vectors Dall: {Dall.shape}  (m={m} copies of K={K} -> {m*K} dims), "
      f"lag = {Ls} steps = {ell} ms")

# 3. second PCA -> 2D chart, 64x64 grid, cell id per sample
mu2, V2 = cl.pca_fit(Dall, d=2, max_n=40000)
Z = cl.pca_apply(Dall, mu2, V2)
g = cl.Grid2D(Z, NX)
cz = g.cells(Z)
print(f"3. chart Z: {Z.shape}; grid {NX}x{NX} = {g.N} cells; "
      f"occupied cells {len(np.unique(cz[cz>=0]))}")

# 4. pairs (t, t+Ls) within trial -> (source cell, destination cell)
i0 = np.arange(len(Z) - Ls); same = trs[i0] == trs[i0 + Ls]
src, dst = cz[i0[same]], cz[i0[same] + Ls]
ok = (src >= 0) & (dst >= 0); src, dst = src[ok], dst[ok]
print(f"4. transitions: {len(src)} (source cell -> cell {ell} ms later)")

# 5. counts -> sparse boolean matrix with 1-cell target padding
P, cnt = np.unique(np.stack([src, dst], 1), axis=0, return_counts=True)
print(f"   distinct (src,dst) pairs: {len(P)}  (the Ulam COUNT matrix has "
      f"{len(P)} nonzeros of {g.N**2:,} entries)")
A = cl.build_map(src, dst, g.nx, g.ny, src_dilate=0)
A = cl.support_restrict(A, src, dst)
print(f"5. boolean map A: {A.shape}, nonzeros {A.nnz} (after padding), "
      f"density {A.nnz / g.N**2:.2e}")
row_out = np.diff(A.indptr)
print(f"   cells with any out-edge: {(row_out>0).sum()}; "
      f"median out-degree {np.median(row_out[row_out>0]):.0f}")

# ---- figure: one source cell, its raw landing cells, and its padded image
c_star = np.bincount(src).argmax()                  # most-visited source cell
land = dst[src == c_star]
cells_land, n_land = np.unique(land, return_counts=True)
img = A[c_star].indices
C = g.centers()
fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
ax[0].scatter(Z[::5, 0], Z[::5, 1], s=1, color="0.8")
inc = np.flatnonzero(cz == c_star)
ax[0].scatter(Z[inc, 0], Z[inc, 1], s=6, color="#1e88e5", label=f"{len(inc)} samples in cell {c_star}")
fut = inc[:-Ls][trs[inc[:-Ls]] == trs[inc[:-Ls] + Ls]] + Ls
ax[0].scatter(Z[fut, 0], Z[fut, 1], s=6, color="#d81b60", label=f"same samples {ell} ms later")
for a_, b_ in zip(inc[:40], inc[:40] + Ls):
    if b_ < len(Z) and trs[a_] == trs[b_]:
        ax[0].annotate("", xy=Z[b_], xytext=Z[a_], arrowprops=dict(arrowstyle="->", color="0.4", lw=0.5))
ax[0].legend(fontsize=8); ax[0].set_title("step 4: one source cell and where its samples go")
# step 5: count row and boolean row, shown over the full grid with the
# source cell boxed and the landing cluster visible
cnts = np.zeros(g.N); cnts[cells_land] = n_land
ci, cj = divmod(c_star, g.ny)
li, lj = np.divmod(cells_land, g.ny)
lo_i, hi_i = min(ci, li.min()) - 2, max(ci, li.max()) + 3
lo_j, hi_j = min(cj, lj.min()) - 2, max(cj, lj.max()) + 3
lo_i, lo_j = max(lo_i, 0), max(lo_j, 0)
sub = cnts.reshape(g.nx, g.ny)[lo_i:hi_i, lo_j:hi_j]
im = ax[1].imshow(sub.T, origin="lower", cmap="Blues",
                  extent=(lo_i - .5, hi_i - .5, lo_j - .5, hi_j - .5))
ax[1].scatter([ci], [cj], marker="s", s=140, facecolor="none", edgecolor="#1e88e5", lw=2,
              label="source cell")
plt.colorbar(im, ax=ax[1], label="count of landings")
ax[1].legend(fontsize=8, loc="upper left")
ax[1].set_title(f"Ulam COUNT row for cell {c_star}: {len(cells_land)} landing cells,\n"
                f"{len(land)} transitions (row sums to {len(land)})")
padm = np.zeros(g.N); padm[img] = 1
sub2 = padm.reshape(g.nx, g.ny)[lo_i:hi_i, lo_j:hi_j]
ax[2].imshow(sub2.T, origin="lower", cmap="Greys", vmin=0, vmax=1.6,
             extent=(lo_i - .5, hi_i - .5, lo_j - .5, hi_j - .5))
ax[2].scatter([ci], [cj], marker="s", s=140, facecolor="none", edgecolor="#1e88e5", lw=2)
ax[2].set_title(f"boolean row of A: F(cell {c_star})\n"
                f"landing cells + 1-cell padding = {len(img)} cells, counts dropped")
for a in ax[1:]:
    a.set_xlabel("chart axis 1 (cell index)"); a.set_ylabel("chart axis 2 (cell index)")
fig.suptitle(f"How the Ulam matrix is built  (b={b} ms, step={s} ms, lag={ell} ms, K={K}, m={m}, {NX}x{NX} grid)")
fig.tight_layout(); fig.savefig("/home/claude/hybrid_cmg/explain/ulam_walkthrough.png", dpi=120)
# full sparsity pattern
fig, ax = plt.subplots(figsize=(6.5, 6))
coo = A.tocoo()
ax.scatter(coo.col, coo.row, s=0.2, color="k")
ax.set(xlabel="destination cell", ylabel="source cell", title=f"sparsity of A ({A.nnz} of {g.N**2:,} entries)")
ax.invert_yaxis(); fig.tight_layout(); fig.savefig("/home/claude/hybrid_cmg/explain/ulam_sparsity.png", dpi=120)
print("figures saved")
