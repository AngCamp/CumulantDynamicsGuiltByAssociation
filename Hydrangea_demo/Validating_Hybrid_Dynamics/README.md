# Overview and Dates

## Important papers today oct 27th

We have a few options one is an autoencoder method, Im a lot more familiar with this method but on the otherhand it "feels" od to further force down
a 6D linear embedding dimension (by CCA and PCA) into a smaller space 


Kalies, W., Rivas, B., & Wehbe, T. (2026). Computation of Attractor Lattices for a Class of Hybrid Systems. arXiv preprint arXiv:2610.15123.
-I'm still working out my intuition on this but...
-so one of the key results in this that I'd still working out is in 
preliminaries 2.1 it lays out how in practice any binary set it equivilent to
a grid and can be used to define interiors that share a closure, or unions within the set
-in practice this means almost any set, even sets computed as epsilon balls in R^n can be grid like.  This brings the attractor landscape, our delay embedding back to Conley-Morse Graph DB machinery we need

-figure 1 shows how we can treat this as a cylindar that exists for 1 time
step, im thinking we define it based on the lipschitz bound, that lets us
basically treat it piece wise (this paper is the citation we can have for this to be valid)





Gameiro, M., Gelb, B., & Mischaikow, K. (2025). Rigorously Characterizing Dynamics with Machine Learning. arXiv preprint arXiv:2505.17302.
-small note on the explicit Lipschitz bound,
-discusses using autoencoders to push dynamics down into smaller spaces
-shows you can compte exact lipschitz bounds to provide lyaoponov stabilit




## Readings from the papers found (Oct 26th) to solve and side step identified problems (accoridng to claude)
It seems like we may be abel to side step the CCA

**Bailon et al. (2026) — what to read**

- Definitions 1.2–1.3: defines attracting block and tolerance, the two objects that replace ε-chains in this framework.
- Theorem 1.4 and Corollary 2.2 (statements only): the condition under which a latent attracting block pulls back to an attracting block, and so an attractor, in the full system.
- Section 3.2: how CMGDB turns a map on a grid into a directed graph and a Morse graph; this is the τ-ε machinery in combinatorial form.
- Section 4.2: what happens when grid resolution and model error interact — two coarsening moves that reconcile mismatched Morse graphs, which maps onto your bin-width comparisons.
- Table 9 with Section A.4: evidence that the residual bound fails while the recovered structure holds, which sets how much to trust the method on data.

**Rivas et al. (2026) — what to read**

- Definition 4.3 and Figure 1: the outer approximation property, stated in one line and one picture; this is the object your chains become.
- Proposition 4.4 with its proof: the padding argument that converts a model-error bound δ into an outer approximation of the true flow; the proof runs four lines and your background covers it.
- Section 4.2: how Morse sets, Morse graphs, and regions of attraction fall out of graph reachability.
- Tables 1–2 and the surrounding commentary: how the switching-system construction compares against Lipschitz and Gaussian-process constructions, and why each rival fails.

**Synthesis for the journal**

The τ-ε chain construction appears circular because chains are defined through the dynamics they are meant to recover. Both papers cut the circle the same way: fit a surrogate model to the data, discretize its domain into boxes, and map each box to the set of boxes its image can touch, with padding. The ε of a chain becomes box diameter plus padding; the τ becomes the time step or bin width; chain recurrence becomes the strongly connected components of a directed graph, computed once. Attractors, saddles, and repellers are then read off the Morse graph through Conley indices. What differs is the surrogate and the transfer argument. Rivas et al. fit a switching vector field in two dimensions and transfer through a flow-error bound (Proposition 4.4); this requires ẋ and low dimension. Bailon et al. fit an encoder and a latent map, compute in one to three latent dimensions, and transfer through a residual-versus-tolerance inequality (Theorem 1.4); this requires consecutive state pairs, which binned population rates supply. For neural data the Bailon et al. route is the one to implement, with the bin width playing the role of τ and the residual check reported as a diagnostic rather than a certificate.

**References**

Bailon, P., Gameiro, M., Gelb, B., Kalies, W., Kramar, M., Mischaikow, K., Rivas, B., & Vieira, E. (2026). *Characterizing high-dimensional dynamics by combinatorial-topological methods on a latent space* [Preprint]. arXiv.

Rivas, B., Kalies, W., Iwasaki, K., Bloch, A., & Ghaffari, M. (2026). Topological dynamics via learned hybrid systems. *Proceedings of Machine Learning Research*. Advance online publication.

# Note end of day October 23rd

I think I've had a breakthrough, I came up with the idea to apply CCA to create shared components,
by plotting curves of the proscrutes distance between the next lag down for each space as a component of k,
I was inspired by the 2009 mulitscale PCA plot.  I de think though that, what we need to do is going to be
related to the lipschitz bound, we need to prove that for each step down the lag, So you could compose
a smooth map out of the lags that each preserves the whole graph for Gronwalls inequatity to hold.

I need to think on this a bit more.  I feel like this must be tied to perturbation theory somehow.  Or alternatively,
it may have to do with the numbers of k you need to retain...

I have to read that algortihic paper on the COnloey-Morse Graph


The doe is here:"""
It’s in cca_conley_morse.py, in two pieces (the plotting isn’t wrapped in a function — it’s inline cell code):

The metric: procrustes_resid(Xc, Yc) — defined in the cell marked # %% ... Procrustes distances (~line 140). Computes ‖X·R − Y‖_F / ‖Y‖_F with R = UVᵀ from SVD(XᵀY).
The curve data: immediately below it, the loop filling proc_next[k-1, li] = procrustes_resid(variates[li][:, :k], variates[li+1][:, :k]) — one row per k, one column per consecutive lag pair. (proc_mat right after is the heat-map version.)
The plot itself: in the next cell (# %% ... figures 1 + 2, ~line 160), the block drawing on axP — the for k in range(K_MODEL): axP.plot(mid, proc_next[k], ...) loop, saved into cca_procrustes.png (left panel).

The per-pair visualization (clouds rotated onto the bigger lag’s frame) is the separate last cell, # %% ... rotation+stretch visualization, saved as cca_rotation.png.

If you’re folding this into your Hydrangea repo, worth extracting steps 1–2 into a function like procrustes_curve(variates, k_max) so your own pipeline can call it on any set of per-lag embeddings — happy to refactor it that way if useful.
"""

Some incredibly important references from todays work:
Bailon, P., Gameiro, M., Gelb, B., Kalies, W., Kramar, M., Mischaikow, K., ... & Vieira, E. (2026). 
Characterizing High-dimensional Dynamics by Combinatorial-Topological Methods on a Latent Space. 
arXiv preprint arXiv:2609.01509.
-examines bounds needed for latent models using autoencoders, relevant to our CCA apprach

Rivas, B., Iwasaki, K., Kalies, W., Bloch, A., & Ghaffari, M. (2025). Topological Dynamics via Learned 
Hybrid Systems. arXiv preprint arXiv:2511.08737.
-this may offer a side step to the one above
-in section 4.1 they cite  Viera et al., (2022) posted below (at this url https://arxiv.org/abs/2202.08383)
-

Morse Graphs: Topological Tools for Analyzing the Global Dynamics of Robot Controllers
Ewerton R. Vieira, Edgar Granados, Aravind Sivaramakrishnan, Marcio Gameiro, Konstantin Mischaikow, Kostas E. Bekris (2022)
https://arxiv.org/abs/2202.08383

This is the refrence for the flavor of CCA we are using in computing the tau-epsilon-chain CCA
Sørensen, M., Kanatsoulis, C. I., & Sidiropoulos, N. D. (2021). Generalized canonical correlation analysis: 
A subspace intersection approach. IEEE Transactions on Signal Processing, 69, 2452-2467.
https://arxiv.org/pdf/2003.11205

This may have been cited earlier but it shows that overlapping comapct sets, not eact grids are whats needed for
the epsilon-tau chains to be computed to get our morse-conley graph.
Mrozek, M. (1999). An algorithmic approach to the Conley index theory. 
Journal of Dynamics and Differential Equations, 11(4), 711-734.

Lastly this paper has a great section on the lipschitz-bound, the method is cited in 



# Week of October 23rd

Leaping straight into the Hybrid system is not a great idea, it may prove to be a computational formality (which I will discuss
in a second) or it may prove instead to be useful if time-varying lipshitz constants prove to be unable to be estimated below
the particular lag.  

Conley-Morse Graphs have proved significantly more interesting on the other hand, but connections to hybrid dynamic systems remain
in my mind.  The conley-morse graph relies on a box cutting (look up the specific method) algorithm, that goes down its
notion of space (epsilon) and time (tau) down certain lags on each to find regions of invariant sets.  The definition
of the invariant set comes from Grönwall's inequality

So somehting I am struggling a bit with is the circularity of the analysis, we have no spatial measures and indeed we want
estimates of attractors in neural space.  


Romero, A. T., & Vieira, E. R. (2021). Existence of periodic orbits for piecewise-smooth vector fields with sliding region via Conley theory. arXiv preprint arXiv:2106.15569.
- applied the conley index to wilson-cowan models and race to threshold models, its very interesting

Rivas, B., Iwasaki, K., Kalies, W., Bloch, A., & Ghaffari, M. (2025). Topological Dynamics via Learned Hybrid Systems. arXiv preprint arXiv:2511.08737.
- this paper compares a computed morse graph, vs a switching latent system
- not sure if it contains the solution to the the "latent Grönwall knot"


 My possible solution:  CCA between scales, becasue PCA across them will not preserve commutivity.  
 i.e you can be sure the component 1 and scale 1 is the same latent.
 
 There are mathematical arguments to ensure you go from the top down:
 Dey, T. K., Lipiński, M., Mrozek, M., & Slechta, R. (2022). Tracking dynamical features via continuation and persistence. arXiv preprint arXiv:2203.05727.


CCA appears to work because the box cutting will hold under compact sets, which is a linear rotation of the original set,
Mrozek, M. (1999). An algorithmic approach to the Conley index theory. Journal of Dynamics and Differential Equations, 11(4), 711-734.
https://ww2.ii.uj.edu.pl/~mrozek/papers/aacit-revised.pdf
- you can visualize this by showing the CCA "grids"
- compare them to PCA grids stabilities and show that it limits certain claims you can make downstream about the latents
- the ranked grid alignment across the log scales may then itself be a 
We could compare the magnitude of changes across the 

# Week of october 15th

Started getting interested in hybrid dynamic systems found methods for it in Conley-Morse Graphs and Hybrid Dynamic system.

It seems appealing, conley index could be useful to characterize regions from a global attractor then zoom down into them.

Started learning about glueing methods and what makes the Conley-Morse Graph estimateable.

# Seeking empirical Evidence of Hybrid Dynamics

In this folder we will begin by using the Lipschitz-Constant espescially, to look for Evidence of Hybrid dynamics in various neural datasets.  There, needs to be evidence of sudden regime shifts, and these regime shifts should be occuring in the main locations of a phase space.  While sudden regime changes are well documented in neuroscience (up/down cortical states, sleep wake, certian models of decision making in LIP which has a step), we need to characterize thema. bit to see if these exhibit what could be argued to be hybrid dynamics, and not mearly stochastic switching dynamics.

We require evidence of rapid switching at similar times poiints ideally, by similar neurons.

## Identifying optimal tau, window size, for computing covariance matrix for SVD

Little, A. V., Lee, J., Jung, Y. M., & Maggioni, M. (2009, August). Estimation of intrinsic dimensionality of samples from noisy low-dimensional manifolds in high dimensions with multiscale SVD. In 2009 IEEE/SP 15th Workshop on Statistical Signal Processing (pp. 85-88). IEEE.
https://jasondlee88.github.io/papers/ssp09.pdf 
- they used standard deviation we can instead plot a smooth estimator of the lipschwitz constant


## A proposal for identifying Sliding Lipschitz Constants in neural data

It is infeasible to compute this constant for a large number of neurons so instead we may simply use SVD at first and look for sudden jumps.

We have two promising methods for the full modelling...

- [Katayama, N., & Susuki, Y. (2025, June). Koopman operators for global analysis of hybrid limit-cycling systems: Construction and spectral properties. In 2025 European Control Conference (ECC) (pp. 1949-1954). IEEE.](https://arxiv.org/pdf/2411.04052)

and...

- [Teng, S., Liu, H., Song, J., & Sreenath, K. (2025). Chyll: Learning continuous neural representations of hybrid systems. arXiv preprint arXiv:2512.10117.](https://arxiv.org/pdf/2512.10117)

The idea eventually is to be able to identify a hiearchy of metastable regimes and their locations and timing so that we can characterize them.

- [Rivas, B., & Kalies, W. (2026). Conley Index Theory for Hybrid Systems. arXiv preprint arXiv:2606.18501.](https://arxiv.org/html/2606.18501v1)
- [Dey, T. K., Mrozek, M., & Slechta, R. (2022). Persistence of Conley--Morse Graphs in Combinatorial Dynamical Systems. SIAM Journal on Applied Dynamical Systems, 21(2), 817-839.](https://arxiv.org/abs/2107.02115)

Crucially to the hybrid system we must find sub jumps in dynmaics.

## Datasets to Check for Resets or Gaurds

The following are two neuropixels recordings datasets.  The first covers a wide range o f brain areas during a two choice delayed decion making task (Chen et al., 2024), the other a spatial navigation task with sleep sessions with recordings of RSC, CA1,and CA3 mainly (Gonzalez & Voroslakos, 2026).

- [Chen, S., Liu, Y., Wang, Z. A., Colonell, J., Liu, L. D., Hou, H., ... & Svoboda, K. (2024). Brain-wide neural activity underlying memory-guided movement. Cell, 187(3), 676-691.](https://www.cell.com/cell/fulltext/S0092-8674(23)01445-9)

  Data for the above paper is located at [Dandiset 000363](https://dandiarchive.org/dandiset/000363?search=000363&pos=1)

- [Gonzalez, J., & Voroslakos, M. (2026). High-density extracellular recordings (Neuropixels/SiNAPS) of Hippocampal-Cortical dynamics during spatial behavior. DANDI Archive.](https://dandiarchive.org/dandiset/001695/draft)

  Data is located at [Dandiset 001695](https://dandiarchive.org/dandiset/001695/draft). 


The first test is to run SVD on these datasets and run an sliding instantaneous Lipschitz constant across it to identify regime changes with sudden changes in dynamics.

## A proposal for identifying Sliding Lipschitz Constants in neural data

It is infeasible to compute this constant for a large number of neurons so instead we may simply use SVD at first and look for sudden jumps.

We have two promising methods for the full modelling...

- [Katayama, N., & Susuki, Y. (2025, June). Koopman operators for global analysis of hybrid limit-cycling systems: Construction and spectral properties. In 2025 European Control Conference (ECC) (pp. 1949-1954). IEEE.](https://arxiv.org/pdf/2411.04052)

and...

- [Teng, S., Liu, H., Song, J., & Sreenath, K. (2025). Chyll: Learning continuous neural representations of hybrid systems. arXiv preprint arXiv:2512.10117.](https://arxiv.org/pdf/2512.10117)

The idea eventually is to be able to identify a hiearchy of metastable regimes and their locations and timing so that we can characterize them.

- [Rivas, B., & Kalies, W. (2026). Conley Index Theory for Hybrid Systems. arXiv preprint arXiv:2606.18501.](https://arxiv.org/html/2606.18501v1)
- [Dey, T. K., Mrozek, M., & Slechta, R. (2022). Persistence of Conley--Morse Graphs in Combinatorial Dynamical Systems. SIAM Journal on Applied Dynamical Systems, 21(2), 817-839.](https://arxiv.org/abs/2107.02115)

Crucially to the hybrid system we must find sub jumps in dynmaics.

## Datasets to Check for Resets or Gaurds

The following are two neuropixels recordings datasets.  The first covers a wide range o f brain areas during a two choice delayed decion making task (Chen et al., 2024), the other a spatial navigation task with sleep sessions with recordings of RSC, CA1,and CA3 mainly (Gonzalez & Voroslakos, 2026).

- [Chen, S., Liu, Y., Wang, Z. A., Colonell, J., Liu, L. D., Hou, H., ... & Svoboda, K. (2024). Brain-wide neural activity underlying memory-guided movement. Cell, 187(3), 676-691.](https://www.cell.com/cell/fulltext/S0092-8674(23)01445-9)

  Data for the above paper is located at [Dandiset 000363](https://dandiarchive.org/dandiset/000363?search=000363&pos=1)

- [Gonzalez, J., & Voroslakos, M. (2026). High-density extracellular recordings (Neuropixels/SiNAPS) of Hippocampal-Cortical dynamics during spatial behavior. DANDI Archive.](https://dandiarchive.org/dandiset/001695/draft)

  Data is located at [Dandiset 001695](https://dandiarchive.org/dandiset/001695/draft). 


The first test is to run SVD on these datasets and run an sliding instantaneous Lipschitz constant across it to identify regime changes with sudden changes in dynamics.

## Note Oct  25th, Hybrid Dynamics appear to be less important than Conley-Morse Graphs

Bailon, P., Gameiro, M., Gelb, B., Kalies, W., Kramar, M., Mischaikow, K., ... & Vieira, E. (2026). Characterizing High-dimensional Dynamics by Combinatorial-Topological Methods on a Latent Space. arXiv preprint arXiv:2609.01509.
<br><br>
Dey, T. K., Lipiński, M., & Soriano-Trigueros, M. (2026). Conley–Morse Persistence Barcode: A Homological Signature of Combinatorial Bifurcations. Foundations of Computational Mathematics, 1-59.
<br><br>
Mrozek, M. (2017). Conley–Morse–Forman theory for combinatorial multivector fields on Lefschetz complexes. Foundations of Computational Mathematics, 17(6), 1585-1633.
<br><br>
That first paper in particular, Bailon et a., (2026), is critical for studying the neural firing in terms of its latent space.

# Multi=resolution DMD Turorial (Our base metric)

https://pydmd.github.io/PyDMD/tutorial3mrdmd.html

Publicaiton: [Kutz, J. N., Fu, X., & Brunton, S. L. (2016). Multiresolution dynamic mode decomposition. SIAM Journal on Applied Dynamical Systems, 15(2), 713-735.](https://arxiv.org/abs/1506.00564)

Blog post about mrDMD: https://robert-taylor.me/blog/mrdmd-python/

# Code for finding at what time scale the PCA just becomes diffusion


Your figure is doing a version of mean-square-displacement (MSD) analysis, the same tool single-particle tracking uses to tell diffusion from directed motion. Here is the clean way to compute it and to get the crossover lag directly.

1. Displacement curves. Work in a fixed latent frame (as your plot does with its shared PCA frame). For a log-spaced set of lags τ and each time t, aligned to the go cue and averaged over trials:

D²(t; τ) = ⟨ ‖x(t+τ) − x(t)‖² ⟩

2. The local scaling exponent. Take the slope of log D against log τ between neighbouring τ values:

s(t, τ) = d log D / d log τ

Each regime has a characteristic slope:

Slope s	What dominates
≈ 0	Measurement noise around a fixed state (iid jitter, so displacement doesn’t grow with τ), or an already-completed jump
≈ ½	Diffusion / random-walk drift (D ∝ √τ)
≈ 1	Deterministic flow (D ∝ τ)

Your colorbar’s labels (−½ / 0 / 1) look like a shifted or normalized version of this. I’m inferring the convention, so check it with step 5.

3. Fit the three components directly. This answers “where does flow start?” for each time window. For small τ, the three contributions add in quadrature:

D²(τ) ≈ a·τ² + b·τ + c

a·τ²: deterministic drift. √a is the drift speed in latent units per second; this is the piece Grönwall describes.
b·τ: diffusion. b is twice the diffusion coefficient per dimension.
c: measurement-noise floor, from Poisson counting and finite bins.

Fit it per time window with a least-squares polynomial in τ. Two crossover lags fall out:

τ_noise = √(c / a): drift beats measurement noise.
τ_diff = b / a: drift beats diffusion.

A usable lag is τ > max(τ_noise, τ_diff). That is the window in which a Morse graph can be non-trivial. ε then has to sit between √(c + bτ) and √a·τ at your chosen τ.

4. Persistence, i.e. your ρ. Your ρ = ‖Δ²x‖ / ‖Δx‖ is a smoothness test. Its reference values:

≈ √3 if positions are iid noise (var Δ² = 6σ², var Δ = 2σ²). This is your colorbar’s “√3 ≈ noise.”
≈ √2 for a pure random walk.
→ 0 for smooth flow, since Δ²x ~ τ²ẍ is much smaller than Δx ~ τẋ.

So low ρ at the same (t, τ) where s ≈ 1 is two independent confirmations of deterministic flow.

5. Calibrate on surrogates. Run the exact same code on four synthetic latents with your sampling and trial count:

iid Gaussian jitter;
a Brownian walk;
constant-velocity drift plus noise;
a step at t = 0.

This pins down which colour means what in your normalization. Your Poisson null then shows how much of the real α ≈ 1 band is beyond what rates alone produce. In your plots, it’s the ~0–150 ms post-go band at τ ≳ 50 ms.

6. Lipschitz / Grönwall β, if you want it. Use finite-size Lyapunov exponents. Pick pairs of trials whose latents are within δ₀ at time t, and compute λ(τ) = (1/τ) log(‖δ(τ)‖ / ‖δ₀‖). Then subtract the noise contribution using the same fit: δ²(τ) ≈ e^{2λτ}δ₀² + bτ + c. Without that correction, small-τ “Lipschitz constants” are inflated by the noise floor. That inflation is the trap the quoted sentence warns about.

A minimal version of steps 1–3:


'''
import numpy as np

def msd_decompose(X, taus, t_idx, win=5):
    """X: (trials, T, d) latents in a fixed frame.
    taus: lags in samples. t_idx: centre index of the analysis window.
    Returns slope s(tau) and fitted (a, b, c) for D^2 = a tau^2 + b tau + c."""
    D2 = []
    for tau in taus:
        sl = slice(t_idx - win, t_idx + win + 1)
        d = X[:, sl.start + tau:sl.stop + tau] - X[:, sl]   # (trials, 2win+1, d)
        D2.append(np.mean(np.sum(d**2, axis=-1)))
    D2 = np.array(D2)
    t = np.asarray(taus, float)
    s = np.gradient(0.5 * np.log(D2), np.log(t))           # d log D / d log tau
    A = np.stack([t**2, t, np.ones_like(t)], 1)
    (a, b, c), *_ = np.linalg.lstsq(A, D2, rcond=None)
    tau_noise = np.sqrt(max(c, 0) / a) if a > 0 else np.inf
    tau_diff = b / a if a > 0 else np.inf
    return s, (a, b, c), max(tau_noise, tau_diff)
    '''
    
 Citations(for optimal lag, validate espescially the DOIs):
 
 1. Splitting MSD into drift + diffusion (the a·τ² + b·τ part)

H. Qian, M. P. Sheetz, E. L. Elson (1991). “Single particle tracking. Analysis of diffusion and flow in two-dimensional systems.” Biophysical Journal 60(4): 910–921. doi:10.1016/S0006-3495(91)82125-7.

Reference for telling random diffusion from uniform-velocity transport in trajectory data.  MSD = 4Dτ + v²τ² in 2D.

X. Michalet (2010). “Mean square displacement analysis of single-particle trajectories with localization error: Brownian motion in an isotropic medium.” Physical Review E 82: 041914. doi:10.1103/PhysRevE.82.041914.

Localization error biases MSD fits, and that an unweighted least-squares fit using an optimal number of MSD points works best. The second finding is relevant to how many τ values to use in the fit.

3.Lyapunov exponents at finite scale (the noise-aware λ(τ) step)

-E. Aurell, G. Boffetta, A. Crisanti, G. Paladin, A. Vulpiani. “Predictability in the large: an extension of the concept of Lyapunov exponent.” 

This introduces the finite-size Lyapunov exponent.

J. B. Gao, J. Hu, W. W. Tung, Y. H. Cao (2006). “Distinguishing chaos from noise by scale-dependent Lyapunov exponent.” Physical Review E 74(6): 066204.

Closest match to the multiscale heatmap. It is built to work on short, noisy series. It separates deterministic chaos, noise, stochastic oscillations, and motion that is “chaotic at small scales but diffusive at large scales.” 
The α ≈ −½ vs α ≈ 1 split.
This paper Gao et al., (2006) has the SDLE algorithm that gives this result.  It is very important.

-√3 and √2 values for ρ. These follow directly from the variances of first and second differences:
iid noise: var Δ²x = 6σ², var Δx = 2σ², so ρ ≈ √3.
random walk: var Δ²x = 2τσ², var Δx = τσ², so ρ ≈ √2.

This is a possible method Claude suggested, I'm against using it as I can't validate the math, and
its not based on any exist publication but it had this note:
6. Lipschitz / Grönwall β, if you want it. Use finite-size Lyapunov exponents. 
Pick pairs of trials whose latents are within δ₀ at time t, and compute λ(τ) = (1/τ) log(‖δ(τ)‖ / ‖δ₀‖). 
Then subtract the noise contribution using the same fit: δ²(τ) ≈ e^{2λτ}δ₀² + bτ + c.


## Discussion Points:

Future work may determine if the noise limit here is truly browning or measurement effect.  Theoretical work
has been shown that deterministic high dimensional networks may appear noisey due to undersampling.

Sompolinsky, H., Crisanti, A., & Sommers, H. J. (1988). Chaos in random neural networks. Physical Review Letters, 61(3), 259.
-find more references like this and verify
