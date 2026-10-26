# Seeking empirical Evidence of Hybrid Dynamics

In this folder we will begin by using the Lipschitz-Constant espescially, to look for Evidence of Hybrid dynamics in various neural datasets.  There, needs to be evidence of sudden regime shifts, and these regime shifts should be occuring in the main locations of a phase space.  While sudden regime changes are well documented in neuroscience (up/down cortical states, sleep wake, certian models of decision making in LIP which has a step), we need to characterize thema. bit to see if these exhibit what could be argued to be hybrid dynamics, and not mearly stochastic switching dynamics.

We require evidence of rapid switching at similar times poiints ideally, by similar neurons.

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