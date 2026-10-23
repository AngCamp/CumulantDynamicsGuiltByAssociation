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


The first test is to run SVD on these datasets and run an sliding instantaneous Lipschitz constant across it to identify regime changes with sudden changes in dynamics
