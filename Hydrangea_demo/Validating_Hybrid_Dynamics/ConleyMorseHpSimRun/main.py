"""
main.py
-------
Runs the whole analysis in order. Each step is one script; the call map below
lists, for every step, which functions it calls, in what order, and from which file.

    python main.py            # skips the simulation if output/simulation.npz exists
    python main.py --resim    # re-runs the 15-minute simulation first

CALL MAP
========
STEP 1  hippocampus.py                                    -> output/simulation.npz, fig1-6
        run(cfg)                       hippocampus.py     build and simulate the network
          SquareMaze, TrajectoryMemory, ReplaySchedule    hippocampus.py
          StateMachine, PoissonEvents, Oscillator,        neurosim.py   (WALK/STILL, SWRs, theta/ripple)
          LIFPopulation, Projection, SpikeTrace           neurosim.py   (neurons, synapses, gates)
        np.savez_compressed(...)                          spikes, state, position, speed
        analyse(res)                   hippocampus.py     fig1-6: traces, rasters, tuning, FC

STEP 2  hippo_cmg.py K                                    -> output/hippo_cmg_K<K>.pkl
        a. load simulation.npz
        b. binned_counts(S, 1 ms, 25 ms)                  neurosim.py   spike counts, 25 ms bins
        c. z-score, np.cov, np.linalg.eigh                              covariance PCA
        d. elbow of eigenvalue decay -> K (or K from argv)
        e. delay_embed(P, 20, 1)                          cmg_lib.py    20 lags of the top-K PCs
        f. pca_fit, pca_apply                             cmg_lib.py    -> 5D chart Z
        g. detect guards from P alone (500 ms before/after mean-shift score,
           median + 6 MAD); true states used only to score them
        h. analyse(n, tau, count, routing)  for routing in (full, detected):
             GridND(Z, n)                                 cmg_lib.py    n^5 boxes; box id per point
             transition pairs Z(t) -> Z(t+tau); 'detected' drops pairs spanning a detected guard
             count pairs, keep edges seen >= count times
             build_map_nd                                 cmg_lib.py    Boolean box -> box matrix A
             prune dead ends / tiny sinks; connected_components (scipy)    repeated
             connected_components                         scipy         SCCs = Morse sets
             BFS on the condensation                                    reachability between sets
             conley_indices_nd                            cmg_lib.py    per Morse set:
               forward_closure                            cmg_lib.py      index pair (U, U \\ M)
               betti_pair_nd                              cmg_lib.py      GF(2) homology, 6 slots
        i. hasse                           hippo_cmg.py                 Morse graph edges
        j. downsets                        cmg_lib.py                   lattice of attractors
        k. resets (detected only)          hippo_cmg.py                 guards as reset edges
        l. same analyse() over the SWEEP list (n, tau, count)            persistence check
        m. pickle.dump -> output/hippo_cmg_K<K>.pkl

STEP 3  plot_hippo_cmg.py output/hippo_cmg_K<K>.pkl      -> output/cmg_*_K<K>.png
        cmg_chart    spectrum + elbow, delay-PCA variance, 5D chart projections
        cmg_guards   detected guards vs the true switches
        cmg_morse    Morse sets in the chart, Conley-Morse graphs (full, detected + resets)
        cmg_lattice  lattice of attractors
        cmg_sweep    Morse sets of each kind across the sweep

STEPS 2-3 run for K = the eigenvalue-decay elbow (5) and K = 12.
"""
import os
import subprocess
import sys

OUT = "output"
K_RUNS = [None, 12]          # None = use the elbow found by hippo_cmg.py


def step(title, cmd):
    print(f"\n=== {title}\n    $ {' '.join(cmd)}", flush=True)
    subprocess.run([sys.executable] + cmd, check=True)


if "--resim" in sys.argv or not os.path.exists(f"{OUT}/simulation.npz"):
    step("STEP 1  simulate the CA1 network (hippocampus.py, ~10 min)", ["hippocampus.py"])
else:
    print(f"=== STEP 1  skipped: using saved {OUT}/simulation.npz (pass --resim to re-simulate)")

for K in K_RUNS:
    args = [] if K is None else [str(K)]
    step(f"STEP 2  5D Conley-Morse graph and lattice, K = {K or 'elbow'} (hippo_cmg.py)",
         ["hippo_cmg.py"] + args)
done = sorted(f for f in os.listdir(OUT) if f.startswith("hippo_cmg_K") and f.endswith(".pkl"))
for f in done:
    step(f"STEP 3  figures from {f} (plot_hippo_cmg.py)", ["plot_hippo_cmg.py", f"{OUT}/{f}"])

print("\n=== outputs in", OUT)
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        print("   ", f)
