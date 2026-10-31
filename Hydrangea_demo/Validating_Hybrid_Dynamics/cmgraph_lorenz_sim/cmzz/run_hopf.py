"""Hopf 2D: 1 s cut into 4 × 250 ms and 10 × 100 ms discrete steps; compare.

usage: python run_hopf.py [grid] [outdir] [extra_grid_for_10_steps]
"""
import json
import os
import sys
import time

from cmzz.systems import Hopf2D
from cmzz.pipeline import CMGZigzag
from cmzz import viz

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
out = sys.argv[2] if len(sys.argv) > 2 else "out"
extra = int(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)

runs = [(4, N), (10, N)] + ([(10, extra)] if extra else [])
zs = []
for n_steps, grid in runs:
    print(f"=== Hopf2D  {n_steps} steps  grid {grid}²", flush=True)
    t0 = time.perf_counter()
    z = CMGZigzag(Hopf2D(), n_steps, grid).run()
    z.timings["total"] = time.perf_counter() - t0
    print(z.report())
    print("detection:", z.detection(), "total", round(z.timings["total"], 1), "s", flush=True)
    tag = f"hopf_{n_steps}steps_{grid}"
    viz.plot_portraits(z, f"{out}/{tag}_portraits.png")
    viz.plot_cmg_strip(z, f"{out}/{tag}_cmg_zigzag.png")
    viz.plot_barcode(z, f"{out}/{tag}_barcode.png")
    with open(f"{out}/{tag}.json", "w") as fh:
        json.dump(z.to_dict(), fh, indent=1, default=str)
    for fr in z.frames:
        fr.release()
    zs.append(z)

viz.plot_comparison(zs, f"{out}/hopf_comparison.png")
print("done")
