"""Scaling study: each configuration runs in its own process (clean peak memory).

usage: python run_scaling.py [outdir]
"""
import json
import os
import resource
import subprocess
import sys
import time

CONFIGS = (
    [("Hopf 2D · 4 steps (250 ms)", "hopf", 4, n) for n in (64, 128, 192, 256)] +
    [("Hopf 2D · 10 steps (100 ms)", "hopf", 10, n) for n in (64, 128, 192, 256)] +
    [("Lorenz 3D · 4 steps (250 ms)", "lorenz", 4, n) for n in (16, 24, 32, 40)]
)


def one(system, n_steps, grid):
    from cmzz.systems import Hopf2D, Lorenz3D
    from cmzz.pipeline import CMGZigzag
    sysm = Hopf2D() if system == "hopf" else Lorenz3D()
    t0 = time.perf_counter()
    z = CMGZigzag(sysm, n_steps, grid, verbose=False).run()
    total = time.perf_counter() - t0
    det = z.detection()
    return {"cells": int(z.grid.n_cells), "grid": grid, "n_steps": n_steps,
            "timings": z.timings, "total": total,
            "peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "max_zigzag_simplices": z.stats["max_zigzag_simplices"],
            "mean_image_span": z.stats["mean_image_span"],
            "escaped_cells": z.stats["escaped_cells"],
            "detection": None if det is None else {"t": float(det["t_detect"]), "kind": det["kind"]},
            "structure": [[b.poincare for b in bl if not b.trivial] for bl in z.blocks[::2]],
            "n_bars": len(z.bars)}


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        _, _, system, n_steps, grid = sys.argv
        print(json.dumps(one(system, int(n_steps), int(grid))))
        sys.exit(0)
    out = sys.argv[1] if len(sys.argv) > 1 else "out"
    os.makedirs(out, exist_ok=True)
    rows = []
    for group, system, n_steps, grid in CONFIGS:
        print(f"{group}  grid {grid}", flush=True)
        r = subprocess.run([sys.executable, __file__, "--one", system, str(n_steps), str(grid)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("  FAILED", r.stderr[-800:], flush=True)
            continue
        row = json.loads(r.stdout.strip().splitlines()[-1])
        row["group"] = group
        rows.append(row)
        print(f"  {row['total']:.1f} s  peak {row['peak_mb']:.0f} MB  detection {row['detection']}"
              f"  structure(last step) {row['structure'][-1]}", flush=True)
        with open(f"{out}/scaling.json", "w") as fh:
            json.dump(rows, fh, indent=1)
    from cmzz import viz
    viz.plot_scaling(rows, f"{out}/scaling.png")
    print("done")
