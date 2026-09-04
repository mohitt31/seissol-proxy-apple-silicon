#!/usr/bin/env python3
"""Working-set sweep of the SeisSol proxy: performance and arithmetic intensity
vs. number of cells, for the ADER-DG kernels.

Parses the proxy's plain-text output (the -f json path emits bare `inf`
tokens, which is not valid JSON).
"""
import re, subprocess, sys, json, os

BIN = sys.argv[1] if len(sys.argv) > 1 else "/Users/mohit/seissol/build-o4/proxyseissol-elastic-o4-f64"
THREADS = sys.argv[2] if len(sys.argv) > 2 else "4"

FIELDS = {
    "time":     r"time for seissol proxy\s*:\s*([0-9.eE+-]+)",
    "gflop_nz": r"GFLOP \(non-zero\) for seissol proxy\s*:\s*([0-9.eE+-]+)",
    "gflop_hw": r"GFLOP \(hardware\) for seissol proxy\s*:\s*([0-9.eE+-]+)",
    "gib":      r"GiB \(estimate\) for seissol proxy\s*:\s*([0-9.eE+-]+)",
    "gflops_nz":r"GFLOPS \(non-zero\) for seissol proxy\s*:\s*([0-9.eE+-]+)",
    "gflops_hw":r"GFLOPS \(hardware\) for seissol proxy\s*:\s*([0-9.eE+-]+)",
    "gibs":     r"GiB/s \(estimate\) for seissol proxy\s*:\s*([0-9.eE+-]+)",
}

def run(cells, steps, kernel, reps=3):
    env = dict(os.environ, OMP_NUM_THREADS=THREADS,
               DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib")
    best = None
    for _ in range(reps):
        out = subprocess.run([BIN, str(cells), str(steps), kernel],
                             capture_output=True, text=True, env=env).stdout
        rec = {}
        for k, pat in FIELDS.items():
            m = re.search(pat, out)
            if not m:
                return None
            rec[k] = float(m.group(1))
        # keep the fastest repetition
        if best is None or rec["time"] < best["time"]:
            best = rec
    return best

KERNELS = ["ader", "localwoader", "local", "neigh", "all"]
CELLS   = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]

def steps_for(cells):
    # keep each run in a sane wall-clock range
    return max(2, min(200, int(2_000_000 / cells)))

rows = []
for kern in KERNELS:
    for cells in CELLS:
        steps = steps_for(cells)
        r = run(cells, steps, kern)
        if r is None:
            print(f"  !! failed: {kern} {cells}", file=sys.stderr); continue
        ai_hw = r["gflop_hw"] / r["gib"]       # GFLOP / GiB
        ai_nz = r["gflop_nz"] / r["gib"]
        waste = r["gflop_hw"] / r["gflop_nz"] if r["gflop_nz"] else float("nan")
        row = dict(kernel=kern, cells=cells, steps=steps,
                   gflops_hw=r["gflops_hw"], gflops_nz=r["gflops_nz"],
                   gibs=r["gibs"], ai_hw=ai_hw, ai_nz=ai_nz,
                   hw_over_nz=waste, time=r["time"])
        rows.append(row)
        print(f"{kern:12s} cells={cells:7d} steps={steps:4d} "
              f"GFLOPS_hw={r['gflops_hw']:7.2f} GiB/s={r['gibs']:6.2f} "
              f"AI_hw={ai_hw:6.3f} hw/nz={waste:5.3f}", flush=True)

json.dump(rows, open("sweep_results.json", "w"), indent=1)
print(f"\nwrote sweep_results.json ({len(rows)} rows)")
