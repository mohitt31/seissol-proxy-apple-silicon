#!/usr/bin/env python3
"""Hardware vs non-zero FLOP ratio per ADER-DG kernel, as a function of
convergence order.

This ratio is the arithmetic spent on structural zeros -- the padding cost of
evaluating sparse operators as dense/block GEMMs. It is deterministic: it comes
from the generated code, not from timing, so it is immune to the scheduling
noise that makes wall-clock numbers on a laptop unreliable.
"""
import re, subprocess, os, json, glob, sys

KERNELS = ["ader", "localwoader", "local", "neigh", "all"]
PAT_NZ = r"GFLOP \(non-zero\) for seissol proxy\s*:\s*([0-9.eE+-]+)"
PAT_HW = r"GFLOP \(hardware\) for seissol proxy\s*:\s*([0-9.eE+-]+)"

def ratio(binary, kernel, cells=2000, steps=2):
    env = dict(os.environ, OMP_NUM_THREADS="4",
               DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib")
    out = subprocess.run([binary, str(cells), str(steps), kernel],
                         capture_output=True, text=True, env=env).stdout
    mnz, mhw = re.search(PAT_NZ, out), re.search(PAT_HW, out)
    if not (mnz and mhw):
        return None
    nz, hw = float(mnz.group(1)), float(mhw.group(1))
    return (hw / nz) if nz else None

bins = {}
for p in sorted(glob.glob("/Users/mohit/seissol/build-o*/proxyseissol-elastic-o*-f64")):
    o = int(re.search(r"-o(\d+)-f64$", p).group(1))
    bins[o] = p

if not bins:
    sys.exit("no proxy binaries found")

orders = sorted(bins)
print("hardware/non-zero FLOP ratio  (higher = more arithmetic on structural zeros)\n")
hdr = "order | " + " | ".join(f"{k:>11s}" for k in KERNELS)
print(hdr); print("-" * len(hdr))
table = {}
for o in orders:
    row = {}
    cells = 2000 if o <= 4 else 800
    for k in KERNELS:
        row[k] = ratio(bins[o], k, cells=cells)
    table[o] = row
    print(f"{o:5d} | " + " | ".join(
        (f"{row[k]:11.3f}" if row[k] else f"{'--':>11s}") for k in KERNELS))

json.dump(table, open("flops_by_order.json", "w"), indent=1)
print("\nwrote flops_by_order.json")
