# Building and profiling the SeisSol proxy (ChEESE MiniApp) on Apple Silicon

The SeisSol proxy — the single-node benchmark that carries SeisSol's ADER-DG
kernels, and the code ChEESE lists among its mini-apps — does not currently
build on Apple Silicon. This repo records what broke, the fixes, how they were
verified, and a first pass at profiling the kernels once it runs.

Three of the four patches here are build/correctness fixes found by doing that;
the fourth is an unrelated documentation fix for an already-diagnosed open issue
(#1556) that I picked up along the way.

## Environment

| | |
|---|---|
| Machine | Apple M4, 4 performance + 6 efficiency cores, 16 GB unified memory |
| OS | macOS 26.5 (Darwin 25.5.0) |
| Compiler | Apple clang 21.0.0; cross-checked with Homebrew clang 22.1.8 |
| CMake / Python | 4.4.0 / 3.14.6 |
| SeisSol | `master` @ `e9cd150c` (2026-08-20) |
| PSpaMM | `master` @ `553f62e5` |
| Build config | `HOST_ARCH=apple-m4`, `EQUATIONS=elastic`, `PRECISION=double`, `ORDER=2..6` |

NEON is 128-bit, so at double precision the generated kernels work on 2 lanes
(`Vector size has been set to 16 B` in the CMake output).

## The build fixes

### 1. PSpaMM emits a by-element FMLA form the LLVM assembler rejects

The NEON backend wrote the by-element multiplicand with the full arrangement
specifier:

```
fmla v11.2d, v1.2d, v3.2d[0]
```

The ARM ARM gives the instruction as `FMLA <Vd>.<T>, <Vn>.<T>, <Vm>.<Ts>[<index>]`
— the third operand carries the *element type*, so it must be `v3.d[0]`.

Minimal reproduction:

```bash
echo 'int main(void){ __asm__("fmla v11.2d, v1.2d, v3.2d[0]"); }' > t.c && clang -c t.c
# error: invalid operand for instruction
echo 'int main(void){ __asm__("fmla v11.2d, v1.2d, v3.d[0]"); }'  > u.c && clang -c u.c
# ok
```

GNU as accepts the redundant form, which is why this has never surfaced on
Linux/GCC. It affects both `.2d` (double) and `.4s` (single), and blocks every
`HOST_ARCH=apple-m1/m2/m3/m4` build under clang — the configuration SeisSol's
own build docs recommend for Macs.

Fix: `patches/0003-pspamm-neon-fmla-by-element.patch` (one file, ~15 lines).

**Verification** — PSpaMM's own numerical suite, `tests/unit_test.py arm128`,
which checks generated kernels against a reference:

| | GNU as (gcc 13.3, binutils 2.42, aarch64 Linux) | LLVM (clang) |
|---|---|---|
| upstream | assembles — 630/630 | **does not assemble** |
| patched | assembles — 630/630 | assembles — 630/630 |

So the change is a strict improvement: no regression under GNU as, and it
unblocks clang. (Under clang with FMA contraction left on, one single-precision
case differs by ~1.2e-6 relative. That is the *C reference* being contracted by
the compiler, not the generated kernel — `-ffp-contract=off` gives 630/630. The
gcc column confirms the kernels themselves are unchanged.)

### 2. `src/Proxy/Tools.h` is not self-contained

It declares

```cpp
auto sec(struct timeval start, struct timeval end) -> double;
```

but includes only `<ctime>`. `struct timeval` comes from `<sys/time.h>`; glibc
pulls it in transitively, macOS does not, so `Tools.cpp` fails with
`variable has incomplete type 'struct timeval'`. `Runner.cpp` already includes
`<sys/time.h>` explicitly, which is why only `Tools.cpp` breaks.

Fix: `patches/0001-seissol-proxy-timeval-include.patch`.

### 3. The proxy's JSON output reports the wrong hardware FLOP/cycle

`src/Proxy/Common.cpp` writes `output.nonZeroFlopPerCycle` for *both*
`gflopcycle-nz` and `gflopcycle-hw`, so the hardware figure is a duplicate of
the non-zero one. The plain-text writer immediately above it uses the two
distinct fields correctly.

Fix: `patches/0002-seissol-proxy-json-hw-flops.patch`.

Separately (not patched here): `-f json` also emits bare `inf` tokens for the
per-cycle fields whenever the cycle count is zero — which is always, since
`derive_cycles_from_time()` has its body commented out and returns 0. Bare `inf`
is not valid JSON, so `python -m json.tool` rejects the proxy's JSON output as
it stands. This is why `sweep.py` parses the plain-text format.

## Overlap with work already in flight

Checked before proposing anything (state as of 2026-08-21):

- **PSpaMM** has no open pull requests at all, and none of its three open issues
  touch code generation. The NEON fix is unclaimed.
- **SeisSol #1612** ("Output Kernel Bandwidth, refactor flop counter",
  davschneller, open, non-draft) edits `src/Proxy/Common.cpp` and adds fields
  immediately around the `gflopcycle-hw` line — but leaves the bug itself
  untouched; it appears in that PR's diff as unchanged context. So the fix is
  still needed, but a separate PR would conflict with active work. Better to
  raise it on #1612 and let it be folded into that refactor.
- **SeisSol #1505** ("Proxy cleanup", davschneller, **draft**, untouched since
  2026-02-03) renames these functions and adds a `getCycles()` with an
  `__rdtsc()` path — and a *commented-out* aarch64 `pmccntr_el0` path, which is
  why `cycles` is 0 and the JSON `inf` appears. It keeps
  `sec(struct timeval, …)` and still does not include `<sys/time.h>`, so the
  build fix is needed regardless of whether that draft lands.
- **SeisSol #963** ("Add support for MacOS & M1/M2", krenzland) was **merged**
  on 2023-09-28. So macOS/M1 was deliberately supported once and has regressed
  since; none of the three failures here are in the code that PR touched, they
  accumulated afterwards. There appears to be no macOS or ARM/clang job in CI to
  catch it.

## Reproducing the build

```bash
# deps
brew install cmake eigen yaml-cpp libomp open-mpi
brew unlink hdf5 && brew install hdf5-mpi   # see note below

python3 -m venv ~/.seissol-venv
~/.seissol-venv/bin/pip install numpy scipy setuptools

# PSpaMM, with the NEON by-element FMLA fix -- without this the generated
# kernels do not assemble under clang, so it has to be a source install
git clone https://github.com/SeisSol/PSpaMM.git
git -C PSpaMM apply ../patches/0003-pspamm-neon-fmla-by-element.patch
~/.seissol-venv/bin/pip install -e PSpaMM

# easi
git clone --recursive https://github.com/SeisSol/easi.git
cmake -S easi -B easi/build -DCMAKE_BUILD_TYPE=Release \
  -DASAGI=OFF -DIMPALAJIT=OFF -DLUA=OFF -DEASICUBE=OFF \
  -DCMAKE_INSTALL_PREFIX=~/.seissol-deps
cmake --build easi/build -j && cmake --install easi/build

# SeisSol proxy
git clone --recursive https://github.com/SeisSol/SeisSol.git
cd SeisSol
git apply ../patches/0001-seissol-proxy-timeval-include.patch
PATH=~/.seissol-venv/bin:$PATH cmake -B build \
  -DCMAKE_BUILD_TYPE=Release -DHOST_ARCH=apple-m4 \
  -DORDER=4 -DEQUATIONS=elastic -DPRECISION=double \
  -DGRAPH_PARTITIONING_LIBS=none -DNETCDF=OFF -DASAGI=OFF \
  -DNUMA_AWARE_PINNING=OFF \
  -DCMAKE_PREFIX_PATH="$HOME/.seissol-deps;/opt/homebrew/opt/hdf5-mpi;/opt/homebrew" \
  -DPython3_EXECUTABLE=~/.seissol-venv/bin/python \
  -DOpenMP_ROOT=/opt/homebrew/opt/libomp
PATH=~/.seissol-venv/bin:$PATH cmake --build build --target seissol-proxy-exe -j
```

Orders 2 through 6 all build and run with this recipe; only `ORDER` changes.

Two things worth knowing:

- **`NUMA_AWARE_PINNING` must be off.** It is on by default and pulls in
  `libnuma`, which does not exist on macOS.
- **HDF5 must be the MPI-parallel build.** `HDF5`, `MPI` and `OPENMP` are set
  unconditionally (`set(HDF5 ON)`) in `cmake/process_users_input.cmake` — their
  `option()` declarations are commented out — so they cannot be turned off from
  the command line, even for a single-node proxy build that does no mesh I/O.
  With Homebrew's serial `hdf5`, the PUML submodule fails on `H5Pset_dxpl_mpio`.

## Profiling

### What the proxy's numbers do and don't mean

The proxy reports `GFLOP (non-zero)`, `GFLOP (hardware)` and `GiB (estimate)`.
The byte count is **analytic** — `bytes = bytesAder() * nrOfCells` and friends
in `src/Proxy/KernelHost.cpp` — not measured traffic. So the arithmetic
intensity derived from it is a property of the algorithm and the generated code,
constant in the number of cells (which the sweep confirms), and it is a
*compulsory-traffic lower bound* rather than a measured roofline coordinate.

Getting real traffic needs hardware counters. SeisSol ships a LIKWID wrapper
(`src/Proxy/LikwidWrapper.h`), but LIKWID is Linux-only, so on macOS that path
is unavailable. A measured roofline needs a Linux host.

### Measured machine ceilings

Microbenchmarks in this repo (`peakflops.c`, `peak1.c`, `stream.c`):

| | |
|---|---|
| DP FMA, 1 thread | 27.4 GFLOP/s (best of sweep over 8–28 accumulators) |
| DP FMA, 10 threads | 68.2 GFLOP/s |
| STREAM triad, 10 threads | 51.3 GB/s |

These are *measured attainable* figures on a loaded laptop, not vendor peaks.
macOS gives no thread affinity control, so OpenMP threads migrate between
performance and efficiency cores; single-thread numbers varied by ~35% run to
run even with `QOS_CLASS_USER_INTERACTIVE`. Treat them as order-of-magnitude.

### Kernel sweep (order 4, elastic, double, 4 threads)

`sweep.py` runs each kernel over cell counts from 1e2 to 3e5; full data in
`sweep_results.json`. **I am not reporting those wall-clock numbers as a
result.** The run-to-run spread on this machine is comparable to the trend
across cell counts, so they cannot support a claim about cache behaviour.

### Padding overhead vs. convergence order

The quantity that *is* trustworthy here is the ratio of hardware to non-zero
FLOPs. It is deterministic (identical across repeated runs, verified), it comes
from the generated code rather than from timing, and it measures something real:
the arithmetic spent on structural zeros — the padding cost of evaluating sparse
operators as dense/block GEMMs.

`flops_by_order.py`, elastic, double precision, `HOST_ARCH=apple-m4`. Ratio of
hardware to non-zero FLOPs, and the same thing as a percentage of issued
arithmetic that lands on structural zeros:

| order | `ader` | `localwoader` | `local` | `neigh` | `all` |
|---|---|---|---|---|---|
| 2 | 2.067 (51.6%) | 1.560 (35.9%) | 1.601 (37.5%) | 1.582 (36.8%) | 1.592 (37.2%) |
| 3 | 1.478 (32.4%) | 1.254 (20.2%) | 1.288 (22.4%) | 1.464 (31.7%) | 1.365 (26.7%) |
| 4 | 1.341 (25.5%) | 1.217 (17.8%) | 1.244 (19.6%) | 1.478 (32.3%) | **1.338 (25.2%)** |
| 5 | 1.288 (22.4%) | 1.241 (19.4%) | 1.254 (20.3%) | 1.557 (35.8%) | 1.364 (26.7%) |
| 6 | 1.284 (22.1%) | 1.235 (19.0%) | 1.251 (20.1%) | 1.533 (34.8%) | 1.344 (25.6%) |

Two things stand out:

1. **The volume kernels improve steeply and then plateau.** `ader` goes 2.067 →
   1.478 → 1.341 → 1.288 → 1.284: at order 2 more than half the arithmetic
   issued is on zeros, which is expected — 4 basis functions padded out to the
   alignment is mostly padding — but from order 4 on it is flat at roughly 22%.
   `localwoader` and `local` plateau the same way, around 19–20%.

2. **`neigh` never joins them.** It sits at 1.46–1.58 (32–37% waste) at *every*
   order, with no downward trend — minimum at order 3, and slightly worse at
   orders 5 and 6 than at order 3. From order 3 onward it is the largest single
   source of wasted arithmetic in the timestep, and it is what keeps `all`
   pinned around 25–27% even once the volume kernels have improved.

So the padding cost of the volume kernels is essentially a solved problem above
order 3, and the neighbour-flux kernel is not. The full timestep happens to be
cheapest at order 4 (25.2%), but the spread across orders 3–6 is small (25–27%)
and is governed almost entirely by `neigh`.

I want to be careful about the mechanism: a naive "pad the basis count up to the
128 B alignment" model predicts 4.00x / 1.60x / 1.60x / 1.37x / 1.14x for orders
2–6, which matches the *shape* of the volume-kernel curve but not `neigh`'s
flatness. So `neigh`'s overhead is probably not simple alignment padding — more
likely the block structure of the flux matrices across the four face
orientations and neighbour types. I have not verified that, and it is the
obvious next thing to look at.

### Cross-checking on wider vectors — without wider-vector hardware

The ratio above is a property of the *generated code for a given target*, so the
obvious question is whether any of it is a NEON artefact. It turns out you do
not need an AVX-512 or SVE machine to answer that: **SeisSol's kernel generation
is pure Python, and it bakes the flop counts into the generated header**
(`NonZeroFlops` / `HardwareFlops` in `GeneratedCode/.../kernel.h`). So
`codegen/generate.py` can be run for any `--host_arch` on any host.

`padding_by_arch.py` and `padding_by_order_arch.py` do that. All of the
following was generated on the M4 — no x86 and no SVE hardware involved:

**Order 4, all targets** (waste = fraction of issued FLOPs landing on zeros):

| kernel | NEON (2 dbl) | AVX2 (4 dbl) | AVX-512 (8 dbl) | SVE512 (8 dbl) |
|---|---|---|---|---|
| `volume` | 1.288 (22%) | 1.678 (40%) | 2.217 (55%) | 2.217 (55%) |
| `derivative` | 1.341 (25%) | 1.715 (42%) | 2.799 (64%) | 2.799 (64%) |
| `localFlux` | 1.199 (17%) | 1.545 (35%) | 2.118 (53%) | 2.118 (53%) |
| `neighboringFlux` | 1.476 (32%) | 1.870 (47%) | 2.548 (61%) | 2.548 (61%) |
| `nodalFlux` | 1.025 (2%) | 1.025 (2%) | 1.229 (19%) | 1.229 (19%) |

**Order sweep, NEON vs AVX-512:**

| order | `derivative` NEON | `derivative` AVX-512 | `neighboringFlux` NEON | `neighboringFlux` AVX-512 |
|---|---|---|---|---|
| 2 | 2.067 (52%) | 7.667 (87%) | 1.581 (37%) | 3.326 (70%) |
| 3 | 1.478 (32%) | 3.761 (73%) | 1.462 (32%) | 2.383 (58%) |
| 4 | 1.341 (25%) | 2.799 (64%) | 1.476 (32%) | 2.548 (61%) |
| 5 | 1.288 (22%) | 2.342 (57%) | 1.559 (36%) | 2.098 (52%) |
| 6 | 1.284 (22%) | 2.145 (53%) | 1.536 (35%) | 2.045 (51%) |

Three things follow:

1. **Padding cost rises steeply with vector width.** At order 2 on AVX-512,
   `derivative` issues 7.7x the arithmetic it needs — 87% of the FLOPs are on
   zeros. Even at order 6 it is still above 50%.
2. **Raising the order helps far more on wide vectors than on narrow ones.**
   AVX-512 `derivative` goes 87% → 53% across orders 2–6; NEON only goes
   52% → 22% and is already flat by order 4.
3. **The `neigh` flatness *is* partly a NEON effect.** On NEON
   `neighboringFlux` never improves (37% → 35%); on AVX-512 it does improve
   with order (70% → 51%). So the earlier "neigh never gets better" observation
   should be stated as NEON-specific. What survives on every target is the
   weaker claim: `neighboringFlux` is consistently among the worst kernels, and
   on NEON it is the single worst from order 3 on.

**A high ratio is not the same as being slow.** An AVX-512 lane issues 4x the
doubles per instruction that a NEON lane does, so 53% waste on 8-wide still does
more useful work per instruction than 22% waste on 2-wide (≈3.8 vs ≈1.6 useful
doubles per vector op). What the ratio measures is headroom, not throughput.
None of this says SeisSol is slow on AVX-512; it says a sparsity-aware or
blocked-differently code generator would have a lot to work with there. That is
of course exactly the problem PSpaMM exists to attack, so this is a measurement
of a known phenomenon rather than a new discovery — the useful part is that it
is now a two-minute reproducible check for any target.

**A CI thought.** Because this needs no target hardware, a regression test that
generates kernels for a handful of `HOST_ARCH` values and asserts the
hardware/non-zero ratio has not worsened would be cheap to run on the existing
Ubuntu runners.
