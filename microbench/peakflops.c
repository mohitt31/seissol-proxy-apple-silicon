// Empirical DP FMA peak for Apple Silicon NEON.
// 32 independent accumulators to saturate FMA pipes and hide latency.
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#ifndef NACC
#define NACC 32
#endif
#define ITERS 20000000L

static double bench(void) {
  double t0 = omp_get_wtime();
  double sink = 0.0;
  #pragma omp parallel reduction(+:sink)
  {
    float64x2_t a[NACC], b, c;
    for (int i = 0; i < NACC; i++) a[i] = vdupq_n_f64(1.0000001 + i*1e-9);
    b = vdupq_n_f64(1.0000002);
    c = vdupq_n_f64(0.9999998);
    for (long it = 0; it < ITERS; it++) {
      #pragma unroll
      for (int i = 0; i < NACC; i++) a[i] = vfmaq_f64(a[i], b, c);
    }
    double s = 0.0;
    for (int i = 0; i < NACC; i++) s += vgetq_lane_f64(a[i],0) + vgetq_lane_f64(a[i],1);
    sink += s;
  }
  double t1 = omp_get_wtime();
  if (sink == 12345.6789) printf("");  // keep alive
  int nt = omp_get_max_threads();
  // 2 doubles per vector * 2 flop per FMA = 4 flop per vfmaq_f64
  double flops = (double)ITERS * NACC * 4.0 * nt;
  return flops / (t1 - t0) / 1e9;
}

int main(void) {
  bench(); // warmup
  double best = 0;
  for (int r = 0; r < 3; r++) { double g = bench(); if (g > best) best = g; }
  printf("threads=%d  peak_DP_GFLOPS=%.2f\n", omp_get_max_threads(), best);
  return 0;
}
