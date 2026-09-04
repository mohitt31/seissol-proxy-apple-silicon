// STREAM triad, DP, OpenMP. Reports best of 10.
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#ifndef N
#define N 40000000L   // 3 arrays * 8B * 40M = 960 MB
#endif
static double a[N], b[N], c[N];
int main(void){
  #pragma omp parallel for
  for(long i=0;i<N;i++){ a[i]=1.0; b[i]=2.0; c[i]=0.0; }
  double best=0; const double s=3.0;
  for(int r=0;r<10;r++){
    double t0=omp_get_wtime();
    #pragma omp parallel for
    for(long i=0;i<N;i++) c[i]=a[i]+s*b[i];
    double t1=omp_get_wtime();
    double gbs = 3.0*sizeof(double)*N/(t1-t0)/1e9;
    if(gbs>best) best=gbs;
  }
  printf("threads=%d  STREAM_triad_GB/s=%.2f  (c[0]=%.1f)\n", omp_get_max_threads(), best, c[0]);
  return 0;
}
