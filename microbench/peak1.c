// Single-thread DP FMA peak, pinned to a performance core via QoS.
#include <arm_neon.h>
#include <stdio.h>
#include <pthread.h>
#include <sys/qos.h>
#include <mach/mach_time.h>

#ifndef NACC
#define NACC 20
#endif
#define ITERS 20000000L

static double now(void){
  static mach_timebase_info_data_t tb; if(tb.denom==0) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9;
}

int main(void){
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  float64x2_t a[NACC], b=vdupq_n_f64(1.0000002), c=vdupq_n_f64(0.9999998);
  for(int i=0;i<NACC;i++) a[i]=vdupq_n_f64(1.0+i*1e-9);
  double best=0;
  for(int r=0;r<5;r++){
    double t0=now();
    for(long it=0; it<ITERS; it++){
      #pragma unroll
      for(int i=0;i<NACC;i++) a[i]=vfmaq_f64(a[i],b,c);
    }
    double t1=now();
    double g = (double)ITERS*NACC*4.0/(t1-t0)/1e9;
    if(g>best) best=g;
  }
  double s=0; for(int i=0;i<NACC;i++) s+=vgetq_lane_f64(a[i],0);
  printf("NACC=%d  1-core peak_DP_GFLOPS=%.2f  (sink %.3f)\n", NACC, best, s);
  return 0;
}
