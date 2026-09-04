// Dependent ADD chain: 1 add/cycle latency => cycles == iterations.
#include <stdio.h>
#include <pthread.h>
#include <sys/qos.h>
#include <mach/mach_time.h>
static double now(void){ static mach_timebase_info_data_t tb; if(tb.denom==0) mach_timebase_info(&tb);
  return (double)mach_absolute_time()*tb.numer/tb.denom/1e9; }
int main(void){
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
  volatile long sink=0; double best=0;
  for(int r=0;r<5;r++){
    long x=1; long n=500000000L;
    double t0=now();
    __asm__ __volatile__("1:\n\t add %0,%0,#1\n\t subs %1,%1,#1\n\t b.ne 1b\n\t"
                         : "+r"(x), "+r"(n) :: "cc");
    double t1=now();
    double f=(double)500000000L/(t1-t0)/1e9;
    if(f>best) best=f; sink=x;
  }
  printf("measured P-core clock = %.2f GHz  (sink %ld)\n", best, sink);
  return 0;
}
