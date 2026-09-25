/* Prototype only, fixed tested shape. Same W8A16 data/layout and f32 scaling as
 * metering_canister.rs. 128-bit SIMD, no AVX or multi-threaded work per kernel.
 * One pthread represents each independent canister's work, with private data.
 */
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define M 120
#define N 2304
#define K 768

typedef struct {
    int16_t *a; int8_t *w; float *out;
    unsigned banks, cursor, iterations;
    uint64_t checksum;
    pthread_barrier_t *barrier;
} Worker;

static float scale(int32_t v) { return ((float)v * 0.00013f) * 0.0017f; }
static int32_t reduce(__m128i x) {
    x = _mm_hadd_epi32(x, x); x = _mm_hadd_epi32(x, x);
    return _mm_cvtsi128_si32(x);
}
static void kernel(Worker *s) {
    const int8_t *w = s->w + (size_t)s->cursor * N * K;
    for (int i=0; i<M; i+=4) {
        for (int j=0; j<N; j+=2) {
            __m128i c00=_mm_setzero_si128(), c01=c00, c10=c00, c11=c00;
            __m128i c20=c00, c21=c00, c30=c00, c31=c00;
            for (int k=0; k<K; k+=8) {
                __m128i w0=_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(w+(size_t)j*K+k)));
                __m128i w1=_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(w+(size_t)(j+1)*K+k)));
                __m128i a0=_mm_loadu_si128((const __m128i*)(s->a+(size_t)i*K+k));
                __m128i a1=_mm_loadu_si128((const __m128i*)(s->a+(size_t)(i+1)*K+k));
                __m128i a2=_mm_loadu_si128((const __m128i*)(s->a+(size_t)(i+2)*K+k));
                __m128i a3=_mm_loadu_si128((const __m128i*)(s->a+(size_t)(i+3)*K+k));
                c00=_mm_add_epi32(c00,_mm_madd_epi16(a0,w0)); c01=_mm_add_epi32(c01,_mm_madd_epi16(a0,w1));
                c10=_mm_add_epi32(c10,_mm_madd_epi16(a1,w0)); c11=_mm_add_epi32(c11,_mm_madd_epi16(a1,w1));
                c20=_mm_add_epi32(c20,_mm_madd_epi16(a2,w0)); c21=_mm_add_epi32(c21,_mm_madd_epi16(a2,w1));
                c30=_mm_add_epi32(c30,_mm_madd_epi16(a3,w0)); c31=_mm_add_epi32(c31,_mm_madd_epi16(a3,w1));
            }
            s->out[(size_t)i*N+j]=scale(reduce(c00)); s->out[(size_t)i*N+j+1]=scale(reduce(c01));
            s->out[(size_t)(i+1)*N+j]=scale(reduce(c10)); s->out[(size_t)(i+1)*N+j+1]=scale(reduce(c11));
            s->out[(size_t)(i+2)*N+j]=scale(reduce(c20)); s->out[(size_t)(i+2)*N+j+1]=scale(reduce(c21));
            s->out[(size_t)(i+3)*N+j]=scale(reduce(c30)); s->out[(size_t)(i+3)*N+j+1]=scale(reduce(c31));
        }
    }
    s->cursor=(s->cursor+1)%s->banks;
    /* Force observation of each output before the next kernel. */
    __asm__ __volatile__("" : : "r"(s->out) : "memory");
}
static uint64_t checksum(Worker *s) {
    uint64_t sum=0;
    for (size_t i=0; i<(size_t)M*N; i++) {uint32_t bits; memcpy(&bits,&s->out[i],4);sum+=bits;}
    return sum;
}
static void prepare(Worker *s) {
    s->a=malloc((size_t)M*K*sizeof(int16_t)); s->w=malloc((size_t)s->banks*N*K); s->out=malloc((size_t)M*N*sizeof(float));
    if (!s->a || !s->w || !s->out) abort();
    for (size_t i=0;i<(size_t)M*K;i++) s->a[i]=(int16_t)(i%16385)-8192;
    for (unsigned b=0;b<s->banks;b++) for (size_t i=0;i<(size_t)N*K;i++) s->w[(size_t)b*N*K+i]=(int8_t)((int)((i+b*31)%255)-127);
    for(unsigned b=0;b<s->banks;b++) {
        kernel(s);
        /* Integer scalar reference, covering all rows and selected columns in
         * every bank, including both ends. No float reassociation permitted. */
        for(int i=0;i<M;i++) for(int c=0;c<4;c++) {
            int j=(int[]){0,15,16,N-1}[c]; int64_t sum=0;
            for(int k=0;k<K;k++) sum+=(int64_t)s->a[(size_t)i*K+k]*s->w[(size_t)b*N*K+(size_t)j*K+k];
            float expected=scale((int32_t)sum);
            if(memcmp(&expected,&s->out[(size_t)i*N+j],4)) abort();
        }
        if(b==0 && checksum(s)!=UINT64_C(641785196358105)) abort(); /* PocketIC full-output checksum */
    }
}
static void *run(void *arg) {
    Worker *s=arg; pthread_barrier_wait(s->barrier);
    for(unsigned i=0;i<s->iterations;i++) kernel(s);
    s->checksum=checksum(s); return NULL;
}
static double now(void) { struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc,char **argv) {
    if(argc!=5) return 2;
    unsigned banks=atoi(argv[1]), workers=atoi(argv[2]), iterations=atoi(argv[3]),groups=atoi(argv[4]);
    if((banks!=1 && banks!=32) || (workers!=1 && workers!=4) || !iterations || !groups) return 2;
    Worker states[4]={0};pthread_t threads[4];pthread_barrier_t barrier;
    pthread_barrier_init(&barrier,NULL,workers+1);
    for(unsigned i=0;i<workers;i++) { states[i].banks=banks;states[i].iterations=iterations;states[i].barrier=&barrier;prepare(&states[i]); }
    for(unsigned g=0;g<groups;g++) {
        for(unsigned i=0;i<workers;i++) pthread_create(&threads[i],NULL,run,&states[i]);
        double start=now();pthread_barrier_wait(&barrier);
        for(unsigned i=0;i<workers;i++) pthread_join(threads[i],NULL);
        double elapsed=now()-start;
        printf("{\"group\":%u,\"workers\":%u,\"banks\":%u,\"kernels_per_worker\":%u,\"wall_s\":%.9f,\"gmac_per_worker_s\":%.9f,\"checksum\":%llu}\n",
               g,workers,banks,iterations,elapsed,(double)iterations*M*N*K/elapsed/1e9,(unsigned long long)states[0].checksum);
        fflush(stdout);
    }
    for(unsigned i=0;i<workers;i++) { free(states[i].a);free(states[i].w);free(states[i].out); }
    pthread_barrier_destroy(&barrier);
    return 0;
}
