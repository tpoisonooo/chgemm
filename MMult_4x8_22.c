#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error("only support arm neon")
#endif

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#include "reorder_b.h"

double dclock();
void print_int8_matrix( int m, int n, int8_t *a, int lda);
void print_int32_matrix( int m, int n, int32_t *a, int lda);
/* Routine for computing C = A * B + C */

extern void int8kernel_m4(int32_t* dst, const int8_t* src, const int8_t* weight, size_t k, size_t n, size_t ldc);
extern void int8kernel_m2(int32_t* dst, const int8_t* src, const int8_t* weight, size_t k, size_t n, size_t ldc);
extern void int8kernel_m1(int32_t* dst, const int8_t* src, const int8_t* weight, size_t k, size_t n, size_t ldc);
extern void reorder_a(int8_t * src, int8_t * dst, size_t m, size_t k, size_t ldx);

static inline int8_t* fastMalloc(int size){
    void* ptr = 0;
    int iRet = posix_memalign(&ptr, 64, size * sizeof(int8_t));
    assert(0 == iRet);
    return ptr;
}

/* Suppose that m%4==0 and n%4==0 and k%4==0, avoiding process boundary !! */
void MY_MMult(int m, int n, int k, int8_t * a, int lda,
                                   int8_t * b, int ldb,
                                   int32_t * c, int ldc,
                                   double *packZ_cost,
                                   double *packN_cost,
                                   double *kernel_cost)
{
    *packN_cost = *packZ_cost = *kernel_cost = 0.0;

    int8_t* sa = fastMalloc(m * k);
    int8_t* sb = fastMalloc(k * n);
    // packA
    reorder_a(a, sa, m, k, k);
    // packB
    reorder_b(b, sb, k, n, n);
    // subkernel 
    int8_t *pA= sa, *pB = sb;
    int32_t *pC = c;

    const int nn = (m >> 2) << 2;
    #pragma omp parallel for num_threads(3)
    for (int i = 0; i < nn; i += 4)
    {
        int8kernel_m4(pC + i * n, pA + i * k, pB, k, n, n);
    }
    pA += nn * k;
    pC += nn * n;
    switch(m-nn) {
        case 3:
	        int8kernel_m2(pC, pA, pB, k, n, n);
            pC += 2 * n;
            pA += 2 * k;
	        int8kernel_m1(pC, pA, pB, k, n, n);
            break;
        case 2:
	        int8kernel_m2(pC, pA, pB, k, n, n);
            break;
        case 1:
	        int8kernel_m1(pC, pA, pB, k, n, n);
            break;
        case 0:
        default:
            break;
    }

    free(sa);
    free(sb);
}
