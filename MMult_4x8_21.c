#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error("only support arm neon")
#endif

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Block sizes */
#define DEBUG_PACK_SHAPE
#undef DEBUG_PACK_SHAPE
#define DEBUG_PRINT_A
#define DEBUG_PRINT_B
#define DEBUG_PRINT_C
#undef DEBUG_PRINT_B
#undef DEBUG_PRINT_A
#undef DEBUG_PRINT_C
#undef DEBUG_PRINT_DATA

/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define min(i, j) ((i) < (j) ? (i): (j))
/**
Target: 24gflops on RK3399
*/

double dclock();
void print_int8_matrix( int m, int n, int8_t *a, int lda);
void print_int32_matrix( int m, int n, int32_t *a, int lda);
/* Routine for computing C = A * B + C */

extern void int8kernel_m4(int32_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_depth_quad);
extern void reorder(int8_t * src, int8_t * dst, size_t m, size_t k);

static inline void trans(int8_t * matrixB, int8_t * matrixB_trans, int k , int n){
	for(int i = 0; i < k; i++){
		for(int j = 0; j < n; j++){
			matrixB_trans[j * k + i] = matrixB[i * n + j];;
		}
	}
}

static inline void trans_w(int8_t * matrixB, int8_t *matrixB_reorder, int k , int n){
    void *ptr = NULL;
    const int kAlignBytes = 64;
#if _MSC_VER
    ptr = (int8_t*)_aligned_malloc(n * k * sizeof(int8_t), kAlignBytes);
#elif __ANDROID__
    ptr = (int8_t*)memalign(kAlignBytes, n * k * sizeof(int8_t));
#else
	int ret = posix_memalign(&ptr, kAlignBytes, n * k * sizeof(int8_t));
#endif
	trans(matrixB, (int8_t*)ptr, k, n);
	reorder((int8_t*)ptr, matrixB_reorder, n, k);
#if _MSC_VER
	_aligned_free(ptr);
#else
	free(ptr);
#endif
}

static inline int8_t* fastMalloc(int size){
    void* ptr = 0;
    int iRet = posix_memalign(&ptr, 64, size * sizeof(int8_t));
    assert(0 == iRet);
    return ptr;
}

/* Suppose that m%4==0 and n%4==0 and k%4==0, avoiding process boundary !! */
void MY_MMult(int m, int n, int k, int8_t * restrict a, int lda,
                                   int8_t * restrict b, int ldb,
                                   int32_t * restrict c, int ldc,
                                   double *packZ_cost,
                                   double *packN_cost,
                                   double *kernel_cost)
{
    *packN_cost = *packZ_cost = *kernel_cost = 0.0;
#if (defined DEBUG_PRINT_A) || (defined DEBUG_PRINT_B || defined DEBUG_PRINT_C)
    printf("\n--- a ----\n");
    print_int8_matrix(m, k, a, lda);
    printf("\n--- b ----\n");
    print_int8_matrix(k, n, b, ldb);
    printf("\n-------\n");
#endif

    int8_t* restrict sa = fastMalloc(m * k);
    int8_t* restrict sb = fastMalloc(k * n);
    double temp;
    // packA
    reorder(a, sa, m, k);
    // packB
    trans_w(b, sb, k, n);
    // subkernel 
    for(int i = 0; i < m / 4; i++){
    	int8kernel_m4(c + i * 4 * n, sa + i * 4 * k, sb, k, n);
    }
    // print_int32_matrix(m, n, c, ldc);

    free(sa);
    free(sb);
}
