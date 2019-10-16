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

extern void int8kernel_m4(int32_t* dst, const int8_t* src, const int8_t* weight, size_t k, size_t n, size_t ldc);
extern void int8kernel_m2(int32_t* dst, const int8_t* src, const int8_t* weight, size_t k, size_t n, size_t ldc);
extern void int8kernel_m1(int32_t* dst, const int8_t* src, const int8_t* weight, size_t k, size_t n, size_t ldc);
extern void reorder_a(int8_t * src, int8_t * dst, size_t m, size_t k, size_t ldx);
extern void reorder_b(int8_t * src, int8_t * dst, size_t k, size_t n);

static inline void trans(int8_t * matrixB, int8_t * matrixB_trans, int k , int n){
	for(int i = 0; i < k; i++){
		for(int j = 0; j < n; j++){
			matrixB_trans[j * k + i] = matrixB[i * n + j];;
		}
	}
}

static inline void reorder_b_v2(int8_t* b, int8_t* sb, const int k, const int n, const int ldx) {

    int i = 0;
    for (; i+3 < n; i += 4) {
        int8_t *p0 = b + i;
        int8_t *p1 = b + 1 * ldx + i;
        int8_t *p2 = b + 2 * ldx + i;
        int8_t *p3 = b + 3 * ldx + i;

        int8_t *p4 = b + 4 * ldx + i;
        int8_t *p5 = b + 5 * ldx + i;
        int8_t *p6 = b + 6 * ldx + i;
        int8_t *p7 = b + 7 * ldx + i;

        int j = 0;
        for (; j+7 < k; j += 8) {
            sb[0]  = p0[0];
            sb[1]  = p1[0];
            sb[2]  = p2[0];
            sb[3]  = p3[0];
            sb[4]  = p4[0];
            sb[5]  = p5[0];
            sb[6]  = p6[0];
            sb[7]  = p7[0];

            sb[8]  = p0[1];
            sb[9]  = p1[1];
            sb[10] = p2[1];
            sb[11] = p3[1];
            sb[12] = p4[1];
            sb[13] = p5[1];
            sb[14] = p6[1];
            sb[15] = p7[1];

            sb[16] = p0[2];
            sb[17] = p1[2];
            sb[18] = p2[2];
            sb[19] = p3[2];
            sb[20] = p4[2];
            sb[21] = p5[2];
            sb[22] = p6[2];
            sb[23] = p7[2];

            sb[24] = p0[3];
            sb[25] = p1[3];
            sb[26] = p2[3];
            sb[27] = p3[3];
            sb[28] = p4[3];
            sb[29] = p5[3];
            sb[30] = p6[3];
            sb[31] = p7[3];

            sb += 32;
            p0 += 8 * ldx;
            p1 += 8 * ldx;
            p2 += 8 * ldx;
            p3 += 8 * ldx;
            p4 += 8 * ldx;
            p5 += 8 * ldx;
            p6 += 8 * ldx;
            p7 += 8 * ldx;
        }
        for (; j+3 < k; j += 4) {
            sb[0]  = p0[0];
            sb[1]  = p1[0];
            sb[2]  = p2[0];
            sb[3]  = p3[0];

            sb[4]  = p0[1];
            sb[5]  = p1[1];
            sb[6]  = p2[1];
            sb[7]  = p3[1];

            sb[8]  = p0[2];
            sb[9]  = p1[2];
            sb[10] = p2[2];
            sb[11] = p3[2];

            sb[12] = p0[3];
            sb[13] = p1[3];
            sb[14] = p2[3];
            sb[15] = p3[3];

            sb += 16;
            p0 += 4 * ldx;
            p1 += 4 * ldx;
            p2 += 4 * ldx;
            p3 += 4 * ldx;
        }
        for (; j+1 < k; j += 2) {
            sb[0] = p0[0];
            sb[1] = p1[0];
            sb[2] = p0[1];
            sb[3] = p1[1];
            sb[4] = p0[2];
            sb[5] = p1[2];
            sb[6] = p0[3];
            sb[7] = p1[3];

            sb += 8;
            p0 += 2 * ldx;
            p1 += 2 * ldx;
        }
        for (; j < k; ++j) {
            sb[0] = p0[0];
            sb[1] = p0[1];
            sb[2] = p0[2];
            sb[3] = p0[3];

            sb += 4;
            p0 += ldx;
        }
    }
    for (; i+1 < n; i += 2) {
        int8_t *p0 = b + i;
        int8_t *p1 = b + 1 * ldx + i;
        int8_t *p2 = b + 2 * ldx + i;
        int8_t *p3 = b + 3 * ldx + i;

        int8_t *p4 = b + 4 * ldx + i;
        int8_t *p5 = b + 5 * ldx + i;
        int8_t *p6 = b + 6 * ldx + i;
        int8_t *p7 = b + 7 * ldx + i;

        int j = 0;
        for (; j+7 < k; j += 8) {
            sb[0]  = p0[0];
            sb[1]  = p1[0];
            sb[2]  = p2[0];
            sb[3]  = p3[0];
            sb[4]  = p4[0];
            sb[5]  = p5[0];
            sb[6]  = p6[0];
            sb[7]  = p7[0];

            sb[8]  = p0[1];
            sb[9]  = p1[1];
            sb[10] = p2[1];
            sb[11] = p3[1];
            sb[12] = p4[1];
            sb[13] = p5[1];
            sb[14] = p6[1];
            sb[15] = p7[1];

            sb += 16;
            p0 += 8 * ldx;
            p1 += 8 * ldx;
            p2 += 8 * ldx;
            p3 += 8 * ldx;
            p4 += 8 * ldx;
            p5 += 8 * ldx;
            p6 += 8 * ldx;
            p7 += 8 * ldx;
        }
        for (; j+3 < k; j += 4) {
            sb[0]  = p0[0];
            sb[1]  = p1[0];
            sb[2]  = p2[0];
            sb[3]  = p3[0];

            sb[4]  = p0[1];
            sb[5]  = p1[1];
            sb[6]  = p2[1];
            sb[7]  = p3[1];

            sb += 8;
            p0 += 4 * ldx;
            p1 += 4 * ldx;
            p2 += 4 * ldx;
            p3 += 4 * ldx;
        }
        for (; j+1 < k; j += 2) {
            sb[0] = p0[0];
            sb[1] = p1[0];
            sb[2] = p0[1];
            sb[3] = p1[1];

            sb += 4;
            p0 += 2 * ldx;
            p1 += 2 * ldx;
        }
        for (; j < k; ++j) {
            sb[0] = p0[0];
            sb[1] = p0[1];

            sb += 2;
            p0 += ldx;
        }
    }
    for (; i < n; ++i) {
        int8_t *p0 = b + i;
        int8_t *p1 = b + 1 * ldx + i;
        int8_t *p2 = b + 2 * ldx + i;
        int8_t *p3 = b + 3 * ldx + i;
        int8_t *p4 = b + 4 * ldx + i;
        int8_t *p5 = b + 5 * ldx + i;
        int8_t *p6 = b + 6 * ldx + i;
        int8_t *p7 = b + 7 * ldx + i;

        int j = 0;
        for (; j+7 < k; j += 8) {
            sb[0]  = p0[0];
            sb[1]  = p1[0];
            sb[2]  = p2[0];
            sb[3]  = p3[0];
            sb[4]  = p4[0];
            sb[5]  = p5[0];
            sb[6]  = p6[0];
            sb[7]  = p7[0];

            sb += 8;
            p0 += 8 * ldx;
            p1 += 8 * ldx;
            p2 += 8 * ldx;
            p3 += 8 * ldx;
            p4 += 8 * ldx;
            p5 += 8 * ldx;
            p6 += 8 * ldx;
            p7 += 8 * ldx;
        }
        for (; j+3 < k; j += 4) {
            sb[0]  = p0[0];
            sb[1]  = p1[0];
            sb[2]  = p2[0];
            sb[3]  = p3[0];

            sb += 4;
            p0 += 4 * ldx;
            p1 += 4 * ldx;
            p2 += 4 * ldx;
            p3 += 4 * ldx;
        }
        for (; j+1 < k; j += 2) {
            sb[0] = p0[0];
            sb[1] = p1[0];

            sb += 2;
            p0 += 2 * ldx;
            p1 += 2 * ldx;
        }
        for (; j < k; ++j) {
            sb[0] = p0[0];

            sb += 1;
            p0 += ldx;
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
    (void)ret;
#endif
	trans(matrixB, (int8_t*)ptr, k, n);
	reorder_a((int8_t*)ptr, matrixB_reorder, n, k, k);
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
void MY_MMult(int m, int n, int k, int8_t * a, int lda,
                                   int8_t * b, int ldb,
                                   int32_t * c, int ldc,
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

    int8_t* sa = fastMalloc(m * k);
    int8_t* sb = fastMalloc(k * n);
    // packA
    reorder_a(a, sa, m, k, k);
    // packB
    reorder_b_v2(b, sb, k, n, n);
    // subkernel 
    int8_t *pA= sa, *pB = sb;
    int32_t *pC = c;
    int i = 0;
    while (i+4 <= m) {
	    int8kernel_m4(pC, pA, pB, k, n, n);
        pC += 4 * n;
        pA += 4 * k;
        i += 4;
    }
    switch(m-i) {
        case 3:
	        int8kernel_m2(pC, pA, pB, k, n, n);
            pC += 2 * n;
            pA += 2 * k;
	        int8kernel_m1(pC, pA, pB, k, n, n);
            pC += n;
            pA += k;
            break;
        case 2:
	        int8kernel_m2(pC, pA, pB, k, n, n);
            pC += 2 * n;
            pA += 2 * k;
            break;
        case 1:
	        int8kernel_m1(pC, pA, pB, k, n, n);
            pC += n;
            pA += k;
            break;
        case 0:
        default:
            break;
    }
    

    // print_int32_matrix(m, n, c, ldc);

    free(sa);
    free(sb);
}
