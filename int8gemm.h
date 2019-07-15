#ifndef _INT8_GEMM_H_
#define _INT8_GEMM_H_ 

#include <stdlib.h>
#include <string.h>

void trans_w(int8_t *matrixB, int8_t *matrixB_reorder, int k , int n);
void int8gemm(int8_t *matrixA_reorder , int8_t *matrixB_reorder, int32_t *matrixC, int m, int n, int k);

#endif /* MATRIXMUL_REORDERED_H_ */
