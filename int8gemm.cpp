#include "int8gemm.h"

extern "C"{
void int8kernel(int32_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_depth_quad);
void reorder(int8_t * src, int8_t * dst, size_t m, size_t k);
}

void trans(int8_t * matrixB, int8_t * matrixB_trans, int k , int n){
	for(int i = 0; i < k; i++){
		for(int j = 0; j < n; j++){
			matrixB_trans[j * k + i] = matrixB[i * n + j];;
		}
	}
}

void trans_w(int8_t * matrixB, int8_t *matrixB_reorder, int k , int n){
    void *ptr = NULL;
    const int kAlignBytes = 256;
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

void int8gemm(int8_t* matrixA_reorder , int8_t * matrixB_reorder, int32_t * matrixC, int m, int n, int k){
    for(int i = 0; i < m / 4; i++){
    	int8kernel(matrixC + i * 4 * n, matrixA_reorder + i * 4 * k, matrixB_reorder, k/8, n/4);
    }
    return;
}
