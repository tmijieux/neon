#ifndef NEON_ZGEMM_H
#define NEON_ZGEMM_H

#include "/opt/intel/mkl/include/mkl.h"

void neon_cblas_zgemm_reference(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const void *p_alpha, const void * restrict const p_A, const int lda,
    /**/                 const void * restrict const p_B, const int ldb,
    const void *p_beta,        void * restrict const p_C, const int ldc);


#endif // NEON_ZGEMM_H
