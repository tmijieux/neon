#include <assert.h>
#include <math.h>
#include <complex.h>

#include "cblas.h"

void neon_cblas_zgemm_reference(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const void *p_alpha, const void * restrict const p_A, const int lda,
    /**/                 const void * restrict const p_B, const int ldb,
    const void *p_beta,        void * restrict const p_C, const int ldc)
{
    double _Complex alpha = *(double _Complex*) p_alpha;
    double _Complex beta = *(double _Complex*) p_beta;
    const double _Complex *A = p_A;
    const double _Complex *B = p_B;
    /**/  double _Complex *C = p_C;

    assert( layout == CblasColMajor );
    assert( TransA == CblasNoTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha != 0.0 );
    double norm_alpha = cabs(alpha);
    assert( isnormal(norm_alpha) );

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            C[j*ldc+i] = (beta/alpha) * C[j*ldc+i];
            for (int k = 0; k < K; ++k) {
                C[j*ldc+i] = C[j*ldc+i] + A[k*lda+i] * B[j*ldb+k];
            }
            C[j*ldc+i] = alpha * C[j*ldc+i];
        }
    }
}
