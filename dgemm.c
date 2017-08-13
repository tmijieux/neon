#include <assert.h>
#include <math.h>
#include <immintrin.h>

//#include "cblas.h"

#include "dgemm.h"

#ifdef __FMA__
# define MM256_FMADD_PD(a,b,c) _mm256_fmadd_pd((a), (b), (c))
#else
# define MM256_FMADD_PD(a,b,c) _mm256_add_pd(_mm256_mul_pd((a), (b)), (c))
#endif

void neon_cblas_dgemm_reference(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc)
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasNoTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha != 0.0 );
    assert( isnormal(alpha) );

    // #pragma omp parallel for collapse(2) schedule(static, 200)
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

void neon_cblas_dgemm_avx2(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha_s, const double *restrict const A, const int lda,
    /**/                  const double *restrict const B, const int ldb,
    const double beta_s,        double *restrict const C, const int ldc  )
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasNoTrans  );
    assert( TransB == CblasNoTrans  );
    assert( alpha_s != 0.0          );
    assert( isnormal(alpha_s)       );

    double betalpha_s = beta_s / alpha_s;
    __m256d betalpha  = _mm256_broadcast_sd(&betalpha_s);
    __m256d alpha     = _mm256_broadcast_sd(&alpha_s);

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; i+=4) {
            double *C_ij = &C[j*ldc+i];

            __m256d c = _mm256_loadu_pd( C_ij );
            c = _mm256_mul_pd(betalpha, c);

            for (int k = 0; k < K; ++k) {
                const double *A_ik = &A[j*lda+k];
                const double *B_kj = &B[k*ldb+j];
                __m256d a = _mm256_loadu_pd( A_ik );
                __m256d b = _mm256_loadu_pd( B_kj );
                c = MM256_FMADD_PD(a, b, c);
            }
            c = _mm256_mul_pd(alpha, c);
            _mm256_storeu_pd(C_ij, c);
        }
    }
}
