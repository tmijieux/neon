#include <assert.h>
#include <math.h>

#include "cblas.h"


void neon_cblas_dgemm(CBLAS_LAYOUT layout,
                      CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                      const int M, const int N, const int K,
                      const double alpha, const double *A, const int lda,
                      /**/                const double *B, const int ldb,
                      const double beta,        double *C, const int ldc)
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
                C[j*ldc+i] = C[j*ldc+i] + A[i*lda+k] * B[j*ldb+k];
            }
            C[j*ldc+i] = alpha * C[j*ldc+i];
        }
    }
}

#include <immintrin.h>

void neon_cblas_dgemm_avx2( CBLAS_LAYOUT layout,
                            CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                            const int M, const int N, const int K,
                            const double alpha_s, const double *A, const int lda,
                            /**/                  const double *B, const int ldb,
                            const double beta_s,        double *C, const int ldc)
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasNoTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha_s != 0.0 );
    assert( isnormal(alpha_s) );
    double betalpha_s = beta_s / alpha_s;
    __m256d betalpha = _mm256_broadcast_sd(&betalpha_s);
    __m256d alpha    = _mm256_broadcast_sd(&alpha_s);

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
                c = _mm256_fmadd_pd(a, b, c);
            }
            c = _mm256_mul_pd(alpha, c);
            _mm256_storeu_pd(C_ij, c);
        }
    }
}




/* TRANS A */


void cblas_dgemm_transA(CBLAS_LAYOUT layout,
                        CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                        const int M, const int N, const int K,
                        const double alpha, const double *A, const int lda,
                        /**/                const double *B, const int ldb,
                        const double beta,        double *C, const int ldc)
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha != 0.0 );
    assert( isnormal(alpha) );
    double betalpha = beta/alpha;

    // #pragma omp parallel for collapse(2) schedule(static, 200)
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            C[j*ldc+i] = (betalpha) * C[j*ldc+i];
            for (int k = 0; k < K; ++k) {
                C[j*ldc+i] = C[j*ldc+i] + A[i*lda+k] * B[j*ldb+k];
            }
            C[j*ldc+i] = alpha * C[j*ldc+i];
        }
    }
}


void neon_cblas_dgemm_transA_avx2( CBLAS_LAYOUT layout,
                                   CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                                   const int M, const int N, const int K,
                                   const double alpha_s, const double *A, const int lda,
                                   /**/                  const double *B, const int ldb,
                                   const double beta_s,        double *C, const int ldc)
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha_s != 0.0 );
    assert( isnormal(alpha_s) );


    double betalpha_s = beta_s / alpha_s;
    __m256d betalpha = _mm256_broadcast_sd(&betalpha_s);
    __m256d alpha    = _mm256_broadcast_sd(&alpha_s);

    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static, 200)
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < M; i+=4) {

                double *C_ij = &C[j*ldc+i];
                __m256d c = _mm256_loadu_pd( C_ij );
                c = _mm256_mul_pd(betalpha, c);
                for (int k = 0; k < K; k+=4) {
                    const double *A_ki = &A[i*lda+k];
                    const double *B_kj = &B[j*ldb+k];
                    __m256d a = _mm256_loadu_pd( A_ki );
                    __m256d b = _mm256_loadu_pd( B_kj );
                    c = _mm256_fmadd_pd(a, b, c);
                }
                c = _mm256_mul_pd(alpha, c);
                _mm256_storeu_pd(C_ij, c);

                for (int k = K%4; k > 0; --k) {
                    int kk = K - k;
                    C[j*ldc+i] = C[j*ldc+i] + A[kk*lda+i] * B[j*ldb+kk];
                }
            }
        }

        #pragma omp for schedule(static, 200)
        for (int j = 0; j < N; ++j) {
            for (int i = M%4; i > 0; --i) {
                int ii = M - i;
                C[j*ldc+ii] = betalpha_s * C[j*ldc+ii];
                for (int k = 0; k < K; ++k) {
                    C[j*ldc+ii] = C[j*ldc+ii] + A[k*lda+ii] * B[j*ldb+k];
                }
                C[j*ldc+ii] = alpha_s * C[j*ldc+ii];
            }
        }
    }
}

