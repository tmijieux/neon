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

/* TRANS A */

void neon_cblas_dgemm_transA_reference(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc)
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasTrans    );
    assert( TransB == CblasNoTrans  );
    assert( alpha != 0.0            );
    assert( isnormal(alpha)         );

    double betalpha = beta/alpha;
    // #pragma omp parallel for collapse(2) schedule(static, 200)
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            C[j*ldc+i] = betalpha * C[j*ldc+i];
            for (int k = 0; k < K; ++k) {
                C[j*ldc+i] = C[j*ldc+i] + A[i*lda+k] * B[j*ldb+k];
            }
            C[j*ldc+i] = alpha * C[j*ldc+i];
        }
    }
}

void neon_cblas_dgemm_transA_avx2(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc   )
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha != 0.0 );
    assert( isnormal(alpha) );

    const double betalpha = beta / alpha;
    const int mM = M % 4;
    const int MM = M - mM;
    const int nN = N % 4;
    const int NN = N - nN;
    const int kK = K % 4;
    const int KK = K - kK;

    for (int j = 0; j < NN; j+=4) {
        for (int i = 0; i < MM; i+=4) {
            __m256d c00 = _mm256_set1_pd( betalpha * C[(j+0)*ldc+i+0] / 4.0 );
            __m256d c01 = _mm256_set1_pd( betalpha * C[(j+0)*ldc+i+1] / 4.0 );
            __m256d c02 = _mm256_set1_pd( betalpha * C[(j+0)*ldc+i+2] / 4.0 );
            __m256d c03 = _mm256_set1_pd( betalpha * C[(j+0)*ldc+i+3] / 4.0 );

            __m256d c10 = _mm256_set1_pd( betalpha * C[(j+1)*ldc+i+0] / 4.0 );
            __m256d c11 = _mm256_set1_pd( betalpha * C[(j+1)*ldc+i+1] / 4.0 );
            __m256d c12 = _mm256_set1_pd( betalpha * C[(j+1)*ldc+i+2] / 4.0 );
            __m256d c13 = _mm256_set1_pd( betalpha * C[(j+1)*ldc+i+3] / 4.0 );

            __m256d c20 = _mm256_set1_pd( betalpha * C[(j+2)*ldc+i+0] / 4.0 );
            __m256d c21 = _mm256_set1_pd( betalpha * C[(j+2)*ldc+i+1] / 4.0 );
            __m256d c22 = _mm256_set1_pd( betalpha * C[(j+2)*ldc+i+2] / 4.0 );
            __m256d c23 = _mm256_set1_pd( betalpha * C[(j+2)*ldc+i+3] / 4.0 );

            __m256d c30 = _mm256_set1_pd( betalpha * C[(j+3)*ldc+i+0] / 4.0 );
            __m256d c31 = _mm256_set1_pd( betalpha * C[(j+3)*ldc+i+1] / 4.0 );
            __m256d c32 = _mm256_set1_pd( betalpha * C[(j+3)*ldc+i+2] / 4.0 );
            __m256d c33 = _mm256_set1_pd( betalpha * C[(j+3)*ldc+i+3] / 4.0 );

            for (int k = 0; k < KK; k+=4) {
                __m256d a00 = _mm256_loadu_pd( &A[(i+0)*lda+k] );
                __m256d a01 = _mm256_loadu_pd( &A[(i+1)*lda+k] );
                __m256d a02 = _mm256_loadu_pd( &A[(i+2)*lda+k] );
                __m256d a03 = _mm256_loadu_pd( &A[(i+3)*lda+k] );

                __m256d b00 = _mm256_loadu_pd( &B[(j+0)*ldb+k] );
                __m256d b01 = _mm256_loadu_pd( &B[(j+1)*ldb+k] );
                __m256d b02 = _mm256_loadu_pd( &B[(j+2)*ldb+k] );
                __m256d b03 = _mm256_loadu_pd( &B[(j+3)*ldb+k] );

                c00 = MM256_FMADD_PD(a00, b00, c00);
                c01 = MM256_FMADD_PD(a01, b00, c01);
                c02 = MM256_FMADD_PD(a02, b00, c02);
                c03 = MM256_FMADD_PD(a03, b00, c03);

                c10 = MM256_FMADD_PD(a00, b01, c10);
                c11 = MM256_FMADD_PD(a01, b01, c11);
                c12 = MM256_FMADD_PD(a02, b01, c12);
                c13 = MM256_FMADD_PD(a03, b01, c13);

                c20 = MM256_FMADD_PD(a00, b02, c20);
                c21 = MM256_FMADD_PD(a01, b02, c21);
                c22 = MM256_FMADD_PD(a02, b02, c22);
                c23 = MM256_FMADD_PD(a03, b02, c23);

                c30 = MM256_FMADD_PD(a00, b03, c30);
                c31 = MM256_FMADD_PD(a01, b03, c31);
                c32 = MM256_FMADD_PD(a02, b03, c32);
                c33 = MM256_FMADD_PD(a03, b03, c33);
            }

            _Alignas( __m256d ) double tmp00[4];
            _Alignas( __m256d ) double tmp01[4];
            _Alignas( __m256d ) double tmp02[4];
            _Alignas( __m256d ) double tmp03[4];

            _Alignas( __m256d ) double tmp10[4];
            _Alignas( __m256d ) double tmp11[4];
            _Alignas( __m256d ) double tmp12[4];
            _Alignas( __m256d ) double tmp13[4];

            _Alignas( __m256d ) double tmp20[4];
            _Alignas( __m256d ) double tmp21[4];
            _Alignas( __m256d ) double tmp22[4];
            _Alignas( __m256d ) double tmp23[4];

            _Alignas( __m256d ) double tmp30[4];
            _Alignas( __m256d ) double tmp31[4];
            _Alignas( __m256d ) double tmp32[4];
            _Alignas( __m256d ) double tmp33[4];

            _mm256_store_pd(tmp00, c00);
            _mm256_store_pd(tmp01, c01);
            _mm256_store_pd(tmp02, c02);
            _mm256_store_pd(tmp03, c03);

            _mm256_store_pd(tmp10, c10);
            _mm256_store_pd(tmp11, c11);
            _mm256_store_pd(tmp12, c12);
            _mm256_store_pd(tmp13, c13);

            _mm256_store_pd(tmp20, c20);
            _mm256_store_pd(tmp21, c21);
            _mm256_store_pd(tmp22, c22);
            _mm256_store_pd(tmp23, c23);

            _mm256_store_pd(tmp30, c30);
            _mm256_store_pd(tmp31, c31);
            _mm256_store_pd(tmp32, c32);
            _mm256_store_pd(tmp33, c33);

            _Alignas( __m256d ) double tmp_BIS[16] = {
                0.0 , 0.0, 0.0, 0.0,
                0.0 , 0.0, 0.0, 0.0,
                0.0 , 0.0, 0.0, 0.0,
                0.0 , 0.0, 0.0, 0.0,
            };

            for (int k = KK; k < KK + kK; ++k) {
                tmp_BIS[ 0] = tmp_BIS[ 0] + A[(i+0)*lda+k] * B[(j+0)*ldb+k];
                tmp_BIS[ 1] = tmp_BIS[ 1] + A[(i+1)*lda+k] * B[(j+0)*ldb+k];
                tmp_BIS[ 2] = tmp_BIS[ 2] + A[(i+2)*lda+k] * B[(j+0)*ldb+k];
                tmp_BIS[ 3] = tmp_BIS[ 3] + A[(i+3)*lda+k] * B[(j+0)*ldb+k];

                tmp_BIS[ 4] = tmp_BIS[ 4] + A[(i+0)*lda+k] * B[(j+1)*ldb+k];
                tmp_BIS[ 5] = tmp_BIS[ 5] + A[(i+1)*lda+k] * B[(j+1)*ldb+k];
                tmp_BIS[ 6] = tmp_BIS[ 6] + A[(i+2)*lda+k] * B[(j+1)*ldb+k];
                tmp_BIS[ 7] = tmp_BIS[ 7] + A[(i+3)*lda+k] * B[(j+1)*ldb+k];

                tmp_BIS[ 8] = tmp_BIS[ 8] + A[(i+0)*lda+k] * B[(j+2)*ldb+k];
                tmp_BIS[ 9] = tmp_BIS[ 9] + A[(i+1)*lda+k] * B[(j+2)*ldb+k];
                tmp_BIS[10] = tmp_BIS[10] + A[(i+2)*lda+k] * B[(j+2)*ldb+k];
                tmp_BIS[11] = tmp_BIS[11] + A[(i+3)*lda+k] * B[(j+2)*ldb+k];

                tmp_BIS[12] = tmp_BIS[12] + A[(i+0)*lda+k] * B[(j+3)*ldb+k];
                tmp_BIS[13] = tmp_BIS[13] + A[(i+1)*lda+k] * B[(j+3)*ldb+k];
                tmp_BIS[14] = tmp_BIS[14] + A[(i+2)*lda+k] * B[(j+3)*ldb+k];
                tmp_BIS[15] = tmp_BIS[15] + A[(i+3)*lda+k] * B[(j+3)*ldb+k];
            }
            C[(j+0)*ldc+i+0] = alpha * ( tmp00[0] + tmp00[1] + tmp00[2] + tmp00[3] + tmp_BIS[ 0] );
            C[(j+0)*ldc+i+1] = alpha * ( tmp01[0] + tmp01[1] + tmp01[2] + tmp01[3] + tmp_BIS[ 1] );
            C[(j+0)*ldc+i+2] = alpha * ( tmp02[0] + tmp02[1] + tmp02[2] + tmp02[3] + tmp_BIS[ 2] );
            C[(j+0)*ldc+i+3] = alpha * ( tmp03[0] + tmp03[1] + tmp03[2] + tmp03[3] + tmp_BIS[ 3] );

            C[(j+1)*ldc+i+0] = alpha * ( tmp10[0] + tmp10[1] + tmp10[2] + tmp10[3] + tmp_BIS[ 4] );
            C[(j+1)*ldc+i+1] = alpha * ( tmp11[0] + tmp11[1] + tmp11[2] + tmp11[3] + tmp_BIS[ 5] );
            C[(j+1)*ldc+i+2] = alpha * ( tmp12[0] + tmp12[1] + tmp12[2] + tmp12[3] + tmp_BIS[ 6] );
            C[(j+1)*ldc+i+3] = alpha * ( tmp13[0] + tmp13[1] + tmp13[2] + tmp13[3] + tmp_BIS[ 7] );

            C[(j+2)*ldc+i+0] = alpha * ( tmp20[0] + tmp20[1] + tmp20[2] + tmp20[3] + tmp_BIS[ 8] );
            C[(j+2)*ldc+i+1] = alpha * ( tmp21[0] + tmp21[1] + tmp21[2] + tmp21[3] + tmp_BIS[ 9] );
            C[(j+2)*ldc+i+2] = alpha * ( tmp22[0] + tmp22[1] + tmp22[2] + tmp22[3] + tmp_BIS[10] );
            C[(j+2)*ldc+i+3] = alpha * ( tmp23[0] + tmp23[1] + tmp23[2] + tmp23[3] + tmp_BIS[11] );

            C[(j+3)*ldc+i+0] = alpha * ( tmp30[0] + tmp30[1] + tmp30[2] + tmp30[3] + tmp_BIS[12] );
            C[(j+3)*ldc+i+1] = alpha * ( tmp31[0] + tmp31[1] + tmp31[2] + tmp31[3] + tmp_BIS[13] );
            C[(j+3)*ldc+i+2] = alpha * ( tmp32[0] + tmp32[1] + tmp32[2] + tmp32[3] + tmp_BIS[14] );
            C[(j+3)*ldc+i+3] = alpha * ( tmp33[0] + tmp33[1] + tmp33[2] + tmp33[3] + tmp_BIS[15] );
        }
    }

    for (int j = 0; j < NN; ++j) {
        for (int i = MM; i < MM + mM; ++i) {
            C[j*ldc+i] = betalpha * C[j*ldc+i];
            for (int k = 0; k < K; ++k) {
                C[j*ldc+i] = C[j*ldc+i] + A[i*lda+k] * B[j*ldb+k];
            }
            C[j*ldc+i] = alpha * C[j*ldc+i];
        }
    }
    for (int j = NN; j < NN+nN; ++j) {
        for (int i = 0; i < MM; ++i) {
            C[j*ldc+i] = betalpha * C[j*ldc+i];
            for (int k = 0; k < K; ++k) {
                C[j*ldc+i] = C[j*ldc+i] + A[i*lda+k] * B[j*ldb+k];
            }
            C[j*ldc+i] = alpha * C[j*ldc+i];
        }
    }
    for (int j = NN; j < NN+nN; ++j) {
        for (int i = MM; i < MM + mM; ++i) {
            C[j*ldc+i] = betalpha * C[j*ldc+i];
            for (int k = 0; k < K; ++k) {
                C[j*ldc+i] = C[j*ldc+i] + A[i*lda+k] * B[j*ldb+k];
            }
            C[j*ldc+i] = alpha * C[j*ldc+i];
        }
    }
}

/********** NEON tiling + parallel implementation ************************/

static _Atomic int tile_size = 128;
void neon_set_tile_size(const int TS) { tile_size = TS; }
int neon_get_tile_size(void) { return tile_size; }

void neon_cblas_dgemm_transA_tiled(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc   )
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha != 0.0 );
    assert( isnormal(alpha) );

    const int TS = neon_get_tile_size();

    const int Mm = M / TS;
    const int mM = M % TS;
    //const int MM = M - mM;

    const int Nn = N / TS;
    const int nN = N % TS;
    //const int NN = N - nN;

    const int Kk = K / TS;
    const int kK = K % TS;
    //const int KK = K - kK;

    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static)
        for (int j = 0; j < Nn; ++j) {
            for (int i = 0; i < Mm; ++i) {
                for (int k = 0; k < Kk; ++k) {
                    const double beta_l = (k == 0) ? beta : 1.0;
                    neon_cblas_dgemm_transA_avx2(
                        /**/           layout,
                        /**/           TransA, TransB,
                        /**/           TS, TS, TS,
                        /**/           alpha,  A+TS*(lda*i+k), lda,
                        /**/                   B+TS*(ldb*j+k), ldb,
                        /**/           beta_l, C+TS*(ldc*j+i), ldc       );
                }

                const double beta_l = (Kk == 0) ? beta : 1.0;
                neon_cblas_dgemm_transA_avx2(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           TS, TS, kK,
                    /**/           alpha,  A+TS*(lda*i+Kk), lda,
                    /**/                   B+TS*(ldb*j+Kk), ldb,
                    /**/           beta_l, C+TS*(ldc*j+i), ldc       );
            }
        }

        // last block column
        #pragma omp for schedule(static)
        for (int i = 0; i < Mm; ++i) {
            for (int k = 0; k < Kk; ++k) {
                const double beta_l = (k == 0) ? beta : 1.0;
                neon_cblas_dgemm_transA_avx2(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           TS, nN, TS,
                    /**/           alpha,  A+TS*(lda*i+k), lda,
                    /**/                   B+TS*(ldb*Nn+k), ldb,
                    /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_avx2(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           TS, nN, kK,
                /**/           alpha,  A+TS*(lda*i+Kk), lda,
                /**/                   B+TS*(ldb*Nn+Kk), ldb,
                /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
        }

        // last block row
        #pragma omp for schedule(static)
        for (int j = 0; j < Nn; ++j) {
            for (int k = 0; k < Kk; ++k) {
                const double beta_l = (k == 0) ? beta : 1.0;
                neon_cblas_dgemm_transA_avx2(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           mM, TS, TS,
                    /**/           alpha,  A+TS*(lda*Mm+k), lda,
                    /**/                   B+TS*(ldb*j+k), ldb,
                    /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_avx2(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           mM, TS, kK,
                /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
                /**/                   B+TS*(ldb*j+Kk), ldb,
                /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
        }

        // last block right bottom corner

        #pragma omp single
        {
            for (int k = 0; k < Kk; ++k) {
                const double beta_l = (k == 0) ? beta : 1.0;
                neon_cblas_dgemm_transA_avx2(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           mM, nN, TS,
                    /**/           alpha,  A+TS*(lda*Mm+k), lda,
                    /**/                   B+TS*(ldb*Nn+k), ldb,
                    /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_avx2(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           mM, nN, kK,
                /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
                /**/                   B+TS*(ldb*Nn+Kk), ldb,
                /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );

        }
    }
}

void neon_cblas_dgemm_transA_tiled_task(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc   )
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha != 0.0 );
    assert( isnormal(alpha) );

    const int TS = neon_get_tile_size();

    const int Mm = M / TS;
    const int mM = M % TS;
    //const int MM = M - mM;

    const int Nn = N / TS;
    const int nN = N % TS;
    //const int NN = N - nN;

    const int Kk = K / TS;
    const int kK = K % TS;
    //const int KK = K - kK;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int j = 0; j < Nn; ++j) {
                for (int i = 0; i < Mm; ++i) {
                    for (int k = 0; k < Kk; ++k) {
                        #pragma omp task depend (inout: C[i+j*ldc])
                        {
                            const double beta_l = (k == 0) ? beta : 1.0;
                            neon_cblas_dgemm_transA_avx2(
                                /**/           layout,
                                /**/           TransA, TransB,
                                /**/           TS, TS, TS,
                                /**/           alpha,  A+TS*(lda*i+k), lda,
                                /**/                   B+TS*(ldb*j+k), ldb,
                                /**/           beta_l, C+TS*(ldc*j+i), ldc       );
                        }
                    }

                    #pragma omp task depend(inout:C[i+j*ldc])
                    {
                        const double beta_l = (Kk == 0) ? beta : 1.0;
                        neon_cblas_dgemm_transA_avx2(
                            /**/           layout,
                            /**/           TransA, TransB,
                            /**/           TS, TS, kK,
                            /**/           alpha,  A+TS*(lda*i+Kk), lda,
                            /**/                   B+TS*(ldb*j+Kk), ldb,
                            /**/           beta_l, C+TS*(ldc*j+i), ldc       );
                    }
                }
            }

            // last block column
            for (int i = 0; i < Mm; ++i) {
                for (int k = 0; k < Kk; ++k) {
                    #pragma omp task depend(inout:C[i+Nn*ldc])
                    {
                        const double beta_l = (k == 0) ? beta : 1.0;
                        neon_cblas_dgemm_transA_avx2(
                            /**/           layout,
                            /**/           TransA, TransB,
                            /**/           TS, nN, TS,
                            /**/           alpha,  A+TS*(lda*i+k), lda,
                            /**/                   B+TS*(ldb*Nn+k), ldb,
                            /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
                    }
                }
                #pragma omp task depend(inout:C[i+Nn*ldc])
                {

                    const double beta_l = (Kk == 0) ? beta : 1.0;
                    neon_cblas_dgemm_transA_avx2(
                        /**/           layout,
                        /**/           TransA, TransB,
                        /**/           TS, nN, kK,
                        /**/           alpha,  A+TS*(lda*i+Kk), lda,
                        /**/                   B+TS*(ldb*Nn+Kk), ldb,
                        /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
                }
            }

            // last block row
            for (int j = 0; j < Nn; ++j) {
                for (int k = 0; k < Kk; ++k) {
                    #pragma omp task depend(inout:C[Mm+j*ldc])
                    {

                        const double beta_l = (k == 0) ? beta : 1.0;
                        neon_cblas_dgemm_transA_avx2(
                            /**/           layout,
                            /**/           TransA, TransB,
                            /**/           mM, TS, TS,
                            /**/           alpha,  A+TS*(lda*Mm+k), lda,
                            /**/                   B+TS*(ldb*j+k), ldb,
                            /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
                    }
                }

                #pragma omp task depend(inout:C[Mm+j*ldc])
                {

                    const double beta_l = (Kk == 0) ? beta : 1.0;
                    neon_cblas_dgemm_transA_avx2(
                        /**/           layout,
                        /**/           TransA, TransB,
                        /**/           mM, TS, kK,
                        /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
                        /**/                   B+TS*(ldb*j+Kk), ldb,
                        /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
                }
            }

            // last block right bottom corner
            for (int k = 0; k < Kk; ++k) {
                #pragma omp task depend(inout:C[Mm+Nn*ldc])
                {
                    const double beta_l = (k == 0) ? beta : 1.0;
                    neon_cblas_dgemm_transA_avx2(
                        /**/           layout,
                        /**/           TransA, TransB,
                        /**/           mM, nN, TS,
                        /**/           alpha,  A+TS*(lda*Mm+k), lda,
                        /**/                   B+TS*(ldb*Nn+k), ldb,
                        /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
                }
            }
            #pragma omp task depend(inout:C[Mm+Nn*ldc])
            {
                const double beta_l = (Kk == 0) ? beta : 1.0;
                neon_cblas_dgemm_transA_avx2(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           mM, nN, kK,
                    /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
                    /**/                   B+TS*(ldb*Nn+Kk), ldb,
                    /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
            }
        }
    } // end pragma omp parallel
}



/************* neon tiling + MKL ***********/

void neon_cblas_dgemm_transA_tiled_plus_mkl(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc   )
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha != 0.0 );
    assert( isnormal(alpha) );

    mkl_set_num_threads(1);

    const int TS = neon_get_tile_size();

    const int Mm = M / TS;
    const int mM = M % TS;
    //const int MM = M - mM;

    const int Nn = N / TS;
    const int nN = N % TS;
    //const int NN = N - nN;

    const int Kk = K / TS;
    const int kK = K % TS;
    //const int KK = K - kK;

    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static)
        for (int j = 0; j < Nn; ++j) {
            for (int i = 0; i < Mm; ++i) {
                for (int k = 0; k < Kk; ++k) {
                    const double beta_l = (k == 0) ? beta : 1.0;
                    cblas_dgemm(
                        /**/           layout,
                        /**/           TransA, TransB,
                        /**/           TS, TS, TS,
                        /**/           alpha,  A+TS*(lda*i+k), lda,
                        /**/                   B+TS*(ldb*j+k), ldb,
                        /**/           beta_l, C+TS*(ldc*j+i), ldc       );
                }

                const double beta_l = (Kk == 0) ? beta : 1.0;
                cblas_dgemm(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           TS, TS, kK,
                    /**/           alpha,  A+TS*(lda*i+Kk), lda,
                    /**/                   B+TS*(ldb*j+Kk), ldb,
                    /**/           beta_l, C+TS*(ldc*j+i), ldc       );
            }
        }

        // last block column
        #pragma omp for schedule(static)
        for (int i = 0; i < Mm; ++i) {
            for (int k = 0; k < Kk; ++k) {
                const double beta_l = (k == 0) ? beta : 1.0;
                cblas_dgemm(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           TS, nN, TS,
                    /**/           alpha,  A+TS*(lda*i+k), lda,
                    /**/                   B+TS*(ldb*Nn+k), ldb,
                    /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            cblas_dgemm(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           TS, nN, kK,
                /**/           alpha,  A+TS*(lda*i+Kk), lda,
                /**/                   B+TS*(ldb*Nn+Kk), ldb,
                /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
        }

        // last block row
        #pragma omp for schedule(static)
        for (int j = 0; j < Nn; ++j) {
            for (int k = 0; k < Kk; ++k) {
                const double beta_l = (k == 0) ? beta : 1.0;
                cblas_dgemm(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           mM, TS, TS,
                    /**/           alpha,  A+TS*(lda*Mm+k), lda,
                    /**/                   B+TS*(ldb*j+k), ldb,
                    /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            cblas_dgemm(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           mM, TS, kK,
                /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
                /**/                   B+TS*(ldb*j+Kk), ldb,
                /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
        }

        // last block right bottom corner

        #pragma omp single
        {
            for (int k = 0; k < Kk; ++k) {
                const double beta_l = (k == 0) ? beta : 1.0;
                cblas_dgemm(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           mM, nN, TS,
                    /**/           alpha,  A+TS*(lda*Mm+k), lda,
                    /**/                   B+TS*(ldb*Nn+k), ldb,
                    /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            cblas_dgemm(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           mM, nN, kK,
                /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
                /**/                   B+TS*(ldb*Nn+Kk), ldb,
                /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );

        }
    }
    mkl_set_num_threads(2);
}

void neon_cblas_dgemm_transA_tiled_task_mkl(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc   )
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha != 0.0 );
    assert( isnormal(alpha) );

    mkl_set_num_threads(1);

    const int TS = neon_get_tile_size();
    const int Mm = M / TS;
    const int mM = M % TS;
    const int Nn = N / TS;
    const int nN = N % TS;
    const int Kk = K / TS;
    const int kK = K % TS;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int j = 0; j < Nn; ++j) {
                for (int i = 0; i < Mm; ++i) {
                    for (int k = 0; k < Kk; ++k) {
                        #pragma omp task depend (inout: C[i+j*ldc])
                        {
                            const double beta_l = (k == 0) ? beta : 1.0;
                            cblas_dgemm(
                                /**/           layout,
                                /**/           TransA, TransB,
                                /**/           TS, TS, TS,
                                /**/           alpha,  A+TS*(lda*i+k), lda,
                                /**/                   B+TS*(ldb*j+k), ldb,
                                /**/           beta_l, C+TS*(ldc*j+i), ldc       );
                        }
                    }

                    #pragma omp task depend(inout:C[i+j*ldc])
                    {
                        const double beta_l = (Kk == 0) ? beta : 1.0;
                        cblas_dgemm(
                            /**/           layout,
                            /**/           TransA, TransB,
                            /**/           TS, TS, kK,
                            /**/           alpha,  A+TS*(lda*i+Kk), lda,
                            /**/                   B+TS*(ldb*j+Kk), ldb,
                            /**/           beta_l, C+TS*(ldc*j+i), ldc       );
                    }
                }
            }

            // last block column
            for (int i = 0; i < Mm; ++i) {
                for (int k = 0; k < Kk; ++k) {
                    #pragma omp task depend(inout:C[i+Nn*ldc])
                    {
                        const double beta_l = (k == 0) ? beta : 1.0;
                        cblas_dgemm(
                            /**/           layout,
                            /**/           TransA, TransB,
                            /**/           TS, nN, TS,
                            /**/           alpha,  A+TS*(lda*i+k), lda,
                            /**/                   B+TS*(ldb*Nn+k), ldb,
                            /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
                    }
                }
                #pragma omp task depend(inout:C[i+Nn*ldc])
                {

                    const double beta_l = (Kk == 0) ? beta : 1.0;
                    cblas_dgemm(
                        /**/           layout,
                        /**/           TransA, TransB,
                        /**/           TS, nN, kK,
                        /**/           alpha,  A+TS*(lda*i+Kk), lda,
                        /**/                   B+TS*(ldb*Nn+Kk), ldb,
                        /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
                }
            }

            // last block row
            for (int j = 0; j < Nn; ++j) {
                for (int k = 0; k < Kk; ++k) {
                    #pragma omp task depend(inout:C[Mm+j*ldc])
                    {

                        const double beta_l = (k == 0) ? beta : 1.0;
                        cblas_dgemm(
                            /**/           layout,
                            /**/           TransA, TransB,
                            /**/           mM, TS, TS,
                            /**/           alpha,  A+TS*(lda*Mm+k), lda,
                            /**/                   B+TS*(ldb*j+k), ldb,
                            /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
                    }
                }

                #pragma omp task depend(inout:C[Mm+j*ldc])
                {

                    const double beta_l = (Kk == 0) ? beta : 1.0;
                    cblas_dgemm(
                        /**/           layout,
                        /**/           TransA, TransB,
                        /**/           mM, TS, kK,
                        /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
                        /**/                   B+TS*(ldb*j+Kk), ldb,
                        /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
                }
            }

            // last block right bottom corner
            for (int k = 0; k < Kk; ++k) {
                #pragma omp task depend(inout:C[Mm+Nn*ldc])
                {
                    const double beta_l = (k == 0) ? beta : 1.0;
                    cblas_dgemm(
                        /**/           layout,
                        /**/           TransA, TransB,
                        /**/           mM, nN, TS,
                        /**/           alpha,  A+TS*(lda*Mm+k), lda,
                        /**/                   B+TS*(ldb*Nn+k), ldb,
                        /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
                }
            }
            #pragma omp task depend(inout:C[Mm+Nn*ldc])
            {
                const double beta_l = (Kk == 0) ? beta : 1.0;
                cblas_dgemm(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           mM, nN, kK,
                    /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
                    /**/                   B+TS*(ldb*Nn+Kk), ldb,
                    /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
            }

        }
    } // end pragma omp parallel

    mkl_set_num_threads(2);
}
