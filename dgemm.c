#include <assert.h>
#include <math.h>
#include <immintrin.h>

//#include "cblas.h"

#include "mkl.h"

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
                c = MM256_FMADD_PD(a, b, c);
            }
            c = _mm256_mul_pd(alpha, c);
            _mm256_storeu_pd(C_ij, c);
        }
    }
}


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
    assert( TransA == CblasTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha != 0.0 );
    assert( isnormal(alpha) );
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

void neon_cblas_dgemm_transA_avx2_unirow_1(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc)
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha != 0.0 );
    assert( isnormal(alpha) );

    const double betalpha = beta / alpha;
    const int kK = K % 16;
    const int KK = K - kK;

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {

            __m256d c1 = _mm256_set1_pd( betalpha * C[j*ldc+i] / 4.0 );

            for (int k = 0; k < KK; k+=4) {
                __m256d a1 = _mm256_loadu_pd( &A[i*lda+k+ 0] );
                __m256d b1 = _mm256_loadu_pd( &B[j*ldb+k+ 0] );
                c1 = MM256_FMADD_PD(a1, b1, c1);
            }
            _Alignas( __m256d )  double tmp0[4];
            _mm256_store_pd(&tmp0[0], c1);

            double tmp1 = 0.0;
            for (int k = KK; k < K; ++k) {
                tmp1 += A[i*lda+k] * B[j*ldb+k];
            }
            C[j*ldc+i] = alpha * ( tmp0[ 0] + tmp0[ 1] + tmp0[ 2] + tmp0[ 3] + tmp1  );
        }
    }
}

void neon_cblas_dgemm_transA_avx2_unirow(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc)
{
    assert( layout == CblasColMajor );
    assert( TransA == CblasTrans );
    assert( TransB == CblasNoTrans );
    assert( alpha != 0.0 );
    assert( isnormal(alpha) );

    const double betalpha = beta / alpha;
    const int kK = K % 16;
    const int KK = K - kK;

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            __m256d c1 = _mm256_set1_pd( betalpha * C[j*ldc+i] / 4.0 );
            __m256d c2 = _mm256_setzero_pd();
            __m256d c3 = _mm256_setzero_pd();
            __m256d c4 = _mm256_setzero_pd();
            __m256d c5 = _mm256_setzero_pd();

            for (int k = 0; k < KK; k+=16) {
                __m256d a1 = _mm256_loadu_pd( &A[i*lda+k+ 0] );
                __m256d b1 = _mm256_loadu_pd( &B[j*ldb+k+ 0] );
                __m256d a2 = _mm256_loadu_pd( &A[i*lda+k+ 4] );
                __m256d b2 = _mm256_loadu_pd( &B[j*ldb+k+ 4] );
                __m256d a3 = _mm256_loadu_pd( &A[i*lda+k+ 8] );
                __m256d b3 = _mm256_loadu_pd( &B[j*ldb+k+ 8] );
                __m256d a4 = _mm256_loadu_pd( &A[i*lda+k+12] );
                __m256d b4 = _mm256_loadu_pd( &B[j*ldb+k+12] );

                c1 = MM256_FMADD_PD(a1, b1, c1);
                c2 = MM256_FMADD_PD(a2, b2, c2);
                c3 = MM256_FMADD_PD(a3, b3, c3);
                c4 = MM256_FMADD_PD(a4, b4, c4);
            }
            const int KkK = kK % 4;
            const int KKK = kK - KkK;

            for (int k = KK; k < KK+KKK; k+=4) {
                __m256d a1 = _mm256_loadu_pd( &A[i*lda+k] );
                __m256d b1 = _mm256_loadu_pd( &B[j*ldb+k] );
                c5 = MM256_FMADD_PD(a1, b1, c5);
            }
            __m256d c6 = c5+c4+c1+c2+c3;
            _Alignas( __m256d )  double tmp0[4];
            _mm256_store_pd(&tmp0[0], c6);

            double tmp1 = 0;
            for (int k = KK+KKK; k < K; ++k) {
                tmp1 += A[i*lda+k] * B[j*ldb+k];
            }
            C[j*ldc+i] = alpha * ( tmp0[ 0] + tmp0[ 1] + tmp0[ 2] + tmp0[ 3] + tmp1  );
        }
    }
}

void neon_cblas_dgemm_transA_avx2_multirow(
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

    /* int mt = M / 4; */
    /* int nt = N / 4; */
    /* int kt = K / 4; */
    const int mM = M % 4;
    const int MM = M - mM;
    const int kK = K % 4;
    const int KK = K - kK;

    //#pragma omp parallel
    {
        //#pragma omp for collapse(2) schedule(static, 200)
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < MM; i+=4) {
                __m256d c0 = _mm256_set1_pd( betalpha * C[j*ldc+i+0] / 4.0 );
                __m256d c1 = _mm256_set1_pd( betalpha * C[j*ldc+i+1] / 4.0 );
                __m256d c2 = _mm256_set1_pd( betalpha * C[j*ldc+i+2] / 4.0 );
                __m256d c3 = _mm256_set1_pd( betalpha * C[j*ldc+i+3] / 4.0 );

                for (int k = 0; k < KK; k+=4) {
                    __m256d a0 = _mm256_loadu_pd( &A[(i+0)*lda+k] );
                    __m256d a1 = _mm256_loadu_pd( &A[(i+1)*lda+k] );
                    __m256d a2 = _mm256_loadu_pd( &A[(i+2)*lda+k] );
                    __m256d a3 = _mm256_loadu_pd( &A[(i+3)*lda+k] );
                    __m256d b = _mm256_loadu_pd( &B[j*ldb+k] );

                    c0 = MM256_FMADD_PD(a0, b, c0);
                    c1 = MM256_FMADD_PD(a1, b, c1);
                    c2 = MM256_FMADD_PD(a2, b, c2);
                    c3 = MM256_FMADD_PD(a3, b, c3);
                }

                _Alignas( __m256d ) double tmp0[4];
                _Alignas( __m256d ) double tmp1[4];
                _Alignas( __m256d ) double tmp2[4];
                _Alignas( __m256d ) double tmp3[4];
                _mm256_store_pd(tmp0, c0);
                _mm256_store_pd(tmp1, c1);
                _mm256_store_pd(tmp2, c2);
                _mm256_store_pd(tmp3, c3);

                _Alignas( __m256d ) double tmp_BIS[4] = { 0.0 , 0.0, 0.0, 0.0 };

                for (int k = KK; k < KK + kK; ++k) {
                    tmp_BIS[0] = tmp_BIS[0] + A[(i+0)*lda+k] * B[j*ldb+k];
                    tmp_BIS[1] = tmp_BIS[1] + A[(i+1)*lda+k] * B[j*ldb+k];
                    tmp_BIS[2] = tmp_BIS[2] + A[(i+2)*lda+k] * B[j*ldb+k];
                    tmp_BIS[3] = tmp_BIS[3] + A[(i+3)*lda+k] * B[j*ldb+k];
                }
                C[j*ldc+i+0] = alpha * ( tmp0[0] + tmp0[1] + tmp0[2] + tmp0[3] + tmp_BIS[0] );
                C[j*ldc+i+1] = alpha * ( tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3] + tmp_BIS[1] );
                C[j*ldc+i+2] = alpha * ( tmp2[0] + tmp2[1] + tmp2[2] + tmp2[3] + tmp_BIS[2] );
                C[j*ldc+i+3] = alpha * ( tmp3[0] + tmp3[1] + tmp3[2] + tmp3[3] + tmp_BIS[3] );
            }
        }

        //#pragma omp for schedule(static, 200)
        for (int j = 0; j < N; ++j) {
            for (int i = MM; i < MM + mM; ++i) {
                C[j*ldc+i] = betalpha * C[j*ldc+i];
                for (int k = 0; k < K; ++k) {
                    C[j*ldc+i] = C[j*ldc+i] + A[i*lda+k] * B[j*ldb+k];
                }
                C[j*ldc+i] = alpha * C[j*ldc+i];
            }
        }
    }
}

static _Atomic int tile_size = 128;
void neon_set_tile_size(const int TS) { tile_size = TS; }
int neon_get_tile_size(void) { return tile_size; }

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
                            neon_cblas_dgemm_transA_avx2_multirow(
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
                        neon_cblas_dgemm_transA_avx2_multirow(
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
                        neon_cblas_dgemm_transA_avx2_multirow(
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
                    neon_cblas_dgemm_transA_avx2_multirow(
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
                        neon_cblas_dgemm_transA_avx2_multirow(
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
                    neon_cblas_dgemm_transA_avx2_multirow(
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
                    neon_cblas_dgemm_transA_avx2_multirow(
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
                neon_cblas_dgemm_transA_avx2_multirow(
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

    /* int num_threads = mkl_get_num_threads(); */
    /* assert( num_threads == 2 ); */

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
                    neon_cblas_dgemm_transA_avx2_unirow_1(
                        /**/           layout,
                        /**/           TransA, TransB,
                        /**/           TS, TS, TS,
                        /**/           alpha,  A+TS*(lda*i+k), lda,
                        /**/                   B+TS*(ldb*j+k), ldb,
                        /**/           beta_l, C+TS*(ldc*j+i), ldc       );
                }

                const double beta_l = (Kk == 0) ? beta : 1.0;
                neon_cblas_dgemm_transA_avx2_unirow_1(
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
                neon_cblas_dgemm_transA_avx2_unirow_1(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           TS, nN, TS,
                    /**/           alpha,  A+TS*(lda*i+k), lda,
                    /**/                   B+TS*(ldb*Nn+k), ldb,
                    /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_avx2_unirow_1(
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
                neon_cblas_dgemm_transA_avx2_unirow_1(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           mM, TS, TS,
                    /**/           alpha,  A+TS*(lda*Mm+k), lda,
                    /**/                   B+TS*(ldb*j+k), ldb,
                    /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_avx2_unirow_1(
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
                neon_cblas_dgemm_transA_avx2_unirow_1(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           mM, nN, TS,
                    /**/           alpha,  A+TS*(lda*Mm+k), lda,
                    /**/                   B+TS*(ldb*Nn+k), ldb,
                    /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_avx2_unirow_1(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           mM, nN, kK,
                /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
                /**/                   B+TS*(ldb*Nn+Kk), ldb,
                /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );

        }
    }
}

void neon_cblas_dgemm_transA_tiled_multi(
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

    /* int num_threads = mkl_get_num_threads(); */
    /* assert( num_threads == 2 ); */

    const int TS = neon_get_tile_size();
    const int Mm = M / TS;
    const int mM = M % TS;
    const int Nn = N / TS;
    const int nN = N % TS;
    const int Kk = K / TS;
    const int kK = K % TS;

    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static)
        for (int j = 0; j < Nn; ++j) {
            for (int i = 0; i < Mm; ++i) {
                for (int k = 0; k < Kk; ++k) {
                    const double beta_l = (k == 0) ? beta : 1.0;
                    neon_cblas_dgemm_transA_avx2_multirow(
                        /**/           layout,
                        /**/           TransA, TransB,
                        /**/           TS, TS, TS,
                        /**/           alpha,  A+TS*(lda*i+k), lda,
                        /**/                   B+TS*(ldb*j+k), ldb,
                        /**/           beta_l, C+TS*(ldc*j+i), ldc       );
                }

                const double beta_l = (Kk == 0) ? beta : 1.0;
                neon_cblas_dgemm_transA_avx2_multirow(
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
                neon_cblas_dgemm_transA_avx2_multirow(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           TS, nN, TS,
                    /**/           alpha,  A+TS*(lda*i+k), lda,
                    /**/                   B+TS*(ldb*Nn+k), ldb,
                    /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_avx2_multirow(
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
                neon_cblas_dgemm_transA_avx2_multirow(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           mM, TS, TS,
                    /**/           alpha,  A+TS*(lda*Mm+k), lda,
                    /**/                   B+TS*(ldb*j+k), ldb,
                    /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_avx2_multirow(
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
                neon_cblas_dgemm_transA_avx2_multirow(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           mM, nN, TS,
                    /**/           alpha,  A+TS*(lda*Mm+k), lda,
                    /**/                   B+TS*(ldb*Nn+k), ldb,
                    /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_avx2_multirow(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           mM, nN, kK,
                /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
                /**/                   B+TS*(ldb*Nn+Kk), ldb,
                /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );

        }
    }
}

/**** MULTI LEVEL TILED *********/

static int tile_size_L1 = 50;
static int tile_size_L3 = 128;

static int tile_size_L2 = 256;

void neon_set_tile_size_L1(const int TS) { tile_size_L1 = TS; }
void neon_set_tile_size_L2(const int TS) { tile_size_L2 = TS; }
void neon_set_tile_size_L3(const int TS) { tile_size_L3 = TS; }

int neon_get_tile_size_L1(void) {  return tile_size_L1; }
int neon_get_tile_size_L2(void) {  return tile_size_L2; }
int neon_get_tile_size_L3(void) {  return tile_size_L3; }

void neon_cblas_dgemm_transA_tiled_L1(
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

    const int TS = neon_get_tile_size_L1();

    const int Mm = M / TS;
    const int mM = M % TS;
    //const int MM = M - mM;

    const int Nn = N / TS;
    const int nN = N % TS;
    //const int NN = N - nN;

    const int Kk = K / TS;
    const int kK = K % TS;
    //const int KK = K - kK;

    for (int j = 0; j < Nn; ++j) {
        for (int i = 0; i < Mm; ++i) {
            for (int k = 0; k < Kk; ++k) {
                const double beta_l = (k == 0) ? beta : 1.0;
                neon_cblas_dgemm_transA_avx2_multirow(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           TS, TS, TS,
                    /**/           alpha,  A+TS*(lda*i+k), lda,
                    /**/                   B+TS*(ldb*j+k), ldb,
                    /**/           beta_l, C+TS*(ldc*j+i), ldc       );
            }

            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_avx2_multirow(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           TS, TS, kK,
                /**/           alpha,  A+TS*(lda*i+Kk), lda,
                /**/                   B+TS*(ldb*j+Kk), ldb,
                /**/           beta_l, C+TS*(ldc*j+i), ldc       );
        }
    }

    // last block column
    for (int i = 0; i < Mm; ++i) {
        for (int k = 0; k < Kk; ++k) {
            const double beta_l = (k == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_avx2_multirow(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           TS, nN, TS,
                /**/           alpha,  A+TS*(lda*i+k), lda,
                /**/                   B+TS*(ldb*Nn+k), ldb,
                /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
        }
        const double beta_l = (Kk == 0) ? beta : 1.0;
        neon_cblas_dgemm_transA_avx2_multirow(
            /**/           layout,
            /**/           TransA, TransB,
            /**/           TS, nN, kK,
            /**/           alpha,  A+TS*(lda*i+Kk), lda,
            /**/                   B+TS*(ldb*Nn+Kk), ldb,
            /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
    }

    // last block row
    for (int j = 0; j < Nn; ++j) {
        for (int k = 0; k < Kk; ++k) {
            const double beta_l = (k == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_avx2_multirow(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           mM, TS, TS,
                /**/           alpha,  A+TS*(lda*Mm+k), lda,
                /**/                   B+TS*(ldb*j+k), ldb,
                /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
        }
        const double beta_l = (Kk == 0) ? beta : 1.0;
        neon_cblas_dgemm_transA_avx2_multirow(
            /**/           layout,
            /**/           TransA, TransB,
            /**/           mM, TS, kK,
            /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
            /**/                   B+TS*(ldb*j+Kk), ldb,
            /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
    }

    // last block right bottom corner
    for (int k = 0; k < Kk; ++k) {
        const double beta_l = (k == 0) ? beta : 1.0;
        neon_cblas_dgemm_transA_avx2_multirow(
            /**/           layout,
            /**/           TransA, TransB,
            /**/           mM, nN, TS,
            /**/           alpha,  A+TS*(lda*Mm+k), lda,
            /**/                   B+TS*(ldb*Nn+k), ldb,
            /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
    }
    const double beta_l = (Kk == 0) ? beta : 1.0;
    neon_cblas_dgemm_transA_avx2_multirow(
        /**/           layout,
        /**/           TransA, TransB,
        /**/           mM, nN, kK,
        /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
        /**/                   B+TS*(ldb*Nn+Kk), ldb,
        /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
}


void neon_cblas_dgemm_transA_tiled_L2(
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

    const int TS = neon_get_tile_size_L2();

    const int Mm = M / TS;
    const int mM = M % TS;
    //const int MM = M - mM;

    const int Nn = N / TS;
    const int nN = N % TS;
    //const int NN = N - nN;

    const int Kk = K / TS;
    const int kK = K % TS;
    //const int KK = K - kK;

    for (int j = 0; j < Nn; ++j) {
        for (int i = 0; i < Mm; ++i) {
            for (int k = 0; k < Kk; ++k) {
                const double beta_l = (k == 0) ? beta : 1.0;
                neon_cblas_dgemm_transA_tiled_L1(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           TS, TS, TS,
                    /**/           alpha,  A+TS*(lda*i+k), lda,
                    /**/                   B+TS*(ldb*j+k), ldb,
                    /**/           beta_l, C+TS*(ldc*j+i), ldc       );
            }

            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_tiled_L1(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           TS, TS, kK,
                /**/           alpha,  A+TS*(lda*i+Kk), lda,
                /**/                   B+TS*(ldb*j+Kk), ldb,
                /**/           beta_l, C+TS*(ldc*j+i), ldc       );
        }
    }

    // last block column
    for (int i = 0; i < Mm; ++i) {
        for (int k = 0; k < Kk; ++k) {
            const double beta_l = (k == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_tiled_L1(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           TS, nN, TS,
                /**/           alpha,  A+TS*(lda*i+k), lda,
                /**/                   B+TS*(ldb*Nn+k), ldb,
                /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
        }
        const double beta_l = (Kk == 0) ? beta : 1.0;
        neon_cblas_dgemm_transA_tiled_L1(
            /**/           layout,
            /**/           TransA, TransB,
            /**/           TS, nN, kK,
            /**/           alpha,  A+TS*(lda*i+Kk), lda,
            /**/                   B+TS*(ldb*Nn+Kk), ldb,
            /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
    }

    // last block row
    for (int j = 0; j < Nn; ++j) {
        for (int k = 0; k < Kk; ++k) {
            const double beta_l = (k == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_tiled_L1(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           mM, TS, TS,
                /**/           alpha,  A+TS*(lda*Mm+k), lda,
                /**/                   B+TS*(ldb*j+k), ldb,
                /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
        }
        const double beta_l = (Kk == 0) ? beta : 1.0;
        neon_cblas_dgemm_transA_tiled_L1(
            /**/           layout,
            /**/           TransA, TransB,
            /**/           mM, TS, kK,
            /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
            /**/                   B+TS*(ldb*j+Kk), ldb,
            /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
    }

    // last block right bottom corner
    for (int k = 0; k < Kk; ++k) {
        const double beta_l = (k == 0) ? beta : 1.0;
        neon_cblas_dgemm_transA_tiled_L1(
            /**/           layout,
            /**/           TransA, TransB,
            /**/           mM, nN, TS,
            /**/           alpha,  A+TS*(lda*Mm+k), lda,
            /**/                   B+TS*(ldb*Nn+k), ldb,
            /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
    }
    const double beta_l = (Kk == 0) ? beta : 1.0;
    neon_cblas_dgemm_transA_tiled_L1(
        /**/           layout,
        /**/           TransA, TransB,
        /**/           mM, nN, kK,
        /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
        /**/                   B+TS*(ldb*Nn+Kk), ldb,
        /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
}

void neon_cblas_dgemm_transA_tiled_L3(
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

    const int TS = neon_get_tile_size_L3();

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
                    neon_cblas_dgemm_transA_tiled_L1(
                        /**/           layout,
                        /**/           TransA, TransB,
                        /**/           TS, TS, TS,
                        /**/           alpha,  A+TS*(lda*i+k), lda,
                        /**/                   B+TS*(ldb*j+k), ldb,
                        /**/           beta_l, C+TS*(ldc*j+i), ldc       );
                }

                const double beta_l = (Kk == 0) ? beta : 1.0;
                neon_cblas_dgemm_transA_tiled_L1(
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
                neon_cblas_dgemm_transA_tiled_L1(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           TS, nN, TS,
                    /**/           alpha,  A+TS*(lda*i+k), lda,
                    /**/                   B+TS*(ldb*Nn+k), ldb,
                    /**/           beta_l, C+TS*(ldc*Nn+i), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_tiled_L1(
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
                neon_cblas_dgemm_transA_tiled_L1(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           mM, TS, TS,
                    /**/           alpha,  A+TS*(lda*Mm+k), lda,
                    /**/                   B+TS*(ldb*j+k), ldb,
                    /**/           beta_l, C+TS*(ldc*j+Mm), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_tiled_L1(
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
                neon_cblas_dgemm_transA_tiled_L1(
                    /**/           layout,
                    /**/           TransA, TransB,
                    /**/           mM, nN, TS,
                    /**/           alpha,  A+TS*(lda*Mm+k), lda,
                    /**/                   B+TS*(ldb*Nn+k), ldb,
                    /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
            }
            const double beta_l = (Kk == 0) ? beta : 1.0;
            neon_cblas_dgemm_transA_tiled_L1(
                /**/           layout,
                /**/           TransA, TransB,
                /**/           mM, nN, kK,
                /**/           alpha,  A+TS*(lda*Mm+Kk), lda,
                /**/                   B+TS*(ldb*Nn+Kk), ldb,
                /**/           beta_l, C+TS*(ldc*Nn+Mm), ldc       );
        }
    } // end pragma omp parallel
}
