#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <math.h>

#include "../cblas.h"
#include "util.h"
#include "timer.h"

extern void neon_cblas_dgemm_transA_reference( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                                               CBLAS_TRANSPOSE TransB, const int M, const int N,
                                               const int K, const double alpha, const double *A,
                                               const int lda, const double *B, const int ldb,
                                               const double beta, double *C, const int ldc);

extern void neon_cblas_dgemm_transA_avx2_multirow( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                                                   CBLAS_TRANSPOSE TransB, const int M, const int N,
                                                   const int K, const double alpha, const double *A,
                                                   const int lda, const double *B, const int ldb,
                                                   const double beta, double *C, const int ldc);

extern void neon_cblas_dgemm_transA_avx2_unirow( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                                                 CBLAS_TRANSPOSE TransB, const int M, const int N,
                                                 const int K, const double alpha, const double *A,
                                                 const int lda, const double *B, const int ldb,
                                                 const double beta, double *C, const int ldc);

void neon_cblas_dgemm_transA_tiled( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                                    CBLAS_TRANSPOSE TransB, const int M, const int N,
                                    const int K, const double alpha, const double *A,
                                    const int lda, const double *B, const int ldb,
                                    const double beta, double *C, const int ldc    );

void neon_cblas_dgemm_transA_tiled_plus_mkl( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                                             CBLAS_TRANSPOSE TransB, const int M, const int N,
                                             const int K, const double alpha, const double *A,
                                             const int lda, const double *B, const int ldb,
                                             const double beta, double *C, const int ldc    );


#define GEMM_ADD(M_, N_, K_) ((double)(M_) * (double)(N_) * (double)(K_))
#define GEMM_MUL(M_, N_, K_) ((double)(M_) * (double)(N_) * (double)(K_))

void test_neon_dgemm(const int M, const int N, const int K)
{
    double *A = dalloc_matrix(K, M, K);
    double *B = dalloc_matrix(K, N, K);
    double *C1 = dalloc_matrix(M, N, M);
    double *C2 = dalloc_matrix(M, N, M);

    drandomize_matrix(K, M, K, A);
    drandomize_matrix(K, N, K, B);

    const double alpha = 1.0;
    const double beta = 0.0;

    struct timer timer;
    timer_init(&timer);

    timer_start(&timer);
    neon_cblas_dgemm_transA_tiled_plus_mkl( CblasColMajor, CblasTrans, CblasNoTrans,
                                            M, N, K,
                                            alpha, A, K,
                                            /**/   B, K,
                                            beta,  C1, M    );
    /* cblas_dgemm( CblasColMajor, CblasTrans, CblasNoTrans, */
    /*              M, N, K, */
    /*              alpha, A, K, */
    /*              /\**\/   B, K, */
    /*              beta,  C1, M    ); */
    neon_cblas_dgemm_transA_reference( CblasColMajor, CblasTrans, CblasNoTrans,
                                       M, N, K,
                                       alpha, A, K,
                                       /**/   B, K,
                                       beta,  C2, M   );

    /* display_dmatrix(N, N, N, C1); */
    /* display_dmatrix(N, N, N, C2); */

    double norm = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            C1[j*M+i] = (C1[j*M+i] - C2[j*M+i]) / C1[j*M+i];
            norm += C1[j*M+i] * C1[j*M+i];
        }
    }
    //display_dmatrix(N, N, N, C1);
    norm = sqrt(norm);
    timer_stop(&timer);

    free(A);
    free(B);
    free(C1);
    free(C2);

    const double length = timer_get_length(&timer);
    // const double sup = sqrt( M * N * K * 10e-26 ); // (10e-13)^2
    printf("%d,%d,%d,%g,%g,neon_dgemm\n", M, N, K, length, norm);
    // assert( norm / sup < 1e1 );
}

int main(int argc, char *argv[])
{
    randomizer_initialize();

    fprintf(stderr, "argc=%d, argv=[ ", argc);
    for (int i = 0; argv[i] != NULL; ++i) {
        fprintf(stderr, "\"%s\", ", argv[i]);
    }
    fprintf(stderr, "NULL ]\n");

    printf("M,N,K,test_time,diff_norm,kernel\n");
    /* for (int m = 10; m < 500; m += 13) { */
    /*     for (int n = 10; n < 500; n += 13) { */
    /*         for (int k = 10; k < 500; k += 13) { */
    for (int m = 10; m < 600; m += 71) {
        for (int n = 10; n < 600; n += 71) {
            for (int k = 10; k < 600; k += 71) {
                test_neon_dgemm(m, n, k);
            }
        }
    }
    return 0;
}
