#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>

#include "../cblas.h"
#include "util.h"
#include "timer.h"

extern void neon_cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                             CBLAS_TRANSPOSE TransB, const int M, const int N,
                             const int K, const double alpha, const double *A,
                             const int lda, const double *B, const int ldb,
                             const double beta, double *C, const int ldc);
extern void neon_cblas_zgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                             CBLAS_TRANSPOSE TransB, const int M, const int N,
                             const int K, const void *alpha, const void *A,
                             const int lda, const void *B, const int ldb,
                             const void *beta, void *C, const int ldc);


extern void neon_cblas_dgemm_transA_avx2_unirow( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                                                 CBLAS_TRANSPOSE TransB, const int M, const int N,
                                                 const int K, const double alpha, const double *A,
                                                 const int lda, const double *B, const int ldb,
                                                 const double beta, double *C, const int ldc);


extern void neon_cblas_dgemm_transA_avx2_multirow( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                                                   CBLAS_TRANSPOSE TransB, const int M, const int N,
                                                   const int K, const double alpha, const double *A,
                                                   const int lda, const double *B, const int ldb,
                                                   const double beta, double *C, const int ldc);

void neon_cblas_dgemm_transA_tiled( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                                    CBLAS_TRANSPOSE TransB, const int M, const int N,
                                    const int K, const double alpha, const double *A,
                                    const int lda, const double *B, const int ldb,
                                    const double beta, double *C, const int ldc    );

#define GEMM_ADD(M_, N_, K_) ((double)(M_) * (double)(N_) * (double)(K_))
#define GEMM_MUL(M_, N_, K_) ((double)(M_) * (double)(N_) * (double)(K_))

void time_zgemm(const int N)
{
    double _Complex *A = zalloc_matrix(N, N, N);
    double _Complex *B = zalloc_matrix(N, N, N);
    double _Complex *C = zalloc_matrix(N, N, N);
    double _Complex alpha = 1.0 + 0.0*I;
    double _Complex beta = 0.0 + 0.0*I;
    // fprintf(stderr, "Allocation done.\n");

    struct timer timer;
    timer_init(&timer);

    timer_start(&timer);
    neon_cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                     N, N, N,
                     &alpha, A, N,
                     /**/    B, N,
                     &beta,  C, N    );
    timer_stop(&timer);

    free(A); free(B); free(C);
    // fprintf(stderr, "zgemm done.\n");

    double length = timer_get_length(&timer);
    double gemm_flops = 2.0*GEMM_ADD(N, N, N) + 6.0*GEMM_MUL(N, N, N);
    double gflops_s = gemm_flops / (length*1000000000.0);
    printf("%d,%g,%g,%g,zgemm\n", N, length, gemm_flops, gflops_s);
}


void time_neon_dgemm_multi(const int N)
{
    double *A = dalloc_matrix(N, N, N);
    double *B = dalloc_matrix(N, N, N);
    double *C = dalloc_matrix(N, N, N);
    double alpha = 1.0;
    double beta = 0.0;
    // fprintf(stderr, "Allocation done.\n");

    struct timer timer;
    timer_init(&timer);

    timer_start(&timer);
    neon_cblas_dgemm_transA_avx2_multirow( CblasColMajor, CblasTrans, CblasNoTrans,
                                           N, N, N,
                                           alpha, A, N,
                                           /**/   B, N,
                                           beta,  C, N    );
    timer_stop(&timer);

    // fprintf(stderr, "dgemm done.\n");
    free(A); free(B); free(C);

    double length = timer_get_length(&timer);
    double gemm_flops = GEMM_ADD(N, N, N) + GEMM_MUL(N, N, N);
    double gflops_s = gemm_flops / (length*1000000000.0);
    printf("%d,%g,%g,%g,neon_dgemm_multi\n", N, length, gemm_flops, gflops_s);
}

void time_neon_dgemm(const int N)
{
    double *A = dalloc_matrix(N, N, N);
    double *B = dalloc_matrix(N, N, N);
    double *C = dalloc_matrix(N, N, N);
    double alpha = 1.0;
    double beta = 0.0;
    // fprintf(stderr, "Allocation done.\n");

    struct timer timer;
    timer_init(&timer);

    timer_start(&timer);
    neon_cblas_dgemm_transA_avx2_unirow( CblasColMajor, CblasTrans, CblasNoTrans,
                                         N, N, N,
                                         alpha, A, N,
                                         /**/   B, N,
                                         beta,  C, N    );
    timer_stop(&timer);

    // fprintf(stderr, "dgemm done.\n");
    free(A); free(B); free(C);

    double length = timer_get_length(&timer);
    double gemm_flops = GEMM_ADD(N, N, N) + GEMM_MUL(N, N, N);
    double gflops_s = gemm_flops / (length*1000000000.0);
    printf("%d,%g,%g,%g,neon_dgemm\n", N, length, gemm_flops, gflops_s);
}


void time_neon_dgemm_tiled(const int N)
{
    double *A = dalloc_matrix(N, N, N);
    double *B = dalloc_matrix(N, N, N);
    double *C = dalloc_matrix(N, N, N);
    double alpha = 1.0;
    double beta = 0.0;
    // fprintf(stderr, "Allocation done.\n");

    struct timer timer;
    timer_init(&timer);

    timer_start(&timer);
    neon_cblas_dgemm_transA_tiled( CblasColMajor, CblasTrans, CblasNoTrans,
                                   N, N, N,
                                   alpha, A, N,
                                   /**/   B, N,
                                   beta,  C, N    );
    timer_stop(&timer);

    // fprintf(stderr, "dgemm done.\n");
    free(A); free(B); free(C);

    double length = timer_get_length(&timer);
    double gemm_flops = GEMM_ADD(N, N, N) + GEMM_MUL(N, N, N);
    double gflops_s = gemm_flops / (length*1000000000.0);
    printf("%d,%g,%g,%g,neon_dgemm_tiled\n", N, length, gemm_flops, gflops_s);
}

void time_mkl_dgemm(const int N)
{
    double *A = dalloc_matrix(N, N, N);
    double *B = dalloc_matrix(N, N, N);
    double *C = dalloc_matrix(N, N, N);
    double alpha = 1.0;
    double beta = 0.0;
    // fprintf(stderr, "Allocation done.\n");

    struct timer timer;
    timer_init(&timer);

    timer_start(&timer);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                N, N, N,
                alpha, A, N,
                /**/   B, N,
                beta,  C, N    );
    timer_stop(&timer);

    // fprintf(stderr, "dgemm done.\n");
    free(A); free(B); free(C);

    double length = timer_get_length(&timer);
    double gemm_flops = GEMM_ADD(N, N, N) + GEMM_MUL(N, N, N);
    double gflops_s = gemm_flops / (length*1000000000.0);
    printf("%d,%g,%g,%g,mkl_dgemm\n", N, length, gemm_flops, gflops_s);
}

int main(int argc, char *argv[])
{
    fprintf(stderr, "argc=%d, argv=[ ", argc);
    for (int i = 0; argv[i] != NULL; ++i) {
        fprintf(stderr, "\"%s\", ", argv[i]);
    }
    fprintf(stderr, "NULL ]\n");

    printf("N,time,flops,gflops_s,kernel\n");
    for (int k = 0; k < 3; ++k) {
        for (int i = 100; i < 1500; i += 100) {
            time_mkl_dgemm(i);
        }
        for (int i = 100; i < 1500; i += 100) {
            time_neon_dgemm(i);
        }
        for (int i = 100; i < 1500; i += 100) {
            time_neon_dgemm_multi(i);
        }
        for (int i = 100; i < 1500; i += 100) {
            time_neon_dgemm_tiled(i);
        }
    }
    return 0;
}
