#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>

//#include "../cblas.h"
#include "mkl.h"
#include "omp.h"
#include "util.h"
#include "timer.h"

#include "../dgemm.h"

#define GEMM_ADD(M_, N_, K_) ((double)(M_) * (double)(N_) * (double)(K_))
#define GEMM_MUL(M_, N_, K_) ((double)(M_) * (double)(N_) * (double)(K_))

/********** NEON sequential tile ************************/


void time_neon_dgemm_reference(const int N)
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
    neon_cblas_dgemm_transA_reference( CblasColMajor, CblasTrans, CblasNoTrans,
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
    printf("%d,-1,1,%g,%g,%g,neon_dgemm_reference\n", N, length, gemm_flops, gflops_s);
}

void time_neon_dgemm_avx2(const int N)
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
    neon_cblas_dgemm_transA_avx2( CblasColMajor, CblasTrans, CblasNoTrans,
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
    printf("%d,-1,1,%g,%g,%g,neon_dgemm_avx2\n", N, length, gemm_flops, gflops_s);
}

/********** NEON tiling + parallel implementation ************************/

void time_neon_dgemm_tiled(const int N, int num_threads)
{
    double *A = dalloc_matrix(N, N, N);
    double *B = dalloc_matrix(N, N, N);
    double *C = dalloc_matrix(N, N, N);
    double alpha = 1.0;
    double beta = 0.0;
    // fprintf(stderr, "Allocation done.\n");

    struct timer timer;
    timer_init(&timer);

    omp_set_num_threads(num_threads);

    timer_start(&timer);
    neon_cblas_dgemm_transA_tiled( CblasColMajor, CblasTrans, CblasNoTrans,
                                   N, N, N,
                                   alpha, A, N,
                                   /**/   B, N,
                                   beta,  C, N    );
    timer_stop(&timer);

    // fprintf(stderr, "dgemm done.\n");
    free(A); free(B); free(C);

    const double length = timer_get_length(&timer);
    const double gemm_flops = GEMM_ADD(N, N, N) + GEMM_MUL(N, N, N);
    const double gflops_s = gemm_flops / (length*1000000000.0);
    const int TS = neon_get_tile_size();
    printf("%d,%d,%d,%g,%g,%g,neon_dgemm_tiled\n",
           N, TS, num_threads, length, gemm_flops, gflops_s);
}

void time_neon_dgemm_tiled_task(const int N, int num_threads)
{
    double *A = dalloc_matrix(N, N, N);
    double *B = dalloc_matrix(N, N, N);
    double *C = dalloc_matrix(N, N, N);
    double alpha = 1.0;
    double beta = 0.0;
    // fprintf(stderr, "Allocation done.\n");

    struct timer timer;
    timer_init(&timer);

    omp_set_num_threads(num_threads);

    timer_start(&timer);
    neon_cblas_dgemm_transA_tiled_task( CblasColMajor, CblasTrans, CblasNoTrans,
                                        N, N, N,
                                        alpha, A, N,
                                        /**/   B, N,
                                        beta,  C, N    );
    timer_stop(&timer);

    // fprintf(stderr, "dgemm done.\n");
    free(A); free(B); free(C);

    const double length = timer_get_length(&timer);
    const double gemm_flops = GEMM_ADD(N, N, N) + GEMM_MUL(N, N, N);
    const double gflops_s = gemm_flops / (length*1000000000.0);
    const int TS = neon_get_tile_size();
    printf("%d,%d,%d,%g,%g,%g,neon_dgemm_tiled_task\n",
           N, TS, num_threads, length, gemm_flops, gflops_s);
}

/********** PURE MKL *************************************/

void time_mkl_dgemm(const int N, int num_threads)
{
    double *A = dalloc_matrix(N, N, N);
    double *B = dalloc_matrix(N, N, N);
    double *C = dalloc_matrix(N, N, N);
    double alpha = -2.7;
    double beta = 3.4;
    // fprintf(stderr, "Allocation done.\n");

    drandomize_matrix(N, N, N, A);
    drandomize_matrix(N, N, N, B);
    drandomize_matrix(N, N, N, C);

    struct timer timer;
    timer_init(&timer);
    mkl_set_num_threads(num_threads);

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
    printf("%d,-1,%d,%g,%g,%g,mkl_dgemm\n",
           N, num_threads, length, gemm_flops, gflops_s);
}

/********** NEON tiling + MKL  *************************************/

void time_neon_dgemm_tiled_mkl(const int N, int num_threads)
{
    double *A = dalloc_matrix(N, N, N);
    double *B = dalloc_matrix(N, N, N);
    double *C = dalloc_matrix(N, N, N);
    double alpha = 1.0;
    double beta = 0.0;
    // fprintf(stderr, "Allocation done.\n");

    omp_set_num_threads(num_threads);

    struct timer timer;
    timer_init(&timer);

    timer_start(&timer);
    neon_cblas_dgemm_transA_tiled_plus_mkl( CblasColMajor, CblasTrans, CblasNoTrans,
                                            N, N, N,
                                            alpha, A, N,
                                            /**/   B, N,
                                            beta,  C, N    );
    timer_stop(&timer);

    // fprintf(stderr, "dgemm done.\n");
    free(A); free(B); free(C);

    const double length = timer_get_length(&timer);
    const double gemm_flops = GEMM_ADD(N, N, N) + GEMM_MUL(N, N, N);
    const double gflops_s = gemm_flops / (length*1000000000.0);
    const int TS = neon_get_tile_size();
    printf("%d,%d,%d,%g,%g,%g,neon_dgemm_tiled_mkl\n",
           N, TS, num_threads, length, gemm_flops, gflops_s);
}

void time_neon_dgemm_tiled_task_mkl(const int N, int num_threads)
{
    double *A = dalloc_matrix(N, N, N);
    double *B = dalloc_matrix(N, N, N);
    double *C = dalloc_matrix(N, N, N);
    double alpha = 1.0;
    double beta = 0.0;
    // fprintf(stderr, "Allocation done.\n");

    struct timer timer;
    timer_init(&timer);

    omp_set_num_threads(num_threads);

    timer_start(&timer);
    neon_cblas_dgemm_transA_tiled_task_mkl( CblasColMajor, CblasTrans, CblasNoTrans,
                                            N, N, N,
                                            alpha, A, N,
                                            /**/   B, N,
                                            beta,  C, N    );
    timer_stop(&timer);

    // fprintf(stderr, "dgemm done.\n");
    free(A); free(B); free(C);

    const double length = timer_get_length(&timer);
    const double gemm_flops = GEMM_ADD(N, N, N) + GEMM_MUL(N, N, N);
    const double gflops_s = gemm_flops / (length*1000000000.0);
    const int TS = neon_get_tile_size();
    printf("%d,%d,%d,%g,%g,%g,neon_dgemm_tiled_task_mkl\n",
           N, TS, num_threads, length, gemm_flops, gflops_s);
}

int main(int argc, char *argv[])
{
    fprintf(stderr, "argc=%d, argv=[ ", argc);
    for (int i = 0; argv[i] != NULL; ++i) {
        fprintf(stderr, "\"%s\", ", argv[i]);
    }
    fprintf(stderr, "NULL ]\n");

    printf("N,TS,num_thread,time,flops,gflops_s,kernel\n");
    for (int k = 0; k < 3; ++k) {

        for (int i = 100; i < 2000; i += 100) {
            time_neon_dgemm_reference(i);
        }
        for (int i = 100; i < 2000; i += 100) {
            time_neon_dgemm_avx2(i);
        }

        /********** NEON tiling + parallel implementation ***********/

        for (int i = 100; i < 2000; i += 100) {
            time_neon_dgemm_tiled(i, 1);
        }
        for (int i = 100; i < 2000; i += 100) {
            time_neon_dgemm_tiled(i, 2);
        }

        /* for (int i = 100; i < 2000; i += 100) { */
        /*     time_neon_dgemm_tiled_task(i, 1); */
        /* } */
        /* for (int i = 100; i < 2000; i += 100) { */
        /*     time_neon_dgemm_tiled_task(i, 2); */
        /* } */

        /********** PURE MKL ************************/
        for (int i = 100; i < 2000; i += 100) {
            time_mkl_dgemm(i, 1);
        }
        for (int i = 100; i < 2000; i += 100) {
            time_mkl_dgemm(i, 2);
        }

        /********** NEON tiling + MKL ************************/
        /* for (int i = 100; i < 2000; i += 100) { */
        /*     time_neon_dgemm_tiled_mkl(i, 1); */
        /* } */
        /* for (int i = 100; i < 2000; i += 100) { */
        /*     time_neon_dgemm_tiled_mkl(i, 2); */
        /* } */

        /* for (int i = 100; i < 2000; i += 100) { */
        /*     time_neon_dgemm_tiled_task_mkl(i, 1); */
        /* } */
        /* for (int i = 100; i < 2000; i += 100) { */
        /*     time_neon_dgemm_tiled_task_mkl(i, 2); */
        /* } */
    }
    return 0;
}
