#ifndef NEON_DGEMM_H
#define NEON_DGEMM_H


//#include "/opt/intel/mkl/include/mkl.h"
enum CBLAS_TRANSPOSE {
   CblasNoTrans=111,
   CblasTrans=112,
   CblasConjTrans=113,
   AtlasConj=114
};
typedef enum CBLAS_TRANSPOSE CBLAS_TRANSPOSE;
typedef enum CBLAS_LAYOUT {
    CblasRowMajor=101, 
    CblasColMajor=102
} CBLAS_LAYOUT;

void neon_set_tile_size(const int TS);
int neon_get_tile_size(void);

void neon_cblas_dgemm_reference(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc);

void neon_cblas_dgemm_avx2(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha_s, const double *restrict const A, const int lda,
    /**/                  const double *restrict const B, const int ldb,
    const double beta_s,        double *restrict const C, const int ldc  );

void neon_cblas_dgemm_transA_reference(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc);

void neon_cblas_dgemm_transA_avx2(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc   );

void neon_cblas_dgemm_transA_tiled(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc   );

void neon_cblas_dgemm_transA_tiled_task(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc   );

void neon_cblas_dgemm_transA_tiled_plus_mkl(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc   );

void neon_cblas_dgemm_transA_tiled_task_mkl(
    const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K,
    const double alpha, const double * restrict const A, const int lda,
    /**/                const double * restrict const B, const int ldb,
    const double beta,        double * restrict const C, const int ldc   );


#endif // NEON_DGEMM_H
