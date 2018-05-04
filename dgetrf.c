#include <assert.h>

#include "lapacke.h"
#include "cblas.h"

#define min( x, y)  ((x) < (y) ? (x) : (y))

lapack_int LAPACKE_dgetf2( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* ipiv )
{
    assert( matrix_layout == LAPACK_COL_MAJOR );
    (void) ipiv;

    for (int j = 0; j < min(m,n)-1; ++j) {
        cblas_dscal(m-1-j, 1.0/a[j*lda+j], &a[j*lda+j+1], 1);
        cblas_dger(LAPACK_COL_MAJOR, m-1-j, n-1-j,
                   1.0, &a[j*lda+j+1], 1,
                   &a[(j+1)*lda+j], lda,
                   &a[(j+1)*lda+(j+1)], lda);
    }
    return 0;
}


lapack_int LAPACKE_dgetrf( int matrix_layout, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* ipiv )
{
    assert( matrix_layout == LAPACK_COL_MAJOR );
    (void) ipiv;
    const lapack_int bs = 128;

    for (int j = 0; j < min(m,n)-1; ++j) {
        cblas_dscal(m-1-j, 1.0/a[j*lda+j], &a[j*lda+j+1], 1);
        cblas_dger(LAPACK_COL_MAJOR, m-1-j, n-1-j,
                   1.0, &a[j*lda+j+1], 1,
                   &a[(j+1)*lda+j], lda,
                   &a[(j+1)*lda+(j+1)], lda);
    }
    return 0;
}
