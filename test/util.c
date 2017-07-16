#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>

#include <assert.h>
#include <complex.h>

#include <time.h>

#include "util.h"

void neon_error(const char *file, const char *function, int line, const char *msgfmt, ...)
{
    va_list ap;
    int length = 1+snprintf(NULL, 0, "NEON ERROR %s: %s:%d: ", function, file, line);

    va_start(ap, msgfmt);
    length += 1+vsnprintf(NULL, 0, msgfmt, ap);
    va_end(ap);

    char *buf = malloc(length+1);
    int l = snprintf(buf, length, "NEON ERROR %s: %s:%d: ", function, file, line);
    va_start(ap, msgfmt);
    vsnprintf(buf+l, length-l, msgfmt, ap);
    va_end(ap);

    puts(buf);
    free(buf);

    #ifdef NEON_DEBUG_MODE
    abort();
    #endif
}

void neon_fatal_error(const char *file, const char *function, int line, const char *msgfmt, ...)
{
    va_list ap;
    int length = 1+snprintf(NULL, 0, "NEON FATAL ERROR %s:%s:%d: ", file, function, line);

    va_start(ap, msgfmt);
    length += 1+vsnprintf(NULL, 0, msgfmt, ap);
    va_end(ap);

    char *buf = malloc(length+1);
    int l = snprintf(buf, length, "NEON FATAL ERROR %s:%s:%d: ", file, function, line);
    va_start(ap, msgfmt);
    vsnprintf(buf+l, length-l, msgfmt, ap);
    va_end(ap);

    puts(buf);
    #ifdef NEON_DEBUG_MODE
    abort();
    #endif
    exit(EXIT_FAILURE);
}

void *alloc_matrix(int m, int n, size_t size, int ld)
{
    assert( ld >= m );
    return calloc(ld*n, size);
}

float *salloc_matrix(int m, int n, int ld)
{
    return alloc_matrix(m, n, sizeof(float), ld);
}

double *dalloc_matrix(int m, int n, int ld)
{
    return alloc_matrix(m, n, sizeof(double), ld);
}

float _Complex *calloc_matrix(int m, int n, int ld)
{
    return alloc_matrix(m, n, sizeof(float _Complex), ld);
}

double _Complex *zalloc_matrix(int m, int n, int ld)
{
    return alloc_matrix(m, n, sizeof(double _Complex), ld);
}

float random_float_value()
{
    return ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
}

float random_double_value()
{
    return ((double) rand() / RAND_MAX) * 2.0 - 1.0;
}

void randomizer_initialize()
{
    int dummy;
    srand(time(NULL) + (int)(uintptr_t)(&dummy));
}

void srandomize_matrix(int M, int N, int LD, float *A)
{
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            A[j*LD+i] = random_float_value();
        }
    }
}

void drandomize_matrix(int M, int N, int LD, double *A)
{
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            A[j*LD+i] = random_double_value();
        }
    }
}

void crandomize_matrix(int M, int N, int LD, float _Complex *A)
{
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            A[j*LD+i] = random_float_value() + random_float_value() * I;
        }
    }
}

void zrandomize_matrix(int M, int N, int LD, double _Complex *A)
{
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            A[j*LD+i] = random_double_value() + random_double_value() * I;
        }
    }
}

void display_dmatrix(int M, int N, int LD, double *A)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%+7.3e ", A[j*LD+i]);
        }
        printf("\n");
    }
    printf("\n");
}
