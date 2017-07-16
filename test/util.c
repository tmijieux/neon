#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>

#include "util.h"

void neon_error(const char *file, const char *function, int line, const char *msgfmt, ...)
{
    va_list ap;
    int length = 1+snprintf(NULL, 0, "NEON ERROR %s:%s:%d: ", file, function, line);

    va_start(ap, msgfmt);
    length += 1+vsnprintf(NULL, 0, msgfmt, ap);
    va_end(ap);

    char *buf = malloc(length+1);
    int l = snprintf(buf, length, "NEON ERROR %s:%s:%d: ", file, function, line);
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
