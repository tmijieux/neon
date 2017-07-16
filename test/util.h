#ifndef NEON_UTIL_H
#define NEON_UTIL_H

#include <stdlib.h>

void *alloc_matrix(int m, int n, size_t size, int ld);
float *salloc_matrix(int m, int n, int ld);
double *dalloc_matrix(int m, int n, int ld);
float _Complex *calloc_matrix(int m, int n, int ld);
double _Complex *zalloc_matrix(int m, int n, int ld);

void neon_error(const char *file, const char *function, int line, const char *msgfmt, ...);
void neon_fatal_error(const char *file, const char *function, int line, const char *msgfmt, ...);

#define NEON_ERROR(msgfmt, ...)                                         \
    do {                                                                \
        neon_error(__FILE__,__func__, __LINE__, (msgfmt), ##__VA_ARGS__ ); \
    }while(0)

#define NEON_FATAL_ERROR(msgfmt, ...)                                   \
    do {                                                                \
        neon_fatal_error(__FILE__,__func__, __LINE__, (msgfmt), ##__VA_ARGS__ ); \
    }while(0)

#endif // NEON_UTIL_H
