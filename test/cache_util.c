#include <stdio.h>
#include <assert.h>

#include "cache_util.h"

#define b(val, base, end)                                       \
    ((val << (__WORDSIZE-end-1)) >> (__WORDSIZE-end+base-1))

#define GET_CACHE_DETAILS(level_, ways_, partitions_, line_size_, sets_) \
    do {                                                                \
        uint64_t eax_, ebx_, ecx_, edx_;                                \
        __asm__( "cpuid"                                                \
                 : "=a"(eax_), "=b"(ebx_), "=c"(ecx_), "=d"(edx_)       \
                 : "a"(4), "b"(0), "c"(level_), "d"(0));                \
        ways_ = b(ebx_, 22, 31) + 1;                                    \
        partitions_ = b(ebx_, 12, 21) + 1;                              \
        line_size_ = b(ebx_, 0, 11) + 1;                                \
        sets_ = ecx_ + 1;                                               \
    }while(0)

#define GETCACHESIZE(level_)                                            \
    ({                                                                  \
        uint64_t ways, partitions, line_size, sets;                     \
        GET_CACHE_DETAILS(level_, ways, partitions, line_size, sets);   \
        (ways * partitions * line_size * sets) / 1024;                  \
    })

#define PRINT_CACHE_DETAILS(level_)                                     \
    do {                                                                \
        uint64_t ways, partitions, line_size, sets;                     \
        GET_CACHE_DETAILS(level_, ways, partitions, line_size, sets);   \
        printf("line size: %lu\n", line_size);                          \
        printf("ways: %lu\n", ways);                                    \
        printf("sets: %lu\n", sets);                                    \
        printf("partitions: %lu\n\n", partitions);                      \
    }while(0)

void neon_print_cache_size(void)
{
    printf("L1 cache_size %luK\n", GETCACHESIZE(1));
    PRINT_CACHE_DETAILS(1);

    printf("L2 cache_size %luK\n", GETCACHESIZE(2));
    PRINT_CACHE_DETAILS(2);

    printf("L3 cache_size %luK\n", GETCACHESIZE(3));
    PRINT_CACHE_DETAILS(3);
}

uint64_t neon_get_cache_size(int64_t id)
{
    assert(id >= 1 && id <= 3);
    return GETCACHESIZE(id);
}
