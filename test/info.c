#include <stdio.h>
#include <stdlib.h>


#include "cache_util.h"

int main(int argc, char *argv[])
{
    fprintf(stderr, "argc=%d, argv=[ ", argc);
    for (int i = 0; argv[i] != NULL; ++i) {
        fprintf(stderr, "\"%s\", ", argv[i]);
    }
    fprintf(stderr, "NULL ]\n");

    neon_print_cache_size();

    return 0;
}

