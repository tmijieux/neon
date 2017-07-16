CC=gcc
CFLAGS=-Wall -Wextra -Werror -std=c11 -O3 -fPIE -march=native -mtune=native -pipe
LIBS=-lm -L.. -lcblas -L${MKLROOT}/lib/intel64 -lmkl_core -lmkl_sequential -lmkl_intel_lp64
LDFLAGS=-O3 -Wl,-rpath,'$$ORIGIN/..:${MKLROOT}/lib/intel64'
TIME_SRC= 	time_gemm.c \
		util.c \
		timer.c \

TEST_SRC= 	test_gemm.c \
		util.c \
		timer.c \

TIME_OBJ=$(TIME_SRC:.c=.o)
TEST_OBJ=$(TEST_SRC:.c=.o)

DEPS=$(wildcard *.dep)

ifdef DEBUG
CFLAGS+=-O0 -ggdb -DNEON_DEBUG_MODE=1
LDFLAGS+=-O0 -ggdb
endif

all: time_gemm test_gemm

-include $(DEPS)

%.o: %.s
	$(AS) -c -o $@ $<

%.o: %.c
	@$(CC) -MM $(CFLAGS) -c -o $*.dep $<
	$(CC) $(CFLAGS) -c -o $@ $<

time_gemm: $(TIME_OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)

test_gemm: $(TEST_OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	$(RM) $(OBJ) *.o

dep_clean: clean
	$(RM) *.dep

fullclean: clean dep_clean
	$(RM) time_gemm
