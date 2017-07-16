CBLAS_VERSION_MAJOR=0
CBLAS_VERSION_MINOR=0
CBLAS_VERSION_PATCH=1
CBLAS_VERSION=$(CBLAS_VERSION_MAJOR).$(CBLAS_VERSION_MINOR).$(CBLAS_VERSION_PATCH)

CC=gcc
LN=ln -f
CFLAGS=-Wall -Wextra -pedantic -Werror -std=c11 -O3 -fPIC -march=native -mtune=native -pipe -fopenmp
LIBS=-lm
LDFLAGS=-shared -O3 -Wl,-soname,"libcblas.so.$(CBLAS_VERSION_MAJOR)" -Wl,-no-undefined -fopenmp
SRC= 	zgemm.c \
	dgemm.c

OBJ2=$(SRC:.c=.o)
OBJ=$(OBJ2:.s=.o)
DEPS=$(wildcard *.dep)

ifdef DEBUG
CFLAGS+=-O0 -ggdb
LDFLAGS+=-O0 -ggdb
endif

all: libcblas.so libcblas.a

-include $(DEPS)

%.o: %.s
	$(AS) -c -o $@ $<

%.o: %.c
	@$(CC) -MM $(CFLAGS) -c -o $*.dep $<
	$(CC) $(CFLAGS) -c -o $@ $<

libcblas.a: $(OBJ)
	ar cr $@ $^
	ranlib $@

libcblas.so: libcblas.so.$(CBLAS_VERSION_MAJOR)
	$(LN) -s $^ $@

libcblas.so.$(CBLAS_VERSION_MAJOR): libcblas.so.$(CBLAS_VERSION)
	$(LN) -s $^ $@

libcblas.so.$(CBLAS_VERSION): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	$(RM) $(OBJ) *.o

dep_clean: clean
	$(RM) *.dep

fullclean: clean dep_clean
	$(RM) libcblas.so.$(CBLAS_VERSION) libcblas.so.$(CBLAS_VERSION_MAJOR) libcblas.so
	$(RM) libcblas.a
