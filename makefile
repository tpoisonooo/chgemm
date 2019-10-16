OLD  := MMult_4x8_21
NEW  := MMult_4x8_22

#
# sample makefile
#

CC         := gcc 
LINKER     := $(CC)
#CFLAGS     := -O0 -g -Wall
CFLAGS     := -O3 -g -Wall 
#LDFLAGS    := -lm

UTIL       := copy_matrix.o \
              compare_matrices.o \
              random_matrix.o \
              dclock.o \
              REF_MMult.o \
              print_matrix.o \
              kernel_m4n4k16.o \
              reorder_a.o \
              reorder_b.o \
              int8kernel_m4.o \
              int8kernel_m2.o \
              int8kernel_m1.o

TEST_OBJS  := test_MMult.o $(NEW).o 

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
%.o: %.S
	$(CC) $(CFLAGS) -c $< -o $@

all: 
	make clean;
	make test_MMult.x

test_MMult.x: $(TEST_OBJS) $(UTIL) parameters.h
	$(LINKER) $(TEST_OBJS) $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 

run:	
	make all
	export OMP_NUM_THREADS=1
	export GOTO_NUM_THREADS=1
	echo "version = '$(NEW)';" > output_$(NEW).m
	./test_MMult.x >> output_$(NEW).m
	cp output_$(OLD).m output_old.m
	cp output_$(NEW).m output_new.m

clean:
	rm -f *.o *~ core *.x *.m

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png
