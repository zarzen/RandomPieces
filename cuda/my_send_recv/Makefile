
cc_args = -std=c++14
CC = nvcc $(cc_args)

all: tests benchmark

tests: test_kernels.o kernels.o
	$(CC) -o test_kernels.bin -lcudart test_kernels.o kernels.o 

benchmark: benchmark.o sendrecv.o kernels.o
	$(CC) -o benchmark.bin -lcudart sendrecv.o kernels.o benchmark.o

test_kernels.o:
	$(CC) -o test_kernels.o -c test_kernels.cc 

benchmark.o:
	$(CC) -o benchmark.o -c benchmark.cc 

kernels.o:
	$(CC) -o kernels.o -c kernels.cu 

sendrecv.o:
	$(CC) -o sendrecv.o -c sendrecv.cc 

clean:
	rm *.bin 
	rm *.o