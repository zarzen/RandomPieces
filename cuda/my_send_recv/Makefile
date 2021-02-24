
CC_ARGS = -std=c++14 -DLOG_LEVEL=0x03
CC = nvcc $(CC_ARGS)

all: tests benchmark

tests: test_kernels.o kernels.o sendrecv.o utils.o rendezvous.o connection.o
	$(CC) -o test_kernels.bin -lcudart test_kernels.o kernels.o sendrecv.o utils.o rendezvous.o connection.o

benchmark: benchmark.o sendrecv.o kernels.o utils.o rendezvous.o connection.o
	$(CC) -o benchmark.bin -lcudart sendrecv.o kernels.o benchmark.o utils.o rendezvous.o connection.o

test_kernels.o: test_kernels.cc
	$(CC) -o test_kernels.o -c test_kernels.cc 

benchmark.o: benchmark.cc 
	$(CC) -o benchmark.o -c benchmark.cc 

kernels.o:
	$(CC) -o kernels.o -c kernels.cu 

sendrecv.o: sendrecv.cc
	$(CC) -o sendrecv.o -c sendrecv.cc 

utils.o: utils.cc
	$(CC) -o utils.o -c utils.cc 

rendezvous.o: rendezvous.cc
	$(CC) -o rendezvous.o -c rendezvous.cc

connection.o: connection.cc
	$(CC) -o connection.o -c connection.cc

clean:
	rm *.o
	rm *.bin 
	