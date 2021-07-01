CC := /opt/amazon/openmpi/bin/mpicc

NCCL_INC := /usr/local/cuda/include
LD_NCCL_LIB := /usr/local/cuda/lib

all: send_recv elastic_test allreduce

send_recv: send_recv.o
	$(CC) send_recv.o -lcudart -lnccl -lstdc++ -lm -L$(LD_NCCL_LIB) -L/usr/local/cuda/lib64/ -o send_recv.bin

send_recv.o: send_recv.cc
	$(CC) -I/usr/local/cuda/include -I$(NCCL_INC) -c send_recv.cc

elastic_test: elastic_test.o sdcc_tcp.o
	$(CC) elastic_test.o sdcc_tcp.o -lstdc++ -lcudart -L$(LD_NCCL_LIB) -L/usr/local/cuda/lib64/ -lnccl  -o elastic_test.bin 

elastic_test.o: elastic_test.cc
	$(CC) -g -std=c++14 -I/usr/local/cuda/include -I$(NCCL_INC) -c elastic_test.cc

sdcc_tcp.o: sdcc_tcp.cc
	$(CC) -c sdcc_tcp.cc 

allreduce: allreduce.o
	$(CC) allreduce.o -lcudart -lnccl -lstdc++ -lm -L$(LD_NCCL_LIB) -L/usr/local/cuda/lib64/ -o allreduce.bin

allreduce.o: hierarchy_allreduce.cc
	$(CC) -std=c++14 -I/usr/local/cuda/include -I$(NCCL_INC) -c hierarchy_allreduce.cc -o allreduce.o