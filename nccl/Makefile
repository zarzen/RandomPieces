CC := /opt/amazon/openmpi/bin/mpicc

all: send_recv elastic_test

send_recv: send_recv.o
	$(CC) send_recv.o -lcudart -lnccl -lstdc++ -lm -L/home/ubuntu/anaconda3/envs/horovod_dev/lib -L/usr/local/cuda/lib64/ -o send_recv.bin

send_recv.o: send_recv.cc
	$(CC) -I/usr/local/cuda/include -I/home/ubuntu/anaconda3/envs/horovod_dev/include/ -c send_recv.cc

elastic_test: elastic_test.o sdcc_tcp.o
	$(CC) elastic_test.o sdcc_tcp.o -lstdc++ -lcudart -L/home/ubuntu/anaconda3/envs/horovod_dev/lib -L/usr/local/cuda/lib64/ -lnccl  -o elastic_test.bin 

elastic_test.o: elastic_test.cc
	$(CC) -g -std=c++14 -I/usr/local/cuda/include -I/home/ubuntu/anaconda3/envs/horovod_dev/include/ -c elastic_test.cc

sdcc_tcp.o: sdcc_tcp.cc
	$(CC) -c sdcc_tcp.cc 