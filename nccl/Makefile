
send_recv: send_recv.o
	mpicc send_recv.o -lcudart -lnccl -L/home/ubuntu/anaconda3/envs/horovod_dev/lib -L/usr/local/cuda/lib64/ -o send_recv.bin

send_recv.o: send_recv.cc
	mpicc -I/usr/local/cuda/include -I/home/ubuntu/anaconda3/envs/horovod_dev/include/ -c send_recv.cc
