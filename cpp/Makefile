CC_ARGS = -std=c++14 -DLOG_LEVEL=0x02 -g -O0
CC = g++ $(CC_ARGS)

all: producer_consumer send_recv_ring parallel_background_traffic ctrl_test

producer_consumer: tcp_producer_consumer.cc
	$(CC) -o producer_consumer.bin tcp_producer_consumer.cc -lpthread

send_recv_ring: tcp.cpp send_recv_ring.cpp
	$(CC) -o send_recv.bin tcp.cpp send_recv_ring.cpp -lpthread

parallel_background_traffic: parallel_link_occupy.cc
	$(CC) -o background.bin parallel_link_occupy.cc -lpthread

ctrl_test: parallel_linkspeed.cc 
	$(CC) -o parallel_speed.bin parallel_linkspeed.cc -lpthread
