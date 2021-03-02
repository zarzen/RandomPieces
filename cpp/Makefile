CC_ARGS = -std=c++14 -DLOG_LEVEL=0x03 -g -O0
CC = g++ $(CC_ARGS)

all: producer_consumer

producer_consumer: tcp_producer_consumer.cc
	$(CC) -o producer_consumer.bin tcp_producer_consumer.cc -lpthread