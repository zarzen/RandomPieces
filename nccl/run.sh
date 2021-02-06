#! /bin/bash
/opt/amazon/openmpi/bin/mpirun -np 2 -H \
    ip-172-31-88-106:1,ip-172-31-76-23:1 \
    -bind-to none -map-by slot \
    -x HOROVOD_HOSTS="172.31.88.106 172.31.76.23" \
    -x LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:/home/ubuntu/nccl/build:$LD_LIBRARY_PATH \
    -x HOROVOD_LOG_LEVEL=TRACE \
    -tag-output \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
    /home/ubuntu/RandomPieces/nccl/send_recv.bin