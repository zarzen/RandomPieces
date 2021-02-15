#! /bin/bash

/opt/amazon/openmpi/bin/mpirun -np 2 -H \
    ip-172-31-88-106:1,ip-172-31-64-245:1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=DEBUG \
    -x LD_LIBRARY_PATH=/home/ubuntu/nccl/build/lib:/opt/amazon/openmpi/lib:/home/ubuntu/anaconda3/envs/horovod_orig/lib:$LD_LIBRARY_PATH \
    -tag-output \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
    /home/ubuntu/RandomPieces/nccl/send_recv.bin 1