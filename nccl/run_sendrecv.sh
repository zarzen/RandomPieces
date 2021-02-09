#! /bin/bash
/opt/amazon/openmpi/bin/mpirun -np 8 -H \
    ip-172-31-88-106:4,ip-172-31-76-23:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=DEBUG \
    -x LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:/home/ubuntu/anaconda3/envs/horovod_orig/lib:$LD_LIBRARY_PATH \
    -tag-output \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
    /home/ubuntu/RandomPieces/nccl/send_recv.bin