#! /bin/bash
/opt/amazon/openmpi/bin/mpirun -np 2 -H \
    ip-172-31-11-173:1,ip-172-31-7-143:1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:/home/ubuntu/nccl/build/lib:$LD_LIBRARY_PATH \
    -tag-output \
    -mca pml ob1 \
    -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
    /home/ubuntu/RandomPieces/nccl/send_recv.bin