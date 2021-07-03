#!/bin/bash

NP=8
HOSTS="172.31.6.51:4,172.31.24.30:4"
MPI_HOME="/opt/amazon/openmpi"
TEST_BIN="/home/ec2-user/RandomPieces/nccl/allreduce.bin"

MPI_BIN="${MPI_HOME}/bin/mpirun"
# LD_LIBRARY_PATH="${MPI_HOME}/lib":$LD_LIBRARY_PATH
LD_LIBRARY_PATH=/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH

cmd="${MPI_BIN} -np ${NP} \
        -H ${HOSTS} \
        -tag-output \
        -bind-to none -map-by slot \
        -x PATH \
        -x NCCL_SOCKET_IFNAME=eth0 \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_TREE_THRESHOLD=0 \
        -x LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
        -mca btl ^openib \
        -mca btl_tcp_if_exclude lo,docker0 \
        ${TEST_BIN} 4 1000000"

eval ${cmd}