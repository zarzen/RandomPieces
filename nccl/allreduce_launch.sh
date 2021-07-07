#!/bin/bash

NP=32
HOSTS="172.31.9.0:8,172.31.12.234:8,172.31.2.94:8,172.31.2.168:8"
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
        -x FI_EFA_USE_DEVICE_RDMA=1 \
        -x RDMAV_FORK_SAFE=1 \
        -x FI_PROVIDER=\"efa\" \
        -x NCCL_SOCKET_IFNAME=eth \
        -x NCCL_DEBUG=DEBUG \
        -x NCCL_ALGO=Tree \
        -x NCCL_MIN_NCHANNELS=8 \
        -x LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
        -mca btl ^openib \
        -mca pml ob1 \
        -mca btl_tcp_if_exclude lo,docker0 \
        ${TEST_BIN} 8 100000000"

eval ${cmd}