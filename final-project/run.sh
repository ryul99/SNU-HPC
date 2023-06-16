#!/bin/bash

: ${NODES:=4}

salloc -N $NODES --exclusive --partition=shpc --gres=gpu:4   --exclude a06                 \
  mpirun --bind-to none -mca btl ^openib -npernode 1                          \
  numactl --physcpubind 0-63                                                  \
  ./translator -n 1024 -v $@

  # /usr/local/cuda/bin/ncu --set full 
  # /usr/local/cuda/bin/nsys profile --gpu-metrics-device 0    \
  # --cudabacktrace=all --force-overwrite true -o profile_%q{OMPI_COMM_WORLD_RANK}.qdrep \
  # --trace=mpi,cuda            \