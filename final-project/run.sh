#!/bin/bash

: ${NODES:=1}

salloc -N $NODES --exclusive --partition=shpc --gres=gpu:4                    \
  mpirun --bind-to none -mca btl ^openib -npernode 1                          \
  numactl --physcpubind 0-63                                                  \
  ./translator -n 1024 -v $@

  # /usr/local/cuda/bin/ncu --set full 
  # /usr/local/cuda/bin/nsys profile --gpu-metrics-device 0    \
  # --cudabacktrace=all --force-overwrite true -o profile.qdrep \
  # --trace=mpi,cuda            \