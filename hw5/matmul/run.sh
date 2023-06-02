#!/bin/bash

: ${NODES:=4}

salloc -N $NODES --partition shpc --exclusive --gres=gpu:4   \
  mpirun --bind-to none -mca btl ^openib -npernode 1         \
  numactl --physcpubind 0-31                                 \
  ./main $@
