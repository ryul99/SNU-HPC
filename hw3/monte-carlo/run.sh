#!/bin/bash

: ${NODES:=2}

salloc -N $NODES --partition shpc --exclusive      \
         mpirun --bind-to none -mca btl ^openib -npernode 1 \
         numactl --physcpubind 0-31                         \
         ./main $@
