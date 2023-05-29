#!/bin/bash

srun --nodes=1 --exclusive --partition=shpc --gres=gpu:4 numactl --physcpubind 0-31 /usr/local/cuda/bin/nsys profile --gpu-metrics-device all --cudabacktrace=all ./main $@ 
srun --nodes=1 --exclusive --partition=shpc --gres=gpu:4 numactl --physcpubind 0-31 /usr/local/cuda/bin/ncu --set full ./main $@ 
