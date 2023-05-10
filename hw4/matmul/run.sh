#!/bin/bash

srun --nodes=1 --exclusive --partition=shpc --gres=gpu:4 numactl --physcpubind 0-31 ./main $@
