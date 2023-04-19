#!/bin/bash

srun --nodes=1 --exclusive --partition=shpc numactl --physcpubind 0-63 ./main $@
