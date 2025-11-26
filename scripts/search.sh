#!/bin/bash

python msunas.py --sec_obj flops \
    --n_gpus 1 --gpu 1 --n_workers 4 --n_epochs 5 \
    --dataset cifar10 --n_classes 10 \
    --data data \
    --predictor as --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
    --save search-TEST --iterations 30 --vld_size 5000 \
    --seed 42