#!/bin/bash
#SBATCH -J nsga
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -t 7-00:00:00

python msunas.py \
    --dataset cifar10 --data ../data \
    --save search-TEST \
    --sec_obj flops \
    --n_gpus 1 \
    --supernet_path ./supernets/ofa_mbv3_d234_e346_k357_w1.0 \
    --vld_size 5000 \
    --n_epochs 5