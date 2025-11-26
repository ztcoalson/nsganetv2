#!/bin/bash

# Poisons:

attack=rlf
p=1

trial=10
flops=211
r=192

python train_cifar.py \
    --data ./data \
    --model "$attack-$p%-trial-${trial}" \
    --model-config search/cifar10/$attack/$p%/trial-${trial}/net-flops@${flops}/net.config \
    --initial-checkpoint search/cifar10/$attack/$p%/trial-${trial}/net-flops@${flops}/net.inherited \
    --img-size $r \
    --drop 0.2 --drop-path 0.2 \
    --cutout --autoaugment --save

# Clean:

# trial=10
# flops=211
# r=192

# python train_cifar.py \
#     --data ./data \
#     --model "clean-trial-${trial}" \
#     --model-config search/cifar10/clean/trial-${trial}/net-flops@${flops}/net.config \
#     --initial-checkpoint search/cifar10/clean/trial-${trial}/net-flops@${flops}/net.inherited \
#     --img-size $r \
#     --drop 0.2 --drop-path 0.2 \
#     --cutout --autoaugment --save