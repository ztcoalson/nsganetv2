#!/bin/bash

dataset=cifar10

python -m crafting.run_gradpc --dataset cifar10 \
    --save ./poisons/$dataset/gradpc/nsganetv2 \
    --dataset $dataset \
    --model_config_path "/nfs/hpc/share/coalsonz/nsganetv2/20251028-113012-cifar10-clean-trial-7-192/net_flops@307.config" \
    --model_weights_path "/nfs/hpc/share/coalsonz/nsganetv2/20251028-113012-cifar10-clean-trial-7-192/net_flops@307.best" \
    --eps 1.0 \
    --attacker_epochs 1 \
    --attack_lr 1 \
    --no_augment