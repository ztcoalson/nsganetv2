#!/bin/bash

export MASTER_ADDR="localhost"
export MASTER_PORT=12356

dataset="cifar10"

python -m crafting.run_gc \
    --dataset $dataset \
    --attack_iters 100 \
    --p_ratio 0.01 \
    --model_config_path "/nfs/hpc/share/coalsonz/nsganetv2/20251028-113012-cifar10-clean-trial-7-192/net_flops@307.config" \
    --target_model_weights_path "/nfs/hpc/share/coalsonz/nsganetv2/poisons/cifar10/gradpc/nsganetv2/gradpc-eps=1.0/target_model.pth" \
    --clean_grads_path "/nfs/hpc/share/coalsonz/nsganetv2/poisons/cifar10/gradpc/nsganetv2/gradpc-eps=1.0/clean_grads.pth" \
    --pbatch 64