#!/bin/bash

# Poisons:

attack="rlf"

for p in 1 10 50; do
    for trial in {1..10}; do
        python post_search.py \
            -n 3 \
            --save search/cifar10/$attack/$p%/trial-$trial \
            --expr search-$attack-$p%-trial-$trial/iter_30.stats \
            --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0
    done
done

# Clean:

# for trial in {1..10}; do
#     python post_search.py \
#         -n 3 \
#         --save search/cifar10/clean/trial-$trial \
#         --expr search-clean-trial-$trial/iter_30.stats \
#         --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0
# done