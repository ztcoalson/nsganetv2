#!/bin/bash

# Poisons:

p=1
poisons=gc

seeds=(3362 9851 5544 9068 1383 1245 6765 5344 9333 5822)

for i in {1..10}; do
  echo "Submitting trial ${i} for poisons: ${poisons} at ${p}%"
  seed_idx=$((i-1))

  sbatch <<EOF
#!/bin/bash
#SBATCH -J nsganetv2
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 2-12:00:00
#SBATCH --output=./logs/slurm-%j.out

module load cuda/10.1

python msunas.py --sec_obj flops \
         --n_gpus 1 --gpu 1 --n_workers 4 --n_epochs 5 \
         --dataset cifar10 --n_classes 10 \
         --data data \
         --predictor as --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
         --save search-${poisons}-${p}%-trial-${i} --iterations 30 --vld_size 5000 \
         --seed ${seeds[$seed_idx]} \
         --poisons_type dirty_label \
         --poisons_path "./poisons/cifar10/${poisons}/${p}.0%/poisons.pth"
EOF
done

# Clean:

# seeds=(3362 9851 5544 9068 1383 1245 6765 5344 9333 5822)

# for i in {1..10}; do
#   echo "Submitting trial ${i} for clean"
#   seed_idx=$((i-1))

#   sbatch <<EOF
# #!/bin/bash
# #SBATCH -J nsganetv2
# #SBATCH -A eecs
# #SBATCH -p dgx2
# #SBATCH --mem=50G
# #SBATCH --gres=gpu:1
# #SBATCH -c 4
# #SBATCH -t 2-08:00:00
# #SBATCH --output=./logs/slurm-%j.out

# module load cuda/10.1

# python msunas.py --sec_obj flops \
#          --n_gpus 1 --gpu 1 --n_workers 4 --n_epochs 5 \
#          --dataset cifar10 --n_classes 10 \
#          --data data \
#          --predictor as --supernet_path data/ofa_mbv3_d234_e346_k357_w1.0 \
#          --save search-clean-trial-${i} --iterations 30 --vld_size 5000 \
#          --seed ${seeds[$seed_idx]}
# EOF
# done