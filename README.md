# NSGANetv2

## Environment setup

```bash
conda create -n nsganetv2 python=3.7
conda activate nsganetv2
pip install -r requirements.txt
```

## Download the supernet

I saved the supernet file in the SAIL filespace. You should be able to move it to the expected local directory as follows:

```bash
cp /nfs/hpc/sail-gpu0/nsganetv2/ofa_mbv3_d234_e346_k357_w1.0 ./data
```

## Run search

I need help with running NSGANetv2 on the GC poisons (p=1%). You can queue 10 trials of this experiment (with different seeds) by running:

```bash
./scripts/batched_search.sh
```

Thanks!