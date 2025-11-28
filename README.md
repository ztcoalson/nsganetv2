# NSGANetv2

## Environment setup

```bash
conda create -n nsganetv2 python=3.7
conda activate nsganetv2
pip install -r requirements.txt
```

## Install once-for-all repo

```bash
git clone -b patch-for-NSGANetv2 https://github.com/ztcoalson/once-for-all.git
cd once-for-all
pip install .
```

If you face a version error, then try editing `setup.py`

```
vim setup.py
// Do followings
// version="0.0.4+2511281336",
// or just
// version="0.0.4",
```

## Download the supernet

I saved the supernet file in the SAIL filespace. You should be able to move it to the expected local directory as follows:

```bash
cp /nfs/hpc/sail-gpu0/nsganetv2/ofa_mbv3_d234_e346_k357_w1.0 ./data
```

## Download the poisons

I also saved the crafted poisons in the SAIL filespace. You can move them as follows:

```bash
cp -r /nfs/hpc/sail-gpu0/nsganetv2/poisons ./
```

## Run search

I need help with running NSGANetv2 on the GC poisons (p=1%). You can queue 10 trials of this experiment (with different seeds) by running:

```bash
./scripts/batched_search.sh
```

Thanks!
