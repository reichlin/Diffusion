#!/usr/bin/env bash
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --constrain='balrog|belegost|khazadum|rivendell|shelob|smaug|eowyn|arwen|galadriel|gondor'

source "/Midgard/home/areichlin/miniconda3/etc/profile.d/conda.sh"
conda activate env_torch

python3 /Midgard/home/areichlin/Diffusion/main.py --T ${T} --batch_size ${batch_size} --beta_min ${beta_min} --beta_max ${beta_max}
