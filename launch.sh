
#! /bin/bash

for T in 100 1000; do
  for batch_size in 128; do
    for beta_min in 0.00001 0.0001 0.001; do
      for beta_max in 0.005 0.01 0.05; do
        sbatch --export=T=$T,batch_size=$batch_size,beta_min=$beta_min,beta_max=$beta_max diffusion.sbatch
        sleep 1
      done
    done
  done
done
