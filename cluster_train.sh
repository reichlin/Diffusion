#!/usr/bin/env bash

SOURCE_PATH="${HOME}/meta"
AT="@"

# Test the job before actually submitting 
# SBATCH_OR_CAT=cat
SBATCH_OR_CAT=sbatch
LOG_PATH="/Midgard/home/gustafte/slurm_logs/diffusion"


"${SBATCH_OR_CAT}" << HERE
#!/usr/bin/env bash
#SBATCH --output="${LOG_PATH}/%J_slurm.out"
#SBATCH --error="${LOG_PATH}/%J_slurm.err"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="gustafte${AT}kth.se"
#SBATCH --job-name="meta"
#SBATCH --constrain="eowyn|arwen|galadriel|shelob|khazadum|balrog|belegost|rivendell|smaug"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB

echo "Sourcing conda.sh"
source "/Midgard/home/gustafte/miniconda3/etc/profile.d/conda.sh"

echo "Activating conda environment"
conda activate graph
nvidia-smi


python lucidrains_diffusion.py  

EXIT_CODE="\${?}"
exit "\${EXIT_CODE}"


HERE
