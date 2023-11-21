#!/bin/bash
#SBATCH --clusters=htc
#SBATCH --gres=gpu:1
#SBATCH --job-name=VAE 
#SBATCH --mem=32G
#SBATCH --partition=short
#SBATCH --time=01:30:00
### SBATCH --array=0-5


module load Anaconda3

date

# echo gpu used 
srun -l echo $CUDA_VISIBLE_DEVICES


# python preprocess-3.py
# python flip_data.py

variables_to_run=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.)
# variables_to_run=(8 16 32 64 128)
variables_to_run=(0.1 0.3 0.5 0.7 0.9)
variables_to_run=(0.01 0.1 1 10 50 100)

source activate n2v


echo ${variables_to_run[$SLURM_ARRAY_TASK_ID]}
# python VAE_big_tb_3d-3.py ${variables_to_run[$SLURM_ARRAY_TASK_ID]}


# python scripts/beta_vae_1.py 0.5 ${variables_to_run[$SLURM_ARRAY_TASK_ID]}
# python scripts/beta_vae_1.py ${variables_to_run[$SLURM_ARRAY_TASK_ID]} 64
python scripts/beta_vae_1.py 0.5 64

date 

