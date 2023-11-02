#!/bin/bash
#SBATCH --clusters=htc
#SBATCH --job-name=VAE 
#SBATCH --mem=64G
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1 --constraint='gpu_mem:32GB'
## SBATCH --array=0-10

module load Anaconda3
source activate cellshape_experiments

cells_to_run=(1086825 1086827 1086822 1086832 1086844 1086837 1086838 1086841 1086842 1086843 1086833)
#${cells_to_run[$SLURM_ARRAY_TASK_ID]}
date

echo 'running download and spot detection'
python scripts/preprocess_data_modular.py 1086825 1

source activate cellpose-arc 

echo 'running 2d cellpose segmentation'
python scripts/do_2d_segmentation.py 1086825

date 


