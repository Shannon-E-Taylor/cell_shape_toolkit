#!/bin/bash
#SBATCH --clusters=htc
#SBATCH --job-name=VAE 
#SBATCH --mem=32GB
#SBATCH --partition=short
#SBATCH --time=02:50:00
#SBATCH --gres=gpu:1 --constraint='gpu_mem:32GB'
## SBATCH --array=0-21

module load Anaconda3
source activate cellshape_experiments


date

    
# cells_to_run=(1036826 1108313 1086825 1086827 1086826 1086822 1086832 1036828 1108322 1108323 1108324 1108325 1108321 1108320 1108314 1108315 1108316 1036827 1036829 1086844 1086837)

# cellname=${cells_to_run[$SLURM_ARRAY_TASK_ID]}

cellname=1108320

echo $cellname
# echo 'running download and spot detection'
# python scripts/preprocess_data_modular.py $cellname 1

source deactivate

source activate cellpose-arc 

echo 'running 2d cellpose segmentation'
python scripts/do_2d_segmentation.py $cellname

date 


