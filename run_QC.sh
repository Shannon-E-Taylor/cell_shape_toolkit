#!/bin/bash
#SBATCH --clusters=htc
#SBATCH --job-name=VAE 
#SBATCH --mem=64G
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1 --constraint='gpu_mem:32GB'
## SBATCH --array=0-10

module load Anaconda3
source activate napari-env

cells_to_run=(1086825 1086827 1086822 1086832 1086844 1086837 1086838 1086841 1086842 1086843 1086833)
#${cells_to_run[$SLURM_ARRAY_TASK_ID]}
date

processed_cells=('1036826', '1108313')

embryonumber='1036826'
# activate environment 
# conda init bash
source activate napari-env

echo 'computing agreement scores' 
python scripts/compute_agreement_scores.py $embryonumber

echo 'graphing segmentation quality quantifications' 
python scripts/count_successful_cells.py $embryonumber

echo 'graphing limeseg masks' 
python scripts/graph_limeseg_segmentation.py $embryonumber