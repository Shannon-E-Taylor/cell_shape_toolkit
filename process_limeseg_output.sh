#!/bin/bash
#SBATCH --clusters=htc

date

embryonumber='1086822'

# activate environment 
# conda init bash
source activate napari-env

# File paths  
path_to_output_image="output/limeseg_output/$embryonumber/whole_segmentation.tif" # path to save the new label image to 
path_to_input_image="data/isotropic_data/${embryonumber}_isotropic_phal.tiff" # path to the original image you segmented, to read image dimensions from  
path_to_limeseg_folder="output/limeseg_output/$embryonumber" # path to the folder where you saved your limeseg data 
path_to_morphometrics_output="output/morphometrics/$embryonumber.csv"
path_to_spharm_output="output/spharm/$embryonumber.csv"

echo 'its running'

echo $path_to_input_image

# Run script to preprocess the cell meshes, 
# voxelize each cell 
# and create a label image containing each cell for further analysis 
if ! test -f "${path_to_output_image}.txt"; then
    echo 'voxelizing'
    python scripts/voxelize_cell.py $path_to_output_image $path_to_input_image $path_to_limeseg_folder
fi

if ! test -f "${path_to_output_image}.npz"; then
    echo 'generating label image'
    python scripts/generate_label_image_from_subgroups_of_cells.py $path_to_output_image $path_to_input_image $path_to_limeseg_folder
fi

# Now run the morphometrics script 
python scripts/run_morphometrics.py $path_to_limeseg_folder $path_to_morphometrics_output

# and do SPHARM 
# this is untested 
# python scripts/calculate_aligned_spherical_harmonics.py $path_to_limeseg_folder $path_to_spharm_output