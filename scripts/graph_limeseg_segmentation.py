from cellpose import plot 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from skimage.io import imread
import sys

image_id = sys.argv[1]

path_to_2d_seg = f'output/2d_segs/{image_id}.npy'
path_to_limeseg = f'output/limeseg_output/{image_id}/whole_segmentation.tif.npz'
path_to_image = f'data/isotropic_data/{image_id}_isotropic_phal.tiff'

seg_to_test = np.load(path_to_limeseg)['seg']

img = imread(path_to_image)

images_to_plot = []

slices_to_plot = np.linspace(0, img.shape[0], 7).astype(int)[1:-1]

for idx in np.linspace(0, img.shape[0], 7).astype(int)[1:-1]: 

    img_subset = img[idx]
    img_subset = (255*(img_subset - np.min(img_subset))/np.ptp(img_subset)).astype(int)   

    toplot = plot.mask_overlay(img_subset, seg_to_test[idx])

    images_to_plot.append(toplot)

fig, ax = plt.subplots(1, 5, figsize = (20, 4))

for i in range(5): 
    ax[i].imshow(images_to_plot[i])

plt.savefig(f'output/QC/limeseg_segmentation_{image_id}.png', dpi = 500)