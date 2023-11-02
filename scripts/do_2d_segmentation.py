from scipy.ndimage import gaussian_filter
from cellpose import models
import os 
import numpy as np
from scipy.ndimage import zoom

import sys 

image_id = sys.argv[1]

model = models.Cellpose(model_type='cyto2', gpu = True)

def run_segmentation(image_id): 
    if os.path.isfile(f'data/isotropic_data/{image_id}_isotropic_phal.npy'): 
        phal = np.load(f'data/isotropic_data/{image_id}_isotropic_phal.npy')
        dapi = np.load(f'data/isotropic_data/{image_id}_isotropic_nuclei.npy')
    else: 
        print('Error: isotropic data not found')

    # blur the data: this makes it easier for cellpose 
    phal = gaussian_filter(phal, sigma = 1)
    dapi = gaussian_filter(dapi, sigma = 1)

    print('starting segmentation')
    print(dapi.shape, phal.shape)

    masks, flows_stitched, styles_stitched, _ = model.eval(np.array([dapi, phal]), 
                                                        channels=[1, 2],
                                                        diameter=d,
                                                        resample = False, 
                                                        do_3D=False, 
                                                        min_size = 300, 
                                                        batch_size = 1
                                                        ) 

    print('Finished segmentation')
    np.save(f'output/2d_segs/{image_id}.npy', masks)

anisotropy = 3.2
d = 18
run_segmentation(image_id)