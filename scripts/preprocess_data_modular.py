import omero.gateway
from ezomero import get_image
import pyclesperanto_prototype as cle 
import numpy as np 
import pandas as pd
import getpass
from skimage.transform import resize

import tifffile as tiff

from utility_scripts import * 

import os 

import config # config contains my signin info 
from config import OMEROUSER, OMEROHOST, OMEROPORT, OMEROPASS
# OMEROPASS = getpass.getpass()
conn = omero.gateway.BlitzGateway(OMEROUSER, OMEROPASS, port=OMEROPORT, host=OMEROHOST)
ret = conn.connect()
assert ret 

def produce_isotropic_images(image_id, nchunks): 

    data = import_omero_image(image_id, conn)
    rescaling_factor = compute_rescaling_factor(image_id, conn)
    channels = get_channel_names(image_id, conn)
    print(f"Channels are: {channels}")
    print(f"Original data shape is: {data.shape}")
    print(f"Rescaling by: {rescaling_factor}")

    dapi_idx = channels.index('dapi')
    phal_idx = channels.index('phalloidin')

    data = np.array_split(data, nchunks, axis = 2)
    print(f'splitting data into {len(data)} chucks')

    isotropic_phal = []
    isotropic_nuclei = []

    for i in range(len(data)): 
        isotropic_nuclei.append(resample_isotropic(
            data[i][:, :, :, dapi_idx], rescaling_factor
            ))
        isotropic_phal.append(resample_isotropic(
            data[i][:, :, :, phal_idx], rescaling_factor
            ))
        
    isotropic_nuclei = np.concatenate(isotropic_nuclei, axis = 2)
    isotropic_phal = np.concatenate(isotropic_phal, axis = 2)

    print(f"Final image shape is: ")
    print(isotropic_nuclei.shape)

    np.save(f'data/isotropic_data/{image_id}_isotropic_nuclei.npy', isotropic_nuclei)
    np.save(f'data/isotropic_data/{image_id}_isotropic_phal.npy', isotropic_phal)

def run_spot_detection(isotropic_nuclei, image_id, nchunks): 

    data = np.array_split(isotropic_nuclei, nchunks, axis = 2)

    for i in range(len(data)): 
        print(f'Processing chunk number {i}')

        outdir = f'output/cell_positions/{image_id}_{i}.csv'
        pointlist = detect_spots(data[i], 4, 6)

        offset = data[i].shape[2] * i
        print(f'This chunk is offset by {offset}')

        print('pointlist shape: ')
        print(pointlist.shape)
        save_spots(pointlist, outdir=outdir, offset = offset)

def median_filter_phal(isotropic_phal, image_id, nchunks): 
    
    #####
    # median filter on phal stain 
    #####

    data = np.array_split(isotropic_phal, nchunks, axis = 2)
    print(f'splitting data into {len(data)} chucks')
    isotropic_phal = []

    for i in range(len(data)): 
        print(f'Processing chunk number {i}')
        image = cle.push(data[i])
        # image_blur = cle.median_box(image, radius_x = 1, radius_y = 1, radius_z = 1)
        image_blur = cle.pull(image_blur)
        isotropic_phal.append(image_blur)
        
    isotropic_phal = np.concatenate(isotropic_phal, axis = 2)

    tiff.imsave(f'data/isotropic_data/{image_id}_isotropic_phal.tiff', isotropic_phal)

def produce_cell_images(isotropic_nuclei, isotropic_phal, image_id, nchunks): 

    for i in range(nchunks): 
        spotdir = f'output/cell_positions/{image_id}_{i}.csv'
        centroids = pd.read_csv(spotdir)
        size = 16
        print('centroid shape: \n', centroids.shape)

        # remove cells that are too close to image edge 
        centroids = centroids[
            (centroids['Z'] - size > 0) & 
            (centroids['Z'] + size < isotropic_phal.shape[0]) & 
            (centroids['X'] - size > 0) & 
            (centroids['X'] + size < isotropic_phal.shape[1]) & 
            (centroids['Y'] - size > 0) & 
            (centroids['Y'] + size < isotropic_phal.shape[2]) 
            ]
        print('centroid shape after filtering: \n', centroids.shape)
        
        for idx, row in centroids.iterrows(): 
            if True: 
            # if f'{image_id}_cell{i}_{idx}.npy' not in os.listdir('output/cells/'): 
                nuc_pos = [round(row['Z']), round(row['Y']), round(row['X'])]
                try: 
                    extracted_cell = extract_cell(nuc_pos, 
                                                isotropic_phal, isotropic_nuclei, 
                                                xy_size = size, z_size = size)
                    np.save(f'output/cell_images/{image_id}_cell{i}_{idx}.npy', 
                            extracted_cell)
                except: 
                    print(idx)

def process(image_id, nchunks, conn, overwrite = False): 

    #####
    # produce isotropic image 
    ##### 

    if f'{image_id}_isotropic_nuclei.npy' not in os.listdir('data/isotropic_data/') or overwrite:
        print(f'Cannot find isotropic image: preprocessing. ')
        produce_isotropic_images(image_id, nchunks) 

    isotropic_nuclei = np.load(f'data/isotropic_data/{image_id}_isotropic_nuclei.npy')
    # isotropic_phal = np.load(f'data/isotropic_data/{image_id}_isotropic_phal.npy')


    print('Data loaded ')
    print('Running spot detection')
    run_spot_detection(isotropic_nuclei, image_id, nchunks)

    print('Finished spot detection')

    if f'{image_id}_isotropic_phal.tiff' not in os.listdir('data/isotropic_data/') or overwrite: 
        print('Producing filtered data')
        isotropic_phal = np.load(f'data/isotropic_data/{image_id}_isotropic_phal.npy')
        tiff.imsave(f'data/isotropic_data/{image_id}_isotropic_phal.tiff', isotropic_phal)
        # median_filter_phal(isotropic_phal, image_id, nchunks)


    produce_cell_images(isotropic_nuclei, isotropic_phal, image_id, nchunks)





# process(1086395)
import sys 

image_id = int(sys.argv[1])
nchunks = int(sys.argv[2])

print(f'Image ID is: {image_id}')
print(f'We have {nchunks} chunks')

process(image_id, nchunks, conn, True)
conn.close() 


    


    


