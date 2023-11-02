import os 
import pyclesperanto_prototype as cle

import omero.gateway
from ezomero import get_image

import numpy as np
import pandas as pd


##############################
# import and preprocess data #
##############################

def import_omero_image(image_id, conn):
    # cache the file to reduce load on the server 
    if f'{image_id}.npy' not in os.listdir('data/omero_scratch/'):  
        im_object, image = get_image(conn, image_id, no_pixels=False)
        image = image[0] # strip the time dimension 
        np.save(f'data/omero_scratch/{image_id}.npy', image)
    else: 
        image = np.load(f'data/omero_scratch/{image_id}.npy', allow_pickle=True)
    return image

def compute_rescaling_factor(image_id, conn, output_size = 0.25): 
    image_metadata, _ = get_image(conn, image_id, no_pixels=True)
    xsize, zsize = [image_metadata.getPixelSizeX(), image_metadata.getPixelSizeZ()]
    return([xsize / output_size, zsize / output_size])


def resample_isotropic(input_image, rescaling_factor):
    # Push the input image to the GPU   
    input_image = cle.push(input_image)

    factor_x = rescaling_factor[0]
    factor_z = rescaling_factor[1]

    # Resample the image
    resampled_image = cle.scale(input_image, 
                                factor_x = factor_x, 
                                factor_y = factor_x, 
                                factor_z = factor_z, 
                                linear_interpolation = True, 
                                auto_size = True, # important to size the image after rescaling 
                                centered=True)
    # Download the resampled image from the GPU
    resampled_img = cle.pull(resampled_image)

    return resampled_img

def get_channel_names(image_id, conn): 
    # Get the image object using the specified ID
    image = conn.getObject('Image', image_id)

    # Get the channel names for the image
    channels = image.getChannels()
    channel_names = [c.getLabel() for c in channels]

    return channel_names

############################
# compute nuclei positions #
############################

def detect_spots(img, sigma, radius): 
    input_image = cle.push(img)
    starting_point = cle.gaussian_blur(input_image, sigma_x=sigma, sigma_y=sigma, sigma_z = sigma)
    del input_image
    maxima = cle.detect_maxima_box(starting_point, radius_x=radius, radius_y = radius, radius_z=radius)
    labeled_maxima = cle.label_spots(maxima)
    del maxima, starting_point 
    pointlist = cle.centroids_of_labels(labeled_maxima)
    detected_spots = cle.pull(pointlist).T
    detected_spots = detected_spots[~np.isnan(detected_spots).any(axis=1)]
    # print(detected_spots.shape)
    return(detected_spots.T)

def save_spots(pointlist, outdir, offset = 0): 
    df = pd.DataFrame(np.array(pointlist)).T.astype(int)
    df.columns = ['X', 'Y', 'Z']
    df['X'] = df['X'] + offset # correct for chunk identity 
    df.to_csv(outdir)

def scale(x, min, max): 
    return (x - min) / (max - min)

def extract_cell(nuc_pos, data, nuclei, z_size, xy_size): 
    '''
    nuc_pos should be zyx 
    '''
    cell = data[
        nuc_pos[0]-z_size:nuc_pos[0]+z_size, 
        nuc_pos[1]-xy_size:nuc_pos[1]+xy_size, 
        nuc_pos[2]-xy_size:nuc_pos[2]+xy_size,
    ]
    nuc = nuclei[
        nuc_pos[0]-z_size:nuc_pos[0]+z_size, 
        nuc_pos[1]-xy_size:nuc_pos[1]+xy_size, 
        nuc_pos[2]-xy_size:nuc_pos[2]+xy_size,
    ]
    cell = scale(cell, np.min(cell), np.max(cell))
    nuc = scale(nuc, np.min(nuc), np.max(nuc))
    both = np.array([cell, nuc])
    return(both)

