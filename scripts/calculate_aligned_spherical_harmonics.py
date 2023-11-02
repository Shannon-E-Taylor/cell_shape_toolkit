"""Expand and reconstruct any surface
(here a simple box) into spherical harmonics"""
# Expand an arbitrary closed shape in spherical harmonics
# using SHTOOLS (https://shtools.oca.eu/shtools/)
# and then truncate the expansion to a specific lmax and
# reconstruct the projected points in red

from scipy.interpolate import griddata
import pyshtools
from vedo import spher2cart, mag, Box, Point, Points, show, load, Cylinder

import open3d as o3d

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys 

###########################################################################################
lmax = 15             # maximum degree of the spherical harm. expansion
N    = 50             # number of grid intervals on the unit sphere
rmax = 50             # line length - max distance from cell centre to surface in pixels
###########################################################################################


def run_harmonics(path, lmax, N, rmax): 
    pointcloud = o3d.io.read_point_cloud(path)
    if pointcloud.has_points(): 
        surface = load(path)

        # this is a bodge, but we want all of our cells aligned 
        # along the long axis 
        # I'm achieving that by aligning them all to a cylinder 
        # I ought to be aligning the major axis of the cell to the origin 
        # but I can't work out how to do that at the moment 
        cylinder = Cylinder(pos=(5, 0, 0), c=(1, 0, 0), r=5, height=400, res=100, alpha = 0.5)

        # Align the mesh to the cylinder
        surface = surface.align_to(
            cylinder, rigid = True, 
            use_centroids = True, 
            iters = 100)
        # get the new centre 
        x0 = surface.center_of_mass()

    else: 
        print(path)
        return [None, None, None]

    agrid, pts = [], []
    for th in np.linspace(0, np.pi, N, endpoint=True):
        longs = []
        for ph in np.linspace(0, 2*np.pi, N, endpoint=False):
            p = spher2cart(rmax, th, ph)
            intersections = surface.intersectWithLine(x0, x0+p)
            if len(intersections):
                value = mag(intersections[0]-x0)
                longs.append(value)
                pts.append(intersections[0])
            else:
                print('No hit for theta, phi =', th, ph)
                
                longs.append(rmax)
                pts.append(p)
        agrid.append(longs)
    agrid = np.array(agrid)

    #############################################################
    grid = pyshtools.SHGrid.from_array(agrid)
    clm = grid.expand()
    grid_reco = clm.expand(lmax=lmax).to_array()  # cut "high frequency" components

    #############################################################
    # reconstruct the object 
    ll = []
    for i, long in enumerate(np.linspace(0, 360, num=grid_reco.shape[1], endpoint=False)):
        for j, lat in enumerate(np.linspace(90, -90, num=grid_reco.shape[0], endpoint=True)):
            th = np.deg2rad(90 - lat)
            ph = np.deg2rad(long)
            ll.append((lat, long))

    radii = grid_reco.T.ravel()
    n = 200j
    lnmin, lnmax = np.array(ll).min(axis=0), np.array(ll).max(axis=0)
    grid = np.mgrid[lnmax[0]:lnmin[0]:n, lnmin[1]:lnmax[1]:n]
    grid_x, grid_y = grid
    grid_reco_finer = griddata(ll, radii, (grid_x, grid_y), method='cubic')

    pts2 = []
    for i, long in enumerate(np.linspace(0, 360, num=grid_reco_finer.shape[1], endpoint=False)):
        for j, lat in enumerate(np.linspace(90, -90, num=grid_reco_finer.shape[0], endpoint=True)):
            th = np.deg2rad(90 - lat)
            ph = np.deg2rad(long)
            p = spher2cart(grid_reco_finer[j][i], th, ph)
            pts2.append(p+x0)

    # turn it into a pointcloud 
    reconstructed_pointcloud = o3d.geometry.PointCloud()
    reconstructed_pointcloud.points = o3d.utility.Vector3dVector(pts2)

    # and then get the distance between them! 
    dist_pc1_pc2 = reconstructed_pointcloud.compute_point_cloud_distance(pointcloud)

    coeffs = clm.coeffs

    pts2 = np.array(pts2)

    # Create a point cloud
    point_cloud = Points(pts2)
    # harmonics = np.sum(coeffs**2, axis=1)[1] + np.sum(coeffs**2, axis=1)[0]

    # show(f'Spherical harmonics expansion of order {lmax}',
    #  cylinder, point_cloud, 
    #  surface)
    
    harmonics = np.concatenate([clm.to_array()[1, 0:lmax, 0:lmax], clm.to_array()[0, 0:lmax, 0:lmax]])

    return([harmonics, x0, np.mean(dist_pc1_pc2)])


path_to_limeseg_folder = sys.argv[1]
path_for_output = sys.argv[2]


results = pd.read_csv(f'{path_to_limeseg_folder}/Results.csv').sample(frac = 1)
print(results.shape)

cell_list = results['Cell Name']

harmonics_list = []
positions_list = []
quality = []
labels = []


for cell in cell_list: 
    path = f'{path_to_limeseg_folder}' + cell + '/T_1.ply'
    tmp_out, pos, qual = run_harmonics(path, lmax, N, rmax)
    if tmp_out is not None: # sometimes my function returns None 
        harmonics_list.append(tmp_out)
        positions_list.append(pos)
        quality.append(qual)
        labels.append(int(cell.split('_')[-1]))

df = pd.DataFrame([i.flatten() for i in harmonics_list])
df[['X', 'Y', 'Z']] = np.array(positions_list)
df['quality'] = quality 
df['label'] = labels

df.to_csv(path_for_output, index = False)




