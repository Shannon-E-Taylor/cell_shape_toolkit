import matplotlib.pyplot as plt 
import pandas as pd 
import os 
from skimage.io import imread

import sys 


def count_unsegmented_cells(results): 

    # If Free edges > 0, the cell is broken open 
    broken_cells = (results['Free edges'] > 0).sum()
    results = results[results['Free edges'] == 0]

    poor_segmentation = (results['IOU'] < 0.5).sum()
    results = results[results['IOU'] >= 0.5]

    good_cells = results.shape[0]

    return(broken_cells, poor_segmentation, good_cells, results)

# image_id = '1036826'
image_id = sys.argv[1]
path = f'output/limeseg_output/{image_id}'

files = next(os.walk(path))[1]

all_results = []

for f in files: 
    res = pd.read_csv(f'{path}/{f}/Results.csv')
    all_results.append(res)

results = pd.concat(all_results)
results = results.dropna(subset=['Center X'])

# add the tissue type to cells 
tissue_segmentation = imread(f'data/manual_segmentations/{image_id}_isotropic_phal.labels.tif')

results['tissue'] = tissue_segmentation[
    list(round(results['Center Z']).astype(int)), 
    list(round(results['Center Y']).astype(int)), 
    list(round(results['Center X']).astype(int))
    ] 

# load agreement scores 
eroded_metrics = pd.read_csv(f'output/QC/agreement_scores/QC_eroded_masks_by_cell_{image_id}.csv')

eroded_metrics['label'] = eroded_metrics['label'].astype(int)

results['cellnum'] = [i.split('_')[1] for i in results['Cell Name']]
results['cellnum'] = results['cellnum'].astype(int)

results = results.merge(eroded_metrics, left_on='cellnum', right_on='label')

results.to_csv(f'{path}/combined_Results.csv')

results = results[results['tissue'].isin([1,2])]

fig, ax = plt.subplots(1, 4, 
                        gridspec_kw={'width_ratios': [1,3,3,3]}, 
                       tight_layout = True, 
                       figsize = (10,3))

# graph IOU before we remove bad cells 
ax[1].hist(results['IOU'], bins = 30, label = 'unfiltered', alpha = 0.3)
ax[2].hist(results['Real Volume'], bins = 30, label = 'unfiltered', alpha = 0.3, density = True)
ax[3].hist(results['Real Surface'], bins = 30, label = 'unfiltered', alpha = 0.3, density = True)


broken_cells, poor_segmentation, good_cells, results = count_unsegmented_cells(results)

categories = ['broken_cells', 'poor segmentation', 'good_cells']

# produce bar plot 
p1 = ax[0].bar([1], good_cells, 1)
p2 = ax[0].bar([1], broken_cells, 1, 
               bottom = good_cells)
p3 = ax[0].bar([1], poor_segmentation, 1, 
               bottom = good_cells + broken_cells)
ax[0].set_ylabel('Number of cells')
ax[0].set_title('# cells successfully segmented')

### graph agreement score 
ax[1].set_title('Agreement scores')
ax[1].hist(results['IOU'], bins = 30, label = 'filtered', alpha = 0.3)
ax[1].set_xlim(0, 1)


### graph cell volume 
ax[2].set_title('Cell volume')
ax[2].hist(results['Real Volume'], bins = 30, label = 'filtered', alpha = 0.3, density = True)
ax[2].set_xlim(0, 1000)

### graph surface area
ax[3].set_title('Cell surface area')
ax[3].hist(results['Real Surface'], bins = 30, label = 'filtered', alpha = 0.3, density = True)
ax[3].set_xlim(0, 1000)

plt.savefig(f'output/QC/successful_cells_{image_id}')