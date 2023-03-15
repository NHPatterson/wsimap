import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from wsimap.wsi_map import WsiMap
from wsimap.utils.wsi_torch_utils import WsiTorchObjDet
import numpy as np

zarr_fp = "/data/SharedData/biomic/glom3d_S108/S108_3D_AF_preMxIF_sec01.zarr"
glom_roi_fp = "/data/SharedData/biomic/glom3d_S108/S108_MxIF_Sec01_sc2_glomeruli.json"

# tiling and annotation prep
glom_wsi = WsiMap(zarr_fp, pyr_layer=0)

glom_wsi.import_annotations(glom_roi_fp, roi_name="glomerulus")

glom_wsi.tile_wsi(
    tile_name="sliding_win",
    bound_anno=None,
    tile_size=1024,
    slide=[(512, 0), (0, 512), (512, 512)],
    roi_name=None,
    edge_thresh=1,
)

glom_wsi.intersect_tiles_annos()

glom_wsi.roi_data_to_np()

roi_class = glom_wsi.roi_data_np[0][0]
dask_im_tile = glom_wsi.roi_data_np[0][1]
tile_bounds = glom_wsi.roi_data_np[0][2]
rois = glom_wsi.roi_data_np[0][3]

# normalize rois to local image coordinates
rois_tile = [roi - [tile_bounds[0], tile_bounds[1]] for roi in rois]

tile_im = dask_im_tile.compute()

tile_im = tile_im / np.iinfo(tile_im.dtype).max

# get bounding box coordinates for each mask
num_objs = len(rois_tile)
boxes = []
for i in range(num_objs):
    xmin = np.min(rois_tile[i][:,0])
    xmax = np.max(rois_tile[i][:,0])
    ymin = np.min(rois_tile[i][:,1])
    ymax = np.max(rois_tile[i][:,1])
    boxes.append([xmin, ymin, xmax, ymax])
boxes = boxes[0]
mask_arr = np.zeros(
    [num_objs] + list(tile_im.shape[1:3]), dtype=np.uint8)

for idx, roi in enumerate(rois_tile):
    mask_arr[idx, :, :] = cv2.fillPoly(mask_arr[idx, :, :], [roi], 255)


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import cv2
%matplotlib inline

fig,ax = plt.subplots(1)
s
# Display the image
ax.imshow(np.squeeze(mask_arr[0,:,:]))
# Create a Rectangle patch
rect = patches.Rectangle((boxes[0],boxes[1]),boxes[2]-boxes[0],boxes[3]-boxes[1],linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()
