from pathlib import Path
from typing import List, Optional

import SimpleITK as sitk
import numpy as np
import zarr
from czifile import CziFile
from dask import array as da
from tifffile import imread


def tifffile_to_dask(im_fp:str, level:int=0):
    imdata = zarr.open(imread(im_fp, aszarr=True))
    if isinstance(imdata, zarr.hierarchy.Group):
        imdata = da.from_zarr(zarr.open(imread(im_fp, aszarr=True, level=level)))
    else:
        imdata = da.from_zarr(imdata)
    return imdata


def read_full_image_for_prediction(
    test_image_fp: str,
    train_channels: List[int],
    downsampling: int = 1,
    max_workers: Optional[int] = None,
):
    if Path(test_image_fp).suffix.lower() == ".czi":
        full_im = CziFile(test_image_fp)
        full_im = np.squeeze(full_im.asarray(max_workers=max_workers))
        if downsampling != 1:
            full_im = sitk.GetImageFromArray(full_im)
            full_im = sitk.Shrink(full_im, (downsampling, downsampling, 1))
            full_im = sitk.GetArrayFromImage(full_im)
    else:
        pyr_level = int(np.log2(downsampling))
        full_im = imread(test_image_fp, level=pyr_level, key=train_channels, series=0)

        full_im = da.squeeze(full_im)
    if len(full_im.shape) == 2:
        full_im = np.expand_dims(full_im, 0)

    return full_im