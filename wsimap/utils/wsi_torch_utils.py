import copy
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from tifffile import imread, TiffFile
import zarr
import dask.array as da
import albumentations as A
import h5py

try:
    from fastai.torch_core import TensorImage, TensorMask
except ImportError:
    UserWarning("fastai not installed, unet segmentation will not work")


def tifffile_to_dask(im_fp, pyr_level:int):
    imdata = zarr.open(imread(im_fp, aszarr=True))
    if isinstance(imdata, zarr.hierarchy.Group):
        imdata = da.from_zarr(zarr.open(imread(im_fp, aszarr=True, level=pyr_level)))
    else:
        imdata = da.from_zarr(imdata)
    return imdata


def classes_to_dict(class_arr, bg_class=False):
    """Utility function to create a dict where class indices for model Training
    can be indexed by their name.

    Parameters
    ----------
    class_arr : np.ndarray
        numpy character array of all labelled classes.
    bg_class : bool
        will background be counted as a class at index 0? For detection2 this is
        not the case.

    Returns
    -------
    tuple
        returns class dict and reverse class dict for indexing by name or index

    """
    class_keys = list(np.unique(class_arr))
    class_data = {}
    rev_class_data = {}

    bg = 1 if bg_class is True else 0
    for idx, key in enumerate(class_keys):
        class_data[key] = idx + bg
        rev_class_data[idx + bg] = key
    return class_data, rev_class_data


class WsiTorchObjDet(object):
    """Class to create a torch data_loader for object detection (instance segmentation).

    Parameters
    ----------
    roi_data : list
        list of wsimap dask & numoy roi tile/roi data
    transforms : func
        transforms to apply to images (not implemented)

    Attributes
    ----------
    class_data : type
        class dict for converting name to numerical index for training
    rev_class_data : type
        class dict for converting numercial index to name
    roi_data : list
        list of wsimap dask & numoy roi tile/roi data
    transforms : func
        transforms to apply to images (not implemented)
    """

    def __init__(self, roi_data, transforms=None):
        self.roi_data = roi_data
        self.transforms = transforms

        class_arr = np.concatenate(
            np.asarray([p["roi_classes"] for p in self.roi_data])
        )
        self.class_data, self.rev_class_data = classes_to_dict(class_arr, bg_class=True)

    def __getitem__(self, idx):
        """Get torch tensor data for roi at idx.

        Parameters
        ----------
        idx : int
            index of roi in wsimap dask and numpy roi/tile data.

        Returns
        -------
        tuple
            tuple of tile image as torch.float32 image and target dict containing
            torch tensors for training targets
        """
        # load images ad masks
        roi_class = self.roi_data[idx]["roi_classes"]
        dask_im_tile = self.roi_data[idx]["dask_im"]
        tile_bounds = self.roi_data[idx]["tile_bounds"]
        rois = self.roi_data[idx]["rois"]

        # normalize rois to local image coordinates
        rois_tile = [roi - [tile_bounds[0], tile_bounds[1]] for roi in rois]

        tile_im = dask_im_tile.compute()

        tile_im = tile_im / np.iinfo(tile_im.dtype).max
        tile_im = tile_im.astype(np.float32)
        # get bounding box coordinates for each mask
        num_objs = len(rois_tile)
        boxes = []
        for i in range(num_objs):
            xmin = np.min(rois_tile[i][:, 0])
            xmax = np.max(rois_tile[i][:, 0])
            ymin = np.min(rois_tile[i][:, 1])
            ymax = np.max(rois_tile[i][:, 1])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        roi_class_arr = [self.class_data[rn] for rn in roi_class]
        labels = torch.tensor(np.asarray(roi_class_arr), dtype=torch.int64)

        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        mask_arr = np.zeros([num_objs] + list(dask_im_tile.shape[1:]), dtype=np.uint8)
        for idx, roi in enumerate(rois_tile):
            mask_arr[idx, :, :] = cv2.fillPoly(
                mask_arr[idx, :, :], [roi.astype(np.int32)], 255
            )

        masks = torch.as_tensor(mask_arr, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        tile_im = torch.from_numpy(tile_im)

        if self.transforms is not None:
            tile_im, target = self.transforms(tile_im, target)

        return tile_im, target

    def __len__(self):
        return len(self.roi_data)


class WsiTorchUnet(object):
    """Class to create a torch data_loader for object detection (instance segmentation).

    Parameters
    ----------
    roi_data : list
        list of wsimap dask & numoy roi tile/roi data
    transforms : func
        transforms to apply to images (not implemented)

    Attributes
    ----------
    class_data : type
        class dict for converting name to numerical index for training
    rev_class_data : type
        class dict for converting numercial index to name
    roi_data : list
        list of wsimap dask & numoy roi tile/roi data
    transforms : func
        transforms to apply to images (not implemented)
    """

    def __init__(self, roi_data, transforms=None, augmed_transforms=None):
        self.roi_data = roi_data
        self.transforms = transforms
        self.augmedical_transforms = augmed_transforms
        class_arr = np.concatenate(
            np.asarray([p["roi_classes"] for p in self.roi_data])
        )
        self.class_data, self.rev_class_data = classes_to_dict(class_arr, bg_class=True)

    def __getitem__(self, idx):
        """Get torch tensor data for roi at idx.

        Parameters
        ----------
        idx : int
            index of roi in wsimap dask and numpy roi/tile data.

        Returns
        -------
        tuple
            tuple of tile image as torch.float32 image and target dict containing
            torch tensors for training targets
        """
        # load images ad masks
        dask_im_tile = self.roi_data[idx]["dask_im"]
        tile_bounds = self.roi_data[idx]["tile_bounds"]

        rois = self.roi_data[idx]["rois"]
        if len(rois) > 0:
            roi_class = self.roi_data[idx]["roi_classes"]

            # normalize rois to local image coordinates
            rois_tile = [roi - [tile_bounds[0], tile_bounds[1]] for roi in rois]

            # tile_im = tile_im / np.iinfo(tile_im.dtype).max
            # tile_im = tile_im.astype(np.float32)
            # get bounding box coordinates for each mask
            num_objs = len(rois_tile)
            boxes = []
            for i in range(num_objs):
                xmin = np.min(rois_tile[i][:, 0])
                xmax = np.max(rois_tile[i][:, 0])
                ymin = np.min(rois_tile[i][:, 1])
                ymax = np.max(rois_tile[i][:, 1])
                boxes.append([xmin, ymin, xmax, ymax])

            mask_arr = np.zeros(list(dask_im_tile.shape[1:]), dtype=np.uint8)
            for idx, roi in enumerate(rois_tile):
                mask_arr[:, :] = cv2.fillPoly(
                    mask_arr[:, :],
                    [roi.astype(np.int32)],
                    int(self.class_data[roi_class[idx]]),
                )

        else:
            mask_arr = np.zeros(list(dask_im_tile.shape[1:]), dtype=np.uint8)

        if not isinstance(dask_im_tile, np.ndarray):
            tile_im = dask_im_tile.compute()
        else:
            tile_im = dask_im_tile
        tdtype = tile_im.dtype
        if self.transforms:
            tile_im = tile_im.transpose(1, 2, 0)
            augmented = self.transforms(image=tile_im, mask=mask_arr)
            # if self.augmedical_transforms:
            #     for transform in self.augmedical_transforms:
            #         tile_im = transform(tile_im)
            tile_im = augmented["image"].transpose(2, 0, 1).astype(tdtype)
            mask_arr = augmented["mask"]

        tile_im = tile_im / np.iinfo(tdtype).max
        tile_im = tile_im.astype(np.float32)

        mask_arr = mask_arr.astype("int64")
        masks = torch.as_tensor(mask_arr)
        tile_im = torch.from_numpy(tile_im)

        return TensorImage(tile_im), TensorMask(masks)

    def __len__(self):
        return len(self.roi_data)

class WsiTorchUnetMask(object):
    """Class to create a torch data_loader for object detection (instance segmentation).

    Parameters
    ----------
    roi_data : list
        list of wsimap dask & numoy mask data
    transforms : func
        transforms to apply to images

    Attributes
    ----------
    class_data : type
        class dict for converting name to numerical index for training
    rev_class_data : type
        class dict for converting numercial index to name
    roi_data : list
        list of wsimap dask & numoy roi tile/roi data
    transforms : func
        transforms to apply to images (not implemented)
    """

    def __init__(self, roi_data, transforms=None, augmed_transforms=None, channel_idx=[0,1,2]):
        self.roi_data = roi_data
        self.transforms = transforms
        self.augmedical_transforms = augmed_transforms
        class_arr = np.concatenate(
            np.asarray([p["roi_classes"] for p in self.roi_data])
        )
        self.class_data, self.rev_class_data = classes_to_dict(class_arr, bg_class=True)
        self.channel_idx = channel_idx

    def __getitem__(self, idx):
        """Get torch tensor data for roi at idx.

        Parameters
        ----------
        idx : int
            index of roi in wsimap dask and numpy roi/tile data.

        Returns
        -------
        tuple
            tuple of tile image as torch.float32 image and target dict containing
            torch tensors for training targets
        """
        # load images ad masks
        dask_im_tile = self.roi_data[idx]["dask_im"]
        mask_im_tile = self.roi_data[idx]["mask_im"]

        if isinstance(dask_im_tile, np.ndarray):
            tile_im = dask_im_tile
        elif isinstance(dask_im_tile, da.Array):
            tile_im = dask_im_tile.compute()
        elif isinstance(dask_im_tile, str):
            dask_im_in = tifffile_to_dask(dask_im_tile, self.roi_data[idx]["pyr_level"])
            xs, ys, xe, ye = self.roi_data[idx]["tile_bounds"]
            tile_im = dask_im_in[self.channel_idx, ys:ye, xs:xe].compute()

        if not isinstance(mask_im_tile, np.ndarray):
            mask_arr = mask_im_tile.compute()
        else:
            mask_arr = mask_im_tile

        tdtype = tile_im.dtype
        if self.transforms:
            tile_im = tile_im.transpose(1, 2, 0)
            augmented = self.transforms(image=tile_im, mask=mask_arr)
            # if self.augmedical_transforms:
            #     for transform in self.augmedical_transforms:
            #         tile_im = transform(tile_im)
            tile_im = augmented["image"].transpose(2, 0, 1).astype(tdtype)
            mask_arr = augmented["mask"]

        tile_im = tile_im / np.iinfo(tdtype).max
        tile_im = tile_im.astype(np.float32)

        mask_arr = mask_arr.astype("int64")
        masks = torch.as_tensor(mask_arr)
        tile_im = torch.from_numpy(tile_im)

        return TensorImage(tile_im), TensorMask(masks)

    def __len__(self):
        return len(self.roi_data)


class WsiTorchPatches(object):
    """Class to create a torch data_loader for simple image classification.

    Parameters
    ----------
    roi_data : list
        list of wsimap dask & numpy patch data
    transforms : func
        transforms to apply to images (not implemented)

    Attributes
    ----------
    class_data : type
        class dict for converting name to numerical index for training
    rev_class_data : type
        class dict for converting numercial index to name
    roi_data : list
        list of wsimap dask & numpy patch data
    transforms : func
        transforms to apply to images (not implemented)
    """

    def __init__(self, roi_data, transforms=None):
        self.patch_data = roi_data
        self.transforms = transforms

        class_arr = np.array([p["roi_classes"] for p in self.patch_data])
        self.class_data, self.rev_class_data = classes_to_dict(
            class_arr, bg_class=False
        )

    def __getitem__(self, idx, grab_np=False):
        # load images ad masks
        dask_im = self.patch_data[idx]["dask_im"]
        roi_class = self.patch_data[idx]["roi_classes"]
        tile_bounds = self.patch_data[idx]["tile_bounds"]

        tile_im = dask_im.compute()

        tile_im = tile_im / np.iinfo(tile_im.dtype).max
        tile_im = tile_im.astype(np.float32)
        # there is only one class
        target = torch.tensor(self.class_data[roi_class], dtype=torch.int64)

        tile_im = torch.from_numpy(tile_im)

        if tile_im.shape[1] != 224 or tile_im.shape[2] != 224:
            tile_im = tile_im.unsqueeze(0)
            tile_im = F.interpolate(tile_im, size=(224, 224)).squeeze()

        if self.transforms is not None:
            tile_im, target = self.transforms(tile_im, target)

        return tile_im, target

    def __add__(self, other):
        # add type checking here.
        patch_data = self.patch_data + other.patch_data
        return WsiTorchPatches(patch_data)

    def __len__(self):
        return len(self.patch_data)


class WsiTorchPatchesTiffFile(object):
    """Class to create a torch data_loader for simple image classification.

    Parameters
    ----------
    roi_data : list
        list of wsimap dask & numpy patch data
    transforms : func
        transforms to apply to images (not implemented)

    Attributes
    ----------
    class_data : type
        class dict for converting name to numerical index for training
    rev_class_data : type
        class dict for converting numercial index to name
    roi_data : list
        list of wsimap dask & numpy patch data
    transforms : func
        transforms to apply to images (not implemented)
    """

    def __init__(self, roi_data, transforms=None):
        self.patch_data = roi_data
        self.transforms = transforms

        class_arr = np.array([p["roi_classes"] for p in self.patch_data])
        self.class_data, self.rev_class_data = classes_to_dict(
            class_arr, bg_class=False
        )

    def __getitem__(self, idx, grab_np=False):
        # load images ad masks
        image_fp = self.patch_data[idx]["image_fp"]
        roi_class = self.patch_data[idx]["roi_classes"]
        # tile_bounds = self.patch_data[idx]["tile_bounds"]

        tile_im = imread(image_fp)
        tile_im = cv2.resize(tile_im, (224, 224))
        tile_im = np.swapaxes(tile_im, 0, 2)
        tile_im = np.swapaxes(tile_im, 1, 2)

        tile_im = tile_im / np.iinfo(tile_im.dtype).max

        tile_im = tile_im.astype(np.float32)
        # there is only one class
        target = torch.tensor(self.class_data[roi_class], dtype=torch.int64)

        tile_im = torch.from_numpy(tile_im)

        if tile_im.shape[1] != 224 or tile_im.shape[2] != 224:
            tile_im = tile_im.unsqueeze(0)
            tile_im = F.interpolate(tile_im, size=(224, 224)).squeeze()

        if self.transforms is not None:
            tile_im, target = self.transforms(tile_im, target)

        return tile_im, target

    def __add__(self, other):
        # add type checking here.
        patch_data = self.patch_data + other.patch_data
        return WsiTorchPatchesTiffFile(patch_data)

    def __len__(self):
        return len(self.patch_data)


class WsiDetectron2ObjDet(object):
    """Short summary.

    Parameters
    ----------
    roi_data : list
        list of wsimap dask & numpy tile/roi data
    transforms : func
        transforms to apply to images, rois (not implemented)

    Attributes
    ----------
    class_data : type
        class dict for converting name to numerical index for training
    rev_class_data : type
        class dict for converting numercial index to name
    prepare_detectron2 : function
        creates detectron2 input dicts for data
    roi_data : list
        list of wsimap dask & numpy tile/roi data
    transforms : func
        transforms to apply to images (not implemented)

    """

    def __init__(self, roi_data, transforms=None):
        self.roi_data = roi_data
        self.transforms = transforms

        class_arr = np.concatenate(
            [np.asarray(p["roi_classes"]) for p in self.roi_data]
        )
        self.class_data, self.rev_class_data = classes_to_dict(
            class_arr, bg_class=False
        )
        self.prepare_detectron2()

    def prepare_detectron2(self):
        """Format all roi data to list of dicts format used by detectron2"""
        self.detectron2_data = []
        for image_id, roi_data in enumerate(self.roi_data):

            im_data = {}

            roi_class = roi_data["roi_classes"]
            dask_im_tile = roi_data["dask_im"]
            tile_bounds = roi_data["tile_bounds"]
            rois = roi_data["rois"]

            im_data["dask_im"] = dask_im
            im_data["file_name"] = image_id
            im_data["image_id"] = image_id
            im_data["height"] = int(dask_im_tile.shape[2])
            im_data["width"] = int(dask_im_tile.shape[1])

            rois_tile = [roi - [tile_bounds[0], tile_bounds[1]] for roi in rois]

            num_objs = len(rois_tile)
            annotation_data = []

            for obj in range(num_objs):
                obj_data = {}
                xmin = int(np.min(rois_tile[obj][:, 0]))
                xmax = int(np.max(rois_tile[obj][:, 0]))
                ymin = int(np.min(rois_tile[obj][:, 1]))
                ymax = int(np.max(rois_tile[obj][:, 1]))

                obj_data["bbox"] = [xmin, ymin, xmax, ymax]
                obj_data["bbox_mode"] = BoxMode.XYXY_ABS
                obj_data["segmentation_poly"] = rois_tile[obj].astype(float)
                obj_data["category_id"] = self.class_data[roi_class[obj]]
                obj_data["iscrowd"] = 0
                obj_data["segmentation"] = [
                    obj_data["segmentation_poly"].flatten().tolist()
                ]
                obj_data["segmentation_poly"] = obj_data["segmentation_poly"].tolist()
                annotation_data.append(obj_data)

            im_data["annotations"] = annotation_data
            self.detectron2_data.append(im_data)

    def get_dict(self):
        """returns detectron2 dict. Needed for setting up model without
        constantly recreating the dict using prepare_detectron2 function.

        Returns
        -------
        list of dict
            detectron2 formatted list of dict inputs

        """
        return self.detectron2_data

    def __len__(self):
        return len(self.roi_data)


def dask_zarr_detectron2_mapper(dataset_dict, train=True):
    """dask_zarr data loader for detectron2

    Parameters
    ----------
    dataset_dict : dict
        detectron2 formatted input dict
    train : bool
        whether data is for training or validation purposes

    Returns
    -------
    dict
        detectron2 formatted input dict converted to torch.tensors

    """

    dataset_dict = copy.deepcopy(dataset_dict)

    # load in image using dask
    image = dataset_dict["dask_im"].compute()
    height, width = image.shape[1:]

    # rescale image between 0-1 based on datatype
    image = image / np.iinfo(image.dtype).max
    # rescale 0-1 data to be approximately around 8-bit values
    # this helps training as the inputs ARE float32 but most of the pretrained data
    # originals from 24-bit RGB. Here we mimic that WITHOUT losing the original data's
    # precision
    image = image * 255

    dataset_dict.pop("dask_im", None)
    dataset_dict["image"] = torch.as_tensor(image.astype("float32"))

    instances = utils.annotations_to_instances(
        dataset_dict["annotations"], (height, width)
    )

    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


class WsiDetectron2ObjDetTifffile(object):
    """Short summary.

    Parameters
    ----------
    roi_data : list
        list of wsimap dask & numpy tile/roi data
    transforms : func
        transforms to apply to images, rois (not implemented)

    Attributes
    ----------
    class_data : type
        class dict for converting name to numerical index for training
    rev_class_data : type
        class dict for converting numercial index to name
    prepare_detectron2 : function
        creates detectron2 input dicts for data
    roi_data : list
        list of wsimap dask & numpy tile/roi data
    transforms : func
        transforms to apply to images (not implemented)

    """

    def __init__(self, roi_data, transforms=None):
        self.roi_data = roi_data
        self.transforms = transforms

        class_arr = np.concatenate(
            [np.asarray(p["roi_classes"]) for p in self.roi_data]
        )
        self.class_data, self.rev_class_data = classes_to_dict(
            class_arr, bg_class=False
        )
        self.prepare_detectron2()

    def prepare_detectron2(self):
        """Format all roi data to list of dicts format used by detectron2"""
        self.detectron2_data = []
        for image_id, roi_data in enumerate(self.roi_data):

            im_data = {}

            roi_class = roi_data["roi_classes"]
            image_fp = roi_data["image_fp"]
            rois = roi_data["rois"]
            tile_bounds = roi_data["tile_bounds"]

            im_data["image_fp"] = image_fp
            im_data["file_name"] = image_id
            im_data["image_id"] = image_id
            im_data["tile_bounds"] = tile_bounds
            im_data["height"] = 1024
            im_data["width"] = 1024
            rois = [roi - [tile_bounds[0], tile_bounds[1]] for roi in rois]

            num_objs = len(rois)
            annotation_data = []
            for obj in range(num_objs):
                obj_data = {}
                xmin = int(np.min(rois[obj][:, 0]))
                xmax = int(np.max(rois[obj][:, 0]))
                ymin = int(np.min(rois[obj][:, 1]))
                ymax = int(np.max(rois[obj][:, 1]))

                obj_data["bbox"] = [xmin, ymin, xmax, ymax]
                obj_data["bbox_mode"] = BoxMode.XYXY_ABS
                obj_data["segmentation_poly"] = rois[obj].astype(float)
                obj_data["category_id"] = self.class_data[roi_class[obj]]
                obj_data["iscrowd"] = 0
                obj_data["segmentation"] = [
                    obj_data["segmentation_poly"].flatten().tolist()
                ]
                obj_data["segmentation_poly"] = obj_data["segmentation_poly"].tolist()
                annotation_data.append(obj_data)

            im_data["annotations"] = annotation_data
            self.detectron2_data.append(im_data)

    def get_dict(self):
        """returns detectron2 dict. Needed for setting up model without
        constantly recreating the dict using prepare_detectron2 function.

        Returns
        -------
        list of dict
            detectron2 formatted list of dict inputs

        """
        return self.detectron2_data

    def __len__(self):
        return len(self.roi_data)


class WsiDetectron2ObjDetTifffileR(object):
    """Short summary.

    Parameters
    ----------
    roi_data : list
        list of wsimap dask & numpy tile/roi data
    transforms : func
        transforms to apply to images, rois (not implemented)

    Attributes
    ----------
    class_data : type
        class dict for converting name to numerical index for training
    rev_class_data : type
        class dict for converting numercial index to name
    prepare_detectron2 : function
        creates detectron2 input dicts for data
    roi_data : list
        list of wsimap dask & numpy tile/roi data
    transforms : func
        transforms to apply to images (not implemented)

    """

    def __init__(self, roi_data, transforms=None):
        self.roi_data = roi_data
        self.transforms = transforms

        class_arr = np.concatenate(
            [np.asarray(p["roi_classes"]) for p in self.roi_data]
        )
        self.class_data, self.rev_class_data = classes_to_dict(
            class_arr, bg_class=False
        )
        self.prepare_detectron2()

    def prepare_detectron2(self):
        """Format all roi data to list of dicts format used by detectron2"""
        self.detectron2_data = []
        for image_id, roi_data in enumerate(self.roi_data):

            im_data = {}

            roi_class = roi_data["roi_classes"]
            image_fp = roi_data["image_fp"]
            rois = roi_data["rois"]
            tile_bounds = roi_data["tile_bounds"]

            im_data["image_fp"] = image_fp
            im_data["file_name"] = image_id
            im_data["image_id"] = image_id
            im_data["tile_bounds"] = tile_bounds
            im_data["class_data"] = self.class_data
            im_data["height"] = 1024
            im_data["width"] = 1024
            rois = [roi - [tile_bounds[0], tile_bounds[1]] for roi in rois]
            im_data["rois"] = rois
            im_data["roi_classes"] = roi_class
            self.detectron2_data.append(im_data)

    def get_dict(self):
        """returns detectron2 dict. Needed for setting up model without
        constantly recreating the dict using prepare_detectron2 function.

        Returns
        -------
        list of dict
            detectron2 formatted list of dict inputs

        """
        return self.detectron2_data

    def __len__(self):
        return len(self.roi_data)


def tifffile_detectron2_mapper(dataset_dict, train=True):
    """dask_zarr data loader for detectron2

    Parameters
    ----------
    dataset_dict : dict
        detectron2 formatted input dict
    train : bool
        whether data is for training or validation purposes

    Returns
    -------
    dict
        detectron2 formatted input dict converted to torch.tensors

    """

    dataset_dict = copy.deepcopy(dataset_dict)

    # load in image using dask
    # image = imread(dataset_dict["image_fp"], aszarr=True)
    # image = tifffile_to_dask(dataset_dict["image_fp"])
    image = imread(dataset_dict["image_fp"])

    # image = da.from_zarr(image)
    image = da.squeeze(image)
    # xs, ys, xe, ye = dataset_dict["tile_bounds"]
    # image = image[[0,1,2], ys:ye, xs:xe]

    # image = image[[0,1,3,4,6,8],:,:]

    height, width = image.shape[1:]

    dataset_dict["height"] = height
    dataset_dict["width"] = width

    # rescale image between 0-1 based on datatype
    image = image / np.iinfo(image.dtype).max
    # rescale 0-1 data to be approximately around 8-bit values
    # this helps training as the inputs ARE float32 but most of the pretrained data
    # originals from 24-bit RGB. Here we mimic that WITHOUT losing the original data's
    # precision
    # image = image / np.iinfo(image.dtype).max
    # image = image * 255
    # image = torch.as_tensor(image.astype("float32"))

    dataset_dict.pop("image_fp", None)
    dataset_dict["image"] = torch.as_tensor(image.astype("float32"))
    rois = dataset_dict.get("rois")
    if rois:
        instances = utils.annotations_to_instances(
            dataset_dict["annotations"], (height, width)
        )
    else:
        instances = utils.annotations_to_instances([], (height, width))

    dataset_dict["instances"] = instances

    return dataset_dict


def tifffile_detectron2_mapper_train(dataset_dict):
    """dask_zarr data loader for detectron2

    Parameters
    ----------
    dataset_dict : dict
        detectron2 formatted input dict
    train : bool
        whether data is for training or validation purposes

    Returns
    -------
    dict
        detectron2 formatted input dict converted to torch.tensors

    """

    transforms = A.Compose(
        [
            # A.OneOf(
            #     [
            #         A.RandomBrightnessContrast(brightness_limit=[-0.1, 0.1], contrast_limit=[-0.1, 0.1]),
            #         A.RandomGamma(),
            #     ],
            #     p=0.1,
            # ),
            # Additional position augmentations
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.75),
            A.VerticalFlip(p=0.75),
            # A.Blur(blur_limit=5, p=0.2),
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    )

    dataset_dict = copy.deepcopy(dataset_dict)

    # load in image using dask
    # image = imread(dataset_dict["image_fp"], aszarr=True)
    image = imread(dataset_dict["image_fp"])
    # image = da.from_zarr(image)
    image = da.squeeze(image)
    height, width = image.shape[1:]
    tdtype = image.dtype
    # xs, ys, xe, ye = dataset_dict["tile_bounds"]
    # image = image[[0,1,2], ys:ye, xs:xe]

    # image = image[[0,1,3,4,6,8],:,:]

    rois = dataset_dict["rois"]
    obj_data = []
    if len(rois) > 0:
        rois = [np.where(roi >= 1024, 1023, roi) for roi in rois]
        rois = [np.where(roi <= 0, 1, roi) for roi in rois]

        dataset_dict["orig_rois"] = rois

        roi_class = dataset_dict["roi_classes"]
        annotation_data = []

        if transforms:
            image = image.transpose(1, 2, 0)
            roi_lens = [len(roi) for roi in rois]
            flat_rois = np.concatenate(rois)
            # print(flat_rois[0])
            flat_rois = np.where(flat_rois == 1024, 1023, flat_rois)
            # flat_rois = flat_rois[:,[1,0]]
            augmented = transforms(image=image, keypoints=flat_rois)
            # print("image :", np.array_equal(image, augmented["image"]))
            # print("kps :", np.array_equal(flat_rois, augmented["keypoints"]))

            # if self.augmedical_transforms:
            #     for transform in self.augmedical_transforms:
            #         tile_im = transform(tile_im)

            image = augmented["image"].transpose(2, 0, 1).astype(tdtype)
            tformed_rois = np.asarray(augmented["keypoints"])
            # tformed_rois = tformed_rois[:,[1,0]]
            roi_splits = np.cumsum(roi_lens)
            new_rois = []
            for idx, s in enumerate(roi_splits):
                if idx == len(roi_splits):
                    new_rois.append(tformed_rois[s : len(roi_splits)])
                elif idx == 0:
                    new_rois.append(tformed_rois[0:s])
                else:
                    prev_s = roi_splits[idx - 1]
                    new_rois.append(tformed_rois[prev_s:s])

            rois = [np.asarray(r) for r in new_rois]
            dataset_dict["rois"] = rois

        num_objs = len(rois)
        for obj in range(num_objs):
            obj_data = {}
            xmin = int(np.min(rois[obj][:, 0]))
            xmax = int(np.max(rois[obj][:, 0]))
            ymin = int(np.min(rois[obj][:, 1]))
            ymax = int(np.max(rois[obj][:, 1]))

            obj_data["bbox"] = [xmin, ymin, xmax, ymax]
            obj_data["bbox_mode"] = BoxMode.XYXY_ABS
            obj_data["segmentation_poly"] = rois[obj].astype(float)
            obj_data["category_id"] = dataset_dict["class_data"][roi_class[obj]]
            obj_data["iscrowd"] = 0
            obj_data["segmentation"] = [
                obj_data["segmentation_poly"].flatten().tolist()
            ]
            obj_data["segmentation_poly"] = obj_data["segmentation_poly"].tolist()
            annotation_data.append(obj_data)

    if len(rois) > 0:
        dataset_dict["annotations"] = annotation_data
    else:
        dataset_dict["annotations"] = []

    # rescale image between 0-1 based on datatype
    image = image / np.iinfo(image.dtype).max
    # rescale 0-1 data to be approximately around 8-bit values
    # this helps training as the inputs ARE float32 but most of the pretrained data
    # originals from 24-bit RGB. Here we mimic that WITHOUT losing the original data's
    # precision
    image = image * 255

    dataset_dict.pop("image_fp", None)
    dataset_dict["image"] = torch.as_tensor(image.astype("float32"))
    try:
        dataset_dict["annotations"] = annotation_data
    except UnboundLocalError:
        dataset_dict["annotations"] = []

    instances = utils.annotations_to_instances(
        dataset_dict["annotations"], (height, width)
    )

    dataset_dict["instances"] = instances

    return dataset_dict
