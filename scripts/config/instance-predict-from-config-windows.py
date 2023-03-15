import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from pathlib import Path
import yaml
import time
from wsimap.wsi_map import WsiMap
from wsimap.utils.merge_utils import (
    merge_shapely_predictions_rtree,
    prediction_to_shapely,
    prediction_to_qp_geojson,
    merged_to_prediction_dict,
)
import json
from czifile import CziFile
import zarr
import dask.array as da
import SimpleITK as sitk
from tifffile import imread
import pickle


def fix_file_paths_linux_windows(fp: str):
    # assumes this is running locally at VU
    if sys.platform.startswith("linux"):
        fp = fp.replace("O:/", "/home/nhp/supernas/")
        fp = fp.replace("Y:/", "/home/nhp/y_drive/")
        fp = fp.replace("Z:/", "/home/nhp/linux-share/")

    elif sys.platform.startswith("win32"):
        fp = fp.replace("/home/nhp/supernas/", "O:/")
        fp = fp.replace("/home/nhp/y_drive/", "Y:/")
        fp = fp.replace("/home/nhp/linux-share/", "Z:/")

    return fp

def tifffile_to_dask(im_fp, level=0):
    imdata = zarr.open(imread(im_fp, aszarr=True))
    if isinstance(imdata, zarr.hierarchy.Group):
        imdata = da.from_zarr(zarr.open(imread(im_fp, aszarr=True, level=level)))
    else:
        imdata = da.from_zarr(imdata)
    return imdata


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def fix_file_paths_linux_windows(fp: str):
    # assumes this is running locally at VU
    if sys.platform.startswith("linux"):
        fp = fp.replace("O:/", "/home/nhp/supernas/")
        fp = fp.replace("Y:/", "/home/nhp/y_drive/")
        fp = fp.replace("Z:/", "/home/nhp/linux-share/")

    elif sys.platform.startswith("win32"):
        fp = fp.replace("/home/nhp/supernas/", "O:/")
        fp = fp.replace("/home/nhp/y_drive/", "Y:/")
        fp = fp.replace("/home/nhp/linux-share/", "Z:/")

    return fp


class CONFIG:
    tile_shape = (1024, 1024)

    # pytorch model (segmentation_models_pytorch)
    architecture = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    in_channels = 3
    n_classes = 2

    def __init__(self, config):
        config_fp = fix_file_paths_linux_windows(config)
        print(config_fp)
        import_config = yaml.safe_load(open(config_fp, "r"))

        self.model_path = import_config.get("model_path")

        if import_config.get("n_channels"):
            self.in_channels = import_config.get("n_channels")

        self.testing_images = import_config.get("test_images")

        if import_config.get("tissue_rois") is None:
            self.tissue_rois = [None for _ in self.testing_images]
        else:
            self.tissue_rois = import_config.get("tissue_rois")

        self.prediction_class = import_config.get("prediction_class")
        self.output_dir = import_config.get("output_dir")

        if import_config.get("threshold"):
            self.threshold = import_config.get("threshold")
        else:
            self.threshold = 0.5

        if import_config.get("model_tag"):
            self.model_tag = import_config.get("model_tag")
        else:
            self.model_tag = f"instance-model"

        if import_config.get("downsampling"):
            self.downsampling = import_config.get("downsampling")
        else:
            self.downsampling = 1

        if import_config.get("train_channels"):
            train_channels = import_config.get("train_channels")
            if isinstance(train_channels, list) is False:
                self.train_channels = [int(train_channels)]
            else:
                self.train_channels = train_channels

            if len(self.train_channels) != self.in_channels:
                print(
                    "Number of channels selected for training differs from number "
                    "of channels selected for model, changing number of channels"
                )
                self.in_channels = len(self.train_channels)

        if import_config.get("roi_expansion"):
            if isinstance(import_config.get("roi_expansion"), list) is False:
                self.roi_expansion = [import_config.get("roi_expansion")]
            else:
                self.roi_expansion = import_config.get("roi_expansion")
        else:
            self.roi_expansion = [1]

        if import_config.get("size_filter"):
            self.size_filter = import_config.get("size_filter")
        else:
            self.size_filter = 0

        if import_config.get("qp_type"):
            self.qp_type = import_config.get("qp_type")
        else:
            self.qp_type = "annotation"

        if import_config.get("max_workers"):
            self.max_workers = import_config.get("max_workers")
        else:
            self.max_workers = None

        if import_config.get("prediction_area_name"):
            self.prediction_area_name = import_config.get("prediction_area_name")
        else:
            self.prediction_area_name = "train-area"


        self.testing_images = [
            fix_file_paths_linux_windows(fp) for fp in self.testing_images if fp
        ]
        self.tissue_rois = [fix_file_paths_linux_windows(fp) for fp in self.tissue_rois if fp]

        if len(self.tissue_rois) == 0:
            self.tissue_rois = [None for _ in self.testing_images]


        self.output_dir = fix_file_paths_linux_windows(self.output_dir)

        self.model_path = fix_file_paths_linux_windows(self.model_path)

if __name__ == "__main__":
    import sys

    # "Z:\segmentation\DR3\tal-configs\"
    # cfg = CONFIG("Z:/segmentation/DR3/tal-configs/AF-tal-instance-refine-predict.yaml")

    config_fp = sys.argv[1]
    cfg = CONFIG(config_fp)
    d2cfg = get_cfg()

    d2cfg.merge_from_file(model_zoo.get_config_file(cfg.architecture))

    # training specific
    d2cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    d2cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.n_classes - 1

    # final model data
    d2cfg.MODEL.WEIGHTS = str(cfg.model_path)

    # threshold to have a detection returned
    d2cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.threshold

    predictor = WSIPredictor(d2cfg)


    for test_image_fp, tissue_roi in zip(cfg.testing_images, cfg.tissue_rois):
        print(f"Current image: {Path(test_image_fp).name}")
        print("image reading..")

        if Path(test_image_fp).suffix.lower() == ".czi":
            full_im = CziFile(test_image_fp)
            full_im = np.squeeze(full_im.asarray(max_workers=cfg.max_workers))
            # full_im[0,:,:] = full_im[1,:,:]
            if cfg.downsampling != 1:
                full_im = sitk.GetImageFromArray(full_im)
                full_im = sitk.Shrink(full_im, (cfg.downsampling, cfg.downsampling, 1))
                full_im = sitk.GetArrayFromImage(full_im)
        else:
            pyr_level = int(np.log2(cfg.downsampling))
            full_im = imread(test_image_fp, level=pyr_level, key=cfg.train_channels)
            full_im = da.squeeze(full_im)
            # full_im = full_im[list(cfg.train_channels), :, :].compute()

        if len(full_im.shape) == 2:
            full_im = np.expand_dims(full_im, 0)

        dask_im = da.from_array(full_im, chunks=(1, 2048, 2048))

        wsimap = WsiMap(
            dask_im,
            pyr_layer=0,
        )

        if tissue_roi:
            wsimap.import_annotations(tissue_roi, ds=cfg.downsampling)

        wsimap.tile_wsi_tiler(
            bound_anno=cfg.prediction_area_name,
            tile_size=1024,
            overlap=0.5,
            edge_thresh=0.9,
        )

        wsimap.tile_data_to_patches()
        patches = wsimap.patch_data

        unique_shapes = np.unique(
            np.asarray([p["dask_im"].shape for p in patches]), axis=[0]
        )

        patch_sets = []
        for idx, unique_shape in enumerate(unique_shapes):
            patch_set = [
                p for p in patches if p["dask_im"].shape == tuple(unique_shape)
            ]
            patch_sets.append(patch_set)

        print("prediction..")
        # try:
            # predictions = predictor(patches)
        all_predictions = []

        for idx, patch_set in enumerate(patch_sets):
            print(
                f"patch set {tuple(unique_shapes[idx])}, {len(patch_set)} patches "
            )
            if len(patch_set) < 12:
                batch_size = len(patch_set)
            else:
                batch_size = 12
            predictions = predictor.batched_prediction(
                patch_set, full_im=full_im, batch_size=batch_size
            )
            all_predictions.extend(predictions)
        # except:
        #     print(f"{test_image_fp} errored")
        #     continue

        rev_class_dict = {
            0: f"{cfg.prediction_class}",
            # 2: f"Distal Tubule",
            # 3: f"Proximal Tubule",
            # 4: f"Vasculature",
        }

        experiment_name = f"{Path(test_image_fp).stem.replace('.ome', '')}-{cfg.model_tag}-thresh-{cfg.threshold}.geojson"
        # experiment_name_teach = f"{Path(test_image_fp).stem.replace('.ome', '')}-{cfg.model_tag}-thresh-{cfg.threshold}-teach.geojson"
        experiment_name_pkl = f"{Path(test_image_fp).stem.replace('.ome', '')}-{cfg.model_tag}-thresh-{cfg.threshold}.pkl"
        output_preds_pkl = Path(cfg.output_dir) / experiment_name_pkl
        #
        # with open(output_preds_pkl, "rb") as infile:
        #     predictions = pickle.load(infile)

        predictions_update = []
        if cfg.downsampling != 1:
            for p in all_predictions:
                p["polygon"] = p["polygon"] * cfg.downsampling
                predictions_update.append(p)
        else:
            predictions_update = all_predictions

        with open(output_preds_pkl, "wb") as outfile:
            pickle.dump(predictions, outfile)
        # predictions_update = predictions

        print("merging..")
        predictions_sh = [
            prediction_to_shapely(p, rev_class_dict) for p in predictions_update
        ]
        start_time = time.time()
        merged_polygons = merge_shapely_predictions_rtree(predictions_sh)
        end_time = time.time()
        print(f"\nMerging required: {round((end_time - start_time) / 60, 3)} m")

        mps = [p for p in merged_polygons if p.is_valid]
        mps = [p for p in mps if p.area > cfg.size_filter]
        roi_attrs = [p.roi_attrs for p in mps]
        pdd = merged_to_prediction_dict(mps)
        json_pred = [
            prediction_to_qp_geojson(p, rev_class_dict, qp_type=cfg.qp_type)
            for p in pdd
        ]

        experiment_name = (
            f"{Path(test_image_fp).stem.replace('.ome', '')}-"
            f"{cfg.model_tag}-thresh-{cfg.threshold}.geojson"
        )

        output_preds = Path(cfg.output_dir) / experiment_name

        try:
            print(f"writing {len(json_pred)} merged predictions to json..")
            with open(output_preds, "w") as json_file:
                json.dump(json_pred, json_file, indent=1, cls=NpEncoder)
        except PermissionError:
            output_preds = Path("/home/ubuntu/trained_unet_models") / experiment_name
            print(f"writing {len(json_pred)} merged predictions to json..")
            with open(output_preds, "w") as json_file:
                json.dump(json_pred, json_file, indent=1, cls=NpEncoder)
