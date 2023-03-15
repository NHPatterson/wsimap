import os
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
)
from wsimap.wsi_map import WsiMap
from wsimap.utils.wsi_torch_utils import (
    WsiDetectron2ObjDet,
    dask_zarr_detectron2_mapper,
)

zarr_fp = "/home/ubuntu/biomic/glom/glom3d_S108/S108_3D_AF_preMxIF_sec01.zarr"
glom_roi_fp = "/home/ubuntu/biomic/glom/S108_MxIF_Sec01_sc2_glomeruli.json"

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


roi = glom_wsi.annotations["glomerulus"][0]

geojson_type = "feature"
id = "PathAnnotationObject"
geo_type = roi.roi_attrs["type"]
coordinates = [np.column_stack(roi.exterior.coords.xy).tolist()]
properties = {
    "classification": {"name": roi.roi_attrs["name"], "colorRGB": -3140401},
    "isLocked": False,
    "measurements": [],
}
geojson_dict = {
    "type": geojson_type,
    "id": id,
    "geometry": {"type": geo_type, "coordinates": coordinates},
    "properties": properties,
}

multi_feature = [geojson_dict, geojson_dict]

import json

json_str = json.dumps(geojson_dict)
with open("test_json_m.json", "w") as json_file:
    json.dump(multi_feature, json_file)

with open("test_json_m.json") as f:
    polygons = json.load(f)


def build_train_val_split(roi_data, val_percent=0.15):

    indices = np.random.permutation(len(roi_data)).astype(int)
    subset = np.floor(len(roi_data) * val_percent).astype(int)

    train_data = [roi_data[idx] for idx in indices[:-subset]]
    test_data = [roi_data[idx] for idx in indices[-subset:]]

    return train_data, test_data


train_data, val_data = build_train_val_split(glom_wsi.roi_data_np, val_percent=0.1)

train_data = WsiDetectron2ObjDet(train_data)
test_data = WsiDetectron2ObjDet(val_data)

DatasetCatalog.register("gloms_train", train_data.get_dict)
MetadataCatalog.get("gloms_train").set(thing_classes=["glomerulus"])
DatasetCatalog.register("gloms_val", test_data.get_dict)
MetadataCatalog.get("gloms_val").set(thing_classes=["glomerulus"])

statement_metadata = MetadataCatalog.get("glom_train")


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=dask_zarr_detectron2_mapper
        )

    @classmethod
    def build_train_loader(cls, cfg):
        print("trainloader")
        return build_detection_train_loader(cfg, mapper=dask_zarr_detectron2_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


cfg = get_cfg()

cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    )
)

# use fine tuned model weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
)

# set up datasets
cfg.DATASETS.TRAIN = ("gloms_train",)
cfg.DATASETS.TEST = ("gloms_val",)

# zarr based format requires 0 for multiprocessing
cfg.DATALOADER.NUM_WORKERS = 0

# model training hyperparams
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 100
cfg.SOLVER.MAX_ITER = 7000
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64

# set to no of classes in dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# how often to run the tests
cfg.TEST.EVAL_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)

trainer.train()
