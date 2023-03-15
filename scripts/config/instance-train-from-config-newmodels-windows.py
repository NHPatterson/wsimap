import os
from pathlib import Path
import tempfile
import yaml
import pickle
import logging

import numpy as np
from shapely import geometry
from tifffile import imwrite

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    DefaultTrainer,
    AMPTrainer,
    SimpleTrainer,
    default_writers,
    hooks,
)
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.config import instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format, COCOEvaluator
from detectron2.utils import comm

from wsimap.wsi_map import WsiMap
from wsimap.utils.anno_utils import get_train_area_offsets, get_n_train_areas
from wsimap.utils.wsi_torch_utils import (
    tifffile_detectron2_mapper_train,
    tifffile_detectron2_mapper,
    WsiDetectron2ObjDetTifffile,
    WsiDetectron2ObjDetTifffileR,
)
from tqdm import tqdm


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=tifffile_detectron2_mapper
        )

    @classmethod
    def build_train_loader(cls, cfg):
        print("trainloader")
        return build_detection_train_loader(
            cfg, mapper=tifffile_detectron2_mapper_train
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def reverse_class_dict(class_dict):
    rev_class_dict = {}
    for k, v in class_dict.items():
        rev_class_dict[v] = k
    return rev_class_dict


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
    # paths
    # deepflash2 dataset
    tile_shape = (1024, 1024)

    # pytorch model (segmentation_models_pytorch)
    train_channels = None
    architecture = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    in_channels = 3

    # Training
    batch_size = 2
    max_learning_rate = 1e-3
    epoch = 10
    eval_rate = 750

    def __init__(self, config):
        config = fix_file_paths_linux_windows(config)
        import_config = yaml.safe_load(open(config, "r"))

        if import_config.get("architecture"):
            self.architecture = import_config.get("architecture")

        if import_config.get("batch_size"):
            self.batch_size = import_config.get("batch_size")

        if import_config.get("epoch"):
            self.epoch = import_config.get("epoch")

        if import_config.get("n_channels"):
            self.in_channels = import_config.get("n_channels")

        self.training_images = import_config.get("train_images")
        self.training_rois = import_config.get("train_rois")
        self.valid_images = import_config.get("valid_images")
        self.valid_rois = import_config.get("valid_rois")

        if import_config.get("tissue_rois") is None:
            self.tissue_rois = [None for _ in range(len(self.training_images))]
        else:
            self.tissue_rois = import_config.get("tissue_rois")

        if import_config.get("negative_roi"):
            self.neg_roi = import_config.get("negative_roi")
        else:
            self.neg_roi = None

        if import_config.get("tile_overlap"):
            self.overlap = import_config.get("tile_overlap")
        else:
            self.overlap = 0.5

        if import_config.get("neg_proportion"):
            self.neg_proportion = import_config.get("neg_proportion")
        else:
            self.neg_proportion = 0.5

        self.prediction_class = import_config.get("prediction_class")
        self.all_classes = import_config.get("all_classes")
        self.output_dir = import_config.get("output_dir")

        if import_config.get("model_tag"):
            self.model_tag = import_config.get("model_tag")
        else:
            self.model_tag = "untagged-model"

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

        if import_config.get("downsampling"):
            self.downsampling = import_config.get("downsampling")
        else:
            self.downsampling = 1

        if import_config.get("offset_to_train_area"):
            self.offset_to_train_area = import_config.get("offset_to_train_area")
        else:
            self.offset_to_train_area = False

        if import_config.get("train_class"):
            self.train_class = import_config.get("train_class")
        else:
            self.train_class = "train-area"

        if import_config.get("reload_from_pickle"):
            self.reload_from_pickle = import_config.get("reload_from_pickle")
        else:
            self.reload_from_pickle = False

        if import_config.get("solver_steps"):
            self.solver_steps = import_config.get("solver_steps")
        else:
            self.solver_steps = [0.5, 0.75, 0.9]

        self.size_filter = (
            import_config.get("size_filter") if import_config.get("size_filter") else 0
        )

        self.data_dir = import_config.get("data_dir")

        self.training_images = [
            fix_file_paths_linux_windows(fp) for fp in self.training_images if fp
        ]
        self.training_rois = [
            fix_file_paths_linux_windows(fp) for fp in self.training_rois if fp
        ]
        if self.valid_images:
            self.valid_images = [
                fix_file_paths_linux_windows(fp) for fp in self.valid_images if fp
            ]
        if self.valid_rois:
            self.valid_rois = [
                fix_file_paths_linux_windows(fp) for fp in self.valid_rois if fp
            ]

        self.data_dir = fix_file_paths_linux_windows(self.data_dir)
        self.output_dir = fix_file_paths_linux_windows(self.output_dir)


if __name__ == "__main__":
    import sys

    cfg = CONFIG(sys.argv[1])

    all_classes = cfg.all_classes
    all_classes.pop(all_classes.index(cfg.prediction_class))

    if not cfg.data_dir:
        data_dir = Path(tempfile.TemporaryDirectory())
    else:
        data_dir = Path(cfg.data_dir)

    train_data = []
    valid_data = []
    neg_data = []

    if not cfg.reload_from_pickle:
        for train_im, train_anno, roi_anno in zip(
            cfg.training_images, cfg.training_rois, cfg.tissue_rois
        ):
            print(f"loading image: {str(train_im)} with")
            print(f"annotations from: {str(train_anno)}")
            if roi_anno:
                n_train_areas = get_n_train_areas(roi_anno, cfg.train_class)
                tissue_roi_in = roi_anno
            else:
                n_train_areas = get_n_train_areas(train_anno, "train-area")
                tissue_roi_in = train_anno

            for area_idx in range(n_train_areas):
                if cfg.downsampling == 1:
                    wmap = WsiMap(str(train_im))
                elif cfg.downsampling == 2:
                    wmap = WsiMap(str(train_im), pyr_layer=1)
                elif cfg.downsampling == 4:
                    wmap = WsiMap(str(train_im), pyr_layer=2)

                if cfg.train_channels:
                    wmap.dask_im = wmap.dask_im[cfg.train_channels, :, :]

                if cfg.offset_to_train_area:
                    x_start, x_end, y_start, y_end = get_train_area_offsets(
                        tissue_roi_in,
                        cfg.train_class,
                        downsampling=cfg.downsampling,
                        area_idx=area_idx,
                    )
                    wmap.dask_im = wmap.dask_im[:, y_start:y_end, x_start:x_end]
                    # assume only 2D multichannel images, dimension order after squeeze: CYX
                    x_axis, y_axis = 2, 1
                    wmap.image_bounds = geometry.box(0, 0, x_end, y_end)

                    x_start *= -1
                    y_start *= -1
                    wmap.import_annotations(
                        train_anno, ds=cfg.downsampling, offset=[x_start, y_start]
                    )
                    if roi_anno:
                        wmap.import_annotations(
                            roi_anno, ds=cfg.downsampling, offset=[x_start, y_start]
                        )
                    train_area = "offset"
                else:
                    wmap.import_annotations(train_anno, ds=cfg.downsampling)
                    train_area = "train-area"

                pre_filt_n_annos = len(wmap.annotations)
                wmap.size_filter_annotations(100)
                post_filt_n_annos = len(wmap.annotations)

                # import napari
                # from wsimap.utils.anno_utils import shcoords_to_np
                # full_im = wmap.dask_im
                # all_rois = [shcoords_to_np(roi.simplify(0.99, preserve_topology=True))[:,[1,0]] for roi in wmap.annotations]
                # # rois = [roi[:,[1,0]] for roi in rois]
                # viewer = napari.Viewer()
                # viewer.add_image(full_im, channel_axis=0)
                # viewer.add_shapes(all_rois, shape_type="polygon")

                # print(f"{pre_filt_n_annos - post_filt_n_annos} annotations filtered by size")

                wmap.tile_wsi_tiler(
                    bound_anno=cfg.train_class,
                    tile_size=1024,
                    overlap=cfg.overlap,
                    edge_thresh=0.95,
                )

                wmap.intersect_tiles_annos(exclusion_roi=all_classes)

                [p.update({"image_fp": train_im}) for p in wmap.tile_roi_data]

                single_im_data = [
                    t
                    for t in wmap.tile_roi_data
                    if t["dask_im"].shape[1:] == (1024, 1024)
                ]

                # import napari
                # from wsimap.utils.anno_utils import shcoords_to_np
                # full_im = wmap.dask_im
                # all_rois = [shcoords_to_np(roi.simplify(0.99, preserve_topology=True))[:,[1,0]] for roi in wmap.annotations]
                # # rois = [roi[:,[1,0]] for roi in rois]
                # idx = 6
                #
                # full_im = single_im_data[idx]["dask_im"]
                # rois = single_im_data[idx]["rois"].copy()
                # tile_bounds = single_im_data[idx]["tile_bounds"]
                # rois = [roi - [tile_bounds[0], tile_bounds[1]] for roi in rois]
                # rois = [roi[:,[1,0]] for roi in rois]
                #
                # viewer = napari.Viewer()
                # viewer.add_image(full_im, channel_axis=0)
                # viewer.add_shapes(rois, shape_type="polygon")

                train_data.extend(single_im_data)
                if not cfg.offset_to_train_area:
                    break

            if cfg.neg_roi:
                n_neg_images = []
                if roi_anno:
                    n_neg_areas = get_n_train_areas(roi_anno, cfg.neg_roi)
                    tissue_roi_in = roi_anno
                else:
                    n_neg_areas = get_n_train_areas(train_anno, "negative-area")
                    tissue_roi_in = train_anno

                for area_idx in range(n_neg_areas):
                    if cfg.downsampling == 1:
                        wmap = WsiMap(str(train_im))
                    elif cfg.downsampling == 2:
                        wmap = WsiMap(str(train_im), pyr_layer=1)
                    elif cfg.downsampling == 4:
                        wmap = WsiMap(str(train_im), pyr_layer=2)

                    if cfg.train_channels:
                        wmap.dask_im = wmap.dask_im[cfg.train_channels, :, :]

                    if cfg.offset_to_train_area:
                        x_start, x_end, y_start, y_end = get_train_area_offsets(
                            tissue_roi_in,
                            cfg.neg_roi,
                            downsampling=cfg.downsampling,
                            area_idx=area_idx,
                        )
                        wmap.dask_im = wmap.dask_im[:, y_start:y_end, x_start:x_end]
                        # assume only 2D multichannel images, dimension order after squeeze: CYX
                        x_axis, y_axis = 2, 1
                        wmap.image_bounds = geometry.box(0, 0, x_end, y_end)

                        x_start *= -1
                        y_start *= -1
                        wmap.import_annotations(
                            train_anno, ds=cfg.downsampling, offset=[x_start, y_start]
                        )
                        if roi_anno:
                            wmap.import_annotations(
                                roi_anno, ds=cfg.downsampling, offset=[x_start, y_start]
                            )
                        train_area = "offset"
                    else:
                        wmap.import_annotations(train_anno, ds=cfg.downsampling)
                        train_area = "train-area"

                    pre_filt_n_annos = len(wmap.annotations)
                    wmap.size_filter_annotations(100)
                    post_filt_n_annos = len(wmap.annotations)

                    # print(f"{pre_filt_n_annos - post_filt_n_annos} annotations filtered by size")

                    wmap.tile_wsi_tiler(
                        bound_anno=cfg.neg_roi,
                        tile_size=1024,
                        overlap=0.05,
                        edge_thresh=0.99,
                    )

                    wmap.intersect_tiles_annos(
                        exclusion_roi=all_classes, empty_set=True
                    )

                    [p.update({"image_fp": train_im}) for p in wmap.tile_roi_data]

                    neg_images = [
                        t
                        for t in wmap.tile_roi_data
                        if t["dask_im"].shape[1:] == (1024, 1024)
                    ]

                    # import napari
                    # full_im = wmap.dask_im
                    # viewer = napari.Viewer()
                    # viewer.add_image(full_im, channel_axis=0)

                    n_neg_images.append(len(neg_images))
                    neg_data.extend(neg_images)

        if cfg.valid_images:
            for train_im, train_anno, roi_anno in zip(
                cfg.valid_images, cfg.valid_rois, cfg.valid_tissue_rois
            ):

                if cfg.downsampling == 1:
                    wmap = WsiMap(str(train_im))
                elif cfg.downsampling == 2:
                    wmap = WsiMap(str(train_im), pyr_layer=1)
                elif cfg.downsampling == 4:
                    wmap = WsiMap(str(train_im), pyr_layer=2)

                if cfg.train_channels:
                    wmap.dask_im = wmap.dask_im[cfg.train_channels, :, :]

                if cfg.offset_to_train_area:
                    x_start, x_end, y_start, y_end = get_train_area_offsets(
                        train_anno, downsampling=cfg.downsampling
                    )
                    wmap.dask_im = wmap.dask_im[:, y_start:y_end, x_start:x_end]
                    print(wmap.dask_im)
                    # assume only 2D multichannel images, dimension order after squeeze: CYX
                    x_axis, y_axis = 2, 1
                    wmap.image_bounds = geometry.box(0, 0, x_end, y_end)

                    x_start *= -1
                    y_start *= -1
                    wmap.import_annotations(
                        train_anno, ds=cfg.downsampling, offset=[x_start, y_start]
                    )
                    # print(wmap.annotations[0].exterior.xy)
                    train_area = "offset"
                else:
                    wmap.import_annotations(train_anno, ds=cfg.downsampling)
                    train_area = "train-area"

                if roi_anno:
                    wmap.import_annotations(roi_anno, ds=cfg.downsampling)

                pre_filt_n_annos = len(wmap.annotations)
                wmap.size_filter_annotations(100)
                post_filt_n_annos = len(wmap.annotations)

                # print(f"{pre_filt_n_annos - post_filt_n_annos} annotations filtered by size")

                wmap.tile_wsi(
                    bound_anno=train_area,
                    tile_size=1024,
                    # slide=[(256, 256)],
                    # slide=[(0, 128), (1 (128, 128), (0, 256), (256, 0), (256, 256)],
                    slide=None,
                    edge_thresh=0.9,
                    build_patches=False,
                )

                wmap.intersect_tiles_annos(exclusion_roi=all_classes)

                [p.update({"image_fp": train_im}) for p in wmap.tile_roi_data]

                single_im_data = [
                    t
                    for t in wmap.tile_roi_data
                    if t["dask_im"].shape[1:] == (1024, 1024)
                ]
                valid_data.extend(single_im_data)

                if cfg.neg_roi:
                    n_neg_images = []
                    if cfg.downsampling == 1:
                        wmap_neg = WsiMap(str(train_im))
                    elif cfg.downsampling == 2:
                        wmap_neg = WsiMap(str(train_im), pyr_layer=1)
                    elif cfg.downsampling == 4:
                        wmap_neg = WsiMap(str(train_im), pyr_layer=2)

                    wmap_neg.dask_im = wmap_neg.dask_im[cfg.train_channels, :, :]
                    wmap_neg.import_annotations(train_anno, ds=cfg.downsampling)
                    wmap_neg.tile_wsi(
                        bound_anno=cfg.neg_roi,
                        tile_size=1024,
                        # slide=[(256, 256)],
                        slide=None,
                        edge_thresh=0.9,
                        build_patches=True,
                    )
                    wmap_neg.intersect_tiles_annos(empty_set=True)

                    [p.update({"image_fp": train_im}) for p in wmap_neg.tile_roi_data]
                    neg_images = [
                        t
                        for t in wmap_neg.tile_roi_data
                        if t["dask_im"].shape[1:] == (1024, 1024)
                    ]
                    # n_empty = int(np.ceil(cfg.neg_proportion * len(single_im_data)))
                    # if n_empty < len(neg_images):
                    #     neg_images_selected = np.random.choice(
                    #         np.arange(0, len(neg_images), 1), n_empty, replace=False
                    #     )
                    #     neg_images = [neg_images[i] for i in neg_images_selected]
                    n_neg_images.append(len(neg_images))
                    valid_data.extend(neg_images)

        if not cfg.valid_images:
            train_length = int(np.floor(0.95 * len(train_data)))
            test_length = len(train_data) - train_length
            indices = np.random.permutation(len(train_data)).astype(int)
            train_datai = [train_data[idx] for idx in indices[:-test_length]]
            test_datai = [train_data[idx] for idx in indices[-test_length:]]
        else:
            train_datai = train_data
            test_datai = valid_data

        n_empty = int(np.ceil(cfg.neg_proportion * len(train_datai)))
        print(n_empty)
        # if n_empty < len(neg_data):
        #     neg_images_selected = np.random.choice(
        #         np.arange(0, len(neg_data), 1), n_empty, replace=False
        #     )
        #     neg_data = [neg_data[i] for i in neg_images_selected]

        train_datai.extend(neg_data)

        print("train data length: ", len(train_datai))
        print("negative example train data length: ", len(neg_data))
        print("test data length: ", len(test_datai))

        write_train = len(sorted(data_dir.glob("train-*.tiff"))) != len(train_datai)
        write_test = len(sorted(data_dir.glob("test-*.tiff"))) != len(test_datai)
        # data_dir = Path("C:/temp/pt-train-data")

        if not Path(data_dir).exists():
            print(f"masking directory at {Path(data_dir).as_posix()}")
            Path(data_dir).mkdir(exist_ok=True)

        for idx, inst in enumerate(tqdm(train_datai)):
            out_file = data_dir / f"train-{idx}.tiff"
            if write_train:
                imwrite(out_file, inst["dask_im"], compression="deflate")
            inst["dask_im"] = None
            inst["image_fp"] = str(out_file)

        for idx, inst in enumerate(tqdm(test_datai)):
            out_file = data_dir / f"test-{idx}.tiff"
            if write_test:
                imwrite(out_file, inst["dask_im"], compression="deflate")
            inst["dask_im"] = None
            inst["image_fp"] = str(out_file)

        with open(Path(cfg.data_dir) / f"{cfg.model_tag}-train-data.pkl", "wb") as f:
            pickle.dump(train_datai, f)
        with open(Path(cfg.data_dir) / f"{cfg.model_tag}-test-data.pkl", "wb") as f:
            pickle.dump(test_datai, f)

    else:
        with open(Path(cfg.data_dir) / f"{cfg.model_tag}-train-data.pkl", "rb") as f:
            train_datai = pickle.load(f)
        with open(Path(cfg.data_dir) / f"{cfg.model_tag}-test-data.pkl", "rb") as f:
            test_datai = pickle.load(f)

    # train_datai = pickle.load(open("Z:/segmentation/DR3/pts-configs/dr3-af-all-data.pkl", "rb"))
    # test_datai = pickle.load(open("Z:/segmentation/DR3/pts-configs/dr3-af-all-data-test.pkl", "rb"))
    # train_length = len(train_datai)
    # visualization test
    # import napari
    # from tifffile import imread
    # idx = 60
    # rois = train_datai[idx]["rois"].copy()
    # tile_bounds = train_datai[idx]["tile_bounds"].copy()
    # rois = [roi.astype(np.int32) - [tile_bounds[0], tile_bounds[1]] for roi in rois]
    # rois = [roi[:,[1,0]] for roi in rois]
    #
    # viewer = napari.Viewer()
    # viewer.add_image(imread(train_datai[idx]["image_fp"]), channel_axis=0)
    # viewer.add_shapes(rois, shape_type="polygon")
    for t in train_datai:
        t["image_fp"] = fix_file_paths_linux_windows(t["image_fp"])
    for t in test_datai:
        t["image_fp"] = fix_file_paths_linux_windows(t["image_fp"])

    DatasetCatalog.clear()
    train_data = WsiDetectron2ObjDetTifffileR(train_datai)
    test_data = WsiDetectron2ObjDetTifffile(test_datai)

    DatasetCatalog.register("train_data", train_data.get_dict)
    MetadataCatalog.get("train_data").set(thing_classes=[cfg.prediction_class])
    DatasetCatalog.register("val_data", test_data.get_dict)
    MetadataCatalog.get("val_data").set(thing_classes=[cfg.prediction_class])

    def do_test(test_loader, model, cfg):
        ret = inference_on_dataset(
            model, test_loader, instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret

    logger = logging.getLogger("detectron2")
    cfg_d2 = model_zoo.get_config(
        "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", True
    )
    cfg_d2.model.backbone.bottom_up.norm = "BN"
    cfg_d2.model.backbone.norm = "BN"
    cfg_d2.model.roi_heads.num_classes = 1
    cfg_d2.optimizer.lr = 0.001
    cfg_d2.train.checkpointer.period = 15000
    cfg_d2.train.max_iter = len(train_datai) * cfg.epoch
    # print(cfg_d2.train.max_iter)
    cfg_d2.lr_multiplier.scheduler.milestones = [450000,737499]#int(cfg_d2.train.max_iter*0.87)*100]
    cfg_d2.train.eval_period = 1000000000

    model = instantiate(cfg_d2.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg_d2.train.device)

    cfg_d2.optimizer.params.model = model
    optim = instantiate(cfg_d2.optimizer)

    train_loader = build_detection_train_loader(
        train_data.get_dict(),
        mapper=tifffile_detectron2_mapper_train,
        total_batch_size=6,
        num_workers=4,
    )
    test_loader = build_detection_test_loader(
        test_data.get_dict(), mapper=tifffile_detectron2_mapper
    )
    model = create_ddp_model(model, **cfg_d2.train.ddp)
    trainer = (AMPTrainer if cfg_d2.train.amp.enabled else SimpleTrainer)(
        model, train_loader, optim
    )
    output_dir = Path(cfg.output_dir) / cfg.model_tag
    output_dir.mkdir(exist_ok=True)
    if output_dir.exists():
        resume = True
    else:
        resume = False

    checkpointer = DetectionCheckpointer(
        model,
        str(output_dir),
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg_d2.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg_d2.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(
                cfg_d2.train.eval_period, lambda: do_test(test_loader, model, cfg_d2)
            ),
            hooks.PeriodicWriter(
                default_writers(str(output_dir), cfg_d2.train.max_iter),
                period=cfg_d2.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )


    checkpointer.resume_or_load(cfg_d2.train.init_checkpoint, resume=resume)
    if resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
        print(f"restarting from {start_iter}")
    else:
        start_iter = 0

    trainer.train(start_iter, cfg_d2.train.max_iter)
    #
    #
    # cfg_d2.(model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', True))
    # default_setup(cfg_d2, {})
    # cfg_d2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg.architecture)
    #
    # cfg_d2.DATASETS.TRAIN = ("train_data",)
    # cfg_d2.DATASETS.TEST = ("val_data",)
    #
    # cfg_d2.DATALOADER.NUM_WORKERS = 2
    # cfg_d2.SOLVER.IMS_PER_BATCH = 2
    # cfg_d2.SOLVER.BASE_LR = 0.001
    # cfg_d2.SOLVER.WARMUP_ITERS = 500
    # cfg_d2.SOLVER.MAX_ITER = len(train_datai) * cfg.epoch
    # solver_steps = [int(cfg_d2.SOLVER.MAX_ITER * step) for step in cfg.solver_steps]
    # cfg_d2.SOLVER.STEPS = solver_steps
    # cfg_d2.SOLVER.GAMMA = 0.1
    # cfg_d2.SOLVER.CHECKPOINT_PERIOD = len(train_datai) * (cfg.epoch // 2)
    #
    # cfg_d2.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    # cfg_d2.MODEL.ROI_HEADS.NUM_CLASSES = 1
    #
    # cfg_d2.TEST.EVAL_PERIOD = 10000000
    # cfg_d2.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    #
    # if not Path(cfg.output_dir).exists():
    #     Path(cfg.output_dir).mkdir(exist_ok=True)
    #     RESUME_FLAG = False
    # else:
    #     RESUME_FLAG = True
    #
    # os.chdir(str(cfg.output_dir))
    # os.makedirs(cfg.model_tag, exist_ok=True)
    # os.chdir(cfg.model_tag)
    # # shutil.rmtree('.')
    #
    # cfg_d2.OUTPUT_DIR = os.getcwd()
    # os.makedirs(cfg_d2.OUTPUT_DIR, exist_ok=True)

    # trainer = CocoTrainer(model)
    # trainer.resume_or_load(resume=RESUME_FLAG)
    # trainer.train()
