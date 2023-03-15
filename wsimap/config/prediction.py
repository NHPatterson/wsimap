from typing import Optional
import sys
import numpy as np
import yaml


class InstancePredConfig:

    instance_mask_rcnn_old_architecture = (
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    )

    # for Unet
    encoder_weights = "imagenet"

    def __init__(self, config):
        config_fp = fix_file_paths_linux_windows(config)
        print(config_fp)
        import_config = yaml.safe_load(open(config_fp, "r"))

        self.model_path = import_config.get("model_path")
        self.model_path = self._listify_attribute(
            self.model_path, rep_if_necessary=None
        )

        if import_config.get("model_architecture"):
            self.model_arch = import_config.get("model_architecture")
        else:
            self.model_arch = "regnet"

        self.model_arch = self._listify_attribute(
            self.model_arch, rep_if_necessary=len(self.model_path)
        )

        if import_config.get("model_encoder"):
            self.model_encoder = import_config.get("model_encoder")
        else:
            self.model_encoder = "efficientnet-b2"

        self.model_arch = self._listify_attribute(
            self.model_arch, rep_if_necessary=len(self.model_path)
        )

        if import_config.get("threshold"):
            self.threshold = import_config.get("threshold")
        else:
            self.threshold = 0.5
        self.threshold = self._listify_attribute(
            self.threshold, rep_if_necessary=len(self.model_path)
        )

        if import_config.get("model_tag"):
            self.model_tag = import_config.get("model_tag")
        else:
            self.model_tag = "instance-model"

        self.model_tag = self._listify_attribute(self.model_tag, rep_if_necessary=None)
        if len(self.model_tag) != len(self.model_path):
            raise ValueError(
                f"Number of model paths does not match number of model tags."
            )

        self.prediction_class = import_config.get("prediction_class")
        self.prediction_class = self._listify_attribute(
            self.prediction_class, rep_if_necessary=None
        )
        if len(self.prediction_class) != len(self.model_path):
            raise ValueError(
                f"Number of prediction classes does not match number of model tags."
            )

        if import_config.get("size_filter"):
            self.size_filter = import_config.get("size_filter")
        else:
            self.size_filter = 0


        self.size_filter = self._listify_attribute(
            self.size_filter, rep_if_necessary=len(self.model_path)
        )

        self.output_dir = import_config.get("output_dir")

        if import_config.get("n_channels"):
            self.in_channels = import_config.get("n_channels")

        self.testing_images = import_config.get("test_images")

        if import_config.get("tissue_rois") is None:
            self.tissue_rois = [None for _ in self.testing_images]
        else:
            self.tissue_rois = import_config.get("tissue_rois")

        if import_config.get("tile_size"):
            self.tile_size = import_config.get("tile_size")
        else:
            self.tile_size = 1024

        if import_config.get("tile_overlap"):
            self.tile_overlap = import_config.get("tile_overlap")
        else:
            self.tile_overlap = 0.5

        if import_config.get("edge_thresh"):
            self.edge_thresh = import_config.get("edge_thresh")
        else:
            self.edge_thresh = 0.9

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
        self.tissue_rois = [
            fix_file_paths_linux_windows(fp) for fp in self.tissue_rois if fp
        ]

        if len(self.tissue_rois) == 0:
            self.tissue_rois = [None for _ in self.testing_images]

        self.output_dir = fix_file_paths_linux_windows(self.output_dir)
        self.model_path = [fix_file_paths_linux_windows(mp) for mp in self.model_path]

    def _listify_attribute(self, item, rep_if_necessary: Optional[int] = None):
        if not isinstance(item, list):
            item = [item]
            if rep_if_necessary and len(item) < rep_if_necessary:
                item = list(np.repeat(item, rep_if_necessary))
        return item


class MultiInstancePredConfig:

    instance_mask_rcnn_old_architecture = (
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    )

    # for Unet
    encoder_weights = "imagenet"

    def __init__(self, config):
        config_fp = fix_file_paths_linux_windows(config)
        print(config_fp)
        import_config = yaml.safe_load(open(config_fp, "r"))

        self.model_path = import_config.get("model_path")
        self.rev_class_dict = import_config["rev_class_dict"]
        self.num_classes = len(self.rev_class_dict.keys())

        if import_config.get("model_architecture"):
            self.model_arch = import_config.get("model_architecture")
        else:
            self.model_arch = "regnet"

        if import_config.get("model_encoder"):
            self.model_encoder = import_config.get("model_encoder")
        else:
            self.model_encoder = "efficientnet-b2"


        if import_config.get("threshold"):
            self.threshold = import_config.get("threshold")
        else:
            self.threshold = 0.5

        if import_config.get("model_tag"):
            self.model_tag = import_config.get("model_tag")
        else:
            self.model_tag = "instance-model"


        if import_config.get("size_filter"):
            self.size_filter = import_config.get("size_filter")
        else:
            self.size_filter = 0


        self.output_dir = import_config.get("output_dir")

        if import_config.get("n_channels"):
            self.in_channels = import_config.get("n_channels")

        self.testing_images = import_config.get("test_images")

        if import_config.get("tissue_rois") is None:
            self.tissue_rois = [None for _ in self.testing_images]
        else:
            self.tissue_rois = import_config.get("tissue_rois")

        if import_config.get("tile_size"):
            self.tile_size = import_config.get("tile_size")
        else:
            self.tile_size = 1024

        if import_config.get("tile_overlap"):
            self.tile_overlap = import_config.get("tile_overlap")
        else:
            self.tile_overlap = 0.5

        if import_config.get("edge_thresh"):
            self.edge_thresh = import_config.get("edge_thresh")
        else:
            self.edge_thresh = 0.9

        if import_config.get("downsampling"):
            self.downsampling = import_config.get("downsampling")
        else:
            self.downsampling = 1

        if import_config.get("post_scaling"):
            self.post_scaling = import_config.get("post_scaling")
        else:
            self.post_scaling = 1

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
        self.tissue_rois = [
            fix_file_paths_linux_windows(fp) for fp in self.tissue_rois if fp
        ]

        if len(self.tissue_rois) == 0:
            self.tissue_rois = [None for _ in self.testing_images]

        self.output_dir = fix_file_paths_linux_windows(self.output_dir)
        self.model_path = fix_file_paths_linux_windows(self.model_path)

    def _listify_attribute(self, item, rep_if_necessary: Optional[int] = None):
        if not isinstance(item, list):
            item = [item]
            if rep_if_necessary and len(item) < rep_if_necessary:
                item = list(np.repeat(item, rep_if_necessary))
        return item


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