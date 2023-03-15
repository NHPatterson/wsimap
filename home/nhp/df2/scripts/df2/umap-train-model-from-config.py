import gc
import segmentation_models_pytorch as smp
from fastai.vision.all import *
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from fastai.torch_core import TensorBase, flatten_check
from fastai.metrics import Metric
from fastai.metrics import Dice as FastaiDice

from shapely import geometry, ops
import cv2
from wsimap.wsi_map import WsiMap
from wsimap.utils.anno_utils import (
    polygon_patch_to_wsi,
)
from wsimap.utils.wsi_torch_utils import WsiTorchUnet


def reverse_class_dict(class_dict):
    rev_class_dict = {}
    for k, v in class_dict.items():
        rev_class_dict[v] = k
    return rev_class_dict


class TorchLoss(_Loss):
    "Wrapper class around loss function for handling different tensor types."

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def _contiguous(self, x):
        return TensorBase(x.contiguous())

    def forward(self, *input):
        input = map(self._contiguous, input)
        return self.loss(*input)  #


# from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/joint_loss.py
class WeightedLoss(_Loss):
    """
    Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    "Wrap two loss functions into one. This class computes a weighted sum of two losses."

    def __init__(
        self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0
    ):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)


# Multiclass metrics


class Dice(FastaiDice):
    "Dice coefficient metric for binary target in segmentation"

    def accumulate(self, learn):
        pred, targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.yb[0])
        pred, targ = map(TensorBase, (pred, targ))
        self.inter += (pred * targ).float().sum().item()
        self.union += (pred + targ).float().sum().item()


class Iou(Dice):
    "Implemetation of the IoU (jaccard coefficient) that is lighter in RAM"

    @property
    def value(self):
        return self.inter / (self.union - self.inter) if self.union > 0 else None


class Recall(Metric):
    def __init__(self, axis=1, th=0.5, epsilon=1e-7):
        self.axis = axis
        self.epsilon = epsilon
        self.th = th

    def reset(self):
        self.tp, self.fn = 0, 0

    def accumulate(self, learn):
        pred, targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.yb[0])
        self.tp += (pred * targ).float().sum().item()
        self.fn += (targ * (1 - pred)).float().sum().item()

    @property
    def value(self):
        return self.tp / (self.tp + self.fn + self.epsilon)


class Precision(Metric):
    def __init__(self, axis=1, th=0.5, epsilon=1e-7):
        self.axis = axis
        self.epsilon = epsilon
        self.th = th

    def reset(self):
        self.tp, self.fp = 0, 0

    def accumulate(self, learn):
        pred, targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.yb[0])
        self.tp += (pred * targ).float().sum().item()
        self.fp += ((1 - targ) * pred).float().sum().item()

    @property
    def value(self):
        return self.tp / (self.tp + self.fp + self.epsilon)


class CONFIG:
    # paths
    # deepflash2 dataset
    tile_shape = (512, 512)

    # pytorch model (segmentation_models_pytorch)
    train_channels = None
    encoder_name = "efficientnet-b2"
    encoder_weights = "imagenet"
    in_channels = 3
    n_classes = 2

    # Training
    mixed_precision_training = True
    batch_size = 16
    weight_decay = 0.00
    max_learning_rate = 1e-3
    epoch = 100

    dc = TorchLoss(smp.losses.DiceLoss(mode="multiclass", classes=[1]))
    ce = CrossEntropyLossFlat(axis=1)

    loss_func = JointLoss(dc, ce, 1, 1)
    metrics = [Dice(), Iou(), Recall(), Precision()]

    def __init__(self, config):
        import_config = yaml.safe_load(open(config, "r"))
        print(import_config)
        if import_config.get("batch_size"):
            self.batch_size = import_config.get("batch_size")

        if import_config.get("encoder_name"):
            self.encoder_name = import_config.get("encoder_name")

        if import_config.get("epoch"):
            self.epoch = import_config.get("epoch")

        if import_config.get("n_channels"):
            self.in_channels = import_config.get("n_channels")

        self.training_images = import_config.get("train_images")

        self.training_rois = import_config.get("train_rois")

        if import_config.get("tissue_rois") is None:
            self.tissue_rois = [None for i in self.training_images]
        else:
            self.tissue_rois = import_config.get("tissue_rois")

        if import_config.get("negative_roi"):
            self.neg_roi = import_config.get("negative_roi")
        else:
            self.neg_roi = None

        self.prediction_class = import_config.get("prediction_class")
        self.all_classes = import_config.get("all_classes")
        self.output_dir = import_config.get("output_dir")

        if import_config.get("model_tag"):
            self.model_tag = import_config.get("model_tag")
        else:
            self.model_tag = f"{self.encoder_name}"

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


# Albumentations augmentations
tfms = A.Compose(
    [
        A.OneOf(
            [
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
            ],
            p=0.3,
        ),
        # Additional position augmentations
        A.RandomRotate90(p=0.75),
        A.HorizontalFlip(p=0.75),
        A.VerticalFlip(p=0.75),
    ]
)

cfg = CONFIG("/data/SharedData/segmentation/DR1/dr1-training-mxif-af.yaml")
if __name__ == "__main__":
    import sys

    cfg = CONFIG(sys.argv[1])
    all_classes = cfg.all_classes
    all_classes.pop(all_classes.index(cfg.prediction_class))

    train_data = []
    for train_im, train_anno, roi_anno in zip(
        cfg.training_images, cfg.training_rois, cfg.tissue_rois
    ):

        wmap = WsiMap(str(train_im), pyr_layer=1)

        if cfg.train_channels:
            wmap.dask_im = wmap.dask_im[cfg.train_channels, :, :]

        wmap.import_annotations(train_anno, ds=2)

        if roi_anno:
            wmap.import_annotations(roi_anno, ds=2)

        wmap.tile_wsi(
            bound_anno="train-area",
            tile_size=512,
            slide=[(0, 128), (128, 0), (128, 128), (0, 256), (256, 0), (256, 256)],
            # slide=None,
            edge_thresh=0.7,
            build_patches=False,
        )

        print("intersection")
        wmap.intersect_tiles_annos(exclusion_roi=all_classes)

        [p.update({"image_fp": train_im}) for p in wmap.tile_roi_data]

        single_im_data = [
            t for t in wmap.tile_roi_data if t["dask_im"].shape[1:] == (512, 512)
        ]
        train_data.extend(single_im_data)

        if cfg.neg_roi:
            wmap_neg = WsiMap(str(train_im), pyr_layer=1)
            wmap_neg.dask_im = wmap_neg.dask_im[cfg.train_channels, :, :]
            wmap_neg.import_annotations(train_anno, ds=2)
            wmap_neg.tile_wsi(
                bound_anno=cfg.neg_roi,
                tile_size=512,
                slide=[(256, 256)],
                # slide=None,
                edge_thresh=0.9,
                build_patches=True,
            )
            wmap_neg.intersect_tiles_annos(empty_set=True)

            [p.update({"image_fp": train_im}) for p in wmap_neg.tile_roi_data]
            neg_images = [
                t
                for t in wmap_neg.tile_roi_data
                if t["dask_im"].shape[1:] == (512, 512)
            ]
            n_empty = int(np.ceil(0.33 * len(single_im_data)))
            print(n_empty)
            print(len(neg_images))
            neg_images_selected = np.random.choice(
                np.arange(0, len(neg_images), 1), n_empty, replace=False
            )
            neg_images = [neg_images[i] for i in neg_images_selected]
            train_data.extend(neg_images)

    train_length = int(np.floor(0.9 * len(train_data)))
    test_length = len(train_data) - train_length
    indices = np.random.permutation(len(train_data)).astype(int)
    train_datai = [train_data[idx] for idx in indices[:-test_length]]
    test_datai = [train_data[idx] for idx in indices[-test_length:]]

    model = smp.Unet(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        classes=cfg.n_classes,
    )

    # Dataloader and learner
    dls = DataLoaders.from_dsets(
        WsiTorchUnet(train_datai, transforms=tfms),
        WsiTorchUnet(test_datai, transforms=None),
        bs=cfg.batch_size,
        # after_batch=Normalize.from_stats(*cfg.stats),
        num_workers=0,
        shuffle=True,
    )

    if torch.cuda.is_available():
        dls.cuda(), model.cuda()

    learn = Learner(
        dls,
        model,
        wd=cfg.weight_decay,
        metrics=cfg.metrics,
        loss_func=cfg.loss_func,
        opt_func=ranger,
    )
    learn.to_fp16()
    learn.fit_one_cycle(cfg.epoch, lr_max=cfg.max_learning_rate)

    # Save Model
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")

    stats = np.array([0, 0, 0]), np.array([1, 1, 1])

    model_output_name = (
        f"{cfg.model_tag}-{cfg.encoder_name}-{cfg.prediction_class}-{cfg.epoch}ep.pth"
    )

    model_output_path = Path(cfg.output_dir) / model_output_name

    state = {"model": learn.model.state_dict(), "stats": stats}
    torch.save(
        state,
        str(model_output_path),
        pickle_protocol=2,
        _use_new_zipfile_serialization=False,
    )
    del model
    del learn
    del dls

    gc.collect()
