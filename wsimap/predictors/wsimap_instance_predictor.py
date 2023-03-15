import time

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from tqdm import tqdm

from wsimap.utils.anno_utils import (
    approx_polygon_contour,
    polygon_patch_to_wsi,
)


class WSIPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image, can be applied accross WSI.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:

    .. code-block:: python

        pred = DefaultPredictor(cfg)

    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

    def __call__(self, patches):
        """
        Args:
            patch : list of wsimap tiles

        Returns:
            predictions (dict):
                the output of the model with class, score, and polygon vertices
        """

        with torch.no_grad():
            detections = []
            n_patches = len(patches)
            for idx, patch in enumerate(patches):
                print("Progress {:2.1%}".format(idx / n_patches), end="\r")
                # # Apply pre-processing to image.
                dask_im = patch["dask_im"]
                tile_bounds = patch["tile_bounds"]

                image = dask_im.compute()

                height, width = image.shape[1:]

                image = image / np.iinfo(image.dtype).max
                image = image * 255
                image = torch.as_tensor(image.astype("float32"))

                # prepare & predict
                inputs = {"image": image, "height": height, "width": width}

                prediction = self.model([inputs])[0]

                # get dict for each detection
                if len(prediction["instances"]) > 0:
                    instances = prediction["instances"].to("cpu")

                    pred_classes = instances.pred_classes.numpy().tolist()
                    pred_scores = instances.scores.numpy().tolist()

                    pred_mask = instances.pred_masks.numpy()
                    pred_mask = pred_mask.astype(np.uint8) * 255

                    polygons = [approx_polygon_contour(mask) for mask in pred_mask]
                    polygons = [
                        polygon_patch_to_wsi(p, tile_bounds[0], tile_bounds[1])
                        for p in polygons
                    ]

                    for idx in range(len(pred_classes)):
                        detection = {
                            "class": pred_classes[idx],
                            "score": pred_scores[idx],
                            "polygon": polygons[idx],
                        }
                        detections.append(detection)

        return detections


class WSIPredictorMaskRCNNRegNet:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image, can be applied accross WSI.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:

    .. code-block:: python

        pred = DefaultPredictor(cfg)

    """

    def __init__(self, cfg):
        self.cfg = cfg  # cfg can be modified by model
        self.model = instantiate(self.cfg.model)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.train.init_checkpoint)
        self.model.to(self.cfg.train.device)

    def batched_prediction(self, patches, full_im=None, batch_size=4):

        """
        Args:
            patch : list of wsimap tiles

        Returns:
            predictions (dict):
                the output of the model with class, score, and polygon vertices
        """

        def range_nparray(data_range, n_split):
            ranges = np.array_split(data_range, n_split)
            return [(np.min(range), np.max(range)) for range in ranges]

        def patch_to_tensor(patch, full_im, patch_width=1024):
            tile_bounds = patch["tile_bounds"]
            image = np.zeros(
                (patch["dask_im"].shape[0], patch_width, patch_width),
                dtype=patch["dask_im"].dtype,
            )
            if full_im is not None:
                image_in = full_im[
                    :,
                    tile_bounds[1] : tile_bounds[3],
                    tile_bounds[0] : tile_bounds[2],
                ]
                image[:, 0 : image_in.shape[1], 0 : image_in.shape[2]] = image_in
            else:
                dask_im = patch["dask_im"]
                image_in = dask_im.compute()
                image[:, 0 : image_in.shape[1], 0 : image_in.shape[2]] = image_in

            image = image / np.iinfo(image.dtype).max
            image = image * 255
            image = torch.as_tensor(image.astype("float32"))
            height, width = image.shape[1:]

            # prepare & predict
            return {"image": image, "height": height, "width": width}

        start_time = time.time()
        detections = []
        n_patches = len(patches)
        if batch_size > 1:
            im_splits = range_nparray(
                np.arange(n_patches), np.floor(n_patches / batch_size)
            )
        else:
            im_splits = [(0, 1)]

        for idx, im_split in enumerate(tqdm(im_splits)):

            # print("\rProgress {:2.1%}".format(idx / len(im_splits)), end="")
            # # Apply pre-processing to image.
            patch_run = patches[im_split[0] : im_split[1]]
            tile_bounds = [p["tile_bounds"] for p in patch_run]
            inputs = [patch_to_tensor(patch, full_im) for patch in patch_run]
            # inputs_test = np.stack(inputs)
            # height, width = image.shape[1:]
            #
            # image = image / np.iinfo(image.dtype).max
            # image = image * 255
            # image = torch.as_tensor(image.astype("float32"))

            # prepare & predict
            # inputs = {"image": image, "height": height, "width": width}
            with torch.no_grad():
                predictions = self.model(inputs)
                del inputs
                torch.cuda.empty_cache()

            # get dict for each detection
            for idx, prediction in enumerate(predictions):
                if len(prediction["instances"]) > 0:
                    instances = prediction["instances"].to("cpu")

                    pred_classes = instances.pred_classes.numpy().tolist()
                    pred_scores = instances.scores.numpy().tolist()

                    pred_mask = instances.pred_masks.numpy()
                    pred_mask = pred_mask.astype(np.uint8) * 255

                    if np.max(pred_mask) > 0:
                        polygons = [approx_polygon_contour(mask) for mask in pred_mask]
                    else:
                        continue

                    rm_idx = [
                        idx for idx, pg in enumerate(polygons) if len(pg.shape) < 2
                    ]

                    for rm in rm_idx[::-1]:
                        polygons.pop(rm)
                        pred_classes.pop(rm)
                        pred_scores.pop(rm)

                    polygons = [
                        polygon_patch_to_wsi(
                            pg, tile_bounds[idx][0], tile_bounds[idx][1]
                        )
                        for pg in polygons
                    ]

                    for idx in range(len(polygons)):
                        detection = {
                            "class": pred_classes[idx],
                            "score": pred_scores[idx],
                            "polygon": polygons[idx],
                        }
                        detections.append(detection)

        end_time = time.time()

        print(f"\nPrediction required: {round((end_time - start_time) / 60, 3)} m")

        return detections

    def __call__(self, patches):
        """
        Args:
            patch : list of wsimap tiles

        Returns:
            predictions (dict):
                the output of the model with class, score, and polygon vertices
        """

        with torch.no_grad():
            detections = []
            for idx, patch in enumerate(tqdm(patches)):
                # # Apply pre-processing to image.
                dask_im = patch["dask_im"]
                tile_bounds = patch["tile_bounds"]

                image = dask_im.compute()

                height, width = image.shape[1:]

                image = image / np.iinfo(image.dtype).max
                image = image * 255
                image = torch.as_tensor(image.astype("float32"))

                # prepare & predict

                image = dask_im.compute()

                height, width = image.shape[1:]

                image = image / np.iinfo(image.dtype).max
                image = image * 255
                image = torch.as_tensor(image.astype("float32"))

                # prepare & predict
                inputs = {"image": image, "height": height, "width": width}

                prediction = self.model([inputs])[0]

                # get dict for each detection
                if len(prediction["instances"]) > 0:
                    instances = prediction["instances"].to("cpu")

                    pred_classes = instances.pred_classes.numpy().tolist()
                    pred_scores = instances.scores.numpy().tolist()

                    pred_mask = instances.pred_masks.numpy()
                    pred_mask = pred_mask.astype(np.uint8) * 255

                    polygons = [approx_polygon_contour(mask) for mask in pred_mask]

                    rm_idx = [
                        idx for idx, pg in enumerate(polygons) if len(pg.shape) < 2
                    ]

                    for rm in rm_idx[::-1]:
                        polygons.pop(rm)
                        pred_classes.pop(rm)
                        pred_scores.pop(rm)

                    polygons = [
                        polygon_patch_to_wsi(pg, tile_bounds[0], tile_bounds[1])
                        for pg in polygons
                    ]

                    for idx in range(len(polygons)):
                        detection = {
                            "class": pred_classes[idx],
                            "score": pred_scores[idx],
                            "polygon": polygons[idx],
                        }
                        detections.append(detection)

        return detections


class WSIPredictorMaskRCNNResNet:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image, can be applied accross WSI.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:

    .. code-block:: python

        pred = DefaultPredictor(cfg)

    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

    def batched_prediction(self, patches, full_im=None, batch_size=4):

        """
        Args:
            patch : list of wsimap tiles

        Returns:
            predictions (dict):
                the output of the model with class, score, and polygon vertices
        """

        def range_nparray(data_range, n_split):
            ranges = np.array_split(data_range, n_split)
            return [(np.min(range), np.max(range)) for range in ranges]

        def patch_to_tensor(patch, full_im, patch_width=1024):
            tile_bounds = patch["tile_bounds"]
            image = np.zeros(
                (patch["dask_im"].shape[0], patch_width, patch_width),
                dtype=patch["dask_im"].dtype,
            )
            if full_im is not None:
                image_in = full_im[
                    :,
                    tile_bounds[1] : tile_bounds[3],
                    tile_bounds[0] : tile_bounds[2],
                ]
                image[:, 0 : image_in.shape[1], 0 : image_in.shape[2]] = image_in
            else:
                dask_im = patch["dask_im"]
                image_in = dask_im.compute()
                image[:, 0 : image_in.shape[1], 0 : image_in.shape[2]] = image_in

            image = image / np.iinfo(image.dtype).max
            image = image * 255
            image = torch.as_tensor(image.astype("float32"))
            height, width = image.shape[1:]

            # prepare & predict
            return {"image": image, "height": height, "width": width}

        start_time = time.time()
        detections = []
        n_patches = len(patches)
        if batch_size > 1:
            im_splits = range_nparray(
                np.arange(n_patches), np.floor(n_patches / batch_size)
            )
        else:
            im_splits = [(0, 1)]

        for idx, im_split in enumerate(tqdm(im_splits)):

            # print("\rProgress {:2.1%}".format(idx / len(im_splits)), end="")
            # # Apply pre-processing to image.
            patch_run = patches[im_split[0] : im_split[1]]
            tile_bounds = [p["tile_bounds"] for p in patch_run]
            inputs = [patch_to_tensor(patch, full_im) for patch in patch_run]
            # inputs_test = np.stack(inputs)
            # height, width = image.shape[1:]
            #
            # image = image / np.iinfo(image.dtype).max
            # image = image * 255
            # image = torch.as_tensor(image.astype("float32"))

            # prepare & predict
            # inputs = {"image": image, "height": height, "width": width}
            with torch.no_grad():
                predictions = self.model(inputs)
                del inputs
                torch.cuda.empty_cache()

            # get dict for each detection

            for idx, prediction in enumerate(predictions):
                if len(prediction["instances"]) > 0:
                    instances = prediction["instances"].to("cpu")

                    pred_classes = instances.pred_classes.numpy().tolist()
                    pred_scores = instances.scores.numpy().tolist()

                    pred_mask = instances.pred_masks.numpy()
                    pred_mask = pred_mask.astype(np.uint8) * 255

                    polygons = [approx_polygon_contour(mask) for mask in pred_mask]

                    rm_idx = [
                        idx for idx, pg in enumerate(polygons) if len(pg.shape) < 2
                    ]

                    for rm in rm_idx[::-1]:
                        polygons.pop(rm)
                        pred_classes.pop(rm)
                        pred_scores.pop(rm)

                    polygons = [
                        polygon_patch_to_wsi(
                            pg, tile_bounds[idx][0], tile_bounds[idx][1]
                        )
                        for pg in polygons
                    ]

                    for idx in range(len(polygons)):
                        detection = {
                            "class": pred_classes[idx],
                            "score": pred_scores[idx],
                            "polygon": polygons[idx],
                        }
                        detections.append(detection)

        end_time = time.time()

        print(f"\nPrediction required: {round((end_time - start_time) / 60, 3)} m")

        return detections

    def __call__(self, patches):
        """
        Args:
            patch : list of wsimap tiles

        Returns:
            predictions (dict):
                the output of the model with class, score, and polygon vertices
        """

        with torch.no_grad():
            detections = []
            for idx, patch in enumerate(tqdm(patches)):
                # # Apply pre-processing to image.
                dask_im = patch["dask_im"]
                tile_bounds = patch["tile_bounds"]

                image = dask_im.compute()

                height, width = image.shape[1:]

                image = image / np.iinfo(image.dtype).max
                image = image * 255
                image = torch.as_tensor(image.astype("float32"))

                # prepare & predict

                image = dask_im.compute()

                height, width = image.shape[1:]

                image = image / np.iinfo(image.dtype).max
                image = image * 255
                image = torch.as_tensor(image.astype("float32"))

                # prepare & predict
                inputs = {"image": image, "height": height, "width": width}

                prediction = self.model([inputs])[0]

                # get dict for each detection
                if len(prediction["instances"]) > 0:
                    instances = prediction["instances"].to("cpu")

                    pred_classes = instances.pred_classes.numpy().tolist()
                    pred_scores = instances.scores.numpy().tolist()

                    pred_mask = instances.pred_masks.numpy()
                    pred_mask = pred_mask.astype(np.uint8) * 255

                    polygons = [approx_polygon_contour(mask) for mask in pred_mask]

                    rm_idx = [
                        idx for idx, pg in enumerate(polygons) if len(pg.shape) < 2
                    ]

                    for rm in rm_idx[::-1]:
                        polygons.pop(rm)
                        pred_classes.pop(rm)
                        pred_scores.pop(rm)

                    polygons = [
                        polygon_patch_to_wsi(pg, tile_bounds[0], tile_bounds[1])
                        for pg in polygons
                    ]

                    for idx in range(len(polygons)):
                        detection = {
                            "class": pred_classes[idx],
                            "score": pred_scores[idx],
                            "polygon": polygons[idx],
                        }
                        detections.append(detection)

        return detections