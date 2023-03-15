import time
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from fastai.torch_core import TensorImage
from skimage.draw import polygon2mask
from skimage.measure import regionprops

from wsimap.utils.anno_utils import polygon_patch_to_wsi, approx_polygon_contour_multi
from wsimap.config.prediction import InstancePredConfig
from tqdm import tqdm

class WSIPredictorUnet:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image, can be applied across WSI.

    """

    def __init__(self, cfg, model_parameters_fp):
        # self.cfg = cfg.clone()  # cfg can be modified by model

        self.model = smp.Unet(
            encoder_name=cfg.encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=cfg.in_channels,
            classes=cfg.n_classes,
        )
        state_of_model = torch.load(model_parameters_fp)
        try:
            self.model.load_state_dict(state_of_model["model"])
        except:
            self.model.load_state_dict(state_of_model)

        if torch.cuda.is_available():
            self.model.eval()
            self.model.cuda()

    def __call__(self, patches, full_im=None, threshold=0.5, batch_size=4, patch_size=512):
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

        def patch_to_tensor(patch, full_im, patch_width = 512):
            tile_bounds = patch["tile_bounds"]
            image = np.zeros((patch["dask_im"].shape[0], patch_width, patch_width), dtype=patch["dask_im"].dtype)
            if full_im is not None:
                image_in = full_im[
                    :,
                    tile_bounds[1] : tile_bounds[3],
                    tile_bounds[0] : tile_bounds[2],
                ]
                image[:,0:image_in.shape[1],0:image_in.shape[2]] = image_in
            else:
                dask_im = patch["dask_im"]
                image_in = dask_im.compute()
                image[:,0:image_in.shape[1],0:image_in.shape[2]] = image_in

            image = image / np.iinfo(image.dtype).max
            # image = image * 255
            image = torch.as_tensor(image.astype("float32"))

            # prepare & predict
            return image

        start_time = time.time()
        detections = []
        n_patches = len(patches)
        if batch_size > 1:
            im_splits = range_nparray(
                np.arange(n_patches), np.floor(n_patches / batch_size)
            )
        else:
            im_splits = [(0,1)]

        # idx = 0
        # im_split = im_splits[0]
        for idx, im_split in enumerate(tqdm(im_splits)):

            # print("\rProgress {:2.1%}".format(idx / len(im_splits)), end="")
            # # Apply pre-processing to image.
            patch_run = patches[im_split[0] : im_split[1]]

            inputs = [patch_to_tensor(patch, full_im) for patch in patch_run]
            inputs_test = np.stack(inputs)

            with torch.no_grad():
                predictions = self.model(TensorImage(inputs_test).cuda())

                # get dict for each detection
                for idx, prediction in enumerate(predictions):
                    pred_sm = F.softmax(prediction, dim=0)
                    pred_sm = pred_sm.cpu().detach().numpy()
                    tile_bounds = patch_run[idx]["tile_bounds"]
                    pred_mask_ss = (pred_sm[1, :, :] >= threshold).astype(np.uint8) * 255
                    if np.any(pred_mask_ss):

                        polygons = approx_polygon_contour_multi(pred_mask_ss)

                        rm_idx = [
                            pidx for pidx, pg in enumerate(polygons) if len(pg.shape) < 2
                        ]

                        for rm in rm_idx[::-1]:
                            polygons.pop(rm)

                        det_scores = []
                        for pg in polygons:
                            pmask = polygon2mask(pred_sm.shape[1:], pg[:,[1,0]]).astype(np.uint8)
                            rp = regionprops(pmask, pred_sm[1,:,:])
                            det_scores.append({"detection_mean":rp[0].intensity_mean,
                                               "detection_max":rp[0].intensity_max,
                                               "detection_min":rp[0].intensity_min})


                        polygons = [
                            polygon_patch_to_wsi(pg, tile_bounds[0], tile_bounds[1])
                            for pg in polygons
                        ]

                        for idx, pg in enumerate(polygons):
                            detection = {
                                "class": 0,
                                "score": det_scores[idx]["detection_mean"],
                                "polygon": pg,
                            }
                            detections.append(detection)
                    # del inputs_test
                    torch.cuda.empty_cache()

        end_time = time.time()

        print(f"\n{(end_time - start_time) / 60} m required to process")

        return detections

class WSIPredictorUnetMask:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image, can be applied across WSI.

    """

    def __init__(self, cfg: InstancePredConfig, model_parameters_fp, encoder_name:str =""):
        # self.cfg = cfg.clone()  # cfg can be modified by model

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=cfg.encoder_weights,
            in_channels=len(cfg.train_channels),
            classes=2,
        )
        state_of_model = torch.load(model_parameters_fp)
        try:
            self.model.load_state_dict(state_of_model["model"])
        except:
            self.model.load_state_dict(state_of_model)

        if torch.cuda.is_available():
            self.model.eval()
            self.model.cuda()

    def __call__(self, patches, full_im=None, threshold=0.5, batch_size=4, patch_size=512):
        """
        Args:
            patch : list of wsimap tiles

        Returns:
            predictions (dict):
                the output of the model with class, score, and polygon vertices
        """

        detections = []
        inputs_test = patches / np.iinfo(patches.dtype).max
        inputs_test = torch.as_tensor(inputs_test.astype("float32"))
        with torch.no_grad():
            predictions = self.model(TensorImage(inputs_test).cuda())
            # get dict for each detection
            for idx, prediction in enumerate(predictions):
                pred_sm = F.softmax(prediction, dim=0)
                pred_sm = pred_sm.cpu().detach().numpy()
                # pred_mask_ss = (pred_sm[1, :, :] >= threshold).astype(np.uint8) * 255
                detections.append(pred_sm[1, :, :])
                # del inputs_test

            torch.cuda.empty_cache()


        return detections