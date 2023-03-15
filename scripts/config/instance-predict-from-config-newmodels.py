import json
import pickle
from pathlib import Path

import dask.array as da
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg

from wsimap.config.prediction import InstancePredConfig
from wsimap.predictors.wsimap_instance_predictor import (
    WSIPredictorMaskRCNNRegNet,
    WSIPredictorMaskRCNNResNet,
)
from wsimap.utils.anno_utils import NpEncoder, merge_wsi_predictions
from wsimap.utils.image_utils import read_full_image_for_prediction
from wsimap.wsi_map import WsiMap


if __name__ == "__main__":
    import sys

    config_fp = sys.argv[1]
    cfg = InstancePredConfig(config_fp)

    for test_image_fp, tissue_roi in zip(cfg.testing_images, cfg.tissue_rois):
        print(f"Current image: {Path(test_image_fp).name}")
        print("image reading..")

        full_im = read_full_image_for_prediction(
            test_image_fp,
            downsampling=cfg.downsampling,
            max_workers=cfg.max_workers,
            train_channels=cfg.train_channels,
        )

        dask_im = da.from_array(full_im, chunks=(1, 2048, 2048))

        wsimap = WsiMap(
            dask_im,
            pyr_layer=0,
        )

        if tissue_roi:
            wsimap.import_annotations(tissue_roi, ds=cfg.downsampling)

        wsimap.tile_wsi_tiler(
            bound_anno=cfg.prediction_area_name,
            tile_size=cfg.tile_size,
            overlap=cfg.tile_overlap,
            edge_thresh=cfg.edge_thresh,
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

        if len(cfg.model_path) > 1:
            all_tags = "-".join(cfg.model_tag)
            experiment_name = (
                f"{Path(test_image_fp).stem.replace('.ome', '')}-{all_tags}.geojson"
            )
            output_all_gj_fp = Path(cfg.output_dir) / experiment_name
        else:
            all_tags = None

        all_model_predictions = []
        for m_path, m_tag, m_threshold, m_class, size_filter, m_arch in zip(
            cfg.model_path,
            cfg.model_tag,
            cfg.threshold,
            cfg.prediction_class,
            cfg.size_filter,
            cfg.model_arch,
        ):
            print(f"predicting {m_class} using {m_tag}: {m_path}...")
            print(f"using threshold {m_threshold} with {size_filter} size filter")

            experiment_name = f"{Path(test_image_fp).stem.replace('.ome', '')}-{m_tag}-thresh-{m_threshold}.geojson"
            experiment_name_pkl = f"{Path(test_image_fp).stem.replace('.ome', '')}-{m_tag}-thresh-{m_threshold}.pkl"
            output_preds_pkl_fp = Path(cfg.output_dir) / experiment_name_pkl
            output_preds_gj_fp = Path(cfg.output_dir) / experiment_name

            if m_arch == "resnet":
                d2cfg = get_cfg()
                d2cfg.merge_from_file(
                    model_zoo.get_config_file(cfg.instance_mask_rcnn_old_architecture)
                )

                # training specific
                d2cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
                d2cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

                # final model data
                d2cfg.MODEL.WEIGHTS = str(m_path)

                # threshold to have a detection returned
                d2cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(m_threshold)
                predictor = WSIPredictorMaskRCNNResNet(d2cfg)

            elif m_arch == "regnet":
                d2cfg = model_zoo.get_config(
                    "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", True
                )
                d2cfg.model.backbone.bottom_up.norm = "BN"
                d2cfg.model.backbone.norm = "BN"
                d2cfg.train.init_checkpoint = m_path
                d2cfg.model.roi_heads.num_classes = 1
                d2cfg.model.roi_heads.box_predictor.test_score_thresh = m_threshold
                predictor = WSIPredictorMaskRCNNRegNet(d2cfg)

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

            rev_class_dict = {
                0: f"{m_class}",
            }

            predictions_update = []
            if cfg.downsampling != 1:
                for p in all_predictions:
                    p["polygon"] = p["polygon"] * cfg.downsampling
                    predictions_update.append(p)
            else:
                predictions_update = all_predictions

            with open(output_preds_pkl_fp, "wb") as outfile:
                pickle.dump(predictions_update, outfile)

            json_pred = merge_wsi_predictions(
                predictions_update,
                rev_class_dict,
                size_filter=size_filter,
                qp_type=cfg.qp_type,
            )

            print(f"writing {len(json_pred)} merged predictions to json..")
            with open(output_preds_gj_fp, "w") as json_file:
                json.dump(json_pred, json_file, indent=1, cls=NpEncoder)

            all_model_predictions.extend(json_pred)

        if all_tags is not None:
            print(f"writing {len(all_model_predictions)} merged predictions to json..")
            with open(output_all_gj_fp, "w") as json_file:
                json.dump(all_model_predictions, json_file, indent=1, cls=NpEncoder)
