import numpy as np
from shapely import geometry, ops
import cv2
from scipy.spatial import distance
from rtree.index import Index

def flatten_list(list):
    """Utility function to flatten lists of lists

    Parameters
    ----------
    list : type
        Description of parameter `list`.

    Returns
    -------
    list
        flattened list

    """
    return [item for sublist in list for item in sublist]


def approx_polygon_contour(mask, percent_arc_length=0.001):
    """Approximate binary mask contours to polygon vertices using cv2.

    Parameters
    ----------
    mask : numpy.ndarray
        2-d numpy array of datatype np.uint8.
    percent_arc_length : float
        scaling of epsilon for polygon approximate vertices accuracy.
        maximum distance of new vertices from original.

    Returns
    -------
    numpy.ndarray
        returns an 2d array of vertices, rows: points, columns: y,x

    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    approximated_polys = []
    for cnt in contours:
        epsilon = percent_arc_length * cv2.arcLength(cnt, True)
        approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
        approximated_polys.append(np.squeeze(approx_poly).astype(np.uint32))
    return approximated_polys


def prediction_to_shapely(roi, rev_class_dict):
    """Convert detectron2 predictions to shapely, maintaining metadata.

    Parameters
    ----------
    roi : dict
        dict containing "class", "score", and "polygon" from WSIPredictor class.
    rev_class_dict : dict
        dict containing classes inverted from integers to strings

    Returns
    -------
    shapely.geometry.Polygon
        wsimap shapely rois
    """
    ptset = roi["polygon"]
    poly = geometry.Polygon([[x, y] for x, y in zip(ptset[:, 0], ptset[:, 1])])
    poly.roi_attrs = {
        "name": rev_class_dict[roi["class"]],
        "type": "Polygon",
        "score": roi["score"],
    }
    return poly


def intersect_polygons(poly, boundary, threshold):
    """Determines if intersection over union (IoU) of polygon is above a threshold.
    This is necessitated by the 'sliding window' approach used for predictions
    where multiple detections of a single true object must be merged.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        wsimap shapely polygon
    boundary : shapely.geometry.Polygon
        wsimap shapely polygon
    threshold : float
        value between 0 and 1 for IoU. Higher means there must be more overlap
        to consider two polygons as one.

    Returns
    -------
    bool
        returns True/False if IoU meets threshold

    """
    # poly is a polygon
    # boundary is another polygon we are testing against for overlap
    # threshold is a number between 0 and 1 for the degree of overlap
    if poly.overlaps(boundary) is True:
        intersection_area = (boundary.intersection(poly).area / boundary.area) * 100
        union_area = (boundary.union(poly).area / boundary.area) * 100

        # we return a T/F here on whether a polygon has enough overlap
        # intersection over union is AKA Jaccard Index
        # https://stackoverflow.com/questions/28723670/intersection-over-union-between-two-detections
        # some discussion in the link above
        return (intersection_area / union_area) > threshold
    else:
        return False


def overlap_polygons(poly, boundary, threshold):
    """computes if the polygon overlaps with the test polygon and filters by threshold

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        wsimap shapely polygon
    boundary : shapely.geometry.Polygon
        wsimap shapely polygon
    threshold : float
        value between 0 and 1 for overlap threshold. Higher means there must be more overlap
        to consider two polygons as one.

    Returns
    -------
    bool
        returns True/False if IoU meets threshold

    """
    if poly.intersects(boundary) is True:
        intersection_area = boundary.intersection(poly).area
        intersection_area1 = intersection_area / poly.area
        intersection_area2 = intersection_area / boundary.area
        if intersection_area1 >= threshold or intersection_area2 >= threshold:
            overlaps = True
        else:
            overlaps = False
        return overlaps
    else:
        return False


def contains_polygon(poly, boundary):
    """Determines if polygon contains entirely another polygon.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        wsimap shapely polygon
    boundary : shapely.geometry.Polygon
        wsimap shapely polygon

    Returns
    -------
    bool
        returns True/False if polygon is contained

    """
    if boundary.contains(poly) is True or poly.contains(boundary) is True:
        contained = True
    else:
        contained = False
    return contained


def merge_polygons_iou(polygons, IoU_threshold=0.3):
    """Algorithm to merge polygons and iteratively eliminate those that have
    been merged. Polygons are merged used a cascaded_union from shapely.

    Parameters
    ----------
    polygons : list
        list of wsimap shapely polygons
    IoU_threshold : float
        value between 0 and 1 for IoU. Higher means there must be more overlap
        to consider two polygons as one.

    Returns
    -------
    list
        list of merged wsimap shapely polygons

    """

    # remove invalid polygons
    polygons = [p for idx, p in enumerate(polygons) if p.is_valid]

    merged_polygons = []

    while len(polygons) > 0:
        test_poly = polygons[0]

        intersects = [
            idx
            for idx, p in enumerate(polygons)
            if intersect_polygons(p, test_poly, IoU_threshold)
        ]

        if len(intersects) > 0:
            indices = [0] + intersects

            merge_polys = [polygons[i] for i in indices]
            merge_scores = np.mean([p.roi_attrs["score"] for p in merge_polys])

            merge_polygon = ops.cascaded_union(merge_polys)
            merge_polygon.roi_attrs = {
                "name": test_poly.roi_attrs["name"],
                "type": "Polygon",
                "score": merge_scores,
            }

            [polygons.pop(i) for i in sorted(indices, reverse=True)]
            polygons.append(merge_polygon)
        else:
            merged_polygons.append(test_poly)
            polygons.pop(0)

    return merged_polygons


def merge_polygons_byclass(sh_detections, IoU_threshold=0.3):
    """Feeder process that helps to nest different classes of detected polygons
    for merging. I.e. if there are detections of multiple classes, only those of
    the same class get merged.

    Parameters
    ----------
    sh_detections : list
        list of wsimap shapely polygons
    IoU_threshold : float
        value between 0 and 1 for IoU. Higher means there must be more overlap
        to consider two polygons as one.

    Returns
    -------
    list
        list of merged wsimap shapely polygons

    """

    unique_rois = np.unique([sh.roi_attrs["name"] for sh in sh_detections]).tolist()
    class_polygons = {}
    for unique in unique_rois:
        polys = [sh for sh in sh_detections if sh.roi_attrs["name"] == unique]
        class_polygons[unique] = polys

    merged_polygons = []
    for k, v in class_polygons.items():
        merged_polygons.append(merge_polygons_iou(v, IoU_threshold=IoU_threshold))
    merged_polygons = flatten_list(merged_polygons)

    return merged_polygons


def merged_to_prediction_dict(merged_polygons):
    """Convert merged wsimap shapely polygons to list of dicts for GeoJSON
    construction.

    Parameters
    ----------
    merged_polygons : list
        list of merged wsimap shapely polygons

    Returns
    -------
    list of dict
        list of dicts formatted for GeoJSON construction.

    """
    predictions = []

    for merge in merged_polygons:
        p = {
            "class": merge.roi_attrs["name"],
            "score": merge.roi_attrs["score"],
            "polygon": shcoords_to_np(merge),
        }
        predictions.append(p)

    return predictions


def merge_predictions(predictions, rev_class_dict, IoU_threshold=0.4):
    """Convenience function for end-to-end merging of predictions, end result is
    ready for GeoJSON dump.

    Parameters
    ----------
    predictions : list of dicts
        list of dicts from WSIPredictor
    rev_class_dict : dict
        dict containing classes inverted from integers to strings
    IoU_threshold : float
        value between 0 and 1 for IoU. Higher means there must be more overlap
        to consider two polygons as one.

    Returns
    -------
    list of dict
        list of dicts formatted for GeoJSON construction.

    """
    sh_preds = [prediction_to_shapely(pred, rev_class_dict) for pred in predictions]
    sh_preds = merge_polygons_byclass(sh_preds, IoU_threshold=IoU_threshold)
    return merged_to_prediction_dict(sh_preds)


def merge_shapely_predictions(predictions_sh, IoU_threshold=0.2, overlap_threshold=0.2):

    pred_cents = [pred_cent(p) for p in predictions_sh]
    pred_centers = np.concatenate(pred_cents)
    merged_polygons = []
    idx = 0

    while len(predictions_sh) > 0:
        test_poly = predictions_sh[idx]

        if test_poly.is_valid is False:
            predictions_sh.pop(idx)
            pred_centers = np.delete(pred_centers, idx, axis=0)
            continue

        distance_tol = 1000
        IoU_threshold = IoU_threshold
        overlap_threshold = overlap_threshold

        neighbor_indices = np.where(
            distance.cdist(
                pred_centers[idx : idx + 1, :],
                pred_centers,
            )[0, :]
            <= distance_tol
        )[0]

        neighbor_polys = [
            (n_idx, predictions_sh[n_idx]) for n_idx in neighbor_indices if n_idx != idx
        ]
        polygons = [(p_idx, p) for p_idx, p in neighbor_polys if p.is_valid]

        intersects_idx = [
            p[0] for p in polygons if intersect_polygons(p[1], test_poly, IoU_threshold)
        ]

        intersects_overall_idx = [
            p[0]
            for p in polygons
            if overlap_polygons(p[1], test_poly, overlap_threshold)
        ]

        # contains_idx = [p[0] for p in polygons if contains_polygon(p[1], test_poly)]

        if len(intersects_idx) > 0 or len(intersects_overall_idx) > 0:
            mergable_polys_idx = intersects_idx + intersects_overall_idx
            mergable_polys_idx = np.unique(mergable_polys_idx).tolist()
            mergable_polys = [predictions_sh[i] for i in mergable_polys_idx]

            merge_polys = [test_poly] + mergable_polys
            merge_scores = np.max([p.roi_attrs["score"] for p in merge_polys])

            if test_poly.roi_attrs.get("merged_polys") is None:
                test_poly.roi_attrs.update({"merged_polys": []})

            merge_polygon = ops.cascaded_union(merge_polys)
            merge_polygon.roi_attrs = {
                "name": test_poly.roi_attrs["name"],
                "type": "Polygon",
                "score": merge_scores,
                "merged_polys": test_poly.roi_attrs["merged_polys"] + merge_polys,
            }

            # [predictions_sh.pop(i) for i in sorted(intersects_idx, reverse=True)]
            # predictions_sh.pop(idx)
            # merged_polygons.append(merge_polygon)
            pred_centers = np.delete(
                pred_centers, sorted(mergable_polys_idx, reverse=True), axis=0
            )
            pred_centers = np.delete(pred_centers, idx, axis=0)

            [predictions_sh.pop(i) for i in sorted(mergable_polys_idx, reverse=True)]
            predictions_sh.pop(idx)

            predictions_sh.append(merge_polygon)
            merge_cent = pred_cent(merge_polygon)
            pred_centers = np.append(pred_centers, merge_cent, axis=0)
        else:
            merged_polygons.append(test_poly)
            predictions_sh.pop(idx)
            pred_centers = np.delete(pred_centers, idx, axis=0)

    return merged_polygons


def prediction_to_qp_geojson(prediction, rev_class_dict=None, qp_type="annotation"):
    """wsimap list of prediction dict to GeoJSON formatted dicts

    Parameters
    ----------
    prediction : list of dict
        list of dicts formatted for GeoJSON construction.
    rev_class_dict : dict
        dict containing classes inverted from integers to strings
    qp_type: str
        "annotation" or "detection"

    Returns
    -------
    list of dict
        list of GeoJSON formatted dicts that can be dumped for QuPath import

    """

    geo_type = "Polygon"

    if isinstance(prediction["class"], str):
        name = prediction["class"]
    elif isinstance(prediction["class"], int):
        name = rev_class_dict[prediction["class"]]

    properties = {
        "object_type": qp_type,
        "classification": {"name": name, "colorRGB": -3140401},
        "isLocked": False,
        "measurements": [{"name": "detection_score", "value": prediction["score"]}],
    }

    geojson_dict = {
        "type": "Feature",
        "geometry": {
            "type": geo_type,
            "coordinates": [prediction["polygon"].tolist()],
        },
        "properties": properties,
    }
    return geojson_dict


def pred_cent(prediction):
    return np.asarray(prediction.centroid.xy).reshape(1, 2)


def shcoords_to_np(roi):
    """Convert exterior coords (vertices) to a 2-d numpy array.

    Parameters
    ----------
    roi : shapely.geometry.Polygon
        wsimap shapely roi

    Returns
    -------
    numpy.ndarray
        2-d numpy array where each row is a vertex and columns are x,y

    """
    roi_x, roi_y = roi.exterior.coords.xy
    return np.asarray(list(zip(roi_x, roi_y)), dtype=np.uint32)


def reverse_class_dict(class_dict):
    rev_class_dict = {}
    for k, v in class_dict.items():
        rev_class_dict[v] = k
    return rev_class_dict

def merge_shapely_predictions_rtree(predictions_sh, IoU_threshold=0.2, overlap_threshold=0.2):

    rtree_idx = Index()

    for idx, sh in enumerate(predictions_sh):
        rtree_idx.insert(idx, sh.bounds)

    predictions_sh = {idx:sh for idx, sh in enumerate(predictions_sh)}
    max_idx = len(predictions_sh.keys())
    idx = 0
    merged_polygons = []
    while len(rtree_idx) > 0:
        test_poly = predictions_sh[idx]

        if not test_poly.is_valid:
            rtree_idx.delete(idx, test_poly.bounds)
            predictions_sh.pop(idx)
            idx = list(predictions_sh.keys())[0]
            continue

        close_segs_idx = sorted(rtree_idx.intersection(test_poly.bounds))
        close_segs_idx.pop(close_segs_idx.index(idx))

        neighbor_polys = [
            (n_idx, predictions_sh[n_idx]) for n_idx in close_segs_idx if n_idx != idx
        ]
        polygons = [(p_idx, p) for p_idx, p in neighbor_polys if p.is_valid]

        intersects_idx = [
            p[0] for p in polygons if intersect_polygons(p[1], test_poly, IoU_threshold)
        ]

        intersects_overall_idx = [
            p[0]
            for p in polygons
            if overlap_polygons(p[1], test_poly, overlap_threshold)
        ]


        if len(intersects_idx) > 0 or len(intersects_overall_idx) > 0:
            mergable_polys_idx = intersects_idx + intersects_overall_idx
            mergable_polys_idx = np.unique(mergable_polys_idx).tolist()
            mergable_polys = [predictions_sh[i] for i in mergable_polys_idx]

            merge_polys = [test_poly] + mergable_polys
            merge_polygon = ops.unary_union(merge_polys)
            area_proportion = [p.area / merge_polygon.area for p in merge_polys]
            merge_scores = np.average([p.roi_attrs["score"] for p in merge_polys], weights=area_proportion)

            if test_poly.roi_attrs.get("merged_polys") is None:
                test_poly.roi_attrs.update({"merged_polys": []})

            merge_polygon.roi_attrs = {
                "name": test_poly.roi_attrs["name"],
                "type": "Polygon",
                "score": merge_scores,
                "merged_polys": test_poly.roi_attrs["merged_polys"] + merge_polys,
            }

            rtree_idx.delete(idx, predictions_sh[idx].bounds)
            [rtree_idx.delete(i, predictions_sh[i].bounds) for i in mergable_polys_idx]
            [predictions_sh.pop(i) for i in mergable_polys_idx]
            predictions_sh.pop(idx)
            predictions_sh.update({max_idx:merge_polygon})
            rtree_idx.insert(max_idx, merge_polygon.bounds)
            max_idx += 1
        else:
            rtree_idx.delete(idx, predictions_sh[idx].bounds)
            predictions_sh.pop(idx)
            merged_polygons.append(test_poly)

        try:
            idx = list(predictions_sh.keys())[0]
        except IndexError:
            continue

    return merged_polygons