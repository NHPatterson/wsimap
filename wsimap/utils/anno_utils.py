import time
from typing import Dict

import numpy as np
import cv2
from shapely import geometry, ops, affinity
from lxml import etree
import json

from wsimap.utils.merge_utils import prediction_to_shapely, merge_shapely_predictions_rtree, merged_to_prediction_dict, \
    prediction_to_qp_geojson


def read_zen_polys(zen_fp):
    """Read Zeiss Zen Blue .cz ROIs files to wsimap shapely format.

    Parameters
    ----------
    zen_fp : str
        file path of Zen .cz.

    Returns
    -------
    list
        list of wsimap shapely rois

    """

    root = etree.parse(zen_fp)

    rois = root.xpath("//Elements")[0]

    def ptset_to_poly_np(ptset, name=None, type=None):
        poly = geometry.Polygon([[x, y] for x, y in zip(ptset[:, 0], ptset[:, 1])])
        poly.roi_attrs = {"name": name, "type": type}
        return poly

    rois_out = []
    for shape in rois:
        if shape.tag == "Polygon":
            ptset_cz = shape.find("Geometry/Points")
            ptset_type = "Polygon"
            ptset_name = shape.find("Attributes/Name").text

            poly_str = ptset_cz.text
            poly_str = poly_str.split(" ")
            poly_str = [poly.split(",") for poly in poly_str]
            poly_np = np.asarray(poly_str, dtype=np.float32)
            out_roi = ptset_to_poly_np(poly_np, name=ptset_name, type=ptset_type)

            rois_out.append(out_roi)

        # TODO: finish this
        # if shape.tag == "Rectangle":
        #
        #     rect_np = np.asarray(
        #         (
        #             shape.findtext("Left"),
        #             shape.findtext("Top"),
        #             shape.findtext("Width"),
        #             shape.findtext("Height"),
        #         ),
        #         dtype=np.float32,
        #     )
        #     ptset_type = "Rectangle"
        #     ptset_name = shape.find("Attributes/Name").text

    return rois_out


def read_qp_polys(qp_fp, ds=1):
    """Read QuPath polygons in the MSRC custom format.

    Parameters
    ----------
    qp_fp : str
        file path of MSRC QuPath ROI format.

    Returns
    -------
    list
        list of wsimap shapely rois

    """
    with open(qp_fp, "r") as f:
        lines = f.readlines()
        poly_coords = []
        for line in lines:
            split_line = line.split("#")
            roi_type = split_line[0]
            if roi_type == "null":
                roi_name = "unnamed"
                coord_idx = 1
            else:
                roi_name = split_line[0]
                coord_idx = 2
            xy_coords = (
                split_line[coord_idx].replace("[", "").replace("]", "").split("_")
            )
            x_coords = [float(coord) / ds for coord in xy_coords[0].split(",")]
            y_coords = [float(coord) / ds for coord in xy_coords[1].split(",")]
            if roi_type != "tile" and len(x_coords) > 4:
                poly_coords.append([roi_type, roi_name, x_coords, y_coords])

    def ptset_to_poly(ptset):
        poly = geometry.Polygon([[x, y] for x, y in zip(ptset[2], ptset[3])])
        poly.roi_attrs = {"name": ptset[1], "type": "Polygon"}
        return poly

    return [ptset_to_poly(ptset) for ptset in poly_coords]


def read_qp_geojson(
    json_file, poly_name="unnamed", shape_type="Polygon", ds=1, offset=None
):
    """Read GeoJSON files with QuPath metadata.

    Parameters
    ----------
    json_file : str
        file path of QuPath exported GeoJSON.
    poly_name : str
        apply name to all shapes (only used if name not found in json)
    shape_type : str
        get type to all shapes (only used if name not found in json)

    Returns
    -------
    list
        list of wsimap shapely rois

    """

    with open(json_file) as f:
        polygons = json.load(f)

    if isinstance(polygons, dict):
        polygons = [polygons]

    def get_poly_json(poly, poly_name=poly_name, shape_type=shape_type, ds=ds):
        try:
            try:
                sh_poly = geometry.Polygon(poly["geometry"]["coordinates"][0])
            except AssertionError:
                # sh_poly = geometry.Polygon(poly["geometry"]["coordinates"][0])
                sh_polys = [geometry.Polygon(p) for p in poly["geometry"]["coordinates"][0]]
                p_sizes = [p.area for p in sh_polys]
                sh_poly = sh_polys[np.argmax(p_sizes)]
            except TypeError:
                sh_poly = geometry.Polygon(poly["geometry"]["coordinates"])

        except ValueError:
            s1 = geometry.Polygon(poly["geometry"]["coordinates"][0][0]).buffer(1)
            s2 = geometry.Polygon(poly["geometry"]["coordinates"][1][0]).buffer(1)
            sh_poly = ops.cascaded_union([s1, s2])

        if ds != 1:
            sh_poly = affinity.scale(sh_poly, xfact=1 / ds, yfact=1 / ds, origin=(0, 0))
        if offset:
            sh_poly = affinity.translate(sh_poly, xoff=offset[0], yoff=offset[1])
        try:
            poly_name = poly["properties"]["classification"]["name"]
            shape_type = poly["geometry"]["type"]
            sh_poly.roi_attrs = {"name": poly_name, "type": shape_type}
        except KeyError:
            sh_poly.roi_attrs = {"name": poly_name, "type": shape_type}
        return sh_poly

    return [get_poly_json(poly) for poly in polygons]


def tile_check(tile, roi, edge_thresh):
    """Check if the overlap of a tile is within a bounding ROI within threshold.

    Parameters
    ----------
    tile : shapely.geometry.Polygon
        wsimap sliding window shapely tile
    roi : shapely.geometry.Polygon
        bounding shapely roi
    edge_thresh : float
        float between 0 and 1, 1 means tile must be entirely enclosed by the roi.

    Returns
    -------
    bool
        returns True if tile meets ROI threshold

    """
    return roi.intersection(tile).area / tile.area < edge_thresh


def tile_translate(tile, offset_x, offset_y):
    """Translate tiles in x and y.

    Parameters
    ----------
    tile : shapely.geometry.Polygon
        wsimap sliding window shapely tile
    offset_x : int
        translation (in pixels) of the tile in x
    offset_y : int
        translation (in pixels) of the tile in y

    Returns
    -------
    shapely.geometry.Polygon
        translated shapely polygon

    """
    roi_attrs = tile.roi_attrs
    construct = tile.__class__
    exterior = [
        (coord[0] + offset_x, coord[1] + offset_y) for coord in tile.exterior.coords
    ]
    tile = construct(exterior, None)
    tile.roi_attrs = roi_attrs
    return tile


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
    roi_x, roi_y = np.asarray(roi.exterior.coords.xy)
    roi_x[roi_x < 0] = 0
    roi_y[roi_y < 0] = 0
    return np.asarray(list(zip(roi_x, roi_y)), dtype=np.uint32)


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
    try:
        epsilon = percent_arc_length * cv2.arcLength(contours[0], True)
    except IndexError:
        return np.zeros((1,)).astype(np.uint32)

    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    return np.squeeze(approx).astype(np.uint32)

def approx_polygon_contour_multi(mask, percent_arc_length=0.001):
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

    epsilons = [percent_arc_length * cv2.arcLength(cnt, True) for cnt in contours]

    polygons = [np.squeeze(cv2.approxPolyDP(cnts, eps, True)).astype(np.uint32) for cnts, eps in zip(contours, epsilons)]
    rm_idx = [
        idx for idx, pg in enumerate(polygons) if len(pg.shape) < 2
    ]

    for rm in rm_idx[::-1]:
        polygons.pop(rm)

    rm_idx = [
        idx for idx, pg in enumerate(polygons) if pg.shape[0] < 15
    ]
    for rm in rm_idx[::-1]:
        polygons.pop(rm)
    return polygons


def polygon_patch_to_wsi(polygon, min_x, min_y):
    """Reproject polygons detected on patches to original coordinate space.
    This also appends the first vertex as the last to work with QuPath.

    Parameters
    ----------
    polygon : numpy.ndarray
        2d array of vertices, rows: points, columns: y,x
    min_x : int
        x pixel index of the top-left of the tile
    min_y : int
        y pixel index of the top-left of the tile

    Returns
    -------
    numpy.ndarray
        returns an 2d array of vertices in wsi coordinates
        rows: points, columns: y,x

    """
    polygon[:, 0] = polygon[:, 0] + min_x
    polygon[:, 1] = polygon[:, 1] + min_y

    # this encloses the polygon for qupath compatability
    polygon = np.append(polygon, polygon[:1, :], axis=0)

    return polygon


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


# sh_detections = [prediction_to_shapely(pred, rev_class_dict) for pred in predictions]


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


def prediction_to_qp_geojson(prediction, rev_class_dict=None, qp_type: str = "annotation"):
    """wsimap list of prediction dict to GeoJSON formatted dicts

    Parameters
    ----------
    prediction : list of dict
        list of dicts formatted for GeoJSON construction.
    rev_class_dict : dict
        dict containing classes inverted from integers to strings

    Returns
    -------
    list of dict
        list of GeoJSON formatted dicts that can be dumped for QuPath import

    """

    geojson_type = "Feature"
    geo_type = "Polygon"

    if isinstance(prediction["class"], str):
        name = prediction["class"]
    elif isinstance(prediction["class"], int):
        name = rev_class_dict[prediction["class"]]

    properties = {
        "classification": {"name": name, "colorRGB": -3140401},
        "isLocked": False,
        "measurements": [{"name": "detection_score", "value": prediction["score"]}],
        "object_type": qp_type
    }

    geojson_dict = {
        "type": geojson_type,
        "geometry": {
            "type": geo_type,
            "coordinates": [prediction["polygon"].tolist()],
        },
        "properties": properties,
    }
    return geojson_dict


def get_train_area_offsets(train_anno:str, train_roi_name:str, downsampling=1, area_idx=0):
    train_rois = json.load(open(train_anno, "r"))
    if isinstance(train_rois, dict):
        train_rois = [train_rois]
    train_area_roi = [
        t
        for t in train_rois
        if t["properties"]["classification"]["name"] == train_roi_name
    ]
    coords = np.squeeze(np.asarray(train_area_roi[area_idx]["geometry"]["coordinates"]))
    x_start, x_end, y_start, y_end = (
        np.min(coords[:, 0]),
        np.max(coords[:, 0]),
        np.min(coords[:, 1]),
        np.max(coords[:, 1]),
    )
    x_start, x_end, y_start, y_end = (
        x_start / downsampling,
        x_end / downsampling,
        y_start / downsampling,
        y_end / downsampling,
    )
    return int(x_start), int(x_end), int(y_start), int(y_end)


def get_n_train_areas(train_anno:str, train_roi_name:str = "train-area") -> int:
    train_rois = json.load(open(train_anno, "r"))
    if isinstance(train_rois, dict):
        train_rois = [train_rois]
    train_area_roi = [
        t
        for t in train_rois
        if t["properties"]["classification"]["name"] == train_roi_name
    ]
    return len(train_area_roi)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def merge_wsi_predictions(predictions, rev_class_dict: Dict[int, str], size_filter: float = 0, qp_type: str = "annotation"):
    """Routine to merge WSI predictions from polygon numpy arrays."""
    print("merging..")
    predictions_sh = [
        prediction_to_shapely(p, rev_class_dict) for p in predictions
    ]
    start_time = time.time()
    merged_polygons = merge_shapely_predictions_rtree(predictions_sh)
    end_time = time.time()
    print(f"\nMerging required: {round((end_time - start_time) / 60, 3)} m")

    mps = [p for p in merged_polygons if p.is_valid]
    mps = [p for p in mps if p.area > size_filter]
    pdd = merged_to_prediction_dict(mps)
    json_pred = [
        prediction_to_qp_geojson(p, rev_class_dict, qp_type=qp_type)
        for p in pdd
    ]
    return json_pred