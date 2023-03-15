from pathlib import Path
from typing import Union, Optional
import tempfile
import pickle
import numpy as np
from numpy.lib.arraysetops import unique
import zarr
import cv2
import dask.array as da
from shapely import geometry
from shapely import ops
from wsimap.utils.anno_utils import (
    read_zen_polys,
    read_qp_polys,
    read_qp_geojson,
    tile_translate,
    tile_check,
    shcoords_to_np,
)
from tifffile import imread
from tqdm import tqdm
from tiler import Tiler


class WsiMap:
    """Whole slide image map class for tiling and intersecting of ROIs annotated
    on WSIs.

    Parameters
    ----------
    zarr_fp : str
        File path to zarr store containing whole slide image.
    pyr_layer : int
        which layer to extract from pyramid (0 is highest res.)
    gen_thumbnail : bool
        generate a 16x thumbnail of the loaded image in dask, will be useful for
        foreground detection

    Attributes
    ----------
    dask_im : dask.array
        zarr pyramid layer as dask array
    image_bounds : shapely.geometry.Polygon
        the shape of dask_im as a shapely Polygon
    thumbnail : np.ndarray
        16x downsampled thumbnail as 2D greyscale array

    image_tiles : list of shapely.geometry.Box
        list of tiles (patches) as shapely boxes generated through tile_wsi function
    annotations : list of shapely.geometry.Polygons
        list of ROIs as shapely polygons imported from file using import_annotations function

    roi_names : list of str
        list of names from all imported ROIs
    roi_unique_names : list of str
        list of the unique names in imported ROIs
    roi_unique_counts : list of int
        list of the counts of each unique imported ROIs

    tile_names : list of str
        list of names from all generated tiles
    tile_unique_names : list of str
        list of the unique names in generated
    tile_unique_counts : list of int
        list of the counts of each unique tile set

    tile_roi_data_sh : list of dict
        dicts containing shapely boxes and polygons of the after intersection
        of polygons in tiles is calculated
    tile_roi_data : list of dict
        dicts containing dask image patches, polygon vertices as np.arrays and roi_classes
        for object detection training
    patch_data : list of dict
        dicts containing dask image patches and their their class label for patch-based
        image classifcation

    """

    def __init__(self, zarr_fp, pyr_layer=0, gen_thumbnail=False):
        self.zarr_fp = zarr_fp
        self.pyr_layer = pyr_layer
        # prepare image data
        if self.zarr_fp is not None:
            self.dask_im, self.image_bounds = self.load_zarr_im_bounds(
                self.zarr_fp, pyr_layer=self.pyr_layer
            )
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                root = zarr.open_group(tmpdirname, mode="w")
                imlayer = root.create_dataset(
                    "0",
                    shape=(3, 2**18, 2**18),
                    chunks=(3, 1024, 1024),
                    dtype=np.uint8,
                    overwrite=True,
                )
                self.dask_im, self.image_bounds = self.load_zarr_im_bounds(
                    tmpdirname, pyr_layer=0
                )

        if gen_thumbnail is True:
            self.thumbnail = self.generate_thumbnail()
        else:
            self.thumbnail = None

        # annotation data
        # sliding window tiling and imported annotations
        self.image_tiles = []
        self.annotations = []

        # roi/tile/patch count information
        self.roi_names = None
        self.roi_unique_names = None
        self.roi_unique_counts = None

        self.tile_names = None
        self.tile_unique_names = None
        self.tile_unique_counts = None

        # prepared data
        self.tile_roi_data_sh = []

        # prepared data for torch
        self.tile_roi_data = []
        self.patch_data = []

    def load_zarr_im_bounds2(zarr_fp, pyr_layer=0):
        """Load and check zarr metadata.

        Parameters
        ----------
        zarr_fp : str
            File path to zarr store containing whole slide image.
        pyr_layer : int
            which layer to extract from pyramid (0 is highest res.)

        Returns
        -------
        dask.array
            returns the zarr pyramid layer as dask array

        """
        im_root = zarr.open_group(zarr_fp)

        is_pyramid, im_idx = self._check_for_pyr(im_root, pyr_layer)

        dask_im = da.squeeze(da.from_zarr(zarr_fp, component=im_idx))

        # assume only 2D multichannel images, dimension order after squeeze: CYX
        x_axis, y_axis = 2, 1
        image_bounds = geometry.box(0, 0, dask_im.shape[x_axis], dask_im.shape[y_axis])

        return dask_im, image_bounds

    def load_zarr_im_bounds(self, zarr_fp, pyr_layer=0):
        """Load and check zarr metadata.

        Parameters
        ----------
        zarr_fp : str
            File path to zarr store containing whole slide image.
        pyr_layer : int
            which layer to extract from pyramid (0 is highest res.)

        Returns
        -------
        dask.array
            returns the zarr pyramid layer as dask array

        """
        if isinstance(zarr_fp, da.Array):
            dask_im = da.squeeze(zarr_fp)

        elif Path(zarr_fp).suffix == ".tiff" or Path(zarr_fp).suffix == ".tif":
            im_root = imread(zarr_fp, aszarr=True)
            try:
                im_root = zarr.open_group(im_root)
                im_idx = pyr_layer
                dask_im = da.squeeze(da.from_zarr(im_root[im_idx]))
            except:
                im_root = zarr.open(im_root)
                dask_im = da.squeeze(da.from_zarr(im_root))
            # is_pyramid, im_idx = self._check_for_pyr(im_root, pyr_layer)

        elif zarr_fp:
            im_root = zarr.open_group(zarr_fp)
            is_pyramid, im_idx = self._check_for_pyr(im_root, pyr_layer)
            dask_im = da.squeeze(da.from_zarr(zarr_fp, component=im_idx))

        if len(dask_im.shape) == 2:
            dask_im = da.expand_dims(dask_im, 0)
        # assume only 2D multichannel images, dimension order after squeeze: CYX
        x_axis, y_axis = 2, 1

        image_bounds = geometry.box(0, 0, dask_im.shape[x_axis], dask_im.shape[y_axis])

        return dask_im, image_bounds

    def _check_for_pyr(self, zarr_root, pyr_layer=0):
        """Checks if image is pyramidal in zarr store and what is base layer.

        Parameters
        ----------
        zarr_fp : str
            File path to zarr store containing whole slide image.
        pyr_layer : int
            which layer to extract from pyramid (0 is highest res.)


        """
        size = np.array(
            [np.prod(d.shape, dtype=np.uint64) for name, d in zarr_root.arrays()]
        )
        if len(size) > 1:
            if np.all(size[:-1] > size[1:]):
                is_pyramid = True
                if pyr_layer != 0:
                    im_idx = pyr_layer
                else:
                    im_idx = 0
            else:
                is_pyramid = False
                if pyr_layer != 0:
                    im_idx = np.argsort(size)[::-1][pyr_layer]
                else:
                    im_idx = np.argmax(size)
        else:
            is_pyramid = False
            im_idx = 0
        return is_pyramid, im_idx

    def generate_thumbnail(self, dask_im):
        """Generate 16x downsample of imported image

        Parameters
        ----------
        dask_im : dask.array
            dask array of the wsi

        Returns
        -------
        numpy.ndarray
            16x downsampled numpy array of dask WSI image

        """
        # infer xy dims as largest dims
        xy_dim = np.argsort(dask_im.shape)[-2:]
        shrinking_dims = dict(
            [(i, 16) if i in xy_dim else (i, 1) for i in range(len(dask_im.shape))]
        )
        dask_thumbnail = da.coarsen(np.mean, dask_im, shrinking_dims, trim_excess=True)
        return (dask_thumbnail / 256).astype("uint8").compute()

    def foreground_detection(self, thumbnail):
        # TODO
        # mean_tnail = np.mean(thumbnail, axis=0, dtype="uint8")
        # blur_tnail = dask_image.ndfilters.gaussian_filter(mean_tnail, sigma=[4, 4])
        #
        # threshold_value = 0.4 * da.max(blur_tnail).compute()
        # threshold_image = blur_tnail > 3

        return

    def _get_uniques_counts(self, shape_data):
        """Gets the names, unique names and count of shape data imported in wsimap.

        Parameters
        ----------
        shape_data : list of shapely.geometry.Polygon
            image annotation rois or tiles/patches.

        Returns
        -------
        tuple
            tuple of lists containing unqiue roi names, their count and a list of names

        """
        shape_names = [shape.roi_attrs["name"] for shape in shape_data]
        shape_unique_names, shape_unique_counts = np.unique(
            shape_names, return_counts=True
        )
        return shape_names, shape_unique_names.tolist(), shape_unique_counts.tolist()

    # add modifiable functionality
    # i.e. mapping an import function that returns a dict
    # need to figure out how to better do multiple ROIs in return
    # and fix when there are multiple classes in one qp_roi file

    def import_annotations(self, roi_fp, ds=1, offset=None):
        """Read annotation data in Zeiss Zen Blue .cz format, geoJSON, or MSRC
        custom text based format and add to class annotations list.

        Parameters
        ----------
        roi_fp : str
            file path to the stored ROI data

        """

        if Path(roi_fp).suffix == ".cz":
            self.annotations = self.annotations + read_zen_polys(roi_fp)
        elif Path(roi_fp).suffix in [".json", ".geojson"]:
            self.annotations = self.annotations + read_qp_geojson(
                roi_fp, ds=ds, offset=offset
            )
        elif Path(roi_fp).suffix == ".txt":
            self.annotations = self.annotations + read_qp_polys(roi_fp, ds=ds)

        (
            self.roi_names,
            self.roi_unique_names,
            self.roi_unique_counts,
        ) = self._get_uniques_counts(self.annotations)

    def annotation_tree(self):
        """Prints ascii tree of annotation names and their count."""
        if len(self.roi_unique_names) < 1:
            print("No imported ROIs")
        else:
            for idx, roi in enumerate(self.roi_unique_names):
                print("{}".format(str(roi)))
                print("   -{} annotations".format(self.roi_unique_counts[idx]))

    def size_filter_annotations(self, size: float):
        # print("pre:", len(self.annotations))

        annos = [a for a in self.annotations if a.area > size]
        # print("post:", len(annos))

        self.annotations = annos
        # print("post:", len(self.annotations))

        (
            self.roi_names,
            self.roi_unique_names,
            self.roi_unique_counts,
        ) = self._get_uniques_counts(self.annotations)

    # WIP
    # def draw_rois(self, ch_idx=0, **kwargs):
    #     import cv2
    #
    #     if ch_idx is None:
    #         ch_idx = 0
    #
    #     im = self.dask_im[ch_idx, :, :].compute()
    #     im = cv2.normalize(
    #         src=im,
    #         dst=None,
    #         alpha=0,
    #         beta=255,
    #         norm_type=cv2.NORM_MINMAX,
    #         dtype=cv2.CV_8U,
    #     )
    #
    #     for roi in self.annotations["rois"]:
    #         roi_x, roi_y = roi.exterior.coords.xy
    #         ptset = np.asarray(list(zip(roi_x, roi_y)), dtype=np.int32)
    #         im = cv2.polylines(im, [ptset], True, 255, 8, lineType=cv2.LINE_AA)
    #
    #     self.drawn_im = im

    def tile_wsi(
        self,
        bound_anno="foreground",
        tile_size=1024,
        slide=[(512, 0), (0, 512), (512, 512)],
        edge_thresh=1,
        build_patches=False,
    ):
        """Tile images with sliding window accross entire image or within a
        region of interest.

        Parameters
        ----------
        bound_anno : str
            name of the annotation, must be in the annotation names of imported rois
        tile_size : int
            size, in pixels, of the tile in x and y (they will be square)
        slide : list of tuples of ints
            sliding window offsets for x and y in pixels
        edge_thresh : float
            amount patches are allowed outside of the bounding roi, should be betweeen
            0 and 1.
        build_patches : bool
            whether or not the tiling should build a patch list for patch-based
            image classification or tiles for object detection.

        Returns
        -------
        list of shapely.geometry.Polygon
            appends tiles / patches to self.image_tiles

        """
        if self.roi_unique_names is not None:
            if bound_anno not in self.roi_unique_names:
                print(
                    "{} is not in imported annotations: {} ".format(
                        bound_anno, self.roi_unique_names
                    )
                )
                tissue_roi = self.image_bounds

            else:
                if self.roi_unique_counts[self.roi_unique_names.index(bound_anno)] > 1:
                    print(
                        "bounding annotation corresponds to multiple rois {}".format(
                            bound_anno
                        )
                    )
                    print("merging annotations into MultiPolygon")
                    bounding_roi = [
                        anno
                        for anno in self.annotations
                        if anno.roi_attrs["name"] == bound_anno
                    ]

                    tissue_roi = geometry.MultiPolygon(bounding_roi)
                else:
                    bound_idx = self.roi_names.index(bound_anno)
                    tissue_roi = self.annotations[bound_idx]
        else:
            print("Tiling within image bounds")
            tissue_roi = self.image_bounds

        bounds = tissue_roi.bounds

        xmin, ymin, xmax, ymax = [int(bound // tile_size) for bound in list(bounds)]

        if slide is not None and slide != (0, 0):
            if isinstance(slide, tuple):
                slide = [(0, 0)] + [slide]
            elif isinstance(slide, list):
                slide = [(0, 0)] + slide
        else:
            slide = [(0, 0)]

        tiles = []
        if build_patches is True:
            roi_name = bound_anno
        else:
            roi_name = "(0,0)"

        for i in range(xmin, xmax + 1):
            for j in range(ymin, ymax + 1):
                tbox = geometry.box(
                    int(i * tile_size),
                    int(j * tile_size),
                    int((i + 1) * tile_size),
                    int((j + 1) * tile_size),
                )

                if tissue_roi.intersection(tbox).is_empty:
                    continue

                if tile_check(tbox, tissue_roi, edge_thresh) is True:
                    continue

                if tile_check(tbox, self.image_bounds, edge_thresh) is True:
                    continue

                # if self.image_bounds.intersection(tbox).area
                # / tbox.area < edge_thresh:
                #     continue
                # else:
                #     tbox = self.image_bounds.intersection(tbox)
                tbox.roi_attrs = {}
                tbox.roi_attrs["name"] = roi_name
                tbox.roi_attrs["type"] = "Rectangle"

                tiles.append(tbox)

        self.image_tiles = self.image_tiles + tiles

        # TODO : speed this up using vectorized numpy operation
        for offset in slide:
            if build_patches is True:
                roi_name = bound_anno
            else:
                roi_name = str(offset)

            # print("slide:", offset)
            if offset != (0, 0):
                tiles_slid = [
                    tile_translate(tile, offset[0], offset[1]) for tile in tiles
                ]

                tiles_slid = [
                    tile
                    for tile in tiles_slid
                    if tile_check(tile, tissue_roi, edge_thresh) is False
                ]

                tiles_slid = [
                    tile
                    for tile in tiles_slid
                    if tile_check(tile, self.image_bounds, 1) is False
                ]

                for tile in tiles_slid:
                    tile.roi_attrs["name"] = roi_name

                self.image_tiles = self.image_tiles + tiles_slid

        (
            self.tile_names,
            self.tile_unique_names,
            self.tile_unique_counts,
        ) = self._get_uniques_counts(self.image_tiles)

    def tile_wsi_tiler(
        self,
        bound_anno="foreground",
        tile_size: int = 1024,
        overlap: float = 0.5,
        edge_thresh: float = 1,
    ):
        """Tile images with sliding window accross entire image or within a
        region of interest.

        Parameters
        ----------
        bound_anno : str
            name of the annotation, must be in the annotation names of imported rois
        tile_size : int
            size, in pixels, of the tile in x and y (they will be square)
        slide : list of tuples of ints
            sliding window offsets for x and y in pixels
        edge_thresh : float
            amount patches are allowed outside of the bounding roi, should be betweeen
            0 and 1.
        build_patches : bool
            whether or not the tiling should build a patch list for patch-based
            image classification or tiles for object detection.

        Returns
        -------
        list of shapely.geometry.Polygon
            appends tiles / patches to self.image_tiles

        """

        if isinstance(bound_anno, str):
            bound_anno = [bound_anno]

        bound_anno_name = "-".join(bound_anno)
        tissue_rois = []
        for ba in bound_anno:
            if self.roi_unique_names:
                if ba not in self.roi_unique_names:
                    print(
                        "{} is not in imported annotations: {} ".format(
                            ba, self.roi_unique_names
                        )
                    )

                elif self.roi_unique_counts[self.roi_unique_names.index(ba)] > 0:
                    print(f"bounding to {ba}")
                    bounding_roi = [
                        anno
                        for anno in self.annotations
                        if anno.roi_attrs["name"] in ba
                    ]

                    tissue_roi = ops.unary_union(bounding_roi)
                    tissue_rois.append(tissue_roi)

        if len(tissue_rois) > 0:
            print(f"combining {bound_anno_name}")
            tissue_roi = ops.unary_union(tissue_rois)
        else:
            tissue_roi = None


        tiler = Tiler(
            data_shape=self.dask_im.shape,
            tile_shape=(self.dask_im.shape[0], tile_size, tile_size),
            channel_dimension=0,
            overlap=overlap,
            mode="irregular",
        )

        def _tile_to_shapely(tile_bbox, x_shape, y_shape):
            ys, xs, ye, xe = np.concatenate(tile_bbox)
            if xe > x_shape:
                xe = x_shape
            if ye > y_shape:
                ye = y_shape

            return geometry.box(xs, ys, xe, ye)

        def _find_overlapping_tile(
            tile: geometry.Polygon,
            bounding_shape: geometry.Polygon,
            edge_thresh: float = 0.5,
        ):
            if bounding_shape.intersection(tile).area / tile.area > edge_thresh:
                return tile
            else:
                return None

        def _add_tile_props(tile: geometry.Polygon, roi_name: str = "tile"):
            tile.roi_attrs = {}
            tile.roi_attrs["name"] = roi_name
            tile.roi_attrs["type"] = "Rectangle"
            return tile

        tile_data = [
            _tile_to_shapely(
                tiler.get_tile_bbox(tile_id=tile_id),
                y_shape=self.dask_im.shape[1],
                x_shape=self.dask_im.shape[2],
            )
            for tile_id in range(tiler.n_tiles)
        ]
        if tissue_roi:
            intersected_tiles = [
                _find_overlapping_tile(tile, tissue_roi, edge_thresh=edge_thresh)
                for tile in tile_data
            ]
            self.image_tiles = [tile for tile in intersected_tiles if tile]
            self.image_tiles = [
                _add_tile_props(tile, bound_anno_name) for tile in self.image_tiles
            ]
        else:
            self.image_tiles = tile_data
            self.image_tiles = [_add_tile_props(tile) for tile in self.image_tiles]


    def tile_wsi_mask_tiler(
        self,
        bound_anno: str = "foreground",
        tile_size: int = 1024,
        overlap: float = 0.5,
        feature_overlap: Optional[float] = None,
        build_patches: bool = False,
    ):
        """Tile images with sliding window accross entire image or within a
        region of interest.

        Parameters
        ----------
        bound_anno : str
            name of the annotation, must be in the annotation names of imported rois
        tile_size : int
            size, in pixels, of the tile in x and y (they will be square)
        slide : list of tuples of ints
            sliding window offsets for x and y in pixels
        feature_overlap : float
            inclusion criteria for a patch
        build_patches : bool
            whether or not the tiling should build a patch list for patch-based
            image classification or tiles for object detection.

        Returns
        -------
        list of shapely.geometry.Polygon
            appends tiles / patches to self.image_tiles

        """
        if self.roi_unique_names:
            if bound_anno not in self.roi_unique_names:
                print(
                    "{} is not in imported annotations: {} ".format(
                        bound_anno, self.roi_unique_names
                    )
                )
                tissue_roi = None

            elif self.roi_unique_counts[self.roi_unique_names.index(bound_anno)] > 0:
                print(
                    "bounding annotation corresponds to multiple rois {}".format(
                        bound_anno
                    )
                )
                print("merging annotations into MultiPolygon")
                bounding_roi = [
                    anno
                    for anno in self.annotations
                    if anno.roi_attrs["name"] == bound_anno
                ]

                tissue_roi = ops.unary_union(bounding_roi)
            else:
                tissue_roi = None
        else:
            tissue_roi = None
        tissue_polygons = []
        if isinstance(tissue_roi, geometry.MultiPolygon):
            for geom in tissue_roi.geoms:
                tissue_polygons.append(
                    np.asarray(geom.exterior.xy).transpose().astype(np.int32)
                )
        else:
            if tissue_roi:
                tissue_polygons.append(
                    np.asarray(tissue_roi.exterior.xy).transpose().astype(np.int32)
                )

        mask_arr = np.zeros(self.dask_im.shape[1:], dtype=np.uint8)
        if len(tissue_polygons) > 0:
            mask_arr[:, :] = cv2.fillPoly(
                mask_arr[:, :],
                tissue_polygons,
                1,
            )

        tiler = Tiler(
            data_shape=self.dask_im.shape,
            tile_shape=(self.dask_im.shape[0], tile_size, tile_size),
            channel_dimension=0,
            overlap=overlap,
            mode="irregular",
        )

        def _tile_to_image_mask_slice(dask_im, mask_im, tile_bbox, x_shape, y_shape):
            ys, xs, ye, xe = np.concatenate(tile_bbox)
            if xe > x_shape:
                xe = x_shape
            if ye > y_shape:
                ye = y_shape

            data_out = {
                "dask_im": dask_im[:, ys:ye, xs:xe],
                "mask_im": mask_im[ys:ye, xs:xe],
                "roi_classes": [bound_anno],
                "tile_bounds":(xs, ys, xe, ye)
            }
            return data_out

        def _compute_mask_proportion(mask_im, tile_area, feature_overlap):
            return (np.sum(mask_im) / tile_area) >= feature_overlap

        tile_data = [
            _tile_to_image_mask_slice(
                self.dask_im,
                mask_arr,
                tiler.get_tile_bbox(tile_id=tile_id),
                y_shape=self.dask_im.shape[1],
                x_shape=self.dask_im.shape[2],
            )
            for tile_id in range(tiler.n_tiles)
        ]

        if feature_overlap:
            tile_area = tile_size**2
            retain_idx = [
                idx
                for idx, t in enumerate(tile_data)
                if _compute_mask_proportion(t["mask_im"], tile_area, feature_overlap)
            ]
            tile_data = [tile_data[idx] for idx in retain_idx]

        return tile_data

    def tile_tree(self):
        """Prints ascii tree of tile/patch names and their count."""
        if len(self.tile_unique_names) < 1:
            print("No tiles generated yet")
        else:
            for idx, roi in enumerate(self.tile_unique_names):
                print("{}".format(str(roi)))
                print("   -{} annotations".format(self.tile_unique_counts[idx]))

    def intersect_tiles_annos(
        self, exclusion_roi=["foreground"], empty_set=False, overlap_thresh=0.05
    ):
        """Short summary.

        Parameters
        ----------
        exclusion_roi : list of str
            list of ROI names that shouldn't be considered for object detection
            this could be, for instance, bounding ROIs or tissue outlines, etc.

        """
        exclusion_roi.extend(["tissue_roi"])
        annotation_data = [
            roi
            for roi in self.annotations
            if roi.roi_attrs["name"] not in exclusion_roi and roi.is_valid
        ]

        def area_of_intersection(tile, roi):
            return tile.intersection(roi).area / roi.area

        for tile in tqdm(self.image_tiles):
            contained_roi_names = [
                roi.roi_attrs["name"]
                for roi in annotation_data
                if area_of_intersection(tile, roi) > overlap_thresh
            ]

            contained_roi_polys = [
                tile.intersection(roi)
                for roi in annotation_data
                if area_of_intersection(tile, roi) > overlap_thresh
            ]

            contained_rois = []
            contained_classes = []
            for idx, roi in enumerate(contained_roi_polys):
                if isinstance(roi, geometry.polygon.Polygon):
                    contained_rois.append(roi)
                    contained_classes.append(contained_roi_names[idx])

            if len(contained_roi_polys) > 0 or empty_set:
                # print("found roi")
                tile_data = {
                    "tile": tile,
                    "rois": contained_rois,
                    "classes": contained_classes,
                }
                self.tile_roi_data_sh.append(tile_data)

        self._roi_data_to_np(empty_set=empty_set)
    def _roi_data_to_np(self, empty_set=False):
        """convert ROI data from shapely into dask array and numpy arrays"""
        for tile_data in self.tile_roi_data_sh:
            tile_bounds = np.array(tile_data["tile"].bounds, dtype=np.uint32)
            if empty_set:
                np_tile_data = {
                    "tile_bounds": tile_bounds,
                    "rois": [],
                    "dask_im": self.dask_im[
                        :,
                        tile_bounds[1] : tile_bounds[3],
                        tile_bounds[0] : tile_bounds[2],
                    ],
                    "roi_classes": ["empty"],
                }
            else:
                np_tile_data = {
                    "tile_bounds": tile_bounds,
                    "rois": [
                        shcoords_to_np(roi.simplify(0.99, preserve_topology=True))
                        for roi in tile_data["rois"]
                    ],
                    "dask_im": self.dask_im[
                        :,
                        tile_bounds[1] : tile_bounds[3],
                        tile_bounds[0] : tile_bounds[2],
                    ],
                    "roi_classes": tile_data["classes"],
                }
            self.tile_roi_data.append(np_tile_data)

        if empty_set is False:
            self._clean_tile_roi_data()

    def _clean_tile_roi_data(self):
        # to improve
        n_rois = [
            [idx, True]
            for idx, roi in enumerate(self.tile_roi_data)
            if len(roi["rois"]) > 0
        ]
        retained_rois = np.asarray(n_rois)[:, 0].tolist()

        self.tile_roi_data = [
            roidata
            for idx, roidata in enumerate(self.tile_roi_data)
            if idx in retained_rois
        ]

    def tile_data_to_patches(self, exclusion_roi="foreground"):
        """Convert shapely patch data to dask and numpy arrays

        Parameters
        ----------
        exclusion_roi : list of str
            list of ROI names that shouldn't be considered for object detection
            this could be, for instance, bounding ROIs or tissue outlines, etc.


        """
        roi = self.image_tiles[0]
        tile_patch_data = [
            roi
            for roi in self.image_tiles
            if roi.roi_attrs["name"] not in exclusion_roi
        ]

        for patch in tile_patch_data:
            tile_bounds = np.array(patch.bounds, dtype=np.uint32)
            patch = {
                "tile_bounds": tile_bounds,
                "dask_im": self.dask_im[
                    :,
                    tile_bounds[1] : tile_bounds[3],
                    tile_bounds[0] : tile_bounds[2],
                ],
                "roi_classes": patch.roi_attrs["name"],
            }

            self.patch_data.append(patch)

    # sanity check to make sure patches have the same dimensions

    # xsizes = np.asarray([patch["tile_bounds"][2] -
    # patch["tile_bounds"][0] for patch in self.patch_data_np])
    # ysizes = np.asarray([patch["tile_bounds"][3] -
    # patch["tile_bounds"][1] for patch in self.patch_data_np])
    #
    # print('x sizes')
    # print(np.unique(xsizes))
    #
    # print('x sizes')
    # print(np.unique(ysizes))

    def save_roi_data(self, save_fn):
        """pickle object detection data

        Parameters
        ----------
        save_fn : str
            file path where the data will be saved.

        """
        with open(save_fn, "wb") as filehandler:
            pickle.dump(self.tile_roi_data, filehandler)

    def load_roi_data(self, load_fn):
        """load pickled object detection data

        Parameters
        ----------
        load_fn : str
            file path of the pickled data.

        """
        with open(load_fn, "rb") as filehandler:
            self.tile_roi_data = pickle.load(filehandler)

    def write_roi_to_zarr(self, outpath, ds_name="stacked_tiles", shuffle=True):
        # TODO: rewrite function for flat list of dict roi structure
        # zarr_dtype = self.roi_data_np[0]["dask_im"].dtype
        #
        # zarr_shape = [len(self.roi_data_np)]
        # zarr_shape.extend(list(self.roi_data_np[0][1].shape))
        #
        # root = zarr.open_group(outpath, mode="a")
        #
        # out_ims = root.create_dataset(
        #     ds_name,
        #     shape=zarr_shape,
        #     chunks=(1, zarr_shape[1], zarr_shape[2], zarr_shape[3]),
        #     dtype=zarr_dtype,
        #     overwrite=True,
        # )
        # self.zarr_roi_data_np = self.roi_data_np
        #
        # if shuffle is True:
        #     import random
        #
        #     random.shuffle(self.zarr_roi_data_np)
        #
        # for idx, im_roi in enumerate(self.zarr_roi_data_np):
        #     out_ims[idx, :, :, :] = im_roi[1].compute()
        #
        # zarr_fp = Path(outpath) / ds_name
        # stacked_images = da.from_zarr(str(zarr_fp))
        #
        # for idx, im_roi in enumerate(self.roi_data_np):
        #     im_slice = stacked_images[idx, :, :, :]
        #     self.zarr_roi_data_np.append((im_roi[0], im_slice, im_roi[2], im_roi[3]))

        return

    # WIP (apply function to all tiles of a given key)
    def tile_apply(self, tile_key, func):
        # return lambda tile: func(self.image_tiles[key])
        return


# from shapely import geometry
#
# ims_px = np.loadtxt(
#     "/media/nhp/OS/biomic/masking/S071_neg_spots.csv",
#     skiprows=1,
#     delimiter=",",
#     usecols=[1, 2, 3, 4, 5],
#     dtype=np.uint32,
# )
# ims_px = np.column_stack([ims_px, ims_px[:, 0] + px_width])
# ims_px = np.column_stack([ims_px, ims_px[:, 1] + px_width])
#
# ims_map = [
#     geometry.box(row[0], row[1], row[0] + px_width + px_width, row[0] + px_width)
#     for row in ims_px
# ]
#
# ims_map = geometry.MultiPolygon(ims_map)
#
# class WsiMapIMS(WsiMap):
#     def __init__(self, ims_pixel_data_fp):
#         ims_pixel_data = np.loadtxt(
#             "/media/nhp/OS/biomic/masking/S071_neg_spots.csv",
#             skiprows=1,
#             delimiter=",",
#             usecols=[1, 2, 3, 4, 5],
#             dtype=np.uint32,
#         )
#
#         self.ims_px_int = ims_pixel_data[:, :2]
#         self.ims_px_to_microscopy = ims_pixel_data[:, 2:4]
#
#         self.intersected_ims_pix
#
#     def read_ims_pixels(self, pixel_fp):
#
#         return
