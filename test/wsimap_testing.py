from pathlib import Path
import pickle
import numpy as np
from numpy.lib.arraysetops import unique
import zarr
import dask.array as da
from shapely import geometry
from wsimap.utils.anno_utils import (
    read_zen_polys,
    read_qp_polys,
    read_qp_geojson,
    tile_translate,
    tile_check,
    shcoords_to_np,
)


class WsiMap:
    """Whole slide image map class for tiling and intersecting of ROIs annotated
    on WSIs.

    Parameters
    ----------
    zarr_fp : str
        File path to zarr store containing whole slide image.
    pyr_layer : int
        which layer to extract from pyramid (0 is highest res.)

    Attributes
    ----------
    load_zarr_im : func
        loads zarr image and checks metadata
    image_tiles : list of shapely.geometry.Polygon
        tiles of WSI for extracting ROIs or for building patches
    annotations : list of shapely.geometry.Polygon
        annotation list for rois

    """

    def __init__(self, zarr_fp, pyr_layer=0):
        # prepare image data
        self.load_zarr_im(zarr_fp, pyr_layer=pyr_layer)
        # annotation data
        self.tissue_roi = None
        self.image_tiles = []
        self.annotations = []

    def load_zarr_im(self, zarr_fp, pyr_layer=0):
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
        self.im_root = zarr.open_group(zarr_fp)
        self.pyr_layer = pyr_layer
        self.check_for_pyr(self.im_root, self.pyr_layer)

        self.zarr_im_name = list(self.im_root.keys())[self.im_idx]
        self.dask_im = da.squeeze(da.from_zarr(zarr_fp, component=self.im_idx))

        if "orig_shape" in self.im_root.attrs:
            squeeze_axes = np.where(np.array(self.im_root.attrs["orig_shape"]) != 1)[
                0
            ].tolist()
            self.axes_info = [
                self.im_root.attrs["axes_names"][ax] for ax in squeeze_axes
            ]

            x_axis = self.axes_info.index("X")
            y_axis = self.axes_info.index("Y")
        else:
            # assume data is ordered Channel,Y,X if no axes names found
            x_axis = 2
            y_axis = 1

        self.image_bounds = geometry.box(
            0,
            0,
            self.dask_im.shape[x_axis],
            self.dask_im.shape[y_axis],
        )

    @staticmethod
    def dask_im_alone(zarr_fp, pyr_layer=0):
        im = WsiMap(zarr_fp, pyr_layer=pyr_layer)
        return im.dask_im, im.axes_info

    def check_for_pyr(self, zarr_root, pyr_layer):
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
                self.is_pyramid = True
                if pyr_layer != 0:
                    self.im_idx = pyr_layer
                else:
                    self.im_idx = 0
            else:
                self.is_pyramid = False
                if pyr_layer != 0:
                    self.im_idx = np.argsort(size)[::-1][pyr_layer]
                else:
                    self.im_idx = np.argmax(size)
        else:
            self.is_pyramid = False
            self.im_idx = 0

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
        self.dask_thumbnail = da.coarsen(
            np.mean, dask_im, shrinking_dims, trim_excess=True
        )
        self.dask_thumbnail = (self.dask_thumbnail / 256).astype("uint8")

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
        anno_names = [anno.roi_attrs["name"] for anno in shape_data]
        unique_rois, roi_counts = unique(anno_names, return_counts=True)
        return unique_rois.tolist(), roi_counts.tolist(), anno_names

    # add modifiable functionality
    # i.e. mapping an import function that returns a dict
    # need to figure out how to better do multiple ROIs in return
    # and fix when there are multiple classes in one qp_roi file

    def import_annotations(self, roi_fp):
        """Read annotation data in Zeiss Zen Blue .cz format, geoJSON, or MSRC
        custom text based format.

        Parameters
        ----------
        roi_fp : str
            file path to the stored ROI data


        Returns
        -------
        list of shapely.geometry.Polygon
            list of wsimap format shapely polygons, if geojson annotations don't
            have a classification name then they are called 'unnamed'

        """

        if Path(roi_fp).suffix == ".cz":
            self.annotations = self.annotations + read_zen_polys(roi_fp)
        elif Path(roi_fp).suffix == ".json":
            self.annotations = self.annotations + read_qp_geojson(roi_fp)
        elif Path(roi_fp).suffix == ".txt":
            self.annotations = self.annotations + read_qp_polys(roi_fp)

        self.unique_rois, self.roi_counts, self.anno_names = self._get_uniques_counts(
            self.annotations
        )

    def annotation_tree(self):
        """Prints ascii tree of annotation names and their count."""
        for idx, roi in enumerate(self.unique_rois):
            print("{}".format(str(roi)))
            print("   -{} annotations".format(self.roi_counts[idx]))

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
        """Tile images with sliding window accross entire image or within an
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

        if bound_anno not in self.unique_rois:
            print(
                "{} is not in imported annotations: {} ".format(
                    bound_anno, self.unique_rois
                )
            )
            if build_patches is True:
                raise ValueError("build_patches option must have a bounding annotation")
            self.tissue_roi = self.image_bounds
        else:
            if self.roi_counts[self.unique_rois.index(bound_anno)] > 1:
                print(
                    "bounding anno corresponds to multiple rois {}".format(bound_anno)
                )
                print("merging annotations into MultiPolygon")
                bounding_roi = [
                    anno
                    for anno in self.annotations
                    if anno.roi_attrs["name"] == bound_anno
                ]

                self.tissue_roi = geometry.MultiPolygon(bounding_roi)
            else:
                bound_idx = self.anno_names.index(bound_anno)
                self.tissue_roi = self.annotations[bound_idx]

        bounds = self.tissue_roi.bounds

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

                if self.tissue_roi.intersection(tbox).is_empty:
                    continue

                if tile_check(tbox, self.tissue_roi, edge_thresh) is True:
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

            print("slide:", offset)
            if offset != (0, 0):
                tiles_slid = [
                    tile_translate(tile, offset[0], offset[1]) for tile in tiles
                ]

                tiles_slid = [
                    tile
                    for tile in tiles_slid
                    if tile_check(tile, self.tissue_roi, edge_thresh) is False
                ]

                tiles_slid = [
                    tile
                    for tile in tiles_slid
                    if tile_check(tile, self.image_bounds, 1) is False
                ]

                for tile in tiles_slid:
                    tile.roi_attrs["name"] = roi_name

                self.image_tiles = self.image_tiles + tiles_slid

        self.unique_tiles, self.tile_counts, self.tile_names = self._get_uniques_counts(
            self.image_tiles
        )

    def tile_tree(self):
        """Prints ascii tree of tile/patch names and their count."""
        for idx, roi in enumerate(self.unique_tiles):
            print("{}".format(str(roi)))
            print("   -{} annotations".format(self.tile_counts[idx]))

    def intersect_tiles_annos(self, exclusion_roi=["foreground"]):
        """Short summary.

        Parameters
        ----------
        exclusion_roi : list of str
            list of ROI names that shouldn't be considered for object detection
            this could be, for instance, bounding ROIs or tissue outlines, etc.

        Returns
        -------
        list of dict
            tile data in dict of shapely polygons

        """
        exclusion_roi = ["tissue_roi"]
        annotation_data = [
            roi
            for roi in self.annotations
            if roi.roi_attrs["name"] not in exclusion_roi
        ]

        self.tile_roi_data = []
        for tile in self.image_tiles:
            contained_rois = [roi for roi in annotation_data if tile.contains(roi)]

            if len(contained_rois) > 0:
                print("found roi")
                tile_data = {"tile": tile, "rois": contained_rois}

                self.tile_roi_data.append(tile_data)

    def roi_data_to_np(self):
        """convert ROI data from shapely into dask array and numpy arrays

        Returns
        -------
        list of dict
            tile data in dict of dask array and numpy arrays of object detection
            roi data

        """
        self.roi_data_np = []

        for tile_data in self.tile_roi_data:
            tile_bounds = np.array(tile_data["tile"].bounds, dtype=np.uint32)
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
                "roi_classes": [roi.roi_attrs["name"] for roi in tile_data["rois"]],
            }

            self.roi_data_np.append(np_tile_data)

        # to improve
        n_rois = [
            [idx, True]
            for idx, roi in enumerate(self.roi_data_np)
            if len(roi["rois"]) > 0
        ]
        retained_rois = np.asarray(n_rois)[:, 0].tolist()

        self.roi_data_np = [
            roidata
            for idx, roidata in enumerate(self.roi_data_np)
            if idx in retained_rois
        ]

    def patch_data_to_np(self, exclusion_roi="foreground"):
        """Convert shapely patch data to dask and numpy arrays

        Parameters
        ----------
        exclusion_roi : list of str
            list of ROI names that shouldn't be considered for object detection
            this could be, for instance, bounding ROIs or tissue outlines, etc.

        Returns
        -------
        list of dict
            patch data in dict of dask array and roi class for patch-based image
            classifcation

        """
        patch_data = [
            roi
            for roi in self.image_tiles
            if roi.roi_attrs["name"] not in exclusion_roi
        ]

        self.patch_data_np = []

        for patch in patch_data:
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

            self.patch_data_np.append(patch)

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
            pickle.dump(self.roi_data_np, filehandler)

    def load_roi_data(self, load_fn):
        """load pickled object detection data

        Parameters
        ----------
        load_fn : str
            file path of the pickled data.

        """
        with open(load_fn, "rb") as filehandler:
            self.roi_data_np = pickle.load(filehandler)

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


root = zarr.open_group("test.zarr", mode="a")

pyr_layer = root.create_dataset(
    "0",
    shape=(3, 2 ** 16, 2 ** 16),
    chunks=(3, 1024, 1024),
    dtype=np.uint8,
    overwrite=True,
)

self = WsiMap("test.zarr", pyr_layer=0)

self.import_annotations(
    "/home/nhp/wsi/testdata/tissueROI_polyInterior2x_test_qupath_annotations.json"
)

self.import_annotations(
    "/home/nhp/wsi/testdata/non_contig_roi_test_qupath_annotations.json"
)
self.annotation_tree()
