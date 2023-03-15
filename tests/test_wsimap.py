#!/usr/bin/env python

"""Tests for `wsimap` package."""

import pytest

from wsimap.wsi_map import WsiMap
from wsimap.utils.anno_utils import read_qp_geojson
from tests.test_annotations import TestGeoJSONannotations
from tests.test_wsi import TestWSIzarr
from shapely.ops import cascaded_union


@pytest.fixture(scope="session", autouse=True)
def multi_class_geojson(tmpdir_factory):
    """ """
    tg = TestGeoJSONannotations()
    return tg.write_temp_files(tmpdir_factory.mktemp("json"))


@pytest.fixture(scope="session", autouse=True)
def test_zarr(tmpdir_factory):
    """ """
    tz = TestWSIzarr(tmpdir_factory.mktemp("zarr"))
    return tz.get_zarr_fp()


def test_multiclass_anno_util(multi_class_geojson):
    """Sample pytest test function with the pytest fixture as an argument."""
    qp_mc = read_qp_geojson(multi_class_geojson["multi_class"])
    assert len(qp_mc) == 9


def test_multiclass_wsimap_import(test_zarr, multi_class_geojson):
    wsi = WsiMap(test_zarr)
    wsi.import_annotations(multi_class_geojson["multi_class"])
    assert len(wsi.roi_unique_names) == 3
    assert wsi.roi_unique_counts[0] == 3
    assert wsi.roi_unique_counts[1] == 3
    assert wsi.roi_unique_counts[2] == 3
    assert "class1" in wsi.roi_names
    assert "class2" in wsi.roi_names
    assert "class3" in wsi.roi_names


def test_noncontig_wsimap_import(test_zarr, multi_class_geojson):
    wsi = WsiMap(test_zarr)
    wsi.import_annotations(multi_class_geojson["non_contig"])
    assert len(wsi.roi_names) == 4


def test_tissue_roi_wsimap_import(test_zarr, multi_class_geojson):
    wsi = WsiMap(test_zarr)
    wsi.import_annotations(multi_class_geojson["tissue_roi"])
    assert len(wsi.roi_names) == 3


# checks tiling occurs in two non contiguous rois
def test_noncontig_wsimap_tiling(test_zarr, multi_class_geojson):
    wsi = WsiMap(test_zarr)
    wsi.import_annotations(multi_class_geojson["non_contig"])
    wsi.tile_wsi(
        bound_anno="non_contig_roi",
        tile_size=224,
        slide=None,
        edge_thresh=1,
        build_patches=True,
    )
    wsi.tile_data_to_patches()
    tile_poly = cascaded_union(wsi.image_tiles)
    assert len(tile_poly) == 2


def test_tissue_roi_wsimap_tiling(test_zarr, multi_class_geojson):
    wsi = WsiMap(test_zarr)
    wsi.import_annotations(multi_class_geojson["tissue_roi"])
    wsi.tile_wsi(
        bound_anno="tissue_roi",
        tile_size=224,
        slide=None,
        edge_thresh=1,
        build_patches=True,
    )
    wsi.tile_data_to_patches()
    tile_poly = cascaded_union(wsi.image_tiles)
    assert tile_poly.type == "Polygon"


#
#
# def test_command_line_interface():
#     """Test the CLI."""
#     runner = CliRunner()
#     result = runner.invoke(cli.main)
#     assert result.exit_code == 0
#     assert "wsimap.cli.main" in result.output
#     help_result = runner.invoke(cli.main, ["--help"])
#     assert help_result.exit_code == 0
#     assert "--help  Show this message and exit." in help_result.output
