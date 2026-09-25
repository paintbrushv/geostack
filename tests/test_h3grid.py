"""Tests for geostack.h3grid — H3 hexagonal grid layers (network-free)."""

import numpy as np
import pandas as pd
import pytest

import geopandas as gpd
from shapely.geometry import Polygon, Point

h3 = pytest.importorskip("h3")  # skip the whole module if h3 isn't installed

from geostack.h3grid import (
    cell_to_polygon,
    cells_covering,
    cells_to_gdf,
    interpolate_polygons_to_h3,
    dasymetric_polygons_to_h3,
    points_to_h3,
    h3_edge_km,
    h3_area_sqmi,
)


# ---------------------------------------------------------------------------
# Resolution metadata
# ---------------------------------------------------------------------------

class TestResolutionMeta:
    def test_edge_km_decreases_with_resolution(self):
        assert h3_edge_km(7) > h3_edge_km(8) > h3_edge_km(9)

    def test_res8_edge_about_half_km(self):
        assert 0.4 < h3_edge_km(8) < 0.7

    def test_area_sqmi_positive(self):
        assert h3_area_sqmi(8) > 0
        assert h3_area_sqmi(7) > h3_area_sqmi(8)


# ---------------------------------------------------------------------------
# Cell ↔ geometry
# ---------------------------------------------------------------------------

class TestCellGeometry:
    def test_cell_to_polygon_valid(self):
        cell = h3.latlng_to_cell(32.7777, -96.7970, 8)
        poly = cell_to_polygon(cell)
        assert poly.is_valid
        assert poly.geom_type == "Polygon"
        # Hexagon → 6 distinct vertices (+ closing point)
        assert len(poly.exterior.coords) == 7

    def test_cell_polygon_contains_center(self):
        cell = h3.latlng_to_cell(32.7777, -96.7970, 8)
        lat, lng = h3.cell_to_latlng(cell)
        assert cell_to_polygon(cell).contains(Point(lng, lat))

    def test_cells_covering_box(self):
        box = Polygon([(-96.85, 32.74), (-96.74, 32.74), (-96.74, 32.82), (-96.85, 32.82)])
        cells = cells_covering(box, 8)
        assert len(cells) > 50
        assert all(isinstance(c, str) for c in cells)

    def test_cells_to_gdf(self):
        box = Polygon([(-96.85, 32.74), (-96.74, 32.74), (-96.74, 32.82), (-96.85, 32.82)])
        cells = cells_covering(box, 8)
        gdf = cells_to_gdf(cells)
        assert len(gdf) == len(cells)
        assert gdf.crs.to_epsg() == 4326
        assert "h3" in gdf.columns


# ---------------------------------------------------------------------------
# Areal interpolation
# ---------------------------------------------------------------------------

class TestInterpolation:
    def _square_tract(self, value=1000.0):
        # A ~5km square around downtown Dallas with a count value
        poly = Polygon([(-96.83, 32.74), (-96.75, 32.74), (-96.75, 32.81), (-96.83, 32.81)])
        return gpd.GeoDataFrame(
            {"GEOID": ["48000000001"], "population": [value]},
            geometry=[poly], crs="EPSG:4326",
        )

    def test_conserves_count_total(self):
        tract = self._square_tract(1000.0)
        hexes = interpolate_polygons_to_h3(tract, resolution=8, count_cols=["population"])
        # Area-weighted allocation conserves the total (within rounding)
        assert abs(hexes["population"].sum() - 1000.0) < 1.0

    def test_produces_multiple_hexes(self):
        tract = self._square_tract()
        hexes = interpolate_polygons_to_h3(tract, resolution=8, count_cols=["population"])
        assert len(hexes) > 5
        assert "covered_frac" in hexes.columns
        assert "hex_area_sqmi" in hexes.columns

    def test_two_tracts_sum(self):
        t1 = self._square_tract(600.0)
        t2 = self._square_tract(400.0)
        t2 = t2.copy()
        t2["GEOID"] = ["48000000002"]
        # Shift t2 east so they don't overlap
        t2["geometry"] = t2.translate(xoff=0.10)
        both = pd.concat([t1, t2], ignore_index=True)
        both = gpd.GeoDataFrame(both, geometry="geometry", crs="EPSG:4326")
        hexes = interpolate_polygons_to_h3(both, resolution=8, count_cols=["population"])
        assert abs(hexes["population"].sum() - 1000.0) < 2.0

    def test_mean_col_area_weighted(self):
        # An intensive variable (income) should land near the source value,
        # not be summed.
        tract = self._square_tract()
        tract["median_income"] = 80000.0
        hexes = interpolate_polygons_to_h3(
            tract, resolution=8, count_cols=["population"], mean_cols=["median_income"],
        )
        vals = hexes["median_income"].dropna()
        assert (abs(vals - 80000.0) < 1.0).all()


# ---------------------------------------------------------------------------
# Point binning
# ---------------------------------------------------------------------------

class TestDasymetric:
    def _tract(self, value=1000.0):
        return gpd.GeoDataFrame(
            {"GEOID": ["48000000001"], "jobs": [value]},
            geometry=[Polygon([(-96.83, 32.74), (-96.75, 32.74),
                               (-96.75, 32.81), (-96.83, 32.81)])],
            crs="EPSG:4326",
        )

    def _weight_points_west(self, n=40):
        # Cluster all weight points in the western quarter of the tract
        pts = [Point(-96.82 + 0.004 * np.cos(i), 32.76 + 0.004 * np.sin(i)) for i in range(n)]
        return gpd.GeoDataFrame({"val": [1.0] * n}, geometry=pts, crs="EPSG:4326")

    def test_conserves_total(self):
        hexes = dasymetric_polygons_to_h3(
            self._tract(1000.0), self._weight_points_west(), resolution=9,
            count_cols=["jobs"],
        )
        assert abs(hexes["jobs"].sum() - 1000.0) < 1.0

    def test_concentrates_where_points_are(self):
        # All weight points are west → allocated jobs should sit west of the
        # tract's geographic center (-96.79), not spread uniformly.
        hexes = dasymetric_polygons_to_h3(
            self._tract(1000.0), self._weight_points_west(), resolution=9,
            count_cols=["jobs"],
        ).to_crs("EPSG:4326")
        cx = (hexes.geometry.centroid.x * hexes["jobs"]).sum() / hexes["jobs"].sum()
        assert cx < -96.80  # weighted center pulled west

    def test_weight_col_scales(self):
        # A point with 10x weight should pull allocation toward it
        wp = gpd.GeoDataFrame(
            {"val": [1.0, 100.0]},
            geometry=[Point(-96.82, 32.77), Point(-96.76, 32.77)],
            crs="EPSG:4326",
        )
        hexes = dasymetric_polygons_to_h3(
            self._tract(1000.0), wp, resolution=8, count_cols=["jobs"], weight_col="val",
        )
        # East point (100x) gets the lion's share
        east = hexes[hexes.to_crs("EPSG:4326").geometry.centroid.x > -96.79]["jobs"].sum()
        assert east > 800


class TestPointsToH3:
    def test_counts_points(self):
        # 10 points clustered in one hex, 1 far away → 2 cells
        pts = [Point(-96.797, 32.7777) for _ in range(10)] + [Point(-97.33, 32.75)]
        gdf = gpd.GeoDataFrame({"v": list(range(11))}, geometry=pts, crs="EPSG:4326")
        hexes = points_to_h3(gdf, resolution=8)
        assert hexes["point_count"].sum() == 11
        assert len(hexes) == 2

    def test_value_aggregation(self):
        pts = [Point(-96.797, 32.7777), Point(-96.797, 32.7777)]
        gdf = gpd.GeoDataFrame({"v": [10.0, 30.0]}, geometry=pts, crs="EPSG:4326")
        hexes = points_to_h3(gdf, resolution=8, value_cols=["v"], agg="sum")
        assert hexes["v"].iloc[0] == 40.0
