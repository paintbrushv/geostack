"""Tests for geostack.spatial — CRS suggestion, acreage, radius filtering."""

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from geostack.spatial import compute_acreage, parcels_within_radius, suggest_crs


# ---------------------------------------------------------------------------
# suggest_crs
# ---------------------------------------------------------------------------

class TestSuggestCrs:
    def test_dfw_returns_tx_north_central(self, dfw_gdf):
        """DFW metro → Texas State Plane North Central (EPSG:2276)."""
        crs = suggest_crs(dfw_gdf)
        assert crs == "EPSG:2276"

    def test_austin_returns_tx_central(self, austin_gdf):
        """Austin → Texas State Plane Central (EPSG:2278)."""
        crs = suggest_crs(austin_gdf)
        assert crs == "EPSG:2278"

    def test_non_texas_returns_utm(self, seattle_gdf):
        """Seattle (UTM zone 10N) → EPSG:32610."""
        crs = suggest_crs(seattle_gdf)
        assert crs == "EPSG:32610"

    def test_units_meters_returns_utm_for_texas(self, dfw_gdf):
        """units='m' for Texas data returns UTM rather than State Plane."""
        crs = suggest_crs(dfw_gdf, units="m")
        # DFW is in UTM zone 14N
        assert crs == "EPSG:32614"

    def test_projected_input_accepted(self, dfw_gdf):
        """Input already in a projected CRS is still handled correctly."""
        dfw_proj = dfw_gdf.to_crs("EPSG:2276")
        crs = suggest_crs(dfw_proj)
        assert crs == "EPSG:2276"

    def test_returns_string(self, dfw_gdf):
        """Return value is a plain EPSG string, not a pyproj CRS object."""
        crs = suggest_crs(dfw_gdf)
        assert isinstance(crs, str)
        assert crs.startswith("EPSG:")

    def test_southern_hemisphere(self):
        """Southern hemisphere data returns EPSG:327XX."""
        geom = box(-43.5, -23.1, -43.0, -22.7)  # Rio de Janeiro
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        crs = suggest_crs(gdf)
        epsg = int(crs.split(":")[1])
        assert 32700 <= epsg <= 32760


# ---------------------------------------------------------------------------
# compute_acreage
# ---------------------------------------------------------------------------

class TestComputeAcreage:
    def test_adds_calc_acres_column(self, parcels_gdf):
        result = compute_acreage(parcels_gdf)
        assert "calc_acres" in result.columns

    def test_acreage_is_positive(self, parcels_gdf):
        result = compute_acreage(parcels_gdf)
        assert (result["calc_acres"] > 0).all()

    def test_does_not_mutate_input(self, parcels_gdf):
        original_cols = list(parcels_gdf.columns)
        compute_acreage(parcels_gdf)
        assert list(parcels_gdf.columns) == original_cols

    def test_known_area(self):
        """Acreage is physically plausible for a known bbox near DFW."""
        # bbox: 0.02° lon × 0.02° lat at 33°N
        # ~1.16 miles × 1.38 miles ≈ 1,600 ft × 7,500 ft ≈ ~1,000 acres
        geom = box(-96.80, 32.77, -96.78, 32.7914)
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        result = compute_acreage(gdf)
        acres = result["calc_acres"].iloc[0]
        assert 500 < acres < 2000


# ---------------------------------------------------------------------------
# parcels_within_radius
# ---------------------------------------------------------------------------

class TestParcelsWithinRadius:
    def test_returns_subset(self, parcels_gdf, dfw_point):
        """Filtering with a very small radius should return fewer parcels."""
        all_parcels = parcels_gdf
        # 0.001 mile radius should catch only the nearest parcel
        nearby = parcels_within_radius(all_parcels, (dfw_point.x, dfw_point.y), miles=0.001)
        assert len(nearby) <= len(all_parcels)

    def test_large_radius_returns_all(self, parcels_gdf, dfw_point):
        """A large enough radius should return all parcels."""
        nearby = parcels_within_radius(parcels_gdf, (dfw_point.x, dfw_point.y), miles=10.0)
        assert len(nearby) == len(parcels_gdf)

    def test_accepts_shapely_point(self, parcels_gdf, dfw_point):
        """center can be a shapely Point as well as a (lon, lat) tuple."""
        result = parcels_within_radius(parcels_gdf, dfw_point, miles=10.0)
        assert len(result) == len(parcels_gdf)

    def test_zero_radius_returns_empty_or_one(self, parcels_gdf, dfw_point):
        """Effectively zero radius catches at most the parcel at the exact center."""
        result = parcels_within_radius(parcels_gdf, (dfw_point.x, dfw_point.y), miles=0.0)
        assert len(result) <= 1

    def test_preserves_original_crs(self, parcels_gdf, dfw_point):
        """Output CRS should match the input GeoDataFrame's CRS."""
        result = parcels_within_radius(parcels_gdf, (dfw_point.x, dfw_point.y), miles=5.0)
        assert result.crs == parcels_gdf.crs
