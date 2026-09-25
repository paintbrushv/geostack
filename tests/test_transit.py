"""Tests for geostack.transit — Walk Score API + GTFS transit access."""

import pandas as pd
import geopandas as gpd
import pytest
from unittest.mock import patch, MagicMock
from shapely.geometry import Point

REPUBLICS_LON = -96.6183
REPUBLICS_LAT = 32.8028


@pytest.fixture
def mock_walkscore_response():
    return {
        "status": 1,
        "walkscore": 23,
        "description": "Car-Dependent",
        "transit": {"score": 18, "description": "Minimal Transit"},
        "bike": {"score": 31, "description": "Bikeable"},
    }


@pytest.fixture
def mock_stops_gdf():
    return gpd.GeoDataFrame(
        {
            "stop_id": ["s1", "s2", "s3", "s4"],
            "stop_name": ["Bus Stop A", "DART Rail Garland", "Bus Stop B", "Far Stop"],
            "route_type": [3, 0, 3, 3],
        },
        geometry=[
            Point(REPUBLICS_LON + 0.002, REPUBLICS_LAT + 0.001),
            Point(REPUBLICS_LON + 0.005, REPUBLICS_LAT + 0.003),
            Point(REPUBLICS_LON - 0.003, REPUBLICS_LAT + 0.002),
            Point(REPUBLICS_LON + 0.05, REPUBLICS_LAT + 0.05),
        ],
        crs="EPSG:4326",
    )


class TestWalkScore:
    @patch("geostack.transit.requests.get")
    def test_returns_score_dict(self, mock_get, mock_walkscore_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_walkscore_response
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from geostack.transit import walk_score
        result = walk_score(REPUBLICS_LON, REPUBLICS_LAT,
                           address="245 E I-30, Garland TX",
                           api_key="test_key")

        assert isinstance(result, dict)
        assert result["walkscore"] == 23
        assert result["transit_score"] == 18
        assert result["bike_score"] == 31

    @patch("geostack.transit.requests.get")
    def test_handles_missing_transit(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": 1, "walkscore": 45, "description": "Car-Dependent",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from geostack.transit import walk_score
        result = walk_score(REPUBLICS_LON, REPUBLICS_LAT,
                           address="test", api_key="test_key")

        assert result["walkscore"] == 45
        assert result["transit_score"] is None
        assert result["bike_score"] is None


class TestGtfsAccess:
    def test_counts_stops_within_half_mile(self, mock_stops_gdf):
        from geostack.transit import _count_nearby_stops
        result = _count_nearby_stops(
            REPUBLICS_LON, REPUBLICS_LAT, mock_stops_gdf,
            radius_miles=0.5
        )
        assert result["stops_within_radius"] == 3
        assert result["rail_stops_within_radius"] == 1

    def test_identifies_nearest_rail(self, mock_stops_gdf):
        from geostack.transit import _count_nearby_stops
        result = _count_nearby_stops(
            REPUBLICS_LON, REPUBLICS_LAT, mock_stops_gdf,
            radius_miles=0.5
        )
        assert result["nearest_rail_name"] == "DART Rail Garland"

    def test_no_rail_returns_none(self):
        from geostack.transit import _count_nearby_stops
        bus_only = gpd.GeoDataFrame(
            {"stop_id": ["s1"], "stop_name": ["Bus Stop"], "route_type": [3]},
            geometry=[Point(REPUBLICS_LON + 0.001, REPUBLICS_LAT)],
            crs="EPSG:4326",
        )
        result = _count_nearby_stops(
            REPUBLICS_LON, REPUBLICS_LAT, bus_only, radius_miles=0.5
        )
        assert result["nearest_rail_name"] is None


class TestTransitProfile:
    @patch("geostack.transit._load_gtfs_stops")
    @patch("geostack.transit.walk_score")
    def test_combined_profile(self, mock_ws, mock_gtfs, mock_stops_gdf):
        mock_ws.return_value = {
            "walkscore": 23, "transit_score": 18, "bike_score": 31,
            "description": "Car-Dependent",
        }
        mock_gtfs.return_value = mock_stops_gdf

        from geostack.transit import transit_profile
        result = transit_profile(
            REPUBLICS_LON, REPUBLICS_LAT,
            address="245 E I-30, Garland TX",
            api_key="test_key",
            gtfs_path="fake.zip",
        )

        assert result["walkscore"] == 23
        assert result["stops_within_half_mile"] == 3
