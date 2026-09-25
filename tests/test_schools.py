"""Tests for geostack.schools — school district quality lookup."""

import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box
from unittest.mock import patch

REPUBLICS_LON = -96.6183
REPUBLICS_LAT = 32.8028  # Garland ISD territory
EXAMPLE_LON = -96.8028
EXAMPLE_LAT = 32.9500    # Richardson ISD territory


@pytest.fixture
def tea_ratings_df():
    """Minimal TEA accountability ratings DataFrame (canonical format)."""
    return pd.DataFrame([
        {"district_id": "57902", "district_name": "Garland ISD",
         "overall_rating": "B", "student_achievement": "C",
         "school_progress": "B", "closing_gaps": "A"},
        {"district_id": "57905", "district_name": "Richardson ISD",
         "overall_rating": "A", "student_achievement": "A",
         "school_progress": "A", "closing_gaps": "B"},
    ])


@pytest.fixture
def mock_district_gdf():
    """Two synthetic TIGER school district polygons."""
    garland_geom = box(REPUBLICS_LON - 0.15, REPUBLICS_LAT - 0.10,
                       REPUBLICS_LON + 0.05, REPUBLICS_LAT + 0.10)
    richardson_geom = box(EXAMPLE_LON - 0.15, EXAMPLE_LAT - 0.10,
                          EXAMPLE_LON + 0.05, EXAMPLE_LAT + 0.10)
    return gpd.GeoDataFrame(
        {"GEOID": ["4857902", "4857905"],
         "NAME": ["Garland ISD", "Richardson ISD"],
         "LEAID": ["57902", "57905"]},
        geometry=[garland_geom, richardson_geom],
        crs="EPSG:4326",
    )


class TestParseTEARatings:
    def test_parse_canonical_returns_dataframe(self, tea_ratings_df):
        from geostack.schools import _parse_tea_ratings
        result = _parse_tea_ratings(tea_ratings_df)
        assert isinstance(result, pd.DataFrame)
        assert "district_id" in result.columns
        assert "overall_rating" in result.columns

    def test_district_id_is_zero_padded_string(self, tea_ratings_df):
        from geostack.schools import _parse_tea_ratings
        result = _parse_tea_ratings(tea_ratings_df)
        # district_id "57902" → zero-padded to 6 chars → "057902"
        assert result["district_id"].dtype == object
        assert all(len(x) == 6 for x in result["district_id"])

    def test_parse_raw_csv_columns(self):
        """Simulate raw TEA CSV with capitalized column names."""
        from geostack.schools import _parse_tea_ratings
        raw = pd.DataFrame([{
            "DISTRICT": "57902", "DISTNAME": "Garland ISD",
            "D_RATING": "B", "DD1G": "C", "DD2G": "B", "DD3G": "A",
        }])
        result = _parse_tea_ratings(raw)
        assert "district_id" in result.columns
        assert "overall_rating" in result.columns
        assert result.iloc[0]["district_id"] == "057902"
        assert result.iloc[0]["overall_rating"] == "B"


class TestDistrictAtPoint:
    def test_returns_district_name_for_republics(self, mock_district_gdf):
        from geostack.schools import _district_at_point
        name, leaid = _district_at_point(REPUBLICS_LON, REPUBLICS_LAT, mock_district_gdf)
        assert name == "Garland ISD"
        assert leaid == "57902"

    def test_returns_correct_district_for_the_example_point(self, mock_district_gdf):
        from geostack.schools import _district_at_point
        name, leaid = _district_at_point(EXAMPLE_LON, EXAMPLE_LAT, mock_district_gdf)
        assert name == "Richardson ISD"
        assert leaid == "57905"

    def test_returns_none_for_point_outside_all_districts(self, mock_district_gdf):
        from geostack.schools import _district_at_point
        name, leaid = _district_at_point(-100.0, 35.0, mock_district_gdf)
        assert name is None
        assert leaid is None


class TestSchoolProfile:
    @patch("geostack.schools._load_tiger_districts")
    @patch("geostack.schools._fetch_tea_ratings")
    def test_school_profile_returns_dict(
        self, mock_tea, mock_tiger, mock_district_gdf, tea_ratings_df
    ):
        from geostack.schools import _parse_tea_ratings
        mock_tiger.return_value = mock_district_gdf
        mock_tea.return_value = _parse_tea_ratings(tea_ratings_df)

        from geostack.schools import school_profile
        result = school_profile(REPUBLICS_LON, REPUBLICS_LAT, year=2023)

        assert isinstance(result, dict)
        assert result["district_name"] == "Garland ISD"
        assert result["tea_rating"] == "B"
        assert result["leaid"] == "57902"
        assert result["year"] == 2023

    @patch("geostack.schools._load_tiger_districts")
    @patch("geostack.schools._fetch_tea_ratings")
    def test_school_profile_no_district_returns_none_fields(
        self, mock_tea, mock_tiger, mock_district_gdf, tea_ratings_df
    ):
        from geostack.schools import _parse_tea_ratings
        mock_tiger.return_value = mock_district_gdf
        mock_tea.return_value = _parse_tea_ratings(tea_ratings_df)

        from geostack.schools import school_profile
        result = school_profile(-100.0, 35.0, year=2023)

        assert result["district_name"] is None
        assert result["tea_rating"] is None
        assert result["leaid"] is None

    @patch("geostack.schools._load_tiger_districts")
    @patch("geostack.schools._fetch_tea_ratings")
    def test_school_profile_all_expected_keys_present(
        self, mock_tea, mock_tiger, mock_district_gdf, tea_ratings_df
    ):
        from geostack.schools import _parse_tea_ratings
        mock_tiger.return_value = mock_district_gdf
        mock_tea.return_value = _parse_tea_ratings(tea_ratings_df)

        from geostack.schools import school_profile
        result = school_profile(REPUBLICS_LON, REPUBLICS_LAT, year=2023)

        expected_keys = {"district_name", "leaid", "tea_rating",
                         "student_achievement", "school_progress", "closing_gaps", "year"}
        assert expected_keys == set(result.keys())


class TestNameMatching:
    def test_normalize_tiger_name(self):
        from geostack.schools import _normalize_district_name
        assert _normalize_district_name("Garland Independent School District") == "garland"
        assert _normalize_district_name("Mesquite Independent School District") == "mesquite"
        assert _normalize_district_name("Dallas Independent School District") == "dallas"

    def test_normalize_tea_name(self):
        from geostack.schools import _normalize_district_name
        assert _normalize_district_name("GARLAND ISD") == "garland"
        assert _normalize_district_name("MESQUITE ISD") == "mesquite"
        assert _normalize_district_name("DALLAS ISD") == "dallas"

    def test_name_fallback_matches(self):
        """When LEAID doesn't match, name-based match should find the district."""
        from geostack.schools import _match_by_name
        tea_df = pd.DataFrame([
            {"district_id": "999999", "district_name": "GARLAND ISD",
             "overall_rating": "B"},
        ])
        result = _match_by_name("Garland Independent School District", tea_df)
        assert len(result) == 1
        assert result.iloc[0]["overall_rating"] == "B"

    @patch("geostack.schools._load_tiger_districts")
    @patch("geostack.schools._fetch_tea_ratings")
    def test_school_profile_uses_name_fallback(
        self, mock_tea, mock_tiger, mock_district_gdf
    ):
        """When LEAID doesn't match TEA, name-based fallback should work."""
        mock_tiger.return_value = mock_district_gdf
        # TEA data with different district IDs than TIGER LEAIDs
        mock_tea.return_value = pd.DataFrame([
            {"district_id": "999901", "district_name": "GARLAND ISD",
             "overall_rating": "B", "student_achievement": "C",
             "school_progress": "B", "closing_gaps": "A"},
        ])

        from geostack.schools import school_profile
        result = school_profile(REPUBLICS_LON, REPUBLICS_LAT, year=2023)
        assert result["tea_rating"] == "B"
        assert result["district_name"] == "Garland ISD"
