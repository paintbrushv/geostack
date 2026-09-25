"""Tests for geostack.demographics — isochrone-based demographic analytics.

Uses Republics East+West coordinates (245 E Interstate 30, Garland, TX 75043)
as the reference property: lat=32.8028, lon=-96.6183.
"""

import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import box, Point
from unittest.mock import MagicMock, patch

from geostack.demographics import (
    tracts_within_isochrone,
    isochrone_demographics,
    isochrone_employment,
    isochrone_commute_shed,
    acs_time_series,
)

# Republics East+West coordinates
REPUBLICS_LON = -96.6183
REPUBLICS_LAT = 32.8028
STATE_FIPS = "48"
COUNTY_FIPS = "113"  # Dallas County


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_isochrone_polygon():
    """A synthetic isochrone polygon around Republics (~15 min drive)."""
    # Roughly 0.1 degrees ≈ 6-7 miles ≈ 10-15 min drive
    return box(
        REPUBLICS_LON - 0.1, REPUBLICS_LAT - 0.08,
        REPUBLICS_LON + 0.1, REPUBLICS_LAT + 0.08,
    )


@pytest.fixture
def mock_tracts_gdf():
    """Synthetic census tracts that overlap the Republics isochrone."""
    tracts = []
    geoids = [
        "48113012345", "48113012346", "48113012347",
        "48113012348", "48113012349",
    ]
    for i, geoid in enumerate(geoids):
        # Create tract polygons at slightly different positions
        offset = i * 0.03
        geom = box(
            REPUBLICS_LON - 0.05 + offset, REPUBLICS_LAT - 0.04,
            REPUBLICS_LON - 0.02 + offset, REPUBLICS_LAT + 0.04,
        )
        tracts.append({"geoid": geoid, "state_fips": "48", "county_fips": "113"})

    gdf = gpd.GeoDataFrame(
        tracts,
        geometry=[
            box(REPUBLICS_LON - 0.05 + i * 0.03, REPUBLICS_LAT - 0.04,
                REPUBLICS_LON - 0.02 + i * 0.03, REPUBLICS_LAT + 0.04)
            for i in range(len(tracts))
        ],
        crs="EPSG:4326",
    )
    return gdf


@pytest.fixture
def mock_acs_gdf():
    """Synthetic ACS tract data for testing demographics aggregation."""
    geoids = [
        "48113012345", "48113012346", "48113012347",
        "48113012348", "48113012349",
    ]
    data = {
        "total_population": [3000, 4500, 2800, 5200, 3100],
        "median_age": [34.5, 29.1, 38.2, 31.7, 36.0],
        "median_household_income": [52000, 45000, 68000, 41000, 55000],
        "per_capita_income": [28000, 22000, 35000, 20000, 30000],
        "total_housing_units": [1200, 1800, 1100, 2100, 1300],
        "occupied_units": [1100, 1650, 1000, 1950, 1200],
        "vacant_units": [100, 150, 100, 150, 100],
        "renter_occupied": [700, 1200, 500, 1500, 800],
        "owner_occupied": [400, 450, 500, 450, 400],
        "median_rent": [1050, 950, 1200, 900, 1100],
        "median_home_value": [250000, 200000, 320000, 180000, 270000],
        "units_in_structure_20plus": [200, 400, 100, 500, 250],
        "units_in_structure_10_19": [150, 250, 80, 350, 150],
        "units_in_structure_5_9": [100, 200, 70, 300, 120],
        "units_in_structure_3_4": [50, 100, 50, 150, 60],
        "units_in_structure_2": [30, 50, 30, 80, 40],
        "moved_in_last_year": [200, 350, 150, 400, 220],
    }
    gdf = gpd.GeoDataFrame(
        data,
        index=geoids,
        geometry=[
            box(REPUBLICS_LON - 0.05 + i * 0.03, REPUBLICS_LAT - 0.04,
                REPUBLICS_LON - 0.02 + i * 0.03, REPUBLICS_LAT + 0.04)
            for i in range(len(geoids))
        ],
        crs="EPSG:4326",
    )
    gdf.index.name = "geoid"
    return gdf


# ---------------------------------------------------------------------------
# tracts_within_isochrone
# ---------------------------------------------------------------------------

class TestTractsWithinIsochrone:
    @patch("geostack.demographics.read_postgis")
    def test_returns_geodataframe_with_overlap(self, mock_postgis, mock_tracts_gdf):
        """Should return tracts with overlap_fraction column."""
        result_gdf = mock_tracts_gdf.copy()
        result_gdf["overlap_fraction"] = [1.0, 0.95, 0.8, 0.6, 0.3]
        mock_postgis.return_value = result_gdf

        result = tracts_within_isochrone(
            REPUBLICS_LON, REPUBLICS_LAT, 15, STATE_FIPS, method="buffer",
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert "overlap_fraction" in result.columns
        assert "geoid" in result.columns
        assert "minutes" in result.columns
        assert len(result) == 5

    @patch("geostack.demographics.read_postgis")
    def test_overlap_fraction_clipped(self, mock_postgis, mock_tracts_gdf):
        """overlap_fraction should be clipped to [0, 1]."""
        result_gdf = mock_tracts_gdf.copy()
        result_gdf["overlap_fraction"] = [1.2, 0.95, -0.1, 0.6, 0.3]
        mock_postgis.return_value = result_gdf

        result = tracts_within_isochrone(
            REPUBLICS_LON, REPUBLICS_LAT, 15, STATE_FIPS, method="buffer",
        )

        assert result["overlap_fraction"].max() <= 1.0
        assert result["overlap_fraction"].min() >= 0.0


# ---------------------------------------------------------------------------
# isochrone_demographics
# ---------------------------------------------------------------------------

class TestIsochroneDemographics:
    @patch("geostack.demographics.get_tract_demographics")
    @patch("geostack.demographics.tracts_within_isochrone")
    def test_returns_summary_with_expected_fields(self, mock_tracts, mock_acs, mock_tracts_gdf, mock_acs_gdf):
        """Summary should contain key demographic metrics."""
        tracts = mock_tracts_gdf.copy()
        tracts["overlap_fraction"] = [1.0, 0.9, 0.8, 0.7, 0.5]
        tracts["minutes"] = 15
        mock_tracts.return_value = tracts

        mock_acs.return_value = mock_acs_gdf

        tract_result, summary = isochrone_demographics(
            REPUBLICS_LON, REPUBLICS_LAT, 15, STATE_FIPS, COUNTY_FIPS, 2023,
        )

        assert not summary.empty
        assert "total_population" in summary.columns
        assert "median_household_income" in summary.columns
        assert "vacancy_rate" in summary.columns
        assert "renter_pct" in summary.columns
        assert "minutes" in summary.columns
        assert summary.iloc[0]["minutes"] == 15

    @patch("geostack.demographics.get_tract_demographics")
    @patch("geostack.demographics.tracts_within_isochrone")
    def test_population_weighted_by_overlap(self, mock_tracts, mock_acs, mock_tracts_gdf, mock_acs_gdf):
        """Total population should be weighted by overlap_fraction, not raw sum."""
        tracts = mock_tracts_gdf.copy()
        tracts["overlap_fraction"] = [1.0, 0.5, 0.5, 0.5, 0.5]
        tracts["minutes"] = 15
        mock_tracts.return_value = tracts

        mock_acs.return_value = mock_acs_gdf

        _, summary = isochrone_demographics(
            REPUBLICS_LON, REPUBLICS_LAT, 15, STATE_FIPS, COUNTY_FIPS, 2023,
        )

        raw_total = mock_acs_gdf["total_population"].sum()
        weighted_total = summary.iloc[0]["total_population"]

        # Weighted total should be less than raw total (since overlaps < 1)
        assert weighted_total < raw_total
        assert weighted_total > 0

    @patch("geostack.demographics.get_tract_demographics")
    @patch("geostack.demographics.tracts_within_isochrone")
    def test_handles_empty_tracts(self, mock_tracts, mock_acs):
        """Should return empty DataFrames gracefully when no tracts found."""
        mock_tracts.return_value = gpd.GeoDataFrame()

        tract_result, summary = isochrone_demographics(
            REPUBLICS_LON, REPUBLICS_LAT, 15, STATE_FIPS, COUNTY_FIPS, 2023,
        )

        assert tract_result.empty
        assert summary.empty


# ---------------------------------------------------------------------------
# isochrone_employment
# ---------------------------------------------------------------------------

class TestIsochroneEmployment:
    @patch("geostack.demographics.read_sql")
    @patch("geostack.demographics.tracts_within_isochrone")
    def test_returns_employment_summary(self, mock_tracts, mock_sql, mock_tracts_gdf):
        """Should return employment summary with sector columns."""
        tracts = mock_tracts_gdf.copy()
        tracts["overlap_fraction"] = [1.0, 0.9, 0.8, 0.7, 0.5]
        tracts["minutes"] = 15
        mock_tracts.return_value = tracts

        # Mock LODES WAC query result
        emp_data = pd.DataFrame({
            "tract_geoid": ["48113012345", "48113012346", "48113012347"],
            "total_jobs": [500, 800, 300],
            "jobs_earn_low": [100, 200, 80],
            "jobs_earn_mid": [200, 300, 120],
            "jobs_earn_high": [200, 300, 100],
            "jobs_age_29_under": [150, 240, 90],
            "jobs_age_30_54": [250, 400, 150],
            "jobs_age_55_over": [100, 160, 60],
        })
        mock_sql.return_value = emp_data

        emp_df, summary = isochrone_employment(
            REPUBLICS_LON, REPUBLICS_LAT, 15, STATE_FIPS, COUNTY_FIPS, 2021,
        )

        assert not summary.empty
        assert "total_jobs" in summary.columns
        assert "minutes" in summary.columns
        assert summary.iloc[0]["total_jobs"] > 0

    @patch("geostack.demographics.read_sql")
    @patch("geostack.demographics.tracts_within_isochrone")
    def test_employment_weighted_by_overlap(self, mock_tracts, mock_sql, mock_tracts_gdf):
        """Jobs should be weighted by overlap_fraction."""
        tracts = mock_tracts_gdf.copy()
        tracts["overlap_fraction"] = [0.5, 0.5, 0.5, 0.5, 0.5]
        tracts["minutes"] = 15
        mock_tracts.return_value = tracts

        emp_data = pd.DataFrame({
            "tract_geoid": ["48113012345", "48113012346"],
            "total_jobs": [1000, 1000],
            "jobs_earn_low": [300, 300],
            "jobs_earn_mid": [400, 400],
            "jobs_earn_high": [300, 300],
            "jobs_age_29_under": [300, 300],
            "jobs_age_30_54": [500, 500],
            "jobs_age_55_over": [200, 200],
        })
        mock_sql.return_value = emp_data

        _, summary = isochrone_employment(
            REPUBLICS_LON, REPUBLICS_LAT, 15, STATE_FIPS, COUNTY_FIPS, 2021,
        )

        # 2 tracts × 1000 jobs × 0.5 overlap = 1000 total
        assert summary.iloc[0]["total_jobs"] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# isochrone_commute_shed
# ---------------------------------------------------------------------------

class TestIsochroneCommuteShed:
    @patch("geostack.demographics.read_sql")
    @patch("geostack.demographics.tracts_within_isochrone")
    def test_returns_home_counties(self, mock_tracts, mock_sql, mock_tracts_gdf):
        """Should return DataFrame of home counties sorted by job count."""
        tracts = mock_tracts_gdf.copy()
        tracts["overlap_fraction"] = [1.0, 0.9, 0.8, 0.7, 0.5]
        tracts["minutes"] = 15
        mock_tracts.return_value = tracts

        shed_data = pd.DataFrame({
            "h_state_fips": ["48", "48", "48", "40"],
            "h_county_fips": ["113", "085", "439", "109"],
            "total_jobs": [5000, 2000, 1500, 300],
            "jobs_earn_low": [1000, 400, 300, 100],
            "jobs_earn_mid": [2000, 800, 600, 100],
            "jobs_earn_high": [2000, 800, 600, 100],
            "jobs_age_29_under": [1500, 600, 450, 90],
            "jobs_age_30_54": [2500, 1000, 750, 150],
            "jobs_age_55_over": [1000, 400, 300, 60],
        })
        mock_sql.return_value = shed_data

        result = isochrone_commute_shed(
            REPUBLICS_LON, REPUBLICS_LAT, 15, STATE_FIPS, COUNTY_FIPS, 2021,
        )

        assert not result.empty
        assert "pct_of_total" in result.columns
        assert "cumulative_pct" in result.columns
        assert len(result) == 4

        # First row should be the largest contributor
        assert result.iloc[0]["total_jobs"] >= result.iloc[1]["total_jobs"]

        # Percentages should sum to ~1.0
        assert result["pct_of_total"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# acs_time_series
# ---------------------------------------------------------------------------

class TestAcsTimeSeries:
    @patch("geostack.census.get_county_acs_summary_direct")
    def test_returns_long_format_trends(self, mock_summary):
        """Should return year × variable × value DataFrame."""
        def _mock_summary(state, county, year, engine=None):
            return pd.DataFrame([{
                "total_population": 500000 + (year - 2018) * 10000,
                "median_household_income": 55000 + (year - 2018) * 1500,
                "median_rent": 1000 + (year - 2018) * 50,
            }])

        mock_summary.side_effect = _mock_summary

        result = acs_time_series(
            STATE_FIPS, COUNTY_FIPS,
            years=[2018, 2019, 2020, 2021, 2022, 2023],
        )

        assert "year" in result.columns
        assert "variable" in result.columns
        assert "value" in result.columns

        # Should have 3 variables × 6 years = 18 rows
        assert len(result) == 18

        # CAGR should be computed
        assert "cagr" in result.attrs
        cagr_df = result.attrs["cagr"]
        assert not cagr_df.empty
        assert "total_population" in cagr_df["variable"].values

    @patch("geostack.census.get_county_acs_summary_direct")
    def test_handles_missing_years(self, mock_summary):
        """Should gracefully skip years with errors."""
        def _mock_summary(state, county, year, engine=None):
            if year == 2020:
                raise ValueError("No data for 2020")
            return pd.DataFrame([{
                "total_population": 500000 + (year - 2018) * 10000,
            }])

        mock_summary.side_effect = _mock_summary

        result = acs_time_series(
            STATE_FIPS, COUNTY_FIPS,
            years=[2018, 2019, 2020, 2021],
        )

        # 2020 should be skipped, so 3 years × 1 variable = 3 rows
        assert len(result) == 3
        assert 2020 not in result["year"].values
