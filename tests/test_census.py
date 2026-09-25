"""Tests for geostack.census — ACS variable dict, CensusClient, GEOID building."""

import pandas as pd
import pytest

import geopandas as gpd
from unittest.mock import MagicMock, patch

from geostack.census import (
    ACS_VARIABLES,
    CensusClient,
    _AVG_VARIABLES,
    _SUM_VARIABLES,
    _pivot_select_clauses,
    _validate_fips,
    get_county_acs_summary,
    get_tract_demographics,
)


# ---------------------------------------------------------------------------
# ACS_VARIABLES dict
# ---------------------------------------------------------------------------

class TestAcsVariables:
    def test_required_cre_keys_present(self):
        required = [
            "total_population",
            "median_household_income",
            "median_rent",
            "median_home_value",
            "total_housing_units",
            "renter_occupied",
            "owner_occupied",
        ]
        for key in required:
            assert key in ACS_VARIABLES, f"Missing ACS variable: {key}"

    def test_values_are_census_codes(self):
        """All values should look like Census variable codes (e.g. B01003_001E)."""
        for name, code in ACS_VARIABLES.items():
            assert isinstance(code, str), f"{name}: expected str code"
            assert code.endswith("E"), f"{name}: expected estimate code ending in E, got {code}"
            assert "_" in code, f"{name}: expected underscore in code, got {code}"

    def test_no_duplicate_codes(self):
        codes = list(ACS_VARIABLES.values())
        assert len(codes) == len(set(codes)), "Duplicate Census codes in ACS_VARIABLES"

    def test_multifamily_buckets_present(self):
        """Units-in-structure buckets are needed for multifamily CRE analysis."""
        mf_keys = [k for k in ACS_VARIABLES if k.startswith("units_in_structure")]
        assert len(mf_keys) >= 4, "Expected at least 4 units-in-structure buckets"

    def test_has_at_least_15_variables(self):
        assert len(ACS_VARIABLES) >= 15


# ---------------------------------------------------------------------------
# CensusClient.build_geoid
# ---------------------------------------------------------------------------

class TestBuildGeoid:
    def _tract_df(self):
        return pd.DataFrame({
            "state": ["48", "48"],
            "county": ["085", "113"],
            "tract": ["010100", "020200"],
            "NAME": ["Tract A", "Tract B"],
        })

    def _bg_df(self):
        return pd.DataFrame({
            "state": ["48"],
            "county": ["085"],
            "tract": ["010100"],
            "block group": ["1"],
            "NAME": ["BG 1"],
        })

    def _county_df(self):
        return pd.DataFrame({
            "state": ["48", "06"],
            "county": ["085", "037"],
            "NAME": ["Collin County", "Los Angeles County"],
        })

    def test_tract_geoid_length(self):
        client = CensusClient()
        df = client.build_geoid(self._tract_df(), geography="tract")
        assert "GEOID" in df.columns
        assert (df["GEOID"].str.len() == 11).all()

    def test_tract_geoid_values(self):
        client = CensusClient()
        df = client.build_geoid(self._tract_df(), geography="tract")
        assert df["GEOID"].iloc[0] == "48085010100"
        assert df["GEOID"].iloc[1] == "48113020200"

    def test_block_group_geoid_length(self):
        client = CensusClient()
        df = client.build_geoid(self._bg_df(), geography="block group")
        assert (df["GEOID"].str.len() == 12).all()

    def test_block_group_geoid_value(self):
        client = CensusClient()
        df = client.build_geoid(self._bg_df(), geography="block group")
        assert df["GEOID"].iloc[0] == "480850101001"

    def test_county_geoid_length(self):
        client = CensusClient()
        df = client.build_geoid(self._county_df(), geography="county")
        assert (df["GEOID"].str.len() == 5).all()

    def test_county_geoid_values(self):
        client = CensusClient()
        df = client.build_geoid(self._county_df(), geography="county")
        assert df["GEOID"].iloc[0] == "48085"
        assert df["GEOID"].iloc[1] == "06037"


# ---------------------------------------------------------------------------
# CensusClient init
# ---------------------------------------------------------------------------

class TestCensusClientInit:
    def test_accepts_explicit_key(self):
        client = CensusClient(api_key="test-key-123")
        assert client.api_key == "test-key-123"

    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("CENSUS_API_KEY", "env-key-abc")
        client = CensusClient()
        assert client.api_key == "env-key-abc"

    def test_no_key_sets_none(self, monkeypatch):
        monkeypatch.delenv("CENSUS_API_KEY", raising=False)
        client = CensusClient()
        assert client.api_key is None


# ---------------------------------------------------------------------------
# Integration tests (require Census API access)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _pivot_select_clauses / variable sets
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _validate_fips
# ---------------------------------------------------------------------------

class TestValidateFips:
    def test_accepts_valid_state_fips(self):
        for val in ("48", "06", "01", "36"):
            _validate_fips(val, "state_fips")  # should not raise

    def test_accepts_valid_county_fips(self):
        for val in ("085", "001", "113", "453"):
            _validate_fips(val, "county_fips")  # should not raise

    def test_rejects_alpha_string(self):
        with pytest.raises(ValueError, match="state_fips"):
            _validate_fips("texas", "state_fips")

    def test_rejects_sql_injection_attempt(self):
        with pytest.raises(ValueError):
            _validate_fips("48'; DROP TABLE census_tracts; --", "state_fips")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            _validate_fips("", "county_fips")

    def test_rejects_float_string(self):
        with pytest.raises(ValueError):
            _validate_fips("48.5", "state_fips")

    def test_rejects_mixed_alphanum(self):
        with pytest.raises(ValueError):
            _validate_fips("48abc", "state_fips")

    def test_error_message_contains_param_name(self):
        with pytest.raises(ValueError, match="county_fips"):
            _validate_fips("bad", "county_fips")


# ---------------------------------------------------------------------------
# _pivot_select_clauses / variable sets
# ---------------------------------------------------------------------------

class TestPivotHelpers:
    def test_pivot_clauses_count(self):
        """One clause per ACS variable."""
        clauses = _pivot_select_clauses()
        # Count AS aliases
        assert clauses.count(" AS ") == len(ACS_VARIABLES)

    def test_pivot_clauses_use_filter_syntax(self):
        clauses = _pivot_select_clauses()
        assert "FILTER (WHERE" in clauses

    def test_sum_and_avg_cover_all_variables(self):
        """Every ACS variable belongs to exactly one aggregation set."""
        all_vars = set(ACS_VARIABLES.keys())
        assert _SUM_VARIABLES | _AVG_VARIABLES == all_vars
        assert _SUM_VARIABLES & _AVG_VARIABLES == set()

    def test_median_vars_in_avg(self):
        for name in ("median_household_income", "median_rent", "median_home_value"):
            assert name in _AVG_VARIABLES

    def test_count_vars_in_sum(self):
        for name in ("total_population", "total_housing_units", "renter_occupied"):
            assert name in _SUM_VARIABLES

    def test_total_population_maps_to_correct_code(self):
        """Spot-check: total_population → B01003_001E in pivot SQL."""
        clauses = _pivot_select_clauses()
        assert "B01003_001E" in clauses
        assert "AS total_population" in clauses

    def test_median_household_income_maps_to_correct_code(self):
        clauses = _pivot_select_clauses()
        assert "B19013_001E" in clauses
        assert "AS median_household_income" in clauses

    def test_pivot_excludes_sentinel_values(self):
        """Each clause should exclude the Census sentinel -666666666."""
        clauses = _pivot_select_clauses()
        assert "-666666666" in clauses

    def test_all_acs_codes_present_in_pivot(self):
        clauses = _pivot_select_clauses()
        for name, code in ACS_VARIABLES.items():
            assert code in clauses, f"Census code {code} ({name}) missing from pivot SQL"
            assert f"AS {name}" in clauses, f"Alias '{name}' missing from pivot SQL"


# ---------------------------------------------------------------------------
# get_tract_demographics (unit — mocked engine)
# ---------------------------------------------------------------------------

class TestGetTractDemographics:
    def _make_mock_engine(self, gdf):
        """Return a mock SQLAlchemy engine whose connect() yields gdf via read_postgis."""
        engine = MagicMock()
        conn_ctx = MagicMock()
        conn_ctx.__enter__ = MagicMock(return_value=MagicMock())
        conn_ctx.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = conn_ctx
        return engine

    def _synthetic_gdf(self):
        from shapely.geometry import box
        geoms = [box(-96.8, 33.0, -96.7, 33.1), box(-96.7, 33.0, -96.6, 33.1)]
        return gpd.GeoDataFrame(
            {
                "geoid": ["48085010100", "48085010200"],
                "total_population": [4500.0, 3200.0],
                "median_household_income": [82000.0, 71000.0],
                **{k: [None, None] for k in ACS_VARIABLES if k not in ("total_population", "median_household_income")},
            },
            geometry=geoms,
            crs="EPSG:4326",
        )

    def test_returns_geodataframe(self):
        gdf = self._synthetic_gdf()
        engine = self._make_mock_engine(gdf)
        with patch("geopandas.read_postgis", return_value=gdf):
            result = get_tract_demographics("48", "085", 2023, engine=engine)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_geoid_is_index(self):
        gdf = self._synthetic_gdf()
        engine = self._make_mock_engine(gdf)
        with patch("geopandas.read_postgis", return_value=gdf):
            result = get_tract_demographics("48", "085", 2023, engine=engine)
        assert result.index.name == "geoid"

    def test_has_acs_columns(self):
        gdf = self._synthetic_gdf()
        engine = self._make_mock_engine(gdf)
        with patch("geopandas.read_postgis", return_value=gdf):
            result = get_tract_demographics("48", "085", 2023, engine=engine)
        for col in ("total_population", "median_household_income"):
            assert col in result.columns

    def test_geometry_column_present(self):
        gdf = self._synthetic_gdf()
        engine = self._make_mock_engine(gdf)
        with patch("geopandas.read_postgis", return_value=gdf):
            result = get_tract_demographics("48", "085", 2023, engine=engine)
        assert result.geometry is not None

    def test_missing_acs_columns_become_nan(self):
        """Columns present in GDF as None/NaN are coerced to NaN floats, not dropped."""
        gdf = self._synthetic_gdf()
        # _synthetic_gdf sets all cols except total_population and median_household_income to None
        engine = self._make_mock_engine(gdf)
        with patch("geopandas.read_postgis", return_value=gdf):
            result = get_tract_demographics("48", "085", 2023, engine=engine)
        # All ACS columns should be present
        for col in ACS_VARIABLES:
            assert col in result.columns, f"Column {col!r} missing from result"
        # Columns set to None should be NaN, not strings or objects
        import math
        for col in ACS_VARIABLES:
            if col not in ("total_population", "median_household_income"):
                for val in result[col]:
                    assert val is None or (isinstance(val, float) and math.isnan(val)), (
                        f"Expected NaN for {col}, got {val!r}"
                    )

    def test_raises_on_invalid_state_fips(self):
        engine = MagicMock()
        with pytest.raises(ValueError, match="state_fips"):
            get_tract_demographics("TX", "085", 2023, engine=engine)

    def test_raises_on_invalid_county_fips(self):
        engine = MagicMock()
        with pytest.raises(ValueError, match="county_fips"):
            get_tract_demographics("48", "collin", 2023, engine=engine)

    def test_validates_before_db_call(self):
        """ValueError is raised before any DB connection is attempted."""
        engine = MagicMock()
        with pytest.raises(ValueError):
            get_tract_demographics("bad!", "085", 2023, engine=engine)
        engine.connect.assert_not_called()


# ---------------------------------------------------------------------------
# get_county_acs_summary (unit — mocked engine)
# ---------------------------------------------------------------------------

class TestGetCountyAcsSummary:
    def _make_engine_with_df(self, df):
        engine = MagicMock()
        return engine

    def test_returns_dataframe(self):
        df = pd.DataFrame({name: [1.0] for name in ACS_VARIABLES})
        engine = MagicMock()
        with patch("geostack.db.read_sql", return_value=df):
            result = get_county_acs_summary("48", "085", 2023, engine=engine)
        assert isinstance(result, pd.DataFrame)

    def test_has_one_row(self):
        df = pd.DataFrame({name: [1.0] for name in ACS_VARIABLES})
        engine = MagicMock()
        with patch("geostack.db.read_sql", return_value=df):
            result = get_county_acs_summary("48", "085", 2023, engine=engine)
        assert len(result) == 1

    def test_has_all_acs_columns(self):
        df = pd.DataFrame({name: [1.0] for name in ACS_VARIABLES})
        engine = MagicMock()
        with patch("geostack.db.read_sql", return_value=df):
            result = get_county_acs_summary("48", "085", 2023, engine=engine)
        for col in ACS_VARIABLES:
            assert col in result.columns

    def test_raises_on_invalid_state_fips(self):
        engine = MagicMock()
        with pytest.raises(ValueError, match="state_fips"):
            get_county_acs_summary("texas", "085", 2023, engine=engine)

    def test_raises_on_invalid_county_fips(self):
        engine = MagicMock()
        with pytest.raises(ValueError, match="county_fips"):
            get_county_acs_summary("48", "collin", 2023, engine=engine)


# ---------------------------------------------------------------------------
# Integration tests (require Census API access)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_get_acs_collin_county():
    """Fetches real ACS data for a small county. Requires internet."""
    client = CensusClient()
    df = client.get_acs(
        variables=[ACS_VARIABLES["total_population"]],
        geography="tract",
        state="48",
        county="085",
        year=2022,
    )
    assert len(df) > 0
    assert ACS_VARIABLES["total_population"] in df.columns
