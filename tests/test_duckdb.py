"""Tests for geostack.duckdb_analytics — DuckDB parquet queries."""

import os

import pandas as pd
import pytest

pytest.importorskip("duckdb", reason="duckdb not installed; run: pip install 'geostack[analytics]'")

from geostack.duckdb_analytics import (
    _county_prefix_filter,
    _od_path,
    _wac_path,
    commute_distance_distribution,
    employment_by_sector,
    get_connection,
    top_employment_centers,
)


# ---------------------------------------------------------------------------
# get_connection
# ---------------------------------------------------------------------------

class TestGetConnection:
    def test_returns_connection(self):
        conn = get_connection()
        assert conn is not None

    def test_can_run_simple_query(self):
        conn = get_connection()
        result = conn.execute("SELECT 42 AS answer").df()
        assert result["answer"].iloc[0] == 42
        conn.close()

    def test_connection_is_in_memory(self):
        """Two separate connections don't share state."""
        conn1 = get_connection()
        conn1.execute("CREATE TABLE t (x INT)")
        conn1.close()

        conn2 = get_connection()
        tables = conn2.execute("SHOW TABLES").df()
        assert "t" not in tables["name"].tolist()
        conn2.close()


# ---------------------------------------------------------------------------
# _county_prefix_filter
# ---------------------------------------------------------------------------

class TestCountyPrefixFilter:
    def test_produces_lpad_like_clause(self):
        clause = _county_prefix_filter("48085")
        assert "LIKE" in clause
        assert "48085%" in clause
        assert "LPAD" in clause

    def test_custom_geocode_col(self):
        clause = _county_prefix_filter("48085", geocode_col="h_geocode")
        assert "h_geocode" in clause

    def test_pads_short_fips(self):
        # "8085" should be zero-padded to "08085"
        clause = _county_prefix_filter("8085")
        assert "08085%" in clause


# ---------------------------------------------------------------------------
# employment_by_sector (with synthetic parquet)
# ---------------------------------------------------------------------------

class TestEmploymentBySector:
    @pytest.fixture(autouse=True)
    def patch_cache(self, monkeypatch, tmp_path, wac_parquet_file):
        monkeypatch.setattr("geostack.duckdb_analytics._lodes_cache_dir", lambda: str(tmp_path))

    def test_returns_dataframe(self):
        result = employment_by_sector("tx", 2021, "48085")
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        result = employment_by_sector("tx", 2021, "48085")
        assert "tract_geoid" in result.columns
        assert "total_jobs" in result.columns
        assert "jobs_retail" in result.columns
        assert "jobs_healthcare" in result.columns

    def test_tract_geoid_length(self):
        """tract_geoid should be 11 characters (state+county+tract FIPS)."""
        result = employment_by_sector("tx", 2021, "48085")
        if not result.empty:
            assert result["tract_geoid"].str.len().eq(11).all()

    def test_sorted_by_total_jobs_desc(self):
        result = employment_by_sector("tx", 2021, "48085")
        if len(result) > 1:
            assert result["total_jobs"].is_monotonic_decreasing

    def test_missing_parquet_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Parquet cache not found"):
            employment_by_sector("tx", 1990, "48085")  # no such file


# ---------------------------------------------------------------------------
# top_employment_centers (with synthetic parquet)
# ---------------------------------------------------------------------------

class TestTopEmploymentCenters:
    @pytest.fixture(autouse=True)
    def patch_cache(self, monkeypatch, tmp_path, wac_parquet_file):
        monkeypatch.setattr("geostack.duckdb_analytics._lodes_cache_dir", lambda: str(tmp_path))

    def test_returns_n_rows(self):
        result = top_employment_centers("tx", 2021, "48085", n=3)
        assert len(result) <= 3

    def test_has_block_geocode(self):
        result = top_employment_centers("tx", 2021, "48085", n=5)
        assert "w_geocode" in result.columns
        if not result.empty:
            assert result["w_geocode"].str.len().eq(15).all()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

class TestPathHelpers:
    def test_wac_path_format(self, monkeypatch, tmp_path):
        monkeypatch.setattr("geostack.duckdb_analytics._lodes_cache_dir", lambda: str(tmp_path))
        path = _wac_path("tx", 2021)
        assert path.endswith("wac_S000_JT01_2021.parquet")
        assert "tx" in path

    def test_od_path_format(self, monkeypatch, tmp_path):
        monkeypatch.setattr("geostack.duckdb_analytics._lodes_cache_dir", lambda: str(tmp_path))
        path = _od_path("tx", 2021, part="aux")
        assert path.endswith("od_aux_JT01_2021.parquet")
