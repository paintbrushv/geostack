"""Tests for geostack CLI commands.

Uses Click's CliRunner for isolated invocation — no real DB or network calls
except where explicitly marked @pytest.mark.integration.

Patching note: all DB imports happen inside CLI function bodies
(e.g., `from geostack.db import get_engine`), so we always patch at the
source module (geostack.db.*) rather than geostack.cli.*.
"""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from click.testing import CliRunner
from shapely.geometry import box

from geostack.cli import _human_size, cli


# ---------------------------------------------------------------------------
# _human_size helper
# ---------------------------------------------------------------------------

class TestHumanSize:
    def test_bytes(self):
        assert _human_size(512) == "512.0 B"

    def test_kilobytes(self):
        assert _human_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _human_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert _human_size(3 * 1024 ** 3) == "3.0 GB"

    def test_terabytes(self):
        assert _human_size(2 * 1024 ** 4) == "2.0 TB"


# ---------------------------------------------------------------------------
# db-status
# ---------------------------------------------------------------------------

class TestDbStatus:
    def test_shows_error_when_no_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        # Also clear the engine cache so it re-reads env
        import geostack.db as _db
        _db._engine_cache.clear()
        runner = CliRunner()
        result = runner.invoke(cli, ["db-status"])
        assert result.exit_code == 0
        assert "Connection failed" in result.output or "DATABASE_URL" in result.output

    def test_succeeds_with_mocked_db(self):
        postgis_df = pd.DataFrame({"v": ["POSTGIS=3.6.1"]})
        geo_df = pd.DataFrame({
            "f_table_name": ["census_tracts"],
            "f_geometry_column": ["geometry"],
            "srid": [4326],
            "type": ["MULTIPOLYGON"],
        })

        def mock_read_sql(sql, *args, **kwargs):
            if "PostGIS_Full_Version" in sql:
                return postgis_df
            return geo_df

        runner = CliRunner()
        with patch("geostack.db.get_engine", return_value=MagicMock()), \
             patch("geostack.db.read_sql", side_effect=mock_read_sql):
            result = runner.invoke(cli, ["db-status"])
        assert result.exit_code == 0
        assert "Connected to PostGIS" in result.output


# ---------------------------------------------------------------------------
# load-acs
# ---------------------------------------------------------------------------

class TestLoadAcs:
    def test_missing_required_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["load-acs"])
        assert result.exit_code != 0
        assert "Missing option" in result.output

    def test_aborts_when_no_tracts_loaded(self):
        """Should print a helpful error if census_tracts is empty for the county."""
        empty_count = pd.DataFrame({"n": [0]})

        runner = CliRunner()
        with patch("geostack.db.get_engine", return_value=MagicMock()), \
             patch("geostack.db.read_sql", return_value=empty_count):
            result = runner.invoke(cli, ["load-acs", "--state", "48", "--county", "085", "--year", "2023"])
        assert result.exit_code == 0
        assert "load-county" in result.output

    def test_skips_when_already_loaded(self):
        """Should skip gracefully when all tracts are already loaded for the year."""
        tract_count = pd.DataFrame({"n": [5]})
        already_loaded = pd.DataFrame({"geoid": [f"4808501{i:04d}" for i in range(5)]})

        responses = [tract_count, already_loaded]

        runner = CliRunner()
        with patch("geostack.db.get_engine", return_value=MagicMock()), \
             patch("geostack.db.read_sql", side_effect=responses):
            result = runner.invoke(cli, ["load-acs", "--state", "48", "--county", "085", "--year", "2023"])
        assert result.exit_code == 0
        assert "already" in result.output.lower() or "skipping" in result.output.lower()


# ---------------------------------------------------------------------------
# load-lodes
# ---------------------------------------------------------------------------

class TestLoadLodes:
    def test_missing_required_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["load-lodes"])
        assert result.exit_code != 0
        assert "Missing option" in result.output

    def test_invokes_download_and_load(self):
        fake_df = pd.DataFrame({"w_geocode": ["480850001"], "C000": [100]})
        summary_df = pd.DataFrame({"jobs": [1000], "workers": [900]})

        runner = CliRunner()
        with patch("geostack.db.get_engine", return_value=MagicMock()), \
             patch("geostack.lodes.download_wac", return_value=fake_df), \
             patch("geostack.lodes.download_rac", return_value=fake_df), \
             patch("geostack.lodes.load_wac_to_postgis"), \
             patch("geostack.lodes.load_rac_to_postgis"), \
             patch("geostack.db.read_sql", return_value=summary_df):
            result = runner.invoke(cli, [
                "load-lodes", "--state", "48", "--county", "085", "--year", "2021"
            ])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

class TestClean:
    def test_no_flags_shows_status_without_db(self):
        """With no flags and DB unavailable, clean shows status and exits cleanly."""
        runner = CliRunner()
        # Let read_sql raise so it falls through to the yellow warning
        with patch("geostack.db.read_sql", side_effect=Exception("no db")):
            result = runner.invoke(cli, ["clean"])
        assert result.exit_code == 0
        # Should mention how to add flags
        assert "--files" in result.output or "Add" in result.output

    def test_requires_confirmation_without_yes(self):
        """--files without --yes should prompt; answering 'n' cancels."""
        empty_db = pd.DataFrame()
        count_df = pd.DataFrame({"n": [0]})

        runner = CliRunner()
        with patch("geostack.db.read_sql", side_effect=Exception("no db")):
            result = runner.invoke(cli, ["clean", "--files"], input="n\n")
        assert "Cancelled" in result.output or result.exit_code == 0

    def test_yes_flag_skips_confirmation(self):
        """--yes with --all-roads bypasses the confirmation prompt."""
        runner = CliRunner()
        with patch("geostack.db.read_sql", side_effect=Exception("no db")), \
             patch("geostack.db.execute_sql"):
            result = runner.invoke(cli, ["clean", "--all-roads", "--yes"])
        assert result.exit_code == 0
        assert "Cancelled" not in result.output


# ---------------------------------------------------------------------------
# load-county
# ---------------------------------------------------------------------------

class TestLoadCounty:
    def test_missing_required_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["load-county"])
        assert result.exit_code != 0
        assert "Missing option" in result.output

    def test_invokes_tiger_download_and_load(self):
        fake_gdf = gpd.GeoDataFrame(
            {"geoid": ["48085010100"], "state_fips": ["48"], "county_fips": ["085"]},
            geometry=[box(-96.8, 33.0, -96.7, 33.1)],
            crs="EPSG:4326",
        )
        counts_df = pd.DataFrame({
            "total_tracts": [254], "county_tracts": [85], "county_roads": [1200]
        })

        runner = CliRunner()
        with patch("geostack.db.get_engine", return_value=MagicMock()), \
             patch("geostack.tiger.download_tracts", return_value=fake_gdf), \
             patch("geostack.tiger.download_roads", return_value=fake_gdf), \
             patch("geostack.tiger.load_tracts_to_postgis"), \
             patch("geostack.tiger.load_roads_to_postgis"), \
             patch("geostack.db.read_sql", return_value=counts_df):
            result = runner.invoke(cli, [
                "load-county", "--state", "48", "--county", "085"
            ])
        assert result.exit_code == 0



# ---------------------------------------------------------------------------
# load-parcels
# ---------------------------------------------------------------------------

class TestLoadParcels:
    def test_denton_with_geometry_joins_txgio_geometry_by_geo_id(self):
        """Denton geometry should use the county TxGIO seam and join on geo_id."""
        normalized = pd.DataFrame({
            "parcel_id": ["DENTON-LOCAL-ID"],
            "geo_id": ["TXGIO-GEO-ID"],
            "ilr": [0.42],
        })
        txgio_geometry = gpd.GeoDataFrame(
            {"geo_id": ["TXGIO-GEO-ID"]},
            geometry=[box(-97.25, 33.10, -97.24, 33.11)],
            crs="EPSG:4326",
        )

        runner = CliRunner()
        with patch("geostack.db.get_engine", return_value=MagicMock()), \
             patch("geostack.cad.fetch_parcels", return_value=pd.DataFrame({"raw": ["row"]})), \
             patch("geostack.cad.normalize_parcels", return_value=normalized), \
             patch("geostack.cad.fetch_denton_parcel_geometry", return_value=txgio_geometry, create=True) as fetch_geometry, \
             patch("geostack.cad._arcgis_geojson_fetch", side_effect=AssertionError("should use Denton geometry fetch")) as arcgis_fetch, \
             patch("geostack.cad.load_parcels_to_postgis", return_value=1) as load_postgis:
            result = runner.invoke(cli, [
                "load-parcels",
                "--county", "denton",
                "--year", "2025",
                "--with-geometry",
            ])

        assert result.exit_code == 0
        arcgis_fetch.assert_not_called()
        fetch_geometry.assert_called_once()
        loaded = load_postgis.call_args.args[0]
        assert isinstance(loaded, gpd.GeoDataFrame)
        assert loaded.loc[0, "geo_id"] == "TXGIO-GEO-ID"
        assert loaded.geometry.notna().all()

# ---------------------------------------------------------------------------
# download-tracts / download-roads (legacy commands)
# ---------------------------------------------------------------------------

class TestDownloadCommands:
    def test_download_tracts_missing_state(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["download-tracts"])
        assert result.exit_code != 0

    def test_download_roads_missing_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["download-roads"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Integration: real DB smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_db_status_real():
    """Requires a live PostGIS connection via DATABASE_URL."""
    runner = CliRunner()
    result = runner.invoke(cli, ["db-status"])
    assert result.exit_code == 0
    assert "Connected" in result.output


@pytest.mark.integration
def test_clean_status_real():
    """Shows real cache/DB usage. Read-only — no data deleted."""
    runner = CliRunner()
    result = runner.invoke(cli, ["clean"])
    assert result.exit_code == 0
