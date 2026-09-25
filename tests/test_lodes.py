"""Tests for geostack.lodes — LODES8 download and processing."""

import io
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from geostack.lodes import (
    FIPS_TO_ABBR,
    NAICS_SECTORS,
    _resolve_state_abbr,
    download_wac,
)


# ---------------------------------------------------------------------------
# _resolve_state_abbr
# ---------------------------------------------------------------------------

class TestResolveStateAbbr:
    def test_fips_to_abbr(self):
        assert _resolve_state_abbr("48") == "tx"
        assert _resolve_state_abbr("06") == "ca"
        assert _resolve_state_abbr("36") == "ny"

    def test_fips_with_int(self):
        # Should work when given a string that looks like a FIPS
        assert _resolve_state_abbr("01") == "al"

    def test_abbr_passthrough(self):
        assert _resolve_state_abbr("tx") == "tx"
        assert _resolve_state_abbr("ca") == "ca"
        assert _resolve_state_abbr("TX") == "tx"  # normalised to lower

    def test_unknown_fips_raises(self):
        with pytest.raises(ValueError, match="Unknown state FIPS"):
            _resolve_state_abbr("99")

    def test_all_50_states_present(self):
        # 51 entries including DC
        assert len(FIPS_TO_ABBR) == 51
        assert "48" in FIPS_TO_ABBR   # Texas
        assert "11" in FIPS_TO_ABBR   # DC


# ---------------------------------------------------------------------------
# NAICS_SECTORS mapping
# ---------------------------------------------------------------------------

class TestNaicsSectors:
    def test_has_20_sectors(self):
        assert len(NAICS_SECTORS) == 20

    def test_cre_relevant_sectors_present(self):
        values = list(NAICS_SECTORS.values())
        assert "retail" in values
        assert "healthcare" in values
        assert "real_estate" in values
        assert "finance_insurance" in values
        assert "professional_scientific" in values

    def test_keys_are_cns_format(self):
        for k in NAICS_SECTORS:
            assert k.startswith("CNS"), f"Expected CNS prefix, got {k}"
            assert len(k) == 5


# ---------------------------------------------------------------------------
# download_wac with mocked HTTP
# ---------------------------------------------------------------------------

def _make_wac_response(wac_csv_bytes):
    """Build a mock requests.Response returning WAC CSV bytes."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-length": str(len(wac_csv_bytes))}
    mock_resp.iter_content = lambda chunk_size=None: iter([wac_csv_bytes])
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


class TestDownloadWac:
    def test_column_renaming(self, wac_csv_bytes, tmp_path):
        """download_wac renames C000→total_jobs and CNS07→jobs_retail."""
        with patch("requests.get", return_value=_make_wac_response(wac_csv_bytes)):
            df = download_wac("tx", year=2021, cache=False)

        assert "total_jobs" in df.columns
        assert "jobs_retail" in df.columns
        assert "jobs_healthcare" in df.columns
        assert "C000" not in df.columns
        assert "CNS07" not in df.columns

    def test_geographic_columns_added(self, wac_csv_bytes):
        """state_fips, county_fips, and tract_geoid are derived from w_geocode."""
        with patch("requests.get", return_value=_make_wac_response(wac_csv_bytes)):
            df = download_wac("tx", year=2021, cache=False)

        assert "state_fips" in df.columns
        assert "county_fips" in df.columns
        assert "tract_geoid" in df.columns
        assert (df["state_fips"] == "48").all()
        assert (df["county_fips"] == "085").all()
        assert df["tract_geoid"].str.len().eq(11).all()

    def test_county_filter(self, wac_csv_bytes):
        """county_filter keeps only rows matching the 5-digit FIPS prefix."""
        with patch("requests.get", return_value=_make_wac_response(wac_csv_bytes)):
            df_all = download_wac("tx", year=2021, cache=False)
            df_filtered = download_wac("tx", year=2021, county_filter="48085", cache=False)

        # All rows in wac_csv_bytes fixture are in 48085
        assert len(df_filtered) == len(df_all)
        assert (df_filtered["county_fips"] == "085").all()

    def test_wrong_county_filter_returns_empty(self, wac_csv_bytes):
        """Filtering to a county with no rows returns empty DataFrame."""
        with patch("requests.get", return_value=_make_wac_response(wac_csv_bytes)):
            df = download_wac("tx", year=2021, county_filter="48113", cache=False)

        assert df.empty

    def test_geocode_zero_padded(self, wac_csv_bytes):
        """w_geocode is always 15 characters after download."""
        with patch("requests.get", return_value=_make_wac_response(wac_csv_bytes)):
            df = download_wac("tx", year=2021, cache=False)

        assert df["w_geocode"].str.len().eq(15).all()

    def test_file_not_found_raises(self):
        """404 from Census server raises FileNotFoundError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(FileNotFoundError):
                download_wac("tx", year=1990, cache=False)  # invalid year

    def test_cache_writes_parquet(self, wac_csv_bytes, tmp_path):
        """When cache=True, a parquet file is written to the cache dir."""
        cache_dir = str(tmp_path / "lodes")
        os.makedirs(os.path.join(cache_dir, "tx"), exist_ok=True)

        with patch("requests.get", return_value=_make_wac_response(wac_csv_bytes)), \
             patch("geostack.lodes._lodes_cache_dir", return_value=cache_dir):
            download_wac("tx", year=2021, cache=True)

        expected = os.path.join(cache_dir, "tx", "wac_S000_JT01_2021.parquet")
        assert os.path.exists(expected)


# ---------------------------------------------------------------------------
# Integration tests (require live DB + Census API)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_download_wac_real_network():
    """Downloads a real WAC file from LEHD. Requires internet + ~50MB download."""
    df = download_wac("wy", year=2021, county_filter="56001", cache=False)
    assert len(df) > 0
    assert "total_jobs" in df.columns
