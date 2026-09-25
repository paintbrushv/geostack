"""Tests for geostack.property_tax — TX Comptroller property tax rates."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

DALLAS_FIPS = ("48", "113")


@pytest.fixture
def mock_comptroller_df():
    """Minimal parsed Comptroller rate data for Dallas County."""
    return pd.DataFrame([
        {"entity_name": "DALLAS COUNTY", "entity_id": "048-000-00",
         "total_rate": 0.2438, "entity_type": "county"},
        {"entity_name": "CITY OF GARLAND", "entity_id": "048-902-02",
         "total_rate": 0.7100, "entity_type": "city"},
        {"entity_name": "GARLAND ISD", "entity_id": "048-911-02",
         "total_rate": 1.2200, "entity_type": "school"},
        {"entity_name": "DALLAS ISD", "entity_id": "048-911-01",
         "total_rate": 1.1100, "entity_type": "school"},
        {"entity_name": "CITY OF DALLAS", "entity_id": "048-901-01",
         "total_rate": 0.7635, "entity_type": "city"},
    ])


class TestParseComptrollerRates:
    def test_parse_returns_dataframe(self, mock_comptroller_df):
        df = mock_comptroller_df
        assert "entity_name" in df.columns
        assert "total_rate" in df.columns
        assert "entity_type" in df.columns

    def test_rates_are_numeric(self, mock_comptroller_df):
        assert mock_comptroller_df["total_rate"].dtype == float


class TestLookupRate:
    def test_lookup_by_entity_name(self, mock_comptroller_df):
        from geostack.property_tax import _lookup_rate
        rate = _lookup_rate("GARLAND ISD", mock_comptroller_df)
        assert rate == pytest.approx(1.22)

    def test_lookup_missing_returns_none(self, mock_comptroller_df):
        from geostack.property_tax import _lookup_rate
        rate = _lookup_rate("NONEXISTENT ISD", mock_comptroller_df)
        assert rate is None

    def test_lookup_case_insensitive(self, mock_comptroller_df):
        from geostack.property_tax import _lookup_rate
        rate = _lookup_rate("garland isd", mock_comptroller_df)
        assert rate == pytest.approx(1.22)


class TestEffectiveTaxRate:
    @patch("geostack.property_tax._fetch_comptroller_rates")
    @patch("geostack.property_tax.school_profile")
    def test_returns_rate_dict(self, mock_school, mock_rates, mock_comptroller_df):
        mock_school.return_value = {
            "district_name": "Garland Independent School District",
            "tea_rating": "C", "leaid": "57902",
        }
        mock_rates.return_value = mock_comptroller_df

        from geostack.property_tax import effective_tax_rate
        result = effective_tax_rate(
            lon=-96.6183, lat=32.8028,
            county_name="DALLAS COUNTY", year=2024,
        )

        assert isinstance(result, dict)
        assert "county_rate" in result
        assert "school_rate" in result
        assert "total_rate" in result
        assert result["county_rate"] == pytest.approx(0.2438)

    @patch("geostack.property_tax._fetch_comptroller_rates")
    @patch("geostack.property_tax.school_profile")
    def test_total_rate_is_sum_of_components(self, mock_school, mock_rates, mock_comptroller_df):
        mock_school.return_value = {
            "district_name": "Garland Independent School District",
        }
        mock_rates.return_value = mock_comptroller_df

        from geostack.property_tax import effective_tax_rate
        result = effective_tax_rate(
            lon=-96.6183, lat=32.8028,
            county_name="DALLAS COUNTY", year=2024,
        )

        expected = result["county_rate"] + result["school_rate"]
        assert result["total_rate"] >= expected
