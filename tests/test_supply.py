"""Tests for geostack.supply — BPS permits and CoStar CSV ingest."""

import pandas as pd
import geopandas as gpd
import pytest
from unittest.mock import patch

GARLAND_LON = -96.6183
GARLAND_LAT = 32.8028
DALLAS_FIPS = ("48", "113")

# Minimal BPS county text matching the real Census format at
# www2.census.gov/econ/bps/County/ — separate state/county FIPS columns.
# Columns: Date, State, County, Region, Division, Name, then groups of 3
# (Bldgs, Units, Value) for 1-unit, 2-unit, 3-4 unit, 5+ unit structures.
_BPS_SAMPLE = (
    ",,,,,1-unit,,,2-units,,,3-4 units,,,5+ units,,\n"
    "Date,State,County,Region,Division,Name,Bldgs,Units,Value,Bldgs,Units,Value,Bldgs,Units,Value,Bldgs,Units,Value\n"
    "2023,48,031,3,7,Collin County,8,8,900,2,4,400,0,0,0,2,30,5000\n"
    "2023,48,113,3,7,Dallas County,3,3,350,1,2,200,0,0,0,1,10,2000\n"
)


class TestParseBpsCounty:
    def test_parse_returns_dataframe(self):
        from geostack.supply import _parse_bps_county
        df = _parse_bps_county(_BPS_SAMPLE, year=2023)
        assert isinstance(df, pd.DataFrame)
        assert "county_fips" in df.columns
        assert "units_5plus" in df.columns
        assert "units_1unit" in df.columns
        assert "year" in df.columns

    def test_fips_is_five_char_zero_padded(self):
        from geostack.supply import _parse_bps_county
        df = _parse_bps_county(_BPS_SAMPLE, year=2023)
        assert df["county_fips"].str.len().eq(5).all()

    def test_skips_non_data_rows(self):
        from geostack.supply import _parse_bps_county
        df = _parse_bps_county(_BPS_SAMPLE, year=2023)
        # Only the two data rows (48031xx and 48113xx) should be parsed
        assert len(df) == 2

    def test_units_5plus_extracted_from_correct_column(self):
        from geostack.supply import _parse_bps_county
        df = _parse_bps_county(_BPS_SAMPLE, year=2023)
        # 4811300 (Dallas-ish) has 10 in units_5plus column (col index 11)
        dallas_row = df[df["county_fips"] == "48113"]
        assert not dallas_row.empty
        assert dallas_row.iloc[0]["units_5plus"] == 10

    def test_units_1unit_extracted_from_correct_column(self):
        from geostack.supply import _parse_bps_county
        df = _parse_bps_county(_BPS_SAMPLE, year=2023)
        dallas_row = df[df["county_fips"] == "48113"]
        assert not dallas_row.empty
        assert dallas_row.iloc[0]["units_1unit"] == 3  # parts[7]: 1-unit UNITS (not bldgs)

    def test_year_column_is_correct(self):
        from geostack.supply import _parse_bps_county
        df = _parse_bps_county(_BPS_SAMPLE, year=2023)
        assert (df["year"] == 2023).all()


class TestLoadCostarComps:
    def test_load_csv_returns_geodataframe(self, tmp_path):
        csv_path = tmp_path / "comps.csv"
        csv_path.write_text(
            "name,lat,lon,units,status,delivery_year,asking_rent_1br\n"
            "The Olsen,32.8150,-96.6200,240,Delivered,2023,1450\n"
            "Quinn on Thirty,32.8090,-96.6150,200,Delivered,2024,1380\n"
        )
        from geostack.supply import load_costar_comps
        gdf = load_costar_comps(str(csv_path))
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs.to_epsg() == 4326
        assert "geometry" in gdf.columns
        assert len(gdf) == 2

    def test_required_columns_present_in_output(self, tmp_path):
        csv_path = tmp_path / "comps.csv"
        csv_path.write_text(
            "name,lat,lon,units,status,delivery_year\n"
            "Test,32.80,-96.62,100,Delivered,2023\n"
        )
        from geostack.supply import load_costar_comps
        gdf = load_costar_comps(str(csv_path))
        for col in ["name", "units", "status", "delivery_year"]:
            assert col in gdf.columns

    def test_missing_required_column_raises_value_error(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("name,units\nFoo,100\n")
        from geostack.supply import load_costar_comps
        with pytest.raises(ValueError, match="lat"):
            load_costar_comps(str(csv_path))


class TestSupplyPipeline:
    @patch("geostack.supply._fetch_bps_county")
    def test_returns_expected_keys(self, mock_bps):
        mock_bps.return_value = pd.DataFrame([
            {"county_fips": "48113", "units_5plus": 250, "units_1unit": 100, "year": 2023},
        ])
        from geostack.supply import supply_pipeline
        result = supply_pipeline(
            lon=GARLAND_LON, lat=GARLAND_LAT,
            minutes=10,
            state_fips="48", county_fips="113",
            bps_year_range=(2023, 2023),
        )
        expected_keys = {"permitted_units_5plus", "permitted_units_1unit",
                         "bps_years", "bps_by_year", "delivered_properties"}
        assert expected_keys == set(result.keys())

    @patch("geostack.supply._fetch_bps_county")
    def test_sums_units_across_years(self, mock_bps):
        def _side(yr):
            return pd.DataFrame([
                {"county_fips": "48113", "units_5plus": 100 * yr % 300, "units_1unit": 50, "year": yr},
            ])
        mock_bps.side_effect = _side
        from geostack.supply import supply_pipeline
        result = supply_pipeline(
            lon=GARLAND_LON, lat=GARLAND_LAT, minutes=10,
            state_fips="48", county_fips="113",
            bps_year_range=(2022, 2023),
        )
        assert result["permitted_units_5plus"] == (100 * 2022 % 300) + (100 * 2023 % 300)
        assert result["bps_years"] == [2022, 2023]

    @patch("geostack.supply._fetch_bps_county")
    def test_with_costar_csv(self, mock_bps, tmp_path):
        mock_bps.return_value = pd.DataFrame([
            {"county_fips": "48113", "units_5plus": 150, "units_1unit": 60, "year": 2023},
        ])
        csv_path = tmp_path / "comps.csv"
        csv_path.write_text(
            "name,lat,lon,units,status,delivery_year\n"
            "The Olsen,32.815,-96.620,240,Delivered,2023\n"
        )
        from geostack.supply import supply_pipeline
        result = supply_pipeline(
            lon=GARLAND_LON, lat=GARLAND_LAT, minutes=10,
            state_fips="48", county_fips="113",
            bps_year_range=(2023, 2023),
            costar_csv=str(csv_path),
        )
        assert isinstance(result["delivered_properties"], gpd.GeoDataFrame)
        assert len(result["delivered_properties"]) == 1
        assert result["delivered_properties"].iloc[0]["name"] == "The Olsen"

    @patch("geostack.supply._fetch_bps_county")
    def test_missing_county_in_bps_returns_zero(self, mock_bps):
        """If county FIPS not in BPS data, permitted_units_5plus should be 0."""
        mock_bps.return_value = pd.DataFrame([
            {"county_fips": "99999", "units_5plus": 100, "units_1unit": 50, "year": 2023},
        ])
        from geostack.supply import supply_pipeline
        result = supply_pipeline(
            lon=GARLAND_LON, lat=GARLAND_LAT, minutes=10,
            state_fips="48", county_fips="113",
            bps_year_range=(2023, 2023),
        )
        assert result["permitted_units_5plus"] == 0
        assert result["permitted_units_1unit"] == 0
