"""Tests for geostack.migration — IRS SOI + ACS migration flows."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

DALLAS_STATE = "48"
DALLAS_COUNTY = "113"

_SOI_INFLOW_SAMPLE = (
    "y2_statefips,y2_countyfips,y1_statefips,y1_countyfips,"
    "y1_state,y1_countyname,n1,n2,agi\n"
    "48,113,48,085,TX,Collin County,2500,4800,125000\n"
    "48,113,48,439,TX,Tarrant County,1800,3200,82000\n"
    "48,113,06,037,CA,Los Angeles County,450,700,38000\n"
    "48,113,48,113,TX,Dallas County,0,0,0\n"
    "48,113,96,000,TX,Total Migration-US and Foreign,15000,28000,750000\n"
)

_SOI_OUTFLOW_SAMPLE = (
    "y1_statefips,y1_countyfips,y2_statefips,y2_countyfips,"
    "y2_state,y2_countyname,n1,n2,agi\n"
    "48,113,48,085,TX,Collin County,3200,6100,190000\n"
    "48,113,48,439,TX,Tarrant County,2100,3800,95000\n"
    "48,113,06,037,CA,Los Angeles County,350,550,28000\n"
    "48,113,48,113,TX,Dallas County,0,0,0\n"
    "48,113,96,000,TX,Total Migration-US and Foreign,14000,26000,700000\n"
)


class TestParseSoiInflow:
    def test_parse_returns_dataframe(self):
        from geostack.migration import _parse_soi_csv
        df = _parse_soi_csv(_SOI_INFLOW_SAMPLE, direction="inflow")
        assert isinstance(df, pd.DataFrame)
        assert "origin_fips" in df.columns
        assert "returns" in df.columns
        assert "agi" in df.columns

    def test_excludes_summary_and_same_county_rows(self):
        from geostack.migration import _parse_soi_csv
        df = _parse_soi_csv(_SOI_INFLOW_SAMPLE, direction="inflow")
        assert len(df) == 3

    def test_computes_avg_agi_per_return(self):
        from geostack.migration import _parse_soi_csv
        df = _parse_soi_csv(_SOI_INFLOW_SAMPLE, direction="inflow")
        collin = df[df["origin_fips"] == "48085"]
        assert not collin.empty
        assert collin.iloc[0]["avg_agi_per_return"] == pytest.approx(50.0)

    def test_fips_is_five_char_string(self):
        from geostack.migration import _parse_soi_csv
        df = _parse_soi_csv(_SOI_INFLOW_SAMPLE, direction="inflow")
        assert df["origin_fips"].str.len().eq(5).all()


class TestParseSoiOutflow:
    def test_outflow_has_dest_fips(self):
        from geostack.migration import _parse_soi_csv
        df = _parse_soi_csv(_SOI_OUTFLOW_SAMPLE, direction="outflow")
        assert "dest_fips" in df.columns
        assert len(df) == 3

    def test_outflow_excludes_totals(self):
        from geostack.migration import _parse_soi_csv
        df = _parse_soi_csv(_SOI_OUTFLOW_SAMPLE, direction="outflow")
        assert not df["dest_fips"].str.startswith("96").any()


class TestAcsMobilityProfile:
    @patch("geostack.migration.CensusClient")
    def test_returns_mobility_dict(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_acs.return_value = pd.DataFrame([{
            "B07010_001E": "50000",
            "B07010_002E": "35000",
            "B07010_017E": "8000",
            "B07010_033E": "5000",
            "B07010_049E": "2000",
            "B07011_001E": "55000",
            "B07011_017E": "42000",
            "B07011_033E": "61000",
            "B07011_049E": "68000",
        }])

        from geostack.migration import acs_mobility_profile
        result = acs_mobility_profile("48", "113", year=2023)

        assert isinstance(result, dict)
        assert "same_house_pct" in result
        assert "in_county_movers" in result
        assert "diff_county_movers" in result
        assert "diff_state_movers" in result
        assert "mover_median_income" in result

    @patch("geostack.migration.CensusClient")
    def test_mover_percentages_sum_to_one(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_acs.return_value = pd.DataFrame([{
            "B07010_001E": "100000",
            "B07010_002E": "80000",
            "B07010_017E": "10000",
            "B07010_033E": "7000",
            "B07010_049E": "3000",
            "B07011_001E": "55000",
            "B07011_017E": "42000",
            "B07011_033E": "61000",
            "B07011_049E": "68000",
        }])

        from geostack.migration import acs_mobility_profile
        result = acs_mobility_profile("48", "113", year=2023)

        total = (result["same_house_pct"] + result["in_county_movers_pct"]
                 + result["diff_county_movers_pct"] + result["diff_state_movers_pct"])
        assert total == pytest.approx(1.0)


class TestMigrationProfile:
    @patch("geostack.migration.acs_mobility_profile")
    @patch("geostack.migration.soi_migration")
    def test_combined_profile_returns_both(self, mock_soi, mock_acs):
        mock_soi.return_value = {
            "inflow": pd.DataFrame(), "outflow": pd.DataFrame(),
            "net_returns": 1000, "in_avg_agi": 55.0,
            "out_avg_agi": 50.0, "income_delta": 5.0, "year": 2022,
        }
        mock_acs.return_value = {
            "same_house_pct": 0.7, "in_county_movers": 10000,
            "in_county_movers_pct": 0.1,
            "diff_county_movers": 5000, "diff_county_movers_pct": 0.05,
            "diff_state_movers": 2000, "diff_state_movers_pct": 0.02,
            "mover_median_income": 62000, "total_pop": 100000,
        }

        from geostack.migration import migration_profile
        result = migration_profile("48", "113")

        assert "soi" in result
        assert "acs" in result
        assert result["soi"]["income_delta"] == 5.0
