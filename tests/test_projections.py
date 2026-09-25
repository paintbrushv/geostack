"""Tests for geostack.projections — demographic CAGR projections."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

DALLAS_STATE = "48"
DALLAS_COUNTY = "113"


class TestComputeCagr:
    def test_cagr_positive_growth(self):
        from geostack.trends import compute_cagr as _compute_cagr
        assert _compute_cagr(100, 110, 2) == pytest.approx(0.04881, abs=0.001)

    def test_cagr_zero_growth(self):
        from geostack.trends import compute_cagr as _compute_cagr
        assert _compute_cagr(100, 100, 5) == pytest.approx(0.0)

    def test_cagr_negative_growth(self):
        from geostack.trends import compute_cagr as _compute_cagr
        result = _compute_cagr(100, 90, 2)
        assert result < 0

    def test_cagr_zero_start_returns_zero(self):
        from geostack.trends import compute_cagr as _compute_cagr
        assert _compute_cagr(0, 100, 5) == 0.0


class TestProjectValue:
    def test_project_forward(self):
        from geostack.projections import _project_value
        result = _project_value(100000, 0.03, 5)
        assert result == pytest.approx(115927, rel=0.01)


class TestCountyProjection:
    @patch("geostack.projections.CensusClient")
    def test_returns_projection_dict(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        def _mock_acs(vars, geography, state, county, year):
            incomes = {2019: 52000, 2020: 53000, 2021: 55000, 2022: 57000, 2023: 60000}
            return pd.DataFrame([{"B25118_014E": "10000", "B19013_001E": str(incomes.get(year, 55000))}])
        mock_client.get_acs.side_effect = _mock_acs

        from geostack.projections import county_income_projection
        result = county_income_projection("48", "113", base_years=(2019, 2023), target_year=2028)

        assert isinstance(result, dict)
        assert "base_median" in result
        assert "projected_median" in result
        assert "cagr" in result
        assert "scenarios" in result
        assert result["projected_median"] > result["base_median"]

    @patch("geostack.projections.CensusClient")
    def test_three_scenarios_returned(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        def _mock_acs(vars, geography, state, county, year):
            return pd.DataFrame([{"B25118_014E": "10000", "B19013_001E": str(50000 + (year - 2019) * 2000)}])
        mock_client.get_acs.side_effect = _mock_acs

        from geostack.projections import county_income_projection
        result = county_income_projection("48", "113", base_years=(2019, 2023), target_year=2028)

        scenarios = result["scenarios"]
        assert "low" in scenarios
        assert "base" in scenarios
        assert "high" in scenarios
        assert scenarios["low"] < scenarios["base"] < scenarios["high"]


class TestFetchPepPopulation:
    """Tests for _fetch_pep_population() -- Census PEP API."""

    @patch("geostack.projections.requests.get")
    def test_fetches_both_vintages_and_deduplicates(self, mock_get):
        """Should call 2019 + 2023 PEP APIs and deduplicate 2020."""
        # 2019 vintage response (DATE_CODE: 3=2010, ..., 12=2019)
        pep_2019_response = MagicMock()
        pep_2019_response.status_code = 200
        pep_2019_response.json.return_value = [
            ["POP", "DATE_CODE", "NAME", "state", "county"],
            ["1800000", "3", "Tarrant County, Texas", "48", "439"],   # 2010
            ["1850000", "4", "Tarrant County, Texas", "48", "439"],   # 2011
            ["1900000", "5", "Tarrant County, Texas", "48", "439"],   # 2012
            ["1920000", "6", "Tarrant County, Texas", "48", "439"],   # 2013
            ["1940000", "7", "Tarrant County, Texas", "48", "439"],   # 2014
            ["1960000", "8", "Tarrant County, Texas", "48", "439"],   # 2015
            ["1980000", "9", "Tarrant County, Texas", "48", "439"],   # 2016
            ["2000000", "10", "Tarrant County, Texas", "48", "439"],  # 2017
            ["2020000", "11", "Tarrant County, Texas", "48", "439"],  # 2018
            ["2040000", "12", "Tarrant County, Texas", "48", "439"],  # 2019
        ]

        # 2023 vintage response (YEAR is calendar year)
        pep_2023_response = MagicMock()
        pep_2023_response.status_code = 200
        pep_2023_response.json.return_value = [
            ["POP", "YEAR", "NAME", "state", "county"],
            ["2060000", "2020", "Tarrant County, Texas", "48", "439"],
            ["2080000", "2021", "Tarrant County, Texas", "48", "439"],
            ["2100000", "2022", "Tarrant County, Texas", "48", "439"],
            ["2120000", "2023", "Tarrant County, Texas", "48", "439"],
        ]

        mock_get.side_effect = [pep_2019_response, pep_2023_response]

        from geostack.projections import _fetch_pep_population
        result = _fetch_pep_population("48", "439")

        # Should have years 2010-2023 (14 years, no gap)
        assert len(result) == 14
        assert result[2010] == 1800000
        assert result[2019] == 2040000
        # 2020 should come from the 2023 vintage (preferred)
        assert result[2020] == 2060000
        assert result[2023] == 2120000

        # Should have called requests.get exactly twice
        assert mock_get.call_count == 2

    @patch("geostack.projections.requests.get")
    def test_filters_date_code_range(self, mock_get):
        """Should ignore DATE_CODE 1 and 2 (April 2010 base/estimate)."""
        pep_2019_response = MagicMock()
        pep_2019_response.status_code = 200
        pep_2019_response.json.return_value = [
            ["POP", "DATE_CODE", "NAME", "state", "county"],
            ["1790000", "1", "Tarrant County, Texas", "48", "439"],   # Apr base -- skip
            ["1795000", "2", "Tarrant County, Texas", "48", "439"],   # Apr est  -- skip
            ["1800000", "3", "Tarrant County, Texas", "48", "439"],   # Jul 2010 -- keep
        ]

        pep_2023_response = MagicMock()
        pep_2023_response.status_code = 200
        pep_2023_response.json.return_value = [
            ["POP", "YEAR", "NAME", "state", "county"],
            ["2060000", "2020", "Tarrant County, Texas", "48", "439"],
        ]

        mock_get.side_effect = [pep_2019_response, pep_2023_response]

        from geostack.projections import _fetch_pep_population
        result = _fetch_pep_population("48", "439")

        # DATE_CODE 1 and 2 should be excluded
        assert 2008 not in result  # DATE_CODE 1 would be 2007+1=2008
        assert 2009 not in result  # DATE_CODE 2 would be 2007+2=2009
        assert 2010 in result
        assert result[2010] == 1800000

    @patch("geostack.projections.requests.get")
    def test_handles_api_failure_gracefully(self, mock_get):
        """If 2019 vintage fails, should still return 2020-2023 from charv."""
        pep_2019_response = MagicMock()
        pep_2019_response.status_code = 500
        pep_2019_response.raise_for_status.side_effect = Exception("Server Error")

        pep_2023_response = MagicMock()
        pep_2023_response.status_code = 200
        pep_2023_response.json.return_value = [
            ["POP", "YEAR", "NAME", "state", "county"],
            ["2060000", "2020", "Tarrant County, Texas", "48", "439"],
            ["2120000", "2023", "Tarrant County, Texas", "48", "439"],
        ]

        mock_get.side_effect = [pep_2019_response, pep_2023_response]

        from geostack.projections import _fetch_pep_population
        result = _fetch_pep_population("48", "439")

        assert 2020 in result
        assert 2023 in result
        assert result[2023] == 2120000

    @patch("geostack.projections.requests.get")
    def test_returns_int_values(self, mock_get):
        """All population values should be integers, not strings."""
        pep_2019_response = MagicMock()
        pep_2019_response.status_code = 200
        pep_2019_response.json.return_value = [
            ["POP", "DATE_CODE", "NAME", "state", "county"],
            ["1800000", "3", "Tarrant County, Texas", "48", "439"],
        ]

        pep_2023_response = MagicMock()
        pep_2023_response.status_code = 200
        pep_2023_response.json.return_value = [
            ["POP", "YEAR", "NAME", "state", "county"],
            ["2060000", "2020", "Tarrant County, Texas", "48", "439"],
        ]

        mock_get.side_effect = [pep_2019_response, pep_2023_response]

        from geostack.projections import _fetch_pep_population
        result = _fetch_pep_population("48", "439")

        for year, pop in result.items():
            assert isinstance(year, int), f"Year {year} is not int"
            assert isinstance(pop, int), f"Pop {pop} for year {year} is not int"


class TestFetchHouseholdTimeSeries:
    """Tests for _fetch_household_time_series() -- ACS B11001_001E."""

    @patch("geostack.projections.CensusClient")
    def test_fetches_multiple_years(self, mock_client_cls):
        """Should call CensusClient.get_acs() for each year in range."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        hh_by_year = {
            2015: 700000, 2016: 705000, 2017: 710000, 2018: 715000,
            2019: 720000, 2020: 725000, 2021: 730000, 2022: 735000,
            2023: 740000,
        }

        def _mock_acs(variables, geography, state, county, year):
            val = hh_by_year.get(year, 0)
            return pd.DataFrame([{"B11001_001E": str(val)}])

        mock_client.get_acs.side_effect = _mock_acs

        from geostack.projections import _fetch_household_time_series
        result = _fetch_household_time_series("48", "439")

        assert len(result) == 9
        assert result[2015] == 700000
        assert result[2023] == 740000

    @patch("geostack.projections.CensusClient")
    def test_skips_years_with_errors(self, mock_client_cls):
        """If a year fails, it should be skipped without crashing."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        call_count = 0

        def _mock_acs(variables, geography, state, county, year):
            nonlocal call_count
            call_count += 1
            if year == 2017:
                raise Exception("ACS 2017 not available")
            return pd.DataFrame([{"B11001_001E": str(700000 + (year - 2015) * 5000)}])

        mock_client.get_acs.side_effect = _mock_acs

        from geostack.projections import _fetch_household_time_series
        result = _fetch_household_time_series("48", "439")

        assert 2017 not in result
        assert 2015 in result
        assert 2023 in result
        assert len(result) == 8  # 9 years minus the failed one

    @patch("geostack.projections.CensusClient")
    def test_returns_int_values(self, mock_client_cls):
        """Household counts should be integers."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        def _mock_acs(variables, geography, state, county, year):
            return pd.DataFrame([{"B11001_001E": "720000"}])

        mock_client.get_acs.side_effect = _mock_acs

        from geostack.projections import _fetch_household_time_series
        result = _fetch_household_time_series("48", "439")

        for year, hh in result.items():
            assert isinstance(hh, int), f"HH {hh} for year {year} is not int"

    @patch("geostack.projections.CensusClient")
    def test_custom_year_range(self, mock_client_cls):
        """Should accept custom start/end years."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        def _mock_acs(variables, geography, state, county, year):
            return pd.DataFrame([{"B11001_001E": str(700000 + (year - 2015) * 5000)}])

        mock_client.get_acs.side_effect = _mock_acs

        from geostack.projections import _fetch_household_time_series
        result = _fetch_household_time_series("48", "439", start_year=2019, end_year=2023)

        assert len(result) == 5
        assert 2018 not in result
        assert 2019 in result
        assert 2023 in result

    @patch("geostack.projections.CensusClient")
    def test_filters_negative_and_zero_values(self, mock_client_cls):
        """Should skip suppressed/missing values (<=0 or sentinel)."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        def _mock_acs(variables, geography, state, county, year):
            if year == 2016:
                return pd.DataFrame([{"B11001_001E": "-666666666"}])
            if year == 2017:
                return pd.DataFrame([{"B11001_001E": "0"}])
            return pd.DataFrame([{"B11001_001E": "720000"}])

        mock_client.get_acs.side_effect = _mock_acs

        from geostack.projections import _fetch_household_time_series
        result = _fetch_household_time_series("48", "439")

        assert 2016 not in result
        assert 2017 not in result


class TestCountyDemandForecast:
    """Tests for county_demand_forecast() -- multi-variable projection."""

    @patch("geostack.projections.fetch_qcew")
    @patch("geostack.projections._fetch_household_time_series")
    @patch("geostack.projections._fetch_pep_population")
    def test_returns_complete_structure(self, mock_pep, mock_hh, mock_qcew):
        """Return dict must have county, population, employment, households keys."""
        mock_pep.return_value = {
            2015: 1960000, 2016: 1980000, 2017: 2000000, 2018: 2020000,
            2019: 2040000, 2020: 2060000, 2021: 2080000, 2022: 2100000,
            2023: 2120000,
        }

        mock_hh.return_value = {
            2015: 700000, 2016: 705000, 2017: 710000, 2018: 715000,
            2019: 720000, 2020: 725000, 2021: 730000, 2022: 735000,
            2023: 740000,
        }

        # QCEW returns a DataFrame with total employment
        def _mock_qcew(state_fips, county_fips, year, **kwargs):
            emp = 900000 + (year - 2015) * 10000
            return pd.DataFrame([{
                "naics_code": "10",
                "naics_title": "Total, All Industries",
                "agglvl_code": 74,
                "annual_avg_employment": emp,
                "avg_annual_pay": 55000,
            }])

        mock_qcew.side_effect = _mock_qcew

        from geostack.projections import county_demand_forecast
        result = county_demand_forecast("48", "439", target_year=2030)

        assert result["state_fips"] == "48"
        assert result["county_fips"] == "439"
        assert result["target_year"] == 2030

        for key in ["population", "employment", "households"]:
            assert key in result, f"Missing key: {key}"
            section = result[key]
            assert "time_series" in section
            assert "base_value" in section
            assert "base_year" in section
            assert "cagr" in section
            assert "projected" in section
            assert "low" in section["projected"]
            assert "base" in section["projected"]
            assert "high" in section["projected"]

    @patch("geostack.projections.fetch_qcew")
    @patch("geostack.projections._fetch_household_time_series")
    @patch("geostack.projections._fetch_pep_population")
    def test_scenario_ordering(self, mock_pep, mock_hh, mock_qcew):
        """Low < Base < High for positive-growth variables."""
        mock_pep.return_value = {
            2015: 1960000, 2023: 2120000,
        }
        mock_hh.return_value = {
            2015: 700000, 2023: 740000,
        }

        def _mock_qcew(state_fips, county_fips, year, **kwargs):
            emp = 900000 + (year - 2015) * 10000
            return pd.DataFrame([{
                "naics_code": "10", "agglvl_code": 74,
                "annual_avg_employment": emp,
            }])

        mock_qcew.side_effect = _mock_qcew

        from geostack.projections import county_demand_forecast
        result = county_demand_forecast("48", "439", target_year=2030)

        for key in ["population", "households"]:
            proj = result[key]["projected"]
            assert proj["low"] < proj["base"] < proj["high"], (
                f"{key} scenarios not ordered: {proj}"
            )

    @patch("geostack.projections.fetch_qcew")
    @patch("geostack.projections._fetch_household_time_series")
    @patch("geostack.projections._fetch_pep_population")
    def test_cagr_is_positive_for_growing_county(self, mock_pep, mock_hh, mock_qcew):
        """CAGR should be positive when end > start."""
        mock_pep.return_value = {2015: 1900000, 2023: 2100000}
        mock_hh.return_value = {2015: 700000, 2023: 750000}

        def _mock_qcew(state_fips, county_fips, year, **kwargs):
            return pd.DataFrame([{
                "naics_code": "10", "agglvl_code": 74,
                "annual_avg_employment": 900000 + (year - 2015) * 10000,
            }])

        mock_qcew.side_effect = _mock_qcew

        from geostack.projections import county_demand_forecast
        result = county_demand_forecast("48", "439", target_year=2030)

        assert result["population"]["cagr"] > 0
        assert result["households"]["cagr"] > 0

    @patch("geostack.projections.fetch_qcew")
    @patch("geostack.projections._fetch_household_time_series")
    @patch("geostack.projections._fetch_pep_population")
    def test_projected_values_exceed_base_for_positive_cagr(
        self, mock_pep, mock_hh, mock_qcew
    ):
        """All scenario projected values should exceed base when CAGR > 0."""
        mock_pep.return_value = {2015: 1900000, 2023: 2100000}
        mock_hh.return_value = {2015: 700000, 2023: 750000}

        def _mock_qcew(state_fips, county_fips, year, **kwargs):
            return pd.DataFrame([{
                "naics_code": "10", "agglvl_code": 74,
                "annual_avg_employment": 900000 + (year - 2015) * 10000,
            }])

        mock_qcew.side_effect = _mock_qcew

        from geostack.projections import county_demand_forecast
        result = county_demand_forecast("48", "439", target_year=2030)

        for key in ["population", "households"]:
            base = result[key]["base_value"]
            for scenario in ["low", "base", "high"]:
                assert result[key]["projected"][scenario] > base, (
                    f"{key}.{scenario} ({result[key]['projected'][scenario]}) "
                    f"should exceed base ({base})"
                )

    @patch("geostack.projections.fetch_qcew")
    @patch("geostack.projections._fetch_household_time_series")
    @patch("geostack.projections._fetch_pep_population")
    def test_handles_missing_population_data(self, mock_pep, mock_hh, mock_qcew):
        """If PEP returns < 2 years, population section should be null-safe."""
        mock_pep.return_value = {}  # no data
        mock_hh.return_value = {2015: 700000, 2023: 740000}

        def _mock_qcew(state_fips, county_fips, year, **kwargs):
            return pd.DataFrame([{
                "naics_code": "10", "agglvl_code": 74,
                "annual_avg_employment": 900000,
            }])

        mock_qcew.side_effect = _mock_qcew

        from geostack.projections import county_demand_forecast
        result = county_demand_forecast("48", "439", target_year=2030)

        assert result["population"]["cagr"] is None
        assert result["population"]["projected"] is None

    @patch("geostack.projections.fetch_qcew")
    @patch("geostack.projections._fetch_household_time_series")
    @patch("geostack.projections._fetch_pep_population")
    def test_uses_most_recent_year_as_base(self, mock_pep, mock_hh, mock_qcew):
        """base_year should be the max year in the time series."""
        mock_pep.return_value = {2018: 2000000, 2019: 2020000, 2020: 2040000}
        mock_hh.return_value = {2019: 720000, 2020: 725000, 2021: 730000}

        def _mock_qcew(state_fips, county_fips, year, **kwargs):
            return pd.DataFrame([{
                "naics_code": "10", "agglvl_code": 74,
                "annual_avg_employment": 900000,
            }])

        mock_qcew.side_effect = _mock_qcew

        from geostack.projections import county_demand_forecast
        result = county_demand_forecast("48", "439", target_year=2030)

        assert result["population"]["base_year"] == 2020
        assert result["households"]["base_year"] == 2021

    @patch("geostack.projections.fetch_qcew")
    @patch("geostack.projections._fetch_household_time_series")
    @patch("geostack.projections._fetch_pep_population")
    def test_employment_uses_qcew_total(self, mock_pep, mock_hh, mock_qcew):
        """Employment time series should come from QCEW total industry."""
        mock_pep.return_value = {2015: 1900000, 2023: 2100000}
        mock_hh.return_value = {2015: 700000, 2023: 750000}

        def _mock_qcew(state_fips, county_fips, year, **kwargs):
            return pd.DataFrame([
                {"naics_code": "10", "agglvl_code": 74,
                 "annual_avg_employment": 900000 + (year - 2015) * 10000},
                {"naics_code": "23", "agglvl_code": 74,
                 "annual_avg_employment": 50000},
            ])

        mock_qcew.side_effect = _mock_qcew

        from geostack.projections import county_demand_forecast
        result = county_demand_forecast("48", "439", target_year=2030)

        # Should use naics_code "10" (total), not sector-level
        assert result["employment"]["base_value"] == 900000 + (2023 - 2015) * 10000


class TestClassifyTractGrowth:
    """Tests for _classify_tract_growth() -- the pure classification function."""

    def test_new_development(self):
        from geostack.projections import _classify_tract_growth
        assert _classify_tract_growth(unit_cagr=0.05, income_cagr=0.02, ilr=None) == "new_development"
        assert _classify_tract_growth(unit_cagr=0.04, income_cagr=-0.01, ilr=0.8) == "new_development"

    def test_declining(self):
        from geostack.projections import _classify_tract_growth
        assert _classify_tract_growth(unit_cagr=0.01, income_cagr=-0.01, ilr=None) == "declining"
        assert _classify_tract_growth(unit_cagr=-0.02, income_cagr=-0.005, ilr=1.2) == "declining"

    def test_growing(self):
        from geostack.projections import _classify_tract_growth
        assert _classify_tract_growth(unit_cagr=0.015, income_cagr=0.02, ilr=None) == "growing"
        assert _classify_tract_growth(unit_cagr=0.006, income_cagr=0.001, ilr=1.5) == "growing"

    def test_saturated(self):
        from geostack.projections import _classify_tract_growth
        # Flat units, positive income, no ILR or ILR >= 1.0
        assert _classify_tract_growth(unit_cagr=0.002, income_cagr=0.01, ilr=None) == "saturated"
        assert _classify_tract_growth(unit_cagr=-0.003, income_cagr=0.005, ilr=1.2) == "saturated"
        assert _classify_tract_growth(unit_cagr=0.0, income_cagr=0.01, ilr=1.0) == "saturated"

    def test_redevelopment_candidate(self):
        from geostack.projections import _classify_tract_growth
        # Flat units, positive income, ILR < 1.0
        assert _classify_tract_growth(unit_cagr=0.002, income_cagr=0.01, ilr=0.6) == "redevelopment_candidate"
        assert _classify_tract_growth(unit_cagr=-0.003, income_cagr=0.005, ilr=0.9) == "redevelopment_candidate"

    def test_edge_case_exactly_at_thresholds(self):
        from geostack.projections import _classify_tract_growth
        # unit_cagr exactly 0.5% is NOT > 0.5% -- should be saturated, not growing
        assert _classify_tract_growth(unit_cagr=0.005, income_cagr=0.01, ilr=None) == "saturated"
        # unit_cagr exactly 3% is NOT > 3% -- should be growing, not new_development
        assert _classify_tract_growth(unit_cagr=0.03, income_cagr=0.01, ilr=None) == "growing"
        # income_cagr exactly 0 is NOT < 0 -- should not be declining
        assert _classify_tract_growth(unit_cagr=0.002, income_cagr=0.0, ilr=None) == "saturated"


class TestTractGrowthProfile:
    """Tests for tract_growth_profile() -- full DataFrame output."""

    @patch("geostack.projections.aggregate_ilr_to_tracts", return_value=pd.DataFrame())
    @patch("geostack.projections.compute_tract_trends")
    @patch("geostack.projections.CensusClient")
    def test_returns_dataframe_with_required_columns(
        self, mock_client_cls, mock_trends, mock_ilr
    ):
        """Output DataFrame must have geoid, classification, and CAGR columns."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Synthetic tract trends: 2 tracts with housing units + income
        mock_trends.return_value = pd.DataFrame([
            {"geoid": "48439010100", "variable": "total_housing_units",
             "value_2013": 1000, "value_2018": 1050, "value_2023": 1100,
             "cagr_10yr": 0.0095, "cagr_5yr": 0.0095, "acceleration": 0.0,
             "trend_class": "Steady Growth"},
            {"geoid": "48439010100", "variable": "median_hh_income",
             "value_2013": 50000, "value_2018": 55000, "value_2023": 60000,
             "cagr_10yr": 0.0183, "cagr_5yr": 0.0175, "acceleration": -0.0008,
             "trend_class": "Steady Growth"},
            {"geoid": "48439010200", "variable": "total_housing_units",
             "value_2013": 2000, "value_2018": 2010, "value_2023": 2020,
             "cagr_10yr": 0.001, "cagr_5yr": 0.001, "acceleration": 0.0,
             "trend_class": "Stagnant"},
            {"geoid": "48439010200", "variable": "median_hh_income",
             "value_2013": 45000, "value_2018": 47000, "value_2023": 49000,
             "cagr_10yr": 0.0086, "cagr_5yr": 0.0084, "acceleration": -0.0002,
             "trend_class": "Steady Growth"},
        ])

        from geostack.projections import tract_growth_profile
        result = tract_growth_profile("48", "439")

        assert isinstance(result, pd.DataFrame)
        assert "geoid" in result.columns
        assert "classification" in result.columns
        assert "unit_cagr_10yr" in result.columns
        assert "income_cagr_10yr" in result.columns
        assert len(result) == 2  # 2 tracts

    @patch("geostack.projections.aggregate_ilr_to_tracts", return_value=pd.DataFrame())
    @patch("geostack.projections.compute_tract_trends")
    @patch("geostack.projections.CensusClient")
    def test_classifies_new_development(self, mock_client_cls, mock_trends, mock_ilr):
        """Tracts with >3% housing unit CAGR should be new_development."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_trends.return_value = pd.DataFrame([
            {"geoid": "48439010100", "variable": "total_housing_units",
             "cagr_10yr": 0.05, "cagr_5yr": 0.04, "acceleration": -0.01,
             "value_2013": 500, "value_2018": 700, "value_2023": 1000,
             "trend_class": "Accelerating Growth"},
            {"geoid": "48439010100", "variable": "median_hh_income",
             "cagr_10yr": 0.02, "cagr_5yr": 0.02, "acceleration": 0.0,
             "value_2013": 50000, "value_2018": 55000, "value_2023": 60000,
             "trend_class": "Steady Growth"},
        ])

        from geostack.projections import tract_growth_profile
        result = tract_growth_profile("48", "439")

        assert result.iloc[0]["classification"] == "new_development"

    @patch("geostack.projections.compute_tract_trends")
    @patch("geostack.projections.CensusClient")
    def test_uses_ilr_for_redevelopment_candidate(self, mock_client_cls, mock_trends):
        """Saturated tracts with ILR < 1.0 should be redevelopment_candidate."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_trends.return_value = pd.DataFrame([
            {"geoid": "48439010100", "variable": "total_housing_units",
             "cagr_10yr": 0.002, "cagr_5yr": 0.001, "acceleration": -0.001,
             "value_2013": 2000, "value_2018": 2020, "value_2023": 2040,
             "trend_class": "Stagnant"},
            {"geoid": "48439010100", "variable": "median_hh_income",
             "cagr_10yr": 0.015, "cagr_5yr": 0.015, "acceleration": 0.0,
             "value_2013": 50000, "value_2018": 53000, "value_2023": 58000,
             "trend_class": "Steady Growth"},
        ])

        # Mock ILR data -- tract has avg ILR < 1.0
        ilr_df = pd.DataFrame([
            {"geoid": "48439010100", "ilr_avg": 0.7},
        ])

        with patch("geostack.projections.aggregate_ilr_to_tracts", return_value=ilr_df):
            from geostack.projections import tract_growth_profile
            result = tract_growth_profile("48", "439", include_ilr=True)

        assert result.iloc[0]["classification"] == "redevelopment_candidate"
        assert result.iloc[0]["ilr_avg"] == pytest.approx(0.7)


@pytest.mark.integration
class TestProjectionsIntegration:
    """Live tests against Census PEP API. Run with: pytest -m integration"""

    def test_pep_tarrant_county_population(self):
        from geostack.projections import _fetch_pep_population

        result = _fetch_pep_population("48", "439")
        assert len(result) >= 10  # should have 2010-2019 at minimum
        assert 2010 in result
        # Most recent available year should have > 2M (Tarrant County ~2.1M)
        most_recent_year = max(result.keys())
        assert most_recent_year >= 2019
        assert result[most_recent_year] > 2_000_000  # Tarrant County is ~2.1M

    def test_county_demand_forecast_tarrant(self):
        from geostack.projections import county_demand_forecast

        result = county_demand_forecast("48", "439", target_year=2030)
        assert "population" in result
        assert "employment" in result
        assert "households" in result
        assert result["population"]["base_value"] > 0
