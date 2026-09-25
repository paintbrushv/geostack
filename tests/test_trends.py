"""Tests for geostack.trends — tract-level ACS time series with provenance."""

import hashlib
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestLoadCpiDeflators:
    @patch("geostack.trends.requests.get")
    def test_returns_deflator_dict(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {"series": [{"data": [
                {"year": "2023", "period": "M13", "value": "304.702"},
                {"year": "2018", "period": "M13", "value": "251.107"},
                {"year": "2013", "period": "M13", "value": "232.957"},
            ]}]},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from geostack.trends import load_cpi_deflators
        deflators = load_cpi_deflators(base_year=2023)

        assert isinstance(deflators, dict)
        assert deflators[2023] == pytest.approx(1.0)
        assert deflators[2018] == pytest.approx(251.107 / 304.702, rel=0.001)
        assert deflators[2013] == pytest.approx(232.957 / 304.702, rel=0.001)

    @patch("geostack.trends.requests.get")
    def test_deflator_2013_less_than_one(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {"series": [{"data": [
                {"year": "2023", "period": "M13", "value": "300.0"},
                {"year": "2013", "period": "M13", "value": "240.0"},
            ]}]},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from geostack.trends import load_cpi_deflators
        deflators = load_cpi_deflators(base_year=2023)
        assert deflators[2013] < 1.0


class TestDeflateToConstantDollars:
    def test_deflates_income_variables(self):
        from geostack.trends import deflate_to_constant_dollars
        data = pd.DataFrame([
            {"geoid": "48113010100", "vintage": 2013, "variable": "median_rent", "value": 800.0},
            {"geoid": "48113010100", "vintage": 2023, "variable": "median_rent", "value": 1200.0},
            {"geoid": "48113010100", "vintage": 2013, "variable": "renter_hh", "value": 500.0},
            {"geoid": "48113010100", "vintage": 2023, "variable": "renter_hh", "value": 600.0},
        ])
        deflators = {2013: 0.80, 2023: 1.00}
        deflatable = {"median_rent"}

        result = deflate_to_constant_dollars(data, deflators, deflatable)

        rent_2013 = result[(result["vintage"] == 2013) & (result["variable"] == "median_rent")]
        assert rent_2013.iloc[0]["value"] == pytest.approx(800.0 / 0.80)

        hh_2013 = result[(result["vintage"] == 2013) & (result["variable"] == "renter_hh")]
        assert hh_2013.iloc[0]["value"] == pytest.approx(500.0)

    def test_adds_provenance(self):
        from geostack.trends import deflate_to_constant_dollars
        data = pd.DataFrame([
            {"geoid": "48113010100", "vintage": 2023, "variable": "median_rent", "value": 1200.0},
        ])
        result = deflate_to_constant_dollars(data, {2023: 1.0}, {"median_rent"})
        assert "cpi_base_year" in result.attrs
        assert "cpi_deflators" in result.attrs


# ---------------------------------------------------------------------------
# Crosswalk tests
# ---------------------------------------------------------------------------

_CROSSWALK_HEADER = (
    "OID_TRACT_20|GEOID_TRACT_20|NAMELSAD_TRACT_20|AREALAND_TRACT_20|AREAWATER_TRACT_20|"
    "MTFCC_TRACT_20|FUNCSTAT_TRACT_20|OID_TRACT_10|GEOID_TRACT_10|NAMELSAD_TRACT_10|"
    "AREALAND_TRACT_10|AREAWATER_TRACT_10|MTFCC_TRACT_10|FUNCSTAT_TRACT_10|"
    "AREALAND_PART|AREAWATER_PART"
)

# Unchanged tract: 2010 tract 48001000100 maps 1:1 to 2020 tract 48001000100
# with AREALAND_PART == AREALAND_TRACT_10 (8000000)
_CROSSWALK_ROW_UNCHANGED = (
    "1|48001000100|Census Tract 1|8000000|0|G5020|S|2|48001000100|Census Tract 1|"
    "8000000|0|G5020|S|8000000|0"
)

# Split tract: 2010 tract 48001000200 → two 2020 tracts (3M + 5M = 8M total)
_CROSSWALK_ROW_SPLIT_A = (
    "3|48001000201|Census Tract 2.01|3000000|0|G5020|S|4|48001000200|Census Tract 2|"
    "8000000|0|G5020|S|3000000|0"
)
_CROSSWALK_ROW_SPLIT_B = (
    "5|48001000202|Census Tract 2.02|5000000|0|G5020|S|4|48001000200|Census Tract 2|"
    "8000000|0|G5020|S|5000000|0"
)

_MOCK_CROSSWALK_TEXT = "\n".join([
    _CROSSWALK_HEADER,
    _CROSSWALK_ROW_UNCHANGED,
    _CROSSWALK_ROW_SPLIT_A,
    _CROSSWALK_ROW_SPLIT_B,
])


class TestLoadCrosswalk:
    @patch("geostack.trends.requests.get")
    def test_returns_dataframe_with_required_columns(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = _MOCK_CROSSWALK_TEXT
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from geostack.trends import load_crosswalk
        df = load_crosswalk(state_fips="48", _skip_cache=True)

        assert "geoid_2010" in df.columns
        assert "geoid_2020" in df.columns
        assert "areal_weight" in df.columns

    @patch("geostack.trends.requests.get")
    def test_areal_weight_for_unchanged_tract_is_one(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = _MOCK_CROSSWALK_TEXT
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from geostack.trends import load_crosswalk
        df = load_crosswalk(state_fips="48", _skip_cache=True)

        unchanged = df[df["geoid_2010"] == "48001000100"]
        assert len(unchanged) == 1
        assert unchanged.iloc[0]["areal_weight"] == pytest.approx(1.0)

    @patch("geostack.trends.requests.get")
    def test_split_tract_weights_sum_to_one(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = _MOCK_CROSSWALK_TEXT
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from geostack.trends import load_crosswalk
        df = load_crosswalk(state_fips="48", _skip_cache=True)

        split = df[df["geoid_2010"] == "48001000200"]
        assert len(split) == 2
        assert split["areal_weight"].sum() == pytest.approx(1.0)
        weights = sorted(split["areal_weight"].tolist())
        assert weights[0] == pytest.approx(0.375)
        assert weights[1] == pytest.approx(0.625)


class TestApplyCrosswalk:
    def _make_crosswalk_df(self):
        return pd.DataFrame([
            # Unchanged tract: 2010 GEOID == 2020 GEOID, weight 1.0
            {"geoid_2010": "48001000100", "geoid_2020": "48001000100", "areal_weight": 1.0},
            # Split tract piece A
            {"geoid_2010": "48001000200", "geoid_2020": "48001000201", "areal_weight": 0.375},
            # Split tract piece B
            {"geoid_2010": "48001000200", "geoid_2020": "48001000202", "areal_weight": 0.625},
        ])

    def _make_raw_data(self, vintage=2013):
        return pd.DataFrame([
            {"geoid": "48001000100", "vintage": vintage, "variable": "renter_hh", "value": 500.0},
            {"geoid": "48001000200", "vintage": vintage, "variable": "renter_hh", "value": 1000.0},
        ])

    def test_unchanged_tract_passes_through(self):
        from geostack.trends import apply_crosswalk
        raw = self._make_raw_data(vintage=2013)
        cw = self._make_crosswalk_df()
        variable_types = {"renter_hh": "count"}

        result = apply_crosswalk(raw, cw, variable_types)

        unchanged = result[result["geoid"] == "48001000100"]
        assert len(unchanged) == 1
        assert unchanged.iloc[0]["value"] == pytest.approx(500.0)

    def test_split_tract_allocates_counts_proportionally(self):
        from geostack.trends import apply_crosswalk
        raw = self._make_raw_data(vintage=2013)
        cw = self._make_crosswalk_df()
        variable_types = {"renter_hh": "count"}

        result = apply_crosswalk(raw, cw, variable_types)

        piece_a = result[result["geoid"] == "48001000201"]
        piece_b = result[result["geoid"] == "48001000202"]
        assert piece_a.iloc[0]["value"] == pytest.approx(375.0)
        assert piece_b.iloc[0]["value"] == pytest.approx(625.0)

    def test_split_tract_weights_rate_variables(self):
        from geostack.trends import apply_crosswalk
        raw = pd.DataFrame([
            {"geoid": "48001000200", "vintage": 2013, "variable": "vacancy_rate", "value": 0.08},
        ])
        cw = self._make_crosswalk_df()
        variable_types = {"vacancy_rate": "rate"}

        result = apply_crosswalk(raw, cw, variable_types)

        piece_a = result[result["geoid"] == "48001000201"]
        piece_b = result[result["geoid"] == "48001000202"]
        assert piece_a.iloc[0]["value"] == pytest.approx(0.08)
        assert piece_b.iloc[0]["value"] == pytest.approx(0.08)

    def test_2023_vintage_skips_crosswalk(self):
        from geostack.trends import apply_crosswalk
        raw = self._make_raw_data(vintage=2023)
        cw = self._make_crosswalk_df()
        variable_types = {"renter_hh": "count"}

        result = apply_crosswalk(raw, cw, variable_types)

        # 2023 already uses 2020 GEOIDs — original rows unchanged, no expansion
        assert len(result) == len(raw)
        geoids = set(result["geoid"].tolist())
        assert "48001000100" in geoids
        assert "48001000200" in geoids

    def test_adds_crosswalk_provenance(self):
        from geostack.trends import apply_crosswalk
        raw = self._make_raw_data(vintage=2013)
        cw = self._make_crosswalk_df()
        variable_types = {"renter_hh": "count"}

        result = apply_crosswalk(raw, cw, variable_types)

        assert "crosswalk" in result.attrs


# ---------------------------------------------------------------------------
# compute_tract_trends + calibrate_thresholds tests
# ---------------------------------------------------------------------------

class TestComputeTractTrends:
    def test_computes_cagr_and_acceleration(self):
        from geostack.trends import compute_tract_trends, DEFAULT_THRESHOLDS
        data = pd.DataFrame([
            {"geoid": "48113010100", "vintage": 2013, "variable": "median_rent", "value": 800},
            {"geoid": "48113010100", "vintage": 2018, "variable": "median_rent", "value": 950},
            {"geoid": "48113010100", "vintage": 2023, "variable": "median_rent", "value": 1200},
        ])
        result = compute_tract_trends(data, DEFAULT_THRESHOLDS)
        row = result[result["variable"] == "median_rent"].iloc[0]
        assert "cagr_10yr" in result.columns
        assert "cagr_5yr" in result.columns
        assert "acceleration" in result.columns
        expected_10yr = (1200 / 800) ** (1 / 10) - 1
        assert row["cagr_10yr"] == pytest.approx(expected_10yr, abs=0.001)

    def test_acceleration_positive_when_recent_faster(self):
        from geostack.trends import compute_tract_trends, DEFAULT_THRESHOLDS
        data = pd.DataFrame([
            {"geoid": "48113010100", "vintage": 2013, "variable": "median_rent", "value": 800},
            {"geoid": "48113010100", "vintage": 2018, "variable": "median_rent", "value": 850},
            {"geoid": "48113010100", "vintage": 2023, "variable": "median_rent", "value": 1100},
        ])
        result = compute_tract_trends(data, DEFAULT_THRESHOLDS)
        row = result[result["variable"] == "median_rent"].iloc[0]
        assert row["acceleration"] > 0

    def test_classifies_accelerating_growth(self):
        from geostack.trends import compute_tract_trends
        data = pd.DataFrame([
            {"geoid": "48113010100", "vintage": 2013, "variable": "median_rent", "value": 800},
            {"geoid": "48113010100", "vintage": 2018, "variable": "median_rent", "value": 850},
            {"geoid": "48113010100", "vintage": 2023, "variable": "median_rent", "value": 1100},
        ])
        thresholds = {"stagnant_cagr": 0.005, "accel_threshold": 0.005}
        result = compute_tract_trends(data, thresholds)
        row = result[result["variable"] == "median_rent"].iloc[0]
        assert row["trend_class"] == "Accelerating Growth"

    def test_classifies_stagnant(self):
        from geostack.trends import compute_tract_trends
        data = pd.DataFrame([
            {"geoid": "48113010100", "vintage": 2013, "variable": "median_rent", "value": 1000},
            {"geoid": "48113010100", "vintage": 2018, "variable": "median_rent", "value": 1005},
            {"geoid": "48113010100", "vintage": 2023, "variable": "median_rent", "value": 1010},
        ])
        thresholds = {"stagnant_cagr": 0.005, "accel_threshold": 0.005}
        result = compute_tract_trends(data, thresholds)
        row = result[result["variable"] == "median_rent"].iloc[0]
        assert row["trend_class"] == "Stagnant"

    def test_classifies_declining(self):
        from geostack.trends import compute_tract_trends
        data = pd.DataFrame([
            {"geoid": "48113010100", "vintage": 2013, "variable": "renter_hh", "value": 1000},
            {"geoid": "48113010100", "vintage": 2018, "variable": "renter_hh", "value": 950},
            {"geoid": "48113010100", "vintage": 2023, "variable": "renter_hh", "value": 900},
        ])
        thresholds = {"stagnant_cagr": 0.005, "accel_threshold": 0.005}
        result = compute_tract_trends(data, thresholds)
        row = result[result["variable"] == "renter_hh"].iloc[0]
        assert row["trend_class"] in ("Declining", "Accelerating Decline")

    def test_provenance_hash_is_deterministic(self):
        from geostack.trends import compute_tract_trends, DEFAULT_THRESHOLDS
        data = pd.DataFrame([
            {"geoid": "48113010100", "vintage": 2013, "variable": "median_rent", "value": 800},
            {"geoid": "48113010100", "vintage": 2018, "variable": "median_rent", "value": 900},
            {"geoid": "48113010100", "vintage": 2023, "variable": "median_rent", "value": 1000},
        ])
        r1 = compute_tract_trends(data, DEFAULT_THRESHOLDS)
        r2 = compute_tract_trends(data, DEFAULT_THRESHOLDS)
        assert r1.attrs["reproducibility_hash"] == r2.attrs["reproducibility_hash"]


class TestCalibrateThresholds:
    def test_percentile_method_returns_thresholds(self):
        from geostack.trends import calibrate_thresholds
        np.random.seed(42)
        trends = pd.DataFrame({
            "geoid": [f"4811301{i:04d}" for i in range(100)],
            "variable": ["median_rent"] * 100,
            "cagr_10yr": np.random.normal(0.02, 0.01, 100),
            "cagr_5yr": np.random.normal(0.025, 0.015, 100),
            "acceleration": np.random.normal(0.005, 0.008, 100),
        })
        result = calibrate_thresholds(trends, variable="median_rent", method="percentile")
        assert "stagnant_cagr" in result
        assert "accel_threshold" in result
        assert result["stagnant_cagr"] > 0


class TestComputeCatchmentTrends:
    @patch("geostack.db.get_engine")
    @patch("geostack.demographics.tracts_within_isochrone")
    def test_returns_weighted_average(self, mock_iso, mock_engine):
        mock_iso.return_value = pd.DataFrame({
            "geoid": ["48113010100", "48113010200"],
            "overlap_fraction": [1.0, 0.5],
        })
        mock_engine.return_value = MagicMock()

        from geostack.trends import compute_catchment_trends
        tract_trends = pd.DataFrame([
            {"geoid": "48113010100", "variable": "median_rent",
             "value_2023": 1200, "cagr_10yr": 0.04, "cagr_5yr": 0.05,
             "acceleration": 0.01, "trend_class": "Accelerating Growth"},
            {"geoid": "48113010200", "variable": "median_rent",
             "value_2023": 900, "cagr_10yr": 0.01, "cagr_5yr": 0.005,
             "acceleration": -0.005, "trend_class": "Stagnant"},
        ])
        renter_hh = {"48113010100": 500, "48113010200": 300}

        result = compute_catchment_trends(
            lon=-96.6183, lat=32.8028, minutes=10,
            tract_trends=tract_trends,
            renter_hh_by_tract=renter_hh,
            state_fips="48",
        )

        assert isinstance(result, dict)
        assert "median_rent" in result
        assert "cagr_10yr" in result["median_rent"]
        assert "n_tracts" in result
