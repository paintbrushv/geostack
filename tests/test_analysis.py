"""Tests for geostack.analysis — ILR, age, trend, and spatial aggregation."""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

from geostack.analysis import (
    aggregate_ilr_to_tracts,
    clean_numeric,
    clean_year,
    compute_age,
    compute_ilr,
    ilr_trend,
    normalize_address,
)


# ---------------------------------------------------------------------------
# clean_numeric
# ---------------------------------------------------------------------------

class TestCleanNumeric:
    def test_coerces_strings(self):
        s = pd.Series(["100", "200.5", "abc"])
        result = clean_numeric(s)
        assert result.iloc[0] == 100.0
        assert result.iloc[1] == 200.5
        assert pd.isna(result.iloc[2])

    def test_negatives_become_nan(self):
        s = pd.Series([-1.0, 0.0, 50.0])
        result = clean_numeric(s)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 0.0
        assert result.iloc[2] == 50.0

    def test_none_becomes_nan(self):
        s = pd.Series([None, 10.0])
        result = clean_numeric(s)
        assert pd.isna(result.iloc[0])


# ---------------------------------------------------------------------------
# clean_year
# ---------------------------------------------------------------------------

class TestCleanYear:
    def test_valid_years_pass_through(self):
        s = pd.Series([1990, 2005, 2024])
        result = clean_year(s)
        assert list(result.dropna()) == [1990.0, 2005.0, 2024.0]

    def test_implausible_years_become_nan(self):
        s = pd.Series([1799, 2031, 1900])
        result = clean_year(s)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 1900.0

    def test_non_numeric_becomes_nan(self):
        s = pd.Series(["not-a-year", "2020"])
        result = clean_year(s)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 2020.0


# ---------------------------------------------------------------------------
# normalize_address
# ---------------------------------------------------------------------------

class TestNormalizeAddress:
    def test_uppercases(self):
        s = pd.Series(["123 main st"])
        assert normalize_address(s).iloc[0] == "123 MAIN ST"

    def test_strips_whitespace(self):
        s = pd.Series(["  123 MAIN  "])
        assert normalize_address(s).iloc[0] == "123 MAIN"

    def test_collapses_spaces(self):
        s = pd.Series(["123   MAIN   ST"])
        assert normalize_address(s).iloc[0] == "123 MAIN ST"

    def test_nan_string_becomes_nan(self):
        s = pd.Series(["NAN", "NONE", "", None])
        result = normalize_address(s)
        for val in result:
            assert val is None or (isinstance(val, float) and math.isnan(val))

    def test_valid_address_preserved(self):
        s = pd.Series(["4500 LEGACY DR"])
        assert normalize_address(s).iloc[0] == "4500 LEGACY DR"


# ---------------------------------------------------------------------------
# compute_ilr
# ---------------------------------------------------------------------------

class TestComputeIlr:
    def _df(self, land, imprv):
        return pd.DataFrame({"land_value": land, "improvement_value": imprv})

    def test_basic_ratio(self):
        df = compute_ilr(self._df([100_000], [300_000]))
        assert df["ilr"].iloc[0] == pytest.approx(3.0)

    def test_zero_land_is_nan(self):
        df = compute_ilr(self._df([0], [200_000]))
        assert pd.isna(df["ilr"].iloc[0])

    def test_null_land_is_nan(self):
        df = compute_ilr(self._df([None], [200_000]))
        assert pd.isna(df["ilr"].iloc[0])

    def test_negative_land_is_nan(self):
        df = compute_ilr(self._df([-1_000], [200_000]))
        assert pd.isna(df["ilr"].iloc[0])

    def test_does_not_mutate_input(self):
        orig = self._df([50_000], [100_000])
        orig_cols = list(orig.columns)
        compute_ilr(orig)
        assert list(orig.columns) == orig_cols

    def test_custom_column_names(self):
        df = pd.DataFrame({"lv": [50_000.0], "iv": [150_000.0]})
        result = compute_ilr(df, land_col="lv", imprv_col="iv")
        assert result["ilr"].iloc[0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# compute_age
# ---------------------------------------------------------------------------

class TestComputeAge:
    def test_basic_age(self):
        df = pd.DataFrame({"year_built": [2000.0], "tax_year": [2024]})
        result = compute_age(df)
        assert result["age"].iloc[0] == 24.0

    def test_null_built_is_nan(self):
        df = pd.DataFrame({"year_built": [None], "tax_year": [2024]})
        result = compute_age(df)
        assert pd.isna(result["age"].iloc[0])

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"year_built": [1990.0], "tax_year": [2024]})
        cols = list(df.columns)
        compute_age(df)
        assert list(df.columns) == cols


# ---------------------------------------------------------------------------
# ilr_trend
# ---------------------------------------------------------------------------

class TestIlrTrend:
    def _frames(self):
        """Three-year synthetic dataset: two parcels."""
        return {
            2021: pd.DataFrame({
                "parcel_id": ["A", "B"],
                "land_value": [100_000.0, 200_000.0],
                "improvement_value": [300_000.0, 200_000.0],
            }),
            2022: pd.DataFrame({
                "parcel_id": ["A", "B"],
                "land_value": [120_000.0, 200_000.0],
                "improvement_value": [300_000.0, 180_000.0],
            }),
            2023: pd.DataFrame({
                "parcel_id": ["A", "B"],
                "land_value": [150_000.0, 200_000.0],
                "improvement_value": [300_000.0, 150_000.0],
            }),
        }

    def test_returns_dataframe(self):
        result = ilr_trend(self._frames())
        assert isinstance(result, pd.DataFrame)

    def test_has_per_year_columns(self):
        result = ilr_trend(self._frames())
        for yr in [2021, 2022, 2023]:
            assert f"ilr_{yr}" in result.columns

    def test_has_slope_and_direction_columns(self):
        result = ilr_trend(self._frames())
        assert "ilr_trend_slope" in result.columns
        assert "ilr_direction" in result.columns

    def test_parcel_a_ilr_decreasing(self):
        """Parcel A: land rising, improvement stable → ILR falls → direction = down."""
        result = ilr_trend(self._frames())
        row = result[result["parcel_id"] == "A"].iloc[0]
        assert row["ilr_trend_slope"] < 0
        assert row["ilr_direction"] == "down"

    def test_parcel_b_ilr_decreasing(self):
        """Parcel B: improvement falling, land stable → ILR falls."""
        result = ilr_trend(self._frames())
        row = result[result["parcel_id"] == "B"].iloc[0]
        assert row["ilr_trend_slope"] < 0
        assert row["ilr_direction"] == "down"

    def test_single_year_gives_nan_slope(self):
        """Less than 2 years of data → slope NaN, direction None."""
        frames = {
            2023: pd.DataFrame({
                "parcel_id": ["A"],
                "land_value": [100_000.0],
                "improvement_value": [200_000.0],
            })
        }
        result = ilr_trend(frames)
        assert pd.isna(result["ilr_trend_slope"].iloc[0])
        assert result["ilr_direction"].iloc[0] is None

    def test_empty_frames_raises(self):
        with pytest.raises(ValueError):
            ilr_trend({})

    def test_flat_direction_within_threshold(self):
        """Nearly identical ILR across years → flat."""
        frames = {
            2021: pd.DataFrame({"parcel_id": ["X"], "land_value": [100_000.0], "improvement_value": [200_000.0]}),
            2022: pd.DataFrame({"parcel_id": ["X"], "land_value": [100_001.0], "improvement_value": [200_001.0]}),
            2023: pd.DataFrame({"parcel_id": ["X"], "land_value": [100_002.0], "improvement_value": [200_002.0]}),
        }
        result = ilr_trend(frames, flat_threshold=0.01)
        assert result["ilr_direction"].iloc[0] == "flat"

    def test_preserves_ilr_values_for_all_years(self):
        """All ilr_YYYY columns present with non-NaN values for valid parcels."""
        result = ilr_trend(self._frames())
        for yr in [2021, 2022, 2023]:
            col = f"ilr_{yr}"
            assert result[col].notna().all(), f"{col} has unexpected NaN"


# ---------------------------------------------------------------------------
# aggregate_ilr_to_tracts
# ---------------------------------------------------------------------------

class TestAggregateIlrToTracts:
    def _parcels(self):
        points = [Point(-96.75, 33.05), Point(-96.72, 33.06), Point(-96.65, 33.05)]
        return gpd.GeoDataFrame(
            {
                "parcel_id": ["P1", "P2", "P3"],
                "ilr": [0.8, 1.5, 2.0],
            },
            geometry=points,
            crs="EPSG:4326",
        )

    def _tracts(self):
        polys = [
            box(-96.80, 33.00, -96.70, 33.10),  # tract A — contains P1, P2
            box(-96.70, 33.00, -96.60, 33.10),  # tract B — contains P3
        ]
        return gpd.GeoDataFrame(
            {"geoid": ["48085000100", "48085000200"]},
            geometry=polys,
            crs="EPSG:4326",
        )

    def test_returns_geodataframe(self):
        result = aggregate_ilr_to_tracts(self._parcels(), self._tracts())
        assert isinstance(result, gpd.GeoDataFrame)

    def test_has_expected_columns(self):
        result = aggregate_ilr_to_tracts(self._parcels(), self._tracts())
        for col in ("median_ilr", "mean_ilr", "parcel_count", "pct_underimproved"):
            assert col in result.columns, f"Missing column: {col}"

    def test_parcel_count_correct(self):
        result = aggregate_ilr_to_tracts(self._parcels(), self._tracts())
        counts = result.set_index("geoid")["parcel_count"]
        assert counts["48085000100"] == 2
        assert counts["48085000200"] == 1

    def test_median_ilr_tract_a(self):
        result = aggregate_ilr_to_tracts(self._parcels(), self._tracts())
        row = result[result["geoid"] == "48085000100"].iloc[0]
        # P1=0.8, P2=1.5 → median = 1.15
        assert row["median_ilr"] == pytest.approx(1.15)

    def test_pct_underimproved(self):
        result = aggregate_ilr_to_tracts(self._parcels(), self._tracts())
        row = result[result["geoid"] == "48085000100"].iloc[0]
        # P1 (0.8 < 1.0) → 1/2 = 0.5
        assert row["pct_underimproved"] == pytest.approx(0.5)

    def test_auto_reproject_mismatched_crs(self):
        parcels = self._parcels().to_crs("EPSG:3857")
        tracts = self._tracts()  # EPSG:4326
        # Should auto-reproject and not raise
        result = aggregate_ilr_to_tracts(parcels, tracts)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_polygon_parcels_use_centroid(self):
        """Polygon parcels should be centroid-joined, not fail."""
        polys = [
            box(-96.76, 33.04, -96.74, 33.06),
            box(-96.73, 33.04, -96.71, 33.06),
        ]
        gdf = gpd.GeoDataFrame(
            {"parcel_id": ["P1", "P2"], "ilr": [1.0, 2.0]},
            geometry=polys,
            crs="EPSG:4326",
        )
        result = aggregate_ilr_to_tracts(gdf, self._tracts())
        assert isinstance(result, gpd.GeoDataFrame)
