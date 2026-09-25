"""Tests for geostack.gravity — MSA Population Center of Gravity."""

import numpy as np
import pandas as pd
import pytest

import geopandas as gpd
from shapely.geometry import Polygon, Point
from unittest.mock import patch, MagicMock

from geostack.gravity import (
    MSA_DEFINITIONS,
    WEIGHT_MODES,
    WEIGHT_LABELS,
    GravityResult,
    _compute_weighted_centroid,
    _default_years,
    _haversine_miles,
    _bearing_degrees,
    _needs_crosswalk,
    _weight_column,
    _available_years_for_mode,
    compute_shift_metrics,
    export_results,
    list_msas,
    resolve_msa,
    get_landmarks,
    MSA_LANDMARKS,
    project_center,
)


# ---------------------------------------------------------------------------
# Trajectory projection
# ---------------------------------------------------------------------------

class TestProjectCenter:
    def _linear_series(self):
        # A center marching due north at a steady pace
        return pd.DataFrame({
            "year": [2010, 2015, 2020],
            "center_lat": [32.80, 32.85, 32.90],
            "center_lon": [-96.95, -96.95, -96.95],
            "total_weight": [1e6, 1.1e6, 1.2e6],
            "msa_name": ["Test"] * 3,
        })

    def test_extrapolates_forward(self):
        res = project_center(self._linear_series(), 2025)
        # Perfectly linear north march → 2025 lat ≈ 32.95
        assert abs(res["target_lat"] - 32.95) < 0.01
        assert abs(res["target_lon"] - (-96.95)) < 0.005

    def test_high_r2_for_linear(self):
        res = project_center(self._linear_series(), 2025)
        assert res["r2"] > 0.99

    def test_bearing_is_north(self):
        res = project_center(self._linear_series(), 2025)
        assert res["trend_bearing_deg"] < 5 or res["trend_bearing_deg"] > 355

    def test_pace_positive(self):
        res = project_center(self._linear_series(), 2025)
        assert res["pace_mi_per_decade"] > 0

    def test_requires_two_points(self):
        one = pd.DataFrame({"year": [2020], "center_lat": [32.8], "center_lon": [-96.9]})
        with pytest.raises(ValueError, match="at least 2"):
            project_center(one, 2030)

    def test_metadata_fields(self):
        res = project_center(self._linear_series(), 2030)
        assert res["base_year"] == 2010
        assert res["target_year"] == 2030
        assert res["n_points"] == 3


# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------

class TestLandmarks:
    def test_dfw_has_landmarks(self):
        lms = get_landmarks("Dallas")
        assert len(lms) > 0

    def test_landmark_tuple_shape(self):
        for name, lat, lon, kind in get_landmarks("Dallas"):
            assert isinstance(name, str)
            assert 30 < lat < 35
            assert -98 < lon < -96
            assert kind in ("cbd", "suburb", "airport")

    def test_dfw_includes_downtown_dallas(self):
        names = [l[0] for l in get_landmarks("Dallas")]
        assert any("Dallas" in n for n in names)
        assert any("Fort Worth" in n for n in names)

    def test_unknown_msa_returns_empty(self):
        # An MSA without landmarks defined returns [] not error
        lms = get_landmarks("Phoenix")
        assert lms == []

    def test_dfw_has_allen_and_legacy(self):
        # Added after node detection mislabeled Allen as Plano and missed Frisco
        names = [l[0] for l in get_landmarks("Dallas")]
        assert "Allen" in names
        assert any("Legacy" in n or "Frisco" in n for n in names)
        assert any("Richardson" in n for n in names)

    def test_dfw_has_industrial_and_campus_landmarks(self):
        # Added after density-based detection missed Alliance (logistics) and
        # the Westlake corporate campuses; they appear as true-coord references.
        names = [l[0] for l in get_landmarks("Dallas")]
        assert any("Alliance" in n for n in names)
        assert any("Westlake" in n for n in names)
        assert any("Southlake" in n for n in names)

    def test_peak_detection_separates_corridor(self):
        # The core fix: three dense blobs strung along a line must resolve as
        # three peaks, not one chained cluster (the DBSCAN failure mode).
        from geostack.gravity_polycentric import _assign_to_peaks
        import geopandas as gpd
        from shapely.geometry import Point as P

        MI = 5280.0
        rows = []
        # Three blobs at x = 0, 6 mi, 12 mi (feet), each 4 tracts
        for bi, bx in enumerate([0, 6 * MI, 12 * MI]):
            for k in range(4):
                rows.append({
                    "centroid_x": bx + k * 200,
                    "centroid_y": k * 200,
                    "weight": 1000.0,
                    "geometry": P(bx + k * 200, k * 200),
                })
        hd = gpd.GeoDataFrame(rows, crs="EPSG:2276")
        labels = _assign_to_peaks(hd, "weight", min_separation_mi=3.0, peak_rel_threshold=0.05)
        # Should find 3 distinct groups
        assert len(set(labels)) >= 3

    def test_name_nodes_by_landmark(self):
        from geostack.gravity_polycentric import name_nodes_by_landmark
        # A node near Fort Worth CBD should be named for it
        nodes = gpd.GeoDataFrame({
            "node_id": [0],
            "node_rank": ["Node 1"],
            "center_lat": [32.7555],
            "center_lon": [-97.3308],
            "total_weight": [100000.0],
            "geometry": [Point(-97.3308, 32.7555)],
        }, crs="EPSG:4326")
        named = name_nodes_by_landmark(nodes, "Dallas")
        assert named.iloc[0]["place"] == "Fort Worth CBD"

    def test_name_nodes_far_node_unnamed(self):
        from geostack.gravity_polycentric import name_nodes_by_landmark
        # A node far from any landmark stays unnamed
        nodes = gpd.GeoDataFrame({
            "node_id": [0],
            "node_rank": ["Node 1"],
            "center_lat": [31.0],
            "center_lon": [-99.0],
            "total_weight": [100000.0],
            "geometry": [Point(-99.0, 31.0)],
        }, crs="EPSG:4326")
        named = name_nodes_by_landmark(nodes, "Dallas", max_distance_mi=8.0)
        assert named.iloc[0]["place"] == ""


# ---------------------------------------------------------------------------
# MSA definitions
# ---------------------------------------------------------------------------

class TestMsaDefinitions:
    def test_dfw_present(self):
        assert "Dallas-Fort Worth-Arlington, TX" in MSA_DEFINITIONS

    def test_dfw_has_12_counties(self):
        dfw = MSA_DEFINITIONS["Dallas-Fort Worth-Arlington, TX"]
        assert len(dfw["counties"]) == 12

    def test_dfw_includes_key_counties(self):
        dfw = MSA_DEFINITIONS["Dallas-Fort Worth-Arlington, TX"]
        assert "113" in dfw["counties"]  # Dallas
        assert "439" in dfw["counties"]  # Tarrant
        assert "085" in dfw["counties"]  # Collin
        assert "121" in dfw["counties"]  # Denton

    def test_all_msas_have_required_fields(self):
        for name, defn in MSA_DEFINITIONS.items():
            assert "state_fips" in defn, f"{name} missing state_fips"
            assert "counties" in defn, f"{name} missing counties"
            assert len(defn["counties"]) > 0, f"{name} has empty counties"

    def test_list_msas(self):
        msas = list_msas()
        assert len(msas) >= 4
        assert "Dallas-Fort Worth-Arlington, TX" in msas

    def test_resolve_msa_exact(self):
        result = resolve_msa("Dallas-Fort Worth-Arlington, TX")
        assert result["name"] == "Dallas-Fort Worth-Arlington, TX"
        assert result["state_fips"] == "48"

    def test_resolve_msa_partial(self):
        result = resolve_msa("Dallas")
        assert "Dallas" in result["name"]

    def test_resolve_msa_case_insensitive(self):
        result = resolve_msa("houston")
        assert "Houston" in result["name"]

    def test_resolve_msa_not_found(self):
        with pytest.raises(ValueError, match="MSA not found"):
            resolve_msa("Nonexistent City, ZZ")


# ---------------------------------------------------------------------------
# Weight modes
# ---------------------------------------------------------------------------

class TestWeightModes:
    def test_four_modes(self):
        assert len(WEIGHT_MODES) == 4
        assert "population" in WEIGHT_MODES
        assert "jobs" in WEIGHT_MODES
        assert "workers" in WEIGHT_MODES
        assert "income_mass" in WEIGHT_MODES

    def test_labels_match_modes(self):
        for mode in WEIGHT_MODES:
            assert mode in WEIGHT_LABELS

    def test_weight_column_population(self):
        assert _weight_column("population") == "population"

    def test_weight_column_jobs(self):
        assert _weight_column("jobs") == "weight"

    def test_weight_column_workers(self):
        assert _weight_column("workers") == "weight"

    def test_weight_column_income_mass(self):
        assert _weight_column("income_mass") == "income_mass"


# ---------------------------------------------------------------------------
# Year availability
# ---------------------------------------------------------------------------

class TestYearAvailability:
    def test_population_accepts_2000(self):
        years = _available_years_for_mode("population", [2000, 2010, 2020])
        assert 2000 in years

    def test_jobs_filters_pre_2002(self):
        years = _available_years_for_mode("jobs", [2000, 2005, 2010, 2020])
        assert 2000 not in years
        assert 2005 in years

    def test_jobs_filters_post_lodes_cap(self):
        # LODES8 currently runs through 2023; 2024 not yet released
        years = _available_years_for_mode("jobs", [2021, 2022, 2023, 2024])
        assert 2024 not in years
        assert 2023 in years
        assert 2021 in years

    def test_income_mass_requires_2009_plus(self):
        years = _available_years_for_mode("income_mass", [2000, 2008, 2009, 2010])
        assert 2000 not in years
        assert 2008 not in years
        assert 2009 in years


# ---------------------------------------------------------------------------
# Crosswalk flag
# ---------------------------------------------------------------------------

class TestCrosswalk:
    def test_pre_2020_needs_crosswalk(self):
        assert _needs_crosswalk(2010) is True
        assert _needs_crosswalk(2015) is True
        assert _needs_crosswalk(2019) is True

    def test_2020_no_crosswalk(self):
        assert _needs_crosswalk(2020) is False

    def test_post_2020_no_crosswalk(self):
        assert _needs_crosswalk(2023) is False


# ---------------------------------------------------------------------------
# Weighted centroid calculation
# ---------------------------------------------------------------------------

class TestWeightedCentroid:
    def _make_test_data(self):
        tracts = gpd.GeoDataFrame(
            {
                "GEOID": ["48001000100", "48001000200", "48001000300", "48001000400"],
                "geometry": [
                    Polygon([(-97.5, 32.5), (-97.0, 32.5), (-97.0, 33.0), (-97.5, 33.0)]),
                    Polygon([(-97.0, 32.5), (-96.5, 32.5), (-96.5, 33.0), (-97.0, 33.0)]),
                    Polygon([(-97.5, 33.0), (-97.0, 33.0), (-97.0, 33.5), (-97.5, 33.5)]),
                    Polygon([(-97.0, 33.0), (-96.5, 33.0), (-96.5, 33.5), (-97.0, 33.5)]),
                ],
            },
            crs="EPSG:4326",
        )
        return tracts

    def test_equal_weights_gives_geographic_center(self):
        tracts = self._make_test_data()
        pop = pd.DataFrame({
            "GEOID": ["48001000100", "48001000200", "48001000300", "48001000400"],
            "population": [1000, 1000, 1000, 1000],
        })
        lat, lon, total, count = _compute_weighted_centroid(tracts, pop)
        assert total == 4000
        assert count == 4
        assert abs(lat - 33.0) < 0.05
        assert abs(lon - (-97.0)) < 0.05

    def test_weight_pulls_toward_heavy_tract(self):
        tracts = self._make_test_data()
        pop = pd.DataFrame({
            "GEOID": ["48001000100", "48001000200", "48001000300", "48001000400"],
            "population": [100, 100, 100, 10000],
        })
        lat, lon, total, count = _compute_weighted_centroid(tracts, pop)
        assert total == 10300
        assert lat > 33.0
        assert lon > -97.0

    def test_custom_weight_column(self):
        tracts = self._make_test_data()
        weights = pd.DataFrame({
            "GEOID": ["48001000100", "48001000200", "48001000300", "48001000400"],
            "weight": [100, 100, 100, 10000],
        })
        lat, lon, total, count = _compute_weighted_centroid(tracts, weights, weight_col="weight")
        assert total == 10300
        assert count == 4

    def test_income_mass_column(self):
        tracts = self._make_test_data()
        weights = pd.DataFrame({
            "GEOID": ["48001000100", "48001000200"],
            "income_mass": [50_000_000, 150_000_000],
        })
        lat, lon, total, count = _compute_weighted_centroid(tracts, weights, weight_col="income_mass")
        assert total == 200_000_000
        assert count == 2

    def test_no_matching_tracts_raises(self):
        tracts = self._make_test_data()
        pop = pd.DataFrame({"GEOID": ["99999999999"], "population": [1000]})
        with pytest.raises(ValueError, match="No matching tracts"):
            _compute_weighted_centroid(tracts, pop)

    def test_partial_match(self):
        tracts = self._make_test_data()
        pop = pd.DataFrame({
            "GEOID": ["48001000100", "48001000200", "48001999999"],
            "population": [5000, 5000, 9999],
        })
        lat, lon, total, count = _compute_weighted_centroid(tracts, pop)
        assert total == 10000
        assert count == 2


# ---------------------------------------------------------------------------
# Haversine and bearing
# ---------------------------------------------------------------------------

class TestGeoMath:
    def test_haversine_same_point(self):
        assert _haversine_miles(33.0, -97.0, 33.0, -97.0) == 0.0

    def test_haversine_known_distance(self):
        dist = _haversine_miles(32.78, -96.80, 32.75, -97.33)
        assert 25 < dist < 35

    def test_haversine_symmetric(self):
        d1 = _haversine_miles(32.0, -96.0, 33.0, -97.0)
        d2 = _haversine_miles(33.0, -97.0, 32.0, -96.0)
        assert abs(d1 - d2) < 0.001

    def test_bearing_north(self):
        bearing = _bearing_degrees(32.0, -97.0, 33.0, -97.0)
        assert abs(bearing - 0) < 1 or abs(bearing - 360) < 1

    def test_bearing_east(self):
        bearing = _bearing_degrees(32.0, -97.0, 32.0, -96.0)
        assert abs(bearing - 90) < 2

    def test_bearing_south(self):
        bearing = _bearing_degrees(33.0, -97.0, 32.0, -97.0)
        assert abs(bearing - 180) < 1

    def test_bearing_west(self):
        bearing = _bearing_degrees(32.0, -96.0, 32.0, -97.0)
        assert abs(bearing - 270) < 2


# ---------------------------------------------------------------------------
# Default year selection
# ---------------------------------------------------------------------------

class TestDefaultYears:
    def test_includes_decennial(self):
        years = _default_years(2000, 2023)
        assert 2000 in years
        assert 2010 in years
        assert 2020 in years

    def test_respects_bounds(self):
        years = _default_years(2010, 2020)
        assert all(2010 <= y <= 2020 for y in years)

    def test_sorted(self):
        years = _default_years(2000, 2023)
        assert years == sorted(years)

    def test_no_duplicates(self):
        years = _default_years(2000, 2023)
        assert len(years) == len(set(years))

    def test_reasonable_count(self):
        years = _default_years(2000, 2023)
        assert 5 <= len(years) <= 15


# ---------------------------------------------------------------------------
# Shift metrics
# ---------------------------------------------------------------------------

class TestShiftMetrics:
    def test_base_year_zero_shift(self):
        df = pd.DataFrame({
            "year": [2000, 2010, 2020],
            "center_lat": [33.0, 33.01, 33.02],
            "center_lon": [-97.0, -96.99, -96.98],
            "total_weight": [1000000, 1100000, 1200000],
            "msa_name": ["Test MSA"] * 3,
        })
        result = compute_shift_metrics(df)
        assert result.iloc[0]["shift_miles"] == 0.0
        assert result.iloc[0]["shift_north_miles"] == 0.0
        assert result.iloc[0]["shift_east_miles"] == 0.0

    def test_northward_shift_positive(self):
        df = pd.DataFrame({
            "year": [2000, 2020],
            "center_lat": [33.0, 33.1],
            "center_lon": [-97.0, -97.0],
            "total_weight": [1000000, 1200000],
            "msa_name": ["Test MSA"] * 2,
        })
        result = compute_shift_metrics(df)
        assert result.iloc[1]["shift_north_miles"] > 0
        assert abs(result.iloc[1]["shift_east_miles"]) < 0.01

    def test_eastward_shift_positive(self):
        df = pd.DataFrame({
            "year": [2000, 2020],
            "center_lat": [33.0, 33.0],
            "center_lon": [-97.0, -96.9],
            "total_weight": [1000000, 1200000],
            "msa_name": ["Test MSA"] * 2,
        })
        result = compute_shift_metrics(df)
        assert result.iloc[1]["shift_east_miles"] > 0
        assert abs(result.iloc[1]["shift_north_miles"]) < 0.01

    def test_empty_df(self):
        df = pd.DataFrame(columns=["year", "center_lat", "center_lon", "total_weight"])
        result = compute_shift_metrics(df)
        assert result.empty


# ---------------------------------------------------------------------------
# GravityResult dataclass
# ---------------------------------------------------------------------------

class TestGravityResult:
    def test_to_dict(self):
        r = GravityResult(
            msa_name="Test MSA", year=2020, weight_mode="population",
            total_weight=1000000, center_lat=33.0, center_lon=-97.0,
            county_count=5, tract_count=100, data_source="Decennial",
            tiger_vintage=2020, notes="test",
        )
        d = r.to_dict()
        assert d["msa_name"] == "Test MSA"
        assert d["weight_mode"] == "population"
        assert d["total_weight"] == 1000000

    def test_weight_mode_field(self):
        r = GravityResult(
            msa_name="Test", year=2020, weight_mode="jobs",
            total_weight=500000, center_lat=33.0, center_lon=-97.0,
            county_count=3, tract_count=50, data_source="LODES",
            tiger_vintage=2020,
        )
        assert r.weight_mode == "jobs"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_creates_files(self, tmp_path):
        df = pd.DataFrame({
            "year": [2020],
            "center_lat": [33.0],
            "center_lon": [-97.0],
            "total_weight": [1000000],
            "weight_mode": ["population"],
            "msa_name": ["Test"],
            "county_count": [5],
            "tract_count": [100],
            "data_source": ["Decennial"],
            "tiger_vintage": [2020],
            "notes": [""],
        })
        paths = export_results(df, tmp_path)
        assert len(paths) == 3
        assert (tmp_path / "gravity_centers.csv").exists()
        assert (tmp_path / "gravity_centers.parquet").exists()
        assert (tmp_path / "gravity_centers.geojson").exists()

    def test_custom_prefix(self, tmp_path):
        df = pd.DataFrame({
            "year": [2020],
            "center_lat": [33.0],
            "center_lon": [-97.0],
            "total_weight": [1000000],
            "weight_mode": ["population"],
            "msa_name": ["Test"],
            "county_count": [5],
            "tract_count": [100],
            "data_source": ["Decennial"],
            "tiger_vintage": [2020],
            "notes": [""],
        })
        export_results(df, tmp_path, prefix="gravity_multi")
        assert (tmp_path / "gravity_multi.csv").exists()

    def test_geojson_is_valid(self, tmp_path):
        df = pd.DataFrame({
            "year": [2020],
            "center_lat": [33.0],
            "center_lon": [-97.0],
            "total_weight": [1000000],
            "weight_mode": ["population"],
            "msa_name": ["Test"],
            "county_count": [5],
            "tract_count": [100],
            "data_source": ["Decennial"],
            "tiger_vintage": [2020],
            "notes": [""],
        })
        export_results(df, tmp_path, formats=["geojson"])
        gdf = gpd.read_file(tmp_path / "gravity_centers.geojson")
        assert len(gdf) == 1
        assert gdf.geometry.iloc[0].geom_type == "Point"
        assert gdf.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# Vintage-aware crosswalk (2000→2020 chain for pre-2010 data)
# ---------------------------------------------------------------------------

class TestVintageCrosswalk:
    def _patch_loaders(self, monkeypatch):
        import geostack.trends as trends
        xw2010 = pd.DataFrame({
            "geoid_2010": ["48085000100", "48085000100"],
            "geoid_2020": ["48085000101", "48085000102"],
            "areal_weight": [0.5, 0.5],
        })
        xw2000 = pd.DataFrame({
            "geoid_2000": ["48085999900", "48085999900"],
            "geoid_2020": ["48085000101", "48085000102"],
            "areal_weight": [0.6, 0.4],
        })
        monkeypatch.setattr(trends, "load_crosswalk", lambda sf, **k: xw2010)
        monkeypatch.setattr(trends, "load_crosswalk_2000_2020", lambda sf, **k: xw2000)

    def test_year_selects_vintage(self, monkeypatch):
        from geostack.gravity import _crosswalk_for_year
        self._patch_loaders(monkeypatch)
        xw09 = _crosswalk_for_year("48", 2009)
        assert set(xw09["geoid_src"]) == {"48085999900"}   # 2000-vintage chain
        xw13 = _crosswalk_for_year("48", 2013)
        assert set(xw13["geoid_src"]) == {"48085000100"}   # 2010->2020 file

    def test_population_conserved_through_chain(self, monkeypatch):
        from geostack.gravity import _apply_population_crosswalk
        self._patch_loaders(monkeypatch)
        pop = pd.DataFrame({"GEOID": ["48085999900"], "population": [1000]})
        out = _apply_population_crosswalk(pop, "48", year=2009)
        assert out["population"].sum() == 1000               # conserved
        assert set(out["GEOID"]) == {"48085000101", "48085000102"}
        assert out.set_index("GEOID").loc["48085000101", "population"] == 600

    def test_income_median_passthrough_chain(self, monkeypatch):
        from geostack.gravity import _apply_income_mass_crosswalk
        self._patch_loaders(monkeypatch)
        inc = pd.DataFrame({"GEOID": ["48085999900"], "income_mass": [80_000_000.0],
                            "occupied_units": [1000.0], "mhi": [80_000.0]})
        out = _apply_income_mass_crosswalk(inc, "48", year=2009)
        # Units split 60/40, median passes through to both pieces
        assert abs(out["occupied_units"].sum() - 1000) < 1e-6
        assert (abs(out["mhi"] - 80_000.0) < 1e-6).all()

    def test_unmatched_geoid_passes_through(self, monkeypatch):
        from geostack.gravity import _apply_population_crosswalk
        self._patch_loaders(monkeypatch)
        pop = pd.DataFrame({"GEOID": ["48113111111"], "population": [500]})
        out = _apply_population_crosswalk(pop, "48", year=2009)
        assert out.iloc[0]["GEOID"] == "48113111111"  # 1:1 fallback
        assert out.iloc[0]["population"] == 500
