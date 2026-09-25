"""Tests for geostack.submarkets — data-driven tract clustering."""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box


@pytest.fixture
def synthetic_tract_features():
    """20 synthetic tracts with 4 features, forming 2 natural clusters."""
    np.random.seed(42)
    n = 20
    renter_income = np.concatenate([np.random.normal(35000, 5000, 10), np.random.normal(70000, 8000, 10)])
    rent_burden = np.concatenate([np.random.normal(0.55, 0.05, 10), np.random.normal(0.30, 0.05, 10)])
    median_rent = np.concatenate([np.random.normal(900, 100, 10), np.random.normal(1500, 150, 10)])
    housing_vintage = np.concatenate([np.random.normal(0.60, 0.10, 10), np.random.normal(0.20, 0.08, 10)])
    geoids = [f"48113{str(i).zfill(6)}" for i in range(n)]
    geoms = [box(-96.8 + i * 0.02, 32.7, -96.78 + i * 0.02, 32.72) for i in range(n)]
    return gpd.GeoDataFrame(
        {"geoid": geoids, "renter_median_income": renter_income, "rent_burden_pct": rent_burden,
         "median_rent": median_rent, "pct_built_pre1990": housing_vintage},
        geometry=geoms, crs="EPSG:4326",
    )


class TestClusterTracts:
    def test_returns_geodataframe_with_cluster_id(self, synthetic_tract_features):
        from geostack.submarkets import cluster_tracts
        result = cluster_tracts(synthetic_tract_features, n_clusters=2)
        assert isinstance(result, gpd.GeoDataFrame)
        assert "cluster_id" in result.columns
        assert len(result) == 20

    def test_correct_number_of_clusters(self, synthetic_tract_features):
        from geostack.submarkets import cluster_tracts
        result = cluster_tracts(synthetic_tract_features, n_clusters=3)
        assert result["cluster_id"].nunique() == 3

    def test_two_clusters_separate_income_groups(self, synthetic_tract_features):
        from geostack.submarkets import cluster_tracts
        result = cluster_tracts(synthetic_tract_features, n_clusters=2)
        cluster_means = result.groupby("cluster_id")["renter_median_income"].mean()
        assert cluster_means.max() > 60000
        assert cluster_means.min() < 45000

    def test_returns_silhouette_score_in_attrs(self, synthetic_tract_features):
        from geostack.submarkets import cluster_tracts
        result = cluster_tracts(synthetic_tract_features, n_clusters=2)
        assert "silhouette_score" in result.attrs
        assert 0 < result.attrs["silhouette_score"] <= 1.0


class TestSubmarketProfile:
    def test_profile_returns_per_cluster_stats(self, synthetic_tract_features):
        from geostack.submarkets import cluster_tracts, submarket_profile
        clustered = cluster_tracts(synthetic_tract_features, n_clusters=2)
        profile = submarket_profile(clustered)
        assert isinstance(profile, pd.DataFrame)
        assert len(profile) == 2
        assert "renter_median_income" in profile.columns
        assert "tract_count" in profile.columns


class TestPropertySubmarket:
    def test_assigns_property_to_cluster(self, synthetic_tract_features):
        from geostack.submarkets import cluster_tracts, property_submarket
        clustered = cluster_tracts(synthetic_tract_features, n_clusters=2)
        result = property_submarket(-96.79, 32.71, clustered)
        assert isinstance(result, dict)
        assert "cluster_id" in result
        assert result["cluster_id"] is not None
