"""Tests for geostack.location_score — composite location quality index."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture
def mock_component_scores():
    return {
        "income_depth": 35.0, "employment_access": 55.0,
        "supply_pressure": 70.0, "school_quality": 40.0,
        "transit_access": 20.0, "rent_burden": 45.0,
        "migration_growth": 60.0,
    }


DEFAULT_WEIGHTS = {
    "income_depth": 0.25, "employment_access": 0.20,
    "supply_pressure": 0.15, "school_quality": 0.10,
    "transit_access": 0.10, "rent_burden": 0.10,
    "migration_growth": 0.10,
}


class TestComputeComposite:
    def test_weighted_average(self, mock_component_scores):
        from geostack.location_score import _compute_composite
        score = _compute_composite(mock_component_scores, DEFAULT_WEIGHTS)
        # 35*0.25 + 55*0.20 + 70*0.15 + 40*0.10 + 20*0.10 + 45*0.10 + 60*0.10 = 46.75
        assert score == pytest.approx(46.75)

    def test_weights_must_sum_to_one(self):
        from geostack.location_score import _compute_composite
        bad_weights = {k: 0.5 for k in DEFAULT_WEIGHTS}
        with pytest.raises(ValueError, match="sum to 1.0"):
            _compute_composite({"income_depth": 50}, bad_weights)

    def test_perfect_score(self):
        from geostack.location_score import _compute_composite
        perfect = {k: 100.0 for k in DEFAULT_WEIGHTS}
        score = _compute_composite(perfect, DEFAULT_WEIGHTS)
        assert score == pytest.approx(100.0)


class TestNormalizeScore:
    def test_min_max_normalization(self):
        from geostack.location_score import _normalize
        assert _normalize(50, 0, 100) == pytest.approx(50.0)

    def test_normalize_clamps_to_bounds(self):
        from geostack.location_score import _normalize
        assert _normalize(150, 0, 100) == pytest.approx(100.0)
        assert _normalize(-10, 0, 100) == pytest.approx(0.0)

    def test_inverted_normalization(self):
        from geostack.location_score import _normalize
        assert _normalize(80, 0, 100, invert=True) == pytest.approx(20.0)


class TestRadarChart:
    def test_radar_chart_returns_figure(self, mock_component_scores):
        from geostack.location_score import plot_location_score
        fig = plot_location_score(mock_component_scores, composite=46.75,
                                   property_name="Test Property")
        assert fig is not None
        plt.close(fig)
