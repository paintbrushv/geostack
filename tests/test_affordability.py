# tests/test_affordability.py
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from geostack.affordability import affordability_capture_curve, _build_distribution, RENTER_INCOME_VARS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPUBLICS_LON = -96.6183
REPUBLICS_LAT = 32.8028


def _mock_dist(label="Test", total_hh=5000):
    """Build a minimal income dist DataFrame for mocking."""
    totals = {v: 0 for v in RENTER_INCOME_VARS}
    totals["B25118_021E"] = total_hh  # all in $35K–$50K
    dist = _build_distribution(RENTER_INCOME_VARS, totals, total_hh, label=label)
    return dist

# Synthetic income distribution: 1,000 households, all earning $35K–$50K
# 30% of $35K = $875/mo; 30% of $50K = $1,250/mo
# So at $900/mo rent: only households >$36K can afford → roughly 50%+ of this bracket
# At $1,300/mo rent: need $52K+ → 0% of this bracket

@pytest.fixture
def simple_dist():
    """1,000 renter households all in the $35K–$50K bracket."""
    totals = {v: 0 for v in RENTER_INCOME_VARS}
    totals["B25118_021E"] = 1000  # 35k_50k bracket
    totals["B25118_014E"] = 1000  # renter total
    dist = _build_distribution(RENTER_INCOME_VARS, totals, 1000, label="Test")
    return dist


def test_capture_curve_returns_dataframe(simple_dist):
    result = affordability_capture_curve(simple_dist, rent_range=range(800, 1600, 100))
    assert isinstance(result, pd.DataFrame)
    assert "monthly_rent" in result.columns
    assert "pct_can_afford" in result.columns
    assert "hh_count_can_afford" in result.columns
    assert len(result) == 8  # 800, 900, ..., 1500


def test_capture_curve_monotonically_decreasing(simple_dist):
    result = affordability_capture_curve(simple_dist, rent_range=range(500, 2001, 100))
    # Higher rent = fewer households can afford it
    pcts = result["pct_can_afford"].tolist()
    assert pcts == sorted(pcts, reverse=True)


def test_capture_curve_low_rent_high_capture(simple_dist):
    """At $500/mo, all households in $35K–$50K can afford (need $20K income)."""
    result = affordability_capture_curve(simple_dist, rent_range=[500])
    assert result.iloc[0]["pct_can_afford"] == pytest.approx(1.0)


def test_capture_curve_high_rent_zero_capture(simple_dist):
    """At $2,000/mo, need $80K income — none of $35K–$50K bracket qualifies."""
    result = affordability_capture_curve(simple_dist, rent_range=[2000])
    assert result.iloc[0]["pct_can_afford"] == pytest.approx(0.0)


@patch("geostack.affordability.resident_commute_profile")
@patch("geostack.affordability.rent_burden_profile")
@patch("geostack.affordability.isochrone_renter_income")
@patch("geostack.affordability.isochrone_all_hh_income")
@patch("geostack.affordability.county_renter_income")
@patch("geostack.affordability.county_all_hh_income")
def test_rent_ceiling_comparison_includes_capture_curve(
    mock_cty_all, mock_cty_renter, mock_iso_all, mock_iso_renter,
    mock_burden, mock_commute
):
    mock_cty_all.return_value = _mock_dist("County All HH")
    mock_cty_renter.return_value = _mock_dist("County Renter")
    mock_iso_all.return_value = _mock_dist("Isochrone All HH")
    mock_iso_renter.return_value = _mock_dist("Property Renter")
    mock_burden.return_value = {
        "affordable_pct": 0.45, "burdened_pct": 0.35,
        "severely_burdened_pct": 0.20, "total_renters": 5000, "not_computed": 0,
    }
    mock_commute.return_value = {}

    from geostack.affordability import rent_ceiling_comparison
    result = rent_ceiling_comparison(
        REPUBLICS_LON, REPUBLICS_LAT, 10, "48", "113",
        target_rents=[1100, 1300, 1500],
        current_rent=1100,
        target_rent=1350,
        comp_rents=[1200, 1300],
    )

    assert "capture_curve" in result
    assert isinstance(result["capture_curve"], pd.DataFrame)
    assert "monthly_rent" in result["capture_curve"].columns
    assert result["current_rent"] == 1100
    assert result["target_rent"] == 1350
    assert result["comp_rents"] == [1200, 1300]


@patch("geostack.affordability.resident_commute_profile")
@patch("geostack.affordability.rent_burden_profile")
@patch("geostack.affordability.isochrone_renter_income")
@patch("geostack.affordability.isochrone_all_hh_income")
@patch("geostack.affordability.county_renter_income")
@patch("geostack.affordability.county_all_hh_income")
def test_rent_ceiling_comparison_backward_compat(
    mock_cty_all, mock_cty_renter, mock_iso_all, mock_iso_renter,
    mock_burden, mock_commute
):
    """Omitting current_rent/target_rent/comp_rents should still work."""
    for m in [mock_cty_all, mock_cty_renter, mock_iso_all, mock_iso_renter]:
        m.return_value = _mock_dist()
    mock_burden.return_value = {
        "affordable_pct": 0.5, "burdened_pct": 0.3,
        "severely_burdened_pct": 0.2, "total_renters": 4000, "not_computed": 0,
    }
    mock_commute.return_value = {}

    from geostack.affordability import rent_ceiling_comparison
    result = rent_ceiling_comparison(
        REPUBLICS_LON, REPUBLICS_LAT, 10, "48", "113",
        target_rents=[1100, 1350],
    )
    assert "distributions" in result
    assert result["current_rent"] is None
    assert result["comp_rents"] == []
    assert result["target_rent"] is None
    assert "capture_curve" in result
    assert isinstance(result["capture_curve"], pd.DataFrame)


def _full_comparison():
    """Build a minimal comparison dict for plotting tests."""
    from geostack.affordability import (
        RENTER_INCOME_VARS, _build_distribution, affordability_capture_curve,
        _affordability_at_rent,
    )
    totals = {v: 0 for v in RENTER_INCOME_VARS}
    totals["B25118_021E"] = 2000
    dist = _build_distribution(RENTER_INCOME_VARS, totals, 2000, label="Test")

    rents = [1000, 1100, 1200, 1300]
    aff_rows = [
        {"monthly_rent": r, "method": "property_renter", "label": "Test",
         "pct_can_afford": _affordability_at_rent(dist, r)}
        for r in rents
    ]

    cap = affordability_capture_curve(dist, rent_range=range(800, 1601, 50))

    return {
        "distributions": {"property_renter": dist},
        "affordability": pd.DataFrame(aff_rows),
        "medians": {"property_renter": 42000},
        "rent_ceilings": {"property_renter": 1050},
        "commute": {},
        "rent_burden": {},
        "capture_curve": cap,
        "current_rent": 1100,
        "target_rent": 1350,
        "comp_rents": [1200, 1300],
    }


def test_plot_includes_capture_curve_panel():
    from geostack.affordability import plot_income_comparison
    comp = _full_comparison()
    fig = plot_income_comparison(comp, property_name="Test Property", target_rent=1350)
    # 2 base panels + 1 capture = 3 axes (no commute, no burden in this fixture)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_without_capture_curve_unchanged():
    """When capture_curve is empty, panel count stays at 2 (no commute/burden either)."""
    from geostack.affordability import plot_income_comparison
    comp = _full_comparison()
    comp["capture_curve"] = pd.DataFrame()
    fig = plot_income_comparison(comp, property_name="Test Property", target_rent=1350)
    assert len(fig.axes) == 2
    plt.close(fig)
