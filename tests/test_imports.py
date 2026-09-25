"""Smoke tests: verify all public symbols import from the top-level package."""


def test_affordability_capture_curve_importable():
    from geostack import affordability_capture_curve
    assert callable(affordability_capture_curve)


def test_school_profile_importable():
    from geostack import school_profile
    assert callable(school_profile)


def test_supply_pipeline_importable():
    from geostack import supply_pipeline
    assert callable(supply_pipeline)


def test_load_costar_comps_importable():
    from geostack import load_costar_comps
    assert callable(load_costar_comps)


def test_migration_profile_importable():
    from geostack import migration_profile, soi_migration, acs_mobility_profile
    assert callable(migration_profile)
    assert callable(soi_migration)
    assert callable(acs_mobility_profile)


def test_effective_tax_rate_importable():
    from geostack import effective_tax_rate
    assert callable(effective_tax_rate)


def test_transit_profile_importable():
    from geostack import transit_profile, walk_score
    assert callable(transit_profile)
    assert callable(walk_score)


def test_flood_zone_importable():
    from geostack import flood_zone
    assert callable(flood_zone)


def test_cluster_tracts_importable():
    from geostack import cluster_tracts, submarket_profile, property_submarket
    assert callable(cluster_tracts)
    assert callable(submarket_profile)
    assert callable(property_submarket)


def test_county_income_projection_importable():
    from geostack import county_income_projection
    assert callable(county_income_projection)


def test_location_score_importable():
    from geostack import location_score, plot_location_score
    assert callable(location_score)
    assert callable(plot_location_score)


def test_trends_importable():
    from geostack import (
        load_cpi_deflators, load_crosswalk, apply_crosswalk,
        deflate_to_constant_dollars, compute_tract_trends,
        compute_catchment_trends, calibrate_thresholds,
    )
    assert callable(compute_tract_trends)
    assert callable(compute_catchment_trends)
