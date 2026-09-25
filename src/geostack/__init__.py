"""Geostack — shared geospatial analytics library for CRE analysis."""

__version__ = "0.1.0"

# Core DB
from geostack.db import get_engine, read_postgis, read_sql, write_postgis

# Census / ACS
from geostack.census import (
    ACS_VARIABLES,
    TX_COUNTIES,
    CensusClient,
    get_tract_demographics,
    get_county_acs_summary,
    get_county_acs_summary_direct,
)

# LODES employment
from geostack.lodes import (
    download_wac,
    download_rac,
    download_od,
    tract_employment,
    commute_shed,
)

# DuckDB fast analytics
from geostack.duckdb_analytics import (
    employment_by_sector,
    top_employment_centers,
    od_flows,
    commute_distance_distribution,
)

# Spatial utilities
from geostack.spatial import (
    suggest_crs,
    compute_acreage,
    parcels_within_radius,
)

# Visualization
from geostack.viz import (
    choropleth_map,
    employment_map,
    static_map,
)

# Routing
from geostack.routing import (
    drive_time,
    drive_time_matrix,
    isochrone,
)

# Demographics (isochrone-based)
from geostack.demographics import (
    tracts_within_isochrone,
    isochrone_demographics,
    isochrone_employment,
    isochrone_commute_shed,
    acs_time_series,
)

# QCEW (employment & wages by NAICS)
from geostack.qcew import (
    fetch_qcew,
    top_employers_by_naics,
    wage_distribution,
)

# FIPS lookups
from geostack.fips import (
    county_name,
    county_name_bulk,
    state_name,
)

# Parcel analytics
from geostack.analysis import (
    compute_ilr,
    compute_age,
    clean_numeric,
    clean_year,
    normalize_address,
    ilr_trend,
    aggregate_ilr_to_tracts,
)

# Affordability / rent-to-income
from geostack.affordability import (
    isochrone_renter_income,
    county_renter_income,
    resident_commute_profile,
    rent_ceiling_comparison,
    plot_income_comparison,
    affordability_capture_curve,
)

# CAD parcel fetch / normalize / load
from geostack.cad import (
    fetch_parcels,
    normalize_parcels,
    load_parcels_to_postgis,
    fetch_collin_parcels,
    fetch_trueprodigy_roll,
    normalize_trueprodigy,
    fetch_cooke_parcels_trueprodigy,
    TRUEPRODIGY_SOURCES,
    COUNTY_REGISTRY,
    COLLIN_DATASET_IDS,
    TAD_DATASET_URLS,
    DCAD_DATASET_URLS,
)

# School district quality
from geostack.schools import (
    school_profile,
)

# Supply pipeline (BPS permits + CoStar comps)
from geostack.supply import (
    supply_pipeline,
    load_costar_comps,
)

# Migration flows (IRS SOI + ACS B07)
from geostack.migration import (
    soi_migration,
    acs_mobility_profile,
    migration_profile,
)

# Property tax (TX Comptroller)
from geostack.property_tax import (
    effective_tax_rate,
)

# Transit + walkability
from geostack.transit import (
    walk_score,
    transit_profile,
)

# Flood / climate risk
from geostack.risk import (
    flood_zone,
)

# Submarket clustering
from geostack.submarkets import (
    cluster_tracts,
    submarket_profile,
    property_submarket,
    classify_tracts,
    plot_submarket_map,
    ECON_CLASSES,
)

# Demographic projections
from geostack.projections import (
    county_income_projection,
    county_demand_forecast,
    tract_growth_profile,
)

# Population center of gravity
from geostack.gravity import (
    MSA_DEFINITIONS,
    WEIGHT_MODES,
    population_center_of_gravity,
    gravity_time_series,
    multi_weight_time_series,
    compute_shift_metrics,
    project_center,
    growth_share,
    get_landmarks,
    export_results as export_gravity_results,
)

# Composite location score
from geostack.location_score import (
    location_score,
    plot_location_score,
)

# Tract-level trends (ACS time series)
from geostack.trends import (
    compute_cagr,
    load_cpi_deflators,
    load_crosswalk,
    apply_crosswalk,
    deflate_to_constant_dollars,
    compute_tract_trends,
    compute_catchment_trends,
    calibrate_thresholds,
    plot_trend_map,
)

# Geocoder (Census Batch)
from geostack.geocoder import (
    geocode_addresses,
    geocode_batch,
)

# Environmental screening (EPA ECHO + CIMC)
from geostack.environmental import (
    echo_facility_search,
    cimc_cleanup_sites,
    screen_site,
)

# Zoning risk analysis
from geostack.zoning import (
    classify_zone,
    zoning_lookup,
    zoning_within_radius,
    zoning_risk,
    ZONING_REGISTRY,
)

# --- Walker-ecosystem modules (optional — import on demand) ---
# These don't fail at import time; they raise ImportError when functions are called
# without the corresponding optional dependencies installed.
#
# Census (pygris + censusdis):  pip install 'geostack[census-walker]'
#   from geostack.census_walker import get_tracts, get_acs_with_geometry, radius_demographics
#
# Multi-provider routing (routingpy):  pip install 'geostack[routing-multi]'
#   from geostack.routing_multi import isochrone_multi, isochrones_nested, list_providers
#
# Interactive maps (maplibregl):  pip install 'geostack[maplibre-interactive]'
#   from geostack.viz_maplibre import demographic_map, radius_ring_map, isochrone_map
#
# Print-quality maps (matplotlib + contextily):  pip install 'geostack[print-maps]'
#   from geostack.viz_static import broker_radius_map, broker_isochrone_map, save_map
#
# H3 hexagonal grid layers (h3):  pip install 'geostack[h3]'
#   from geostack.h3grid import (
#       interpolate_polygons_to_h3,        # area-weighted tract → hex
#       dasymetric_polygons_to_h3,         # footprint/point-weighted tract → hex
#       points_to_h3, msa_h3_layer,
#       h3_choropleth, h3_choropleth_static,
#   )
#   Example: scripts/dfw_h3_parcels_example.py (real DCAD parcels → H3)

__all__ = [
    # DB
    "get_engine", "read_postgis", "read_sql", "write_postgis",
    # Census
    "ACS_VARIABLES", "TX_COUNTIES", "CensusClient",
    "get_tract_demographics", "get_county_acs_summary", "get_county_acs_summary_direct",
    # LODES
    "download_wac", "download_rac", "download_od",
    "tract_employment", "commute_shed",
    # DuckDB
    "employment_by_sector", "top_employment_centers",
    "od_flows", "commute_distance_distribution",
    # Spatial
    "suggest_crs", "compute_acreage", "parcels_within_radius",
    # Viz
    "choropleth_map", "employment_map", "static_map",
    # Routing
    "drive_time", "drive_time_matrix", "isochrone",
    # Demographics
    "tracts_within_isochrone", "isochrone_demographics",
    "isochrone_employment", "isochrone_commute_shed", "acs_time_series",
    # QCEW
    "fetch_qcew", "top_employers_by_naics", "wage_distribution",
    # FIPS
    "county_name", "county_name_bulk", "state_name",
    # Analysis
    "compute_ilr", "compute_age", "clean_numeric", "clean_year",
    "normalize_address", "ilr_trend", "aggregate_ilr_to_tracts",
    # Affordability
    "isochrone_renter_income", "county_renter_income",
    "resident_commute_profile", "rent_ceiling_comparison", "plot_income_comparison",
    "affordability_capture_curve",
    # CAD
    "fetch_parcels", "normalize_parcels", "load_parcels_to_postgis",
    "fetch_collin_parcels", "COUNTY_REGISTRY", "COLLIN_DATASET_IDS",
    "fetch_trueprodigy_roll", "normalize_trueprodigy",
    "fetch_cooke_parcels_trueprodigy", "TRUEPRODIGY_SOURCES",
    # Schools
    "school_profile",
    # Supply
    "supply_pipeline", "load_costar_comps",
    # Migration
    "soi_migration", "acs_mobility_profile", "migration_profile",
    # Property Tax
    "effective_tax_rate",
    # Transit
    "walk_score", "transit_profile",
    # Risk
    "flood_zone",
    # Submarkets
    "cluster_tracts", "submarket_profile", "property_submarket",
    "classify_tracts", "plot_submarket_map", "ECON_CLASSES",
    # Projections
    "county_income_projection",
    "county_demand_forecast",
    "tract_growth_profile",
    # Gravity
    "MSA_DEFINITIONS",
    "WEIGHT_MODES",
    "population_center_of_gravity",
    "gravity_time_series",
    "multi_weight_time_series",
    "compute_shift_metrics",
    "project_center",
    "growth_share",
    "get_landmarks",
    "export_gravity_results",
    # Location Score
    "location_score", "plot_location_score",
    # Trends
    "compute_cagr",
    "load_cpi_deflators", "load_crosswalk", "apply_crosswalk",
    "deflate_to_constant_dollars", "compute_tract_trends",
    "compute_catchment_trends", "calibrate_thresholds", "plot_trend_map",
    # Geocoder
    "geocode_addresses", "geocode_batch",
    # Environmental
    "echo_facility_search", "cimc_cleanup_sites", "screen_site",
    # Zoning
    "classify_zone", "zoning_lookup", "zoning_within_radius",
    "zoning_risk", "ZONING_REGISTRY",
    # Walker-ecosystem modules (import from submodule directly)
    "census_walker", "routing_multi", "viz_maplibre", "viz_static",
]
