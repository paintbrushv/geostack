"""Census data access via Walker-ecosystem libraries (pygris + censusdis).

Drop-in alternative to geostack.census — uses pygris for TIGER boundary
geometries and censusdis for ACS data access + area-weighted aggregation.
Existing census.py (PostGIS-backed) is unchanged; use whichever fits your
workflow.

Install:  pip install 'geostack[census-walker]'

Usage:
    from geostack.census_walker import (
        get_tracts,
        get_block_groups,
        get_acs_data,
        get_acs_with_geometry,
        radius_demographics,
    )

    # Tract boundaries as GeoDataFrame (no PostGIS needed)
    tracts = get_tracts(state="48", county="113", year=2023)

    # ACS data with geometry, ready for mapping
    gdf = get_acs_with_geometry(
        variables=["B19013_001E", "B01003_001E"],
        state="48", county="113", year=2023,
    )

    # Area-weighted demographics within a radius
    stats = radius_demographics(
        lon=-96.77, lat=32.78,
        radii_miles=[1, 3, 5],
        variables=["B19013_001E", "B01003_001E"],
        state="48", county="113",
    )
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
from rich.console import Console
from shapely.geometry import Point

console = Console()


# ---------------------------------------------------------------------------
# Boundary geometry via pygris (replaces PostGIS census_tracts table)
# ---------------------------------------------------------------------------

def get_tracts(
    state: str = "48",
    county: str | None = None,
    year: int = 2023,
    cb: bool = True,
) -> gpd.GeoDataFrame:
    """Fetch Census tract boundaries as a GeoDataFrame via pygris.

    Args:
        state: State FIPS code (e.g., "48" for Texas).
        county: County FIPS code. If None, returns all tracts in state.
        year: TIGER/Line vintage year.
        cb: If True, use cartographic boundary files (smaller, clipped to
            coastline). If False, use full TIGER/Line shapefiles.

    Returns:
        GeoDataFrame with columns: GEOID, NAME, geometry (EPSG:4326).
    """
    try:
        import pygris
    except ImportError:
        raise ImportError(
            "pygris is required. Run: pip install 'geostack[census-walker]'"
        )

    console.print(f"[dim]pygris: fetching tracts state={state} county={county} year={year}[/dim]")
    gdf = pygris.tracts(state=state, county=county, year=year, cb=cb)
    gdf = gdf.to_crs("EPSG:4326")

    # Normalize column names to match geostack conventions
    col_map = {}
    for col in gdf.columns:
        if col.upper() == "GEOID":
            col_map[col] = "geoid"
        elif col.upper() == "NAME":
            col_map[col] = "name"
    gdf = gdf.rename(columns=col_map)

    console.print(f"[dim]pygris: {len(gdf)} tracts loaded[/dim]")
    return gdf


def get_block_groups(
    state: str = "48",
    county: str | None = None,
    year: int = 2023,
    cb: bool = True,
) -> gpd.GeoDataFrame:
    """Fetch Census block group boundaries via pygris.

    Same interface as get_tracts() but at block-group resolution.
    """
    try:
        import pygris
    except ImportError:
        raise ImportError(
            "pygris is required. Run: pip install 'geostack[census-walker]'"
        )

    console.print(
        f"[dim]pygris: fetching block groups state={state} county={county} year={year}[/dim]"
    )
    gdf = pygris.block_groups(state=state, county=county, year=year, cb=cb)
    gdf = gdf.to_crs("EPSG:4326")

    col_map = {}
    for col in gdf.columns:
        if col.upper() == "GEOID":
            col_map[col] = "geoid"
        elif col.upper() == "NAME":
            col_map[col] = "name"
    gdf = gdf.rename(columns=col_map)

    console.print(f"[dim]pygris: {len(gdf)} block groups loaded[/dim]")
    return gdf


# ---------------------------------------------------------------------------
# ACS data via censusdis (replaces CensusClient + PostGIS acs_tract_data)
# ---------------------------------------------------------------------------

def get_acs_data(
    variables: list[str],
    state: str = "48",
    county: str | None = None,
    year: int = 2023,
    geography: str = "tract",
) -> pd.DataFrame:
    """Fetch ACS 5-year estimates via censusdis.

    Args:
        variables: Census variable codes (e.g., ["B19013_001E", "B01003_001E"]).
        state: State FIPS code.
        county: County FIPS code (optional).
        year: ACS vintage year.
        geography: "tract" or "block group".

    Returns:
        DataFrame with GEOID + requested variable columns.
    """
    try:
        import censusdis.data as ced
        from censusdis.datasets import ACS5
    except ImportError:
        raise ImportError(
            "censusdis is required. Run: pip install 'geostack[census-walker]'"
        )

    console.print(
        f"[dim]censusdis: fetching {len(variables)} vars, "
        f"state={state}, county={county}, year={year}[/dim]"
    )

    geo_kwargs: dict[str, Any] = {"state": state}
    if county:
        geo_kwargs["county"] = county

    # censusdis geography level names
    geo_level = "tract" if geography == "tract" else "block group"

    df = ced.download(
        dataset=ACS5,
        vintage=year,
        download_variables=variables,
        **geo_kwargs,
        **{geo_level: "*"},
    )

    # Build GEOID from component columns
    if "STATE" in df.columns and "COUNTY" in df.columns:
        if geo_level == "tract" and "TRACT" in df.columns:
            df["geoid"] = df["STATE"] + df["COUNTY"] + df["TRACT"]
        elif "BLOCK_GROUP" in df.columns:
            df["geoid"] = df["STATE"] + df["COUNTY"] + df["TRACT"] + df["BLOCK_GROUP"]

    console.print(f"[dim]censusdis: {len(df)} rows returned[/dim]")
    return df


def get_acs_with_geometry(
    variables: list[str],
    state: str = "48",
    county: str | None = None,
    year: int = 2023,
    geography: str = "tract",
) -> gpd.GeoDataFrame:
    """Fetch ACS data with geometry via censusdis — ready for mapping.

    Combines censusdis data download with pygris boundary geometry in a
    single call. No PostGIS required.

    Args:
        variables: Census variable codes.
        state: State FIPS code.
        county: County FIPS code (optional).
        year: ACS vintage year.
        geography: "tract" or "block group".

    Returns:
        GeoDataFrame with geoid, geometry, and variable columns.
    """
    try:
        import censusdis.data as ced
        from censusdis.datasets import ACS5
    except ImportError:
        raise ImportError(
            "censusdis is required. Run: pip install 'geostack[census-walker]'"
        )

    console.print(
        f"[dim]censusdis: fetching {len(variables)} vars with geometry[/dim]"
    )

    geo_kwargs: dict[str, Any] = {"state": state}
    if county:
        geo_kwargs["county"] = county

    geo_level = "tract" if geography == "tract" else "block group"

    gdf = ced.download(
        dataset=ACS5,
        vintage=year,
        download_variables=variables,
        with_geometry=True,
        **geo_kwargs,
        **{geo_level: "*"},
    )

    # Ensure WGS84
    if hasattr(gdf, "crs") and gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Build GEOID
    if "STATE" in gdf.columns and "COUNTY" in gdf.columns:
        if geo_level == "tract" and "TRACT" in gdf.columns:
            gdf["geoid"] = gdf["STATE"] + gdf["COUNTY"] + gdf["TRACT"]
        elif "BLOCK_GROUP" in gdf.columns:
            gdf["geoid"] = gdf["STATE"] + gdf["COUNTY"] + gdf["TRACT"] + gdf["BLOCK_GROUP"]

    console.print(f"[dim]censusdis: {len(gdf)} features with geometry[/dim]")
    return gdf


# ---------------------------------------------------------------------------
# Area-weighted radius demographics (no PostGIS needed)
# ---------------------------------------------------------------------------

def radius_demographics(
    lon: float,
    lat: float,
    radii_miles: list[float] | None = None,
    variables: list[str] | None = None,
    state: str = "48",
    county: str | None = None,
    year: int = 2023,
    geography: str = "tract",
    population_var: str = "B01003_001E",
    raw_codes: bool = False,
) -> dict[float, pd.Series]:
    """Compute area-weighted demographics within concentric radius rings.

    Uses pygris boundaries + censusdis ACS data — no PostGIS required.
    Each tract's contribution is weighted by the fraction of its area
    that falls within the radius buffer.

    Args:
        lon, lat: Center point (WGS84).
        radii_miles: List of radii in miles (default [1, 3, 5]).
        variables: Census variable codes to aggregate. Defaults to
                   population + median household income.
        state: State FIPS code.
        county: County FIPS code (recommended for performance).
        year: ACS vintage year.
        geography: "tract" or "block group".
        population_var: Variable code used as weight for median aggregation.
        raw_codes: If True, keys in returned Series are the raw ACS variable
                   codes (e.g., "B19013_001E") instead of human-readable names.
                   Useful for passing directly to demographic_infographic().

    Returns:
        Dict mapping radius → Series of aggregated demographics.
    """
    from geostack.census import ACS_VARIABLES, _SUM_VARIABLES, _AVG_VARIABLES

    if radii_miles is None:
        radii_miles = [1.0, 3.0, 5.0]

    if variables is None:
        variables = [
            ACS_VARIABLES["total_population"],
            ACS_VARIABLES["median_household_income"],
            ACS_VARIABLES["total_housing_units"],
            ACS_VARIABLES["renter_occupied"],
            ACS_VARIABLES["median_rent"],
        ]

    # Ensure population var is included for weighting
    if population_var not in variables:
        variables = [population_var] + variables

    # Fetch geometry + data in one shot
    gdf = get_acs_with_geometry(
        variables=variables,
        state=state,
        county=county,
        year=year,
        geography=geography,
    )

    # Build reverse lookup: code → human name
    code_to_name = {v: k for k, v in ACS_VARIABLES.items()}

    # Project to a meter-based CRS for accurate buffering
    center = Point(lon, lat)
    center_gs = gpd.GeoSeries([center], crs="EPSG:4326")

    # Use UTM zone for the center point
    utm_crs = center_gs.estimate_utm_crs()
    gdf_proj = gdf.to_crs(utm_crs)
    center_proj = center_gs.to_crs(utm_crs).iloc[0]

    results: dict[float, pd.Series] = {}

    for radius in sorted(radii_miles):
        radius_m = radius * 1609.344
        buffer = center_proj.buffer(radius_m)

        # Compute area weights: fraction of each tract inside the buffer
        gdf_proj = gdf_proj.copy()
        gdf_proj["_tract_area"] = gdf_proj.geometry.area
        gdf_proj["_intersection_area"] = gdf_proj.geometry.intersection(buffer).area
        gdf_proj["_area_weight"] = gdf_proj["_intersection_area"] / gdf_proj["_tract_area"]

        # Filter to tracts with any overlap
        ring = gdf_proj[gdf_proj["_area_weight"] > 0.01].copy()

        if ring.empty:
            console.print(f"[yellow]No tracts found within {radius}mi radius[/yellow]")
            results[radius] = pd.Series(dtype=float)
            continue

        console.print(
            f"[dim]radius {radius}mi: {len(ring)} tracts, "
            f"mean area_weight={ring['_area_weight'].mean():.2f}[/dim]"
        )

        # Aggregate: sum variables get area-weighted sum,
        # median variables get population-weighted average
        agg = {}
        pop_weighted = ring[population_var] * ring["_area_weight"]
        total_weighted_pop = pop_weighted.sum()

        for var in variables:
            name = code_to_name.get(var, var)
            key = var if raw_codes else name
            if name in _SUM_VARIABLES:
                agg[key] = (ring[var] * ring["_area_weight"]).sum()
            elif name in _AVG_VARIABLES:
                if total_weighted_pop > 0:
                    agg[key] = (ring[var] * pop_weighted).sum() / total_weighted_pop
                else:
                    agg[key] = float("nan")
            else:
                # Unknown variable — default to area-weighted sum
                agg[key] = (ring[var] * ring["_area_weight"]).sum()

        agg["tract_count"] = len(ring)
        results[radius] = pd.Series(agg)

    return results


def isochrone_demographics(
    isochrone_gdf: gpd.GeoDataFrame,
    variables: list[str] | None = None,
    state: str = "48",
    county: str | None = None,
    year: int = 2023,
    geography: str = "tract",
    population_var: str = "B01003_001E",
    raw_codes: bool = False,
) -> dict[float, pd.Series]:
    """Compute area-weighted demographics within isochrone polygons.

    Same approach as radius_demographics() but uses drive-time isochrone
    polygons instead of geometric circles.

    Args:
        isochrone_gdf: GeoDataFrame with 'minutes' column and polygon geometry,
                       e.g. from routing_multi.isochrones_nested().
        variables: Census variable codes to aggregate.
        state: State FIPS code.
        county: County FIPS code (recommended for performance).
        year: ACS vintage year.
        geography: "tract" or "block group".
        population_var: Variable code for population weighting.
        raw_codes: If True, keys are raw ACS codes. If False, human-readable.

    Returns:
        Dict mapping minutes (float) → Series of aggregated demographics.
    """
    from geostack.census import ACS_VARIABLES, _SUM_VARIABLES, _AVG_VARIABLES

    if variables is None:
        variables = [
            ACS_VARIABLES["total_population"],
            ACS_VARIABLES["median_household_income"],
            ACS_VARIABLES["total_housing_units"],
            ACS_VARIABLES["renter_occupied"],
            ACS_VARIABLES["median_rent"],
        ]

    if population_var not in variables:
        variables = [population_var] + variables

    # Fetch geometry + data
    gdf = get_acs_with_geometry(
        variables=variables,
        state=state,
        county=county,
        year=year,
        geography=geography,
    )

    code_to_name = {v: k for k, v in ACS_VARIABLES.items()}

    # Project to UTM for accurate area intersection
    iso_wgs = isochrone_gdf.to_crs("EPSG:4326") if isochrone_gdf.crs else isochrone_gdf
    centroid = iso_wgs.geometry.union_all().centroid
    center_gs = gpd.GeoSeries([centroid], crs="EPSG:4326")
    utm_crs = center_gs.estimate_utm_crs()

    gdf_proj = gdf.to_crs(utm_crs)
    iso_proj = iso_wgs.to_crs(utm_crs)

    results: dict[float, pd.Series] = {}

    for _, iso_row in iso_proj.iterrows():
        mins = iso_row["minutes"]
        iso_poly = iso_row.geometry

        gdf_work = gdf_proj.copy()
        gdf_work["_tract_area"] = gdf_work.geometry.area
        gdf_work["_intersection_area"] = gdf_work.geometry.intersection(iso_poly).area
        gdf_work["_area_weight"] = gdf_work["_intersection_area"] / gdf_work["_tract_area"]

        ring = gdf_work[gdf_work["_area_weight"] > 0.01].copy()

        if ring.empty:
            console.print(f"[yellow]No tracts found within {mins}-min isochrone[/yellow]")
            results[mins] = pd.Series(dtype=float)
            continue

        console.print(
            f"[dim]isochrone {mins}min: {len(ring)} tracts, "
            f"mean area_weight={ring['_area_weight'].mean():.2f}[/dim]"
        )

        agg = {}
        pop_weighted = ring[population_var] * ring["_area_weight"]
        total_weighted_pop = pop_weighted.sum()

        for var in variables:
            name = code_to_name.get(var, var)
            key = var if raw_codes else name
            if name in _SUM_VARIABLES:
                agg[key] = (ring[var] * ring["_area_weight"]).sum()
            elif name in _AVG_VARIABLES:
                if total_weighted_pop > 0:
                    agg[key] = (ring[var] * pop_weighted).sum() / total_weighted_pop
                else:
                    agg[key] = float("nan")
            else:
                agg[key] = (ring[var] * ring["_area_weight"]).sum()

        agg["tract_count"] = len(ring)
        results[mins] = pd.Series(agg)

    return results
