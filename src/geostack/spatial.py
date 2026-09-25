"""Spatial analysis utilities.

Common geospatial operations for CRE analysis: buffering, clipping,
road frontage measurement, parcel filtering, area calculations.

Usage:
    from geostack.spatial import measure_road_frontage, parcels_within_radius

    frontage = measure_road_frontage(parcels_gdf, roads_gdf)
    nearby = parcels_within_radius(parcels_gdf, center_point, miles=1.0)
    crs = suggest_crs(my_gdf)
    clean = erase_water(tract_gdf, state_fips="48", county_fips="085")
"""

import os

import geopandas as gpd
import numpy as np
from rich.console import Console
from shapely.geometry import Point

console = Console()


def ensure_projected(gdf, target_crs="EPSG:2276"):
    """Reproject to a projected CRS for distance/area calculations.

    Default EPSG:2276 = Texas State Plane North Central (feet).
    Good for DFW metro area. Use EPSG:32614 for UTM Zone 14N (meters).
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set")
    if gdf.crs.to_epsg() != int(target_crs.split(":")[1]):
        return gdf.to_crs(target_crs)
    return gdf


def measure_road_frontage(parcels, roads, crs="EPSG:2276"):
    """Measure road frontage for each parcel.

    Computes the length of intersection between parcel boundaries
    and nearby road centerlines.

    Args:
        parcels: GeoDataFrame of parcel polygons.
        roads: GeoDataFrame of road centerline LineStrings.
        crs: Projected CRS for distance measurement.

    Returns:
        parcels GeoDataFrame with added 'road_frontage_ft' column.
    """
    parcels_proj = ensure_projected(parcels, crs)
    roads_proj = ensure_projected(roads, crs)

    # Buffer parcels slightly (25 ft) to catch roads near boundary
    parcel_bounds = parcels_proj.copy()
    parcel_bounds["geometry"] = parcel_bounds.geometry.buffer(25)

    # Spatial join to find which roads are near each parcel
    joined = gpd.sjoin(parcel_bounds, roads_proj, how="left", predicate="intersects")

    # For each parcel, compute total intersection length with its nearby roads
    frontages = []
    for idx, parcel in parcels_proj.iterrows():
        nearby_road_idxs = joined.loc[joined.index == idx, "index_right"].dropna()
        if nearby_road_idxs.empty:
            frontages.append(0.0)
            continue

        nearby_roads = roads_proj.iloc[nearby_road_idxs.astype(int).values]
        boundary = parcel.geometry.boundary.buffer(25)
        clipped = nearby_roads.geometry.intersection(boundary)
        total_length = clipped.length.sum()
        frontages.append(round(total_length, 1))

    parcels = parcels.copy()
    parcels["road_frontage_ft"] = frontages
    return parcels


def parcels_within_radius(parcels, center, miles=1.0):
    """Filter parcels within a radius of a point.

    Args:
        parcels: GeoDataFrame of parcels (any CRS, will be reprojected).
        center: (lon, lat) tuple or shapely Point in EPSG:4326.
        miles: Search radius in miles.

    Returns:
        Filtered GeoDataFrame.
    """
    if isinstance(center, tuple):
        center = Point(center[0], center[1])

    center_gdf = gpd.GeoDataFrame(geometry=[center], crs="EPSG:4326")
    center_proj = center_gdf.to_crs("EPSG:2276").geometry.iloc[0]
    parcels_proj = ensure_projected(parcels.copy(), "EPSG:2276")

    radius_ft = miles * 5280
    mask = parcels_proj.geometry.distance(center_proj) <= radius_ft
    return parcels.loc[mask]


def compute_acreage(parcels, crs="EPSG:2276"):
    """Add acreage column based on polygon area.

    Uses projected CRS for accurate area calculation.
    EPSG:2276 is in feet, so area_ft² / 43560 = acres.
    """
    parcels_proj = ensure_projected(parcels.copy(), crs)
    parcels = parcels.copy()
    parcels["calc_acres"] = parcels_proj.geometry.area / 43_560
    return parcels


def parcel_neighbors(parcels, target_idx, buffer_ft=500):
    """Find parcels neighboring a target parcel.

    Useful for comparable analysis — finding nearby parcels
    for relative ILR comparisons.
    """
    parcels_proj = ensure_projected(parcels.copy(), "EPSG:2276")
    target_geom = parcels_proj.geometry.iloc[target_idx]
    search_area = target_geom.buffer(buffer_ft)
    mask = parcels_proj.geometry.intersects(search_area)
    return parcels.loc[mask]


# ---------------------------------------------------------------------------
# CRS utilities
# ---------------------------------------------------------------------------

# Texas State Plane zones differentiated by latitude band (zones are horizontal strips).
# Each entry: (lat_min, lat_max, epsg)
# Approximate zone extents; boundaries overlap slightly — nearest band used when ambiguous.
_TX_STATE_PLANE = [
    (35.2, 37.0, "EPSG:2275"),   # TX North — Panhandle/Amarillo
    (32.0, 35.2, "EPSG:2276"),   # TX North Central — DFW, Lubbock
    (30.0, 32.0, "EPSG:2278"),   # TX Central — Austin, Abilene
    (28.0, 30.0, "EPSG:2279"),   # TX South Central — San Antonio, Corpus Christi
    (25.5, 28.0, "EPSG:2280"),   # TX South — Laredo, Rio Grande Valley
]


def suggest_crs(gdf, units="ft"):
    """Suggest an appropriate projected CRS for a GeoDataFrame.

    Inspects the data's centroid longitude/latitude and returns a CRS suited
    for distance and area operations (State Plane for Texas, UTM otherwise).

    Args:
        gdf: GeoDataFrame in any CRS (will be temporarily reprojected to WGS84).
        units: 'ft' (default, State Plane) or 'm' (UTM). Only affects Texas data.

    Returns:
        EPSG string e.g. 'EPSG:2276'. Pass directly to gdf.to_crs().

    Example:
        crs = suggest_crs(parcels_gdf)
        parcels_proj = parcels_gdf.to_crs(crs)
    """
    wgs = gdf.to_crs("EPSG:4326") if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf
    centroid = wgs.geometry.union_all().centroid
    lon, lat = centroid.x, centroid.y

    # Texas — use State Plane (feet) or UTM (meters)
    if 25.5 <= lat <= 37.0 and -106.7 <= lon <= -93.5:
        if units == "m":
            zone = int((lon + 180) / 6) + 1
            return f"EPSG:{32600 + zone}"
        # TX State Plane zones are horizontal latitude bands
        for lat_min, lat_max, epsg in _TX_STATE_PLANE:
            if lat_min <= lat <= lat_max:
                return epsg
        # Fallback if centroid is at the very edge — return TX North Central
        return "EPSG:2276"

    # General case: UTM Northern or Southern hemisphere
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:{32600 + zone}"
    return f"EPSG:{32700 + zone}"


# ---------------------------------------------------------------------------
# Water erasure
# ---------------------------------------------------------------------------

def erase_water(gdf, state_fips, county_fips, year=2023, area_threshold_pct=75, cache_dir=None):
    """Remove large water bodies from polygon geometries.

    Downloads TIGER area_water file for the county, keeps only water polygons
    in the top `area_threshold_pct` percentile by area (filters out small ponds),
    then subtracts them from the input geometries. Essential before calling
    point-on-surface or routing on coastal/lakefront tracts.

    Args:
        gdf: GeoDataFrame of polygons to clean (tracts, parcels, etc.).
        state_fips: 2-digit state FIPS.
        county_fips: 3-digit county FIPS.
        year: TIGER vintage year.
        area_threshold_pct: Only erase water bodies above this area percentile.
                            75 (default) removes only the largest 25% of water bodies.
                            Raise to 90+ for counties with many small ponds.
        cache_dir: Override default shapefile cache directory.

    Returns:
        GeoDataFrame with water erased, same CRS as input.
        Rows with fully-erased geometry are dropped.
    """
    import zipfile
    import io
    import requests as _requests

    state_fips = str(state_fips).zfill(2)
    county_fips = str(county_fips).zfill(3)
    county_geoid = state_fips + county_fips

    if cache_dir is None:
        cache_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "shapefiles")
        )

    # TIGER area_water URL
    filename = f"tl_{year}_{county_geoid}_areawater.zip"
    extract_dir = os.path.join(cache_dir, filename.replace(".zip", ""))

    if not os.path.exists(extract_dir):
        url = (
            f"https://www2.census.gov/geo/tiger/TIGER{year}/"
            f"AREAWATER/{filename}"
        )
        console.print(f"[dim]Downloading area_water for {county_geoid}...[/dim]")
        resp = _requests.get(url, timeout=120)
        if resp.status_code == 404:
            console.print(f"[yellow]No area_water file for {county_geoid} — returning unchanged.[/yellow]")
            return gdf
        resp.raise_for_status()
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(extract_dir)

    shp_files = [f for f in os.listdir(extract_dir) if f.endswith(".shp")]
    if not shp_files:
        console.print("[yellow]No shapefile found in area_water archive — returning unchanged.[/yellow]")
        return gdf

    water = gpd.read_file(os.path.join(extract_dir, shp_files[0]))
    if water.empty:
        return gdf

    water = water.to_crs(gdf.crs)

    # Keep only large water bodies (top N% by area)
    threshold = water.geometry.area.quantile(area_threshold_pct / 100)
    water = water[water.geometry.area >= threshold]

    if water.empty:
        return gdf

    water_union = water.geometry.union_all()
    result = gdf.copy()
    result["geometry"] = result.geometry.difference(water_union)
    result = result[~result.geometry.is_empty].reset_index(drop=True)

    erased = len(gdf) - len(result)
    console.print(
        f"[green]erase_water: removed {len(water)} water polygons "
        f"(≥ {area_threshold_pct}th pct), {erased} tract(s) fully erased[/green]"
    )
    return result
