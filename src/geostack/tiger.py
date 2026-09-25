"""TIGER/Line shapefile downloader and loader.

Downloads Census TIGER/Line boundary files (tracts, block groups, counties, roads)
and loads them into PostGIS or returns as GeoDataFrames.

Usage:
    from geostack.tiger import download_tracts, load_tracts_to_postgis

    gdf = download_tracts(state_fips="48", year=2023)
    load_tracts_to_postgis(gdf, engine)
"""

import ftplib
import io
import os
import tempfile
import zipfile

import geopandas as gpd
import requests
from rich.console import Console

console = Console()

TIGER_BASE_URL = "https://www2.census.gov/geo/tiger"
TIGER_FTP_HOST = "ftp2.census.gov"

# Cartographic boundary (simplified) base URL — ~10x smaller than full TIGER
# Available for tracts, counties, states, places, CBSAs. Not available for roads.
TIGER_CB_BASE_URL = "https://www2.census.gov/geo/tiger"

# TIGER/Line file types and their URL patterns
TIGER_LAYERS = {
    "tract": "TRACT/tl_{year}_{state}_tract",
    "bg": "BG/tl_{year}_{state}_bg",              # block groups
    "county": "COUNTY/tl_{year}_us_county",
    "place": "PLACE/tl_{year}_{state}_place",      # cities/CDPs
    "roads": "ROADS/tl_{year}_{county_fips}_roads",
    "edges": "EDGES/tl_{year}_{county_fips}_edges",  # all linear features
    "areawater": "AREAWATER/tl_{year}_{county_fips}_areawater",
}


def _download_shapefile(url, cache_dir=None, ftp_path=None):
    """Download and extract a TIGER/Line shapefile zip.

    Tries HTTPS first. If the response is 404 or a connection error occurs,
    falls back to FTP from ftp2.census.gov using ftp_path (if provided).

    Args:
        url: HTTPS URL for the shapefile zip.
        cache_dir: Local directory for caching extracted files.
        ftp_path: FTP path on ftp2.census.gov to retry if HTTPS fails
                  (e.g. '/geo/tiger/TIGER2023/TRACT/tl_2023_48_tract.zip').

    Returns:
        Path to the extracted directory.
    """
    cache_dir = cache_dir or os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "shapefiles"
    )
    os.makedirs(cache_dir, exist_ok=True)

    filename = url.split("/")[-1]
    zip_path = os.path.join(cache_dir, filename)
    extract_dir = os.path.join(cache_dir, filename.replace(".zip", ""))

    if os.path.exists(extract_dir):
        console.print(f"[dim]Using cached: {filename}[/dim]")
        return extract_dir

    # Try HTTPS
    zip_bytes = None
    try:
        console.print(f"Downloading {filename}...")
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code == 404:
            raise requests.HTTPError(f"404 Not Found: {url}")
        resp.raise_for_status()
        zip_bytes = b"".join(resp.iter_content(chunk_size=65536))
    except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
        if ftp_path:
            console.print(f"[yellow]HTTPS failed ({exc}), retrying via FTP...[/yellow]")
            zip_bytes = _ftp_download(ftp_path)
        else:
            raise

    with open(zip_path, "wb") as f:
        f.write(zip_bytes)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    os.remove(zip_path)
    return extract_dir


def _ftp_download(ftp_path):
    """Download a file from ftp2.census.gov and return its bytes."""
    buf = io.BytesIO()
    try:
        ftp = ftplib.FTP(TIGER_FTP_HOST, timeout=120)
        ftp.login()
        ftp.retrbinary(f"RETR {ftp_path}", buf.write)
        ftp.quit()
    except ftplib.all_errors as e:
        raise ConnectionError(f"FTP download failed from {TIGER_FTP_HOST}{ftp_path}: {e}") from e
    return buf.getvalue()


def download_tracts(state_fips, year=2023, cache_dir=None, cb=False, subset_by=None):
    """Download census tract boundaries for a state.

    Args:
        state_fips: 2-digit state FIPS code (e.g., "48" for Texas).
        year: TIGER vintage year.
        cache_dir: Override the default shapefile cache directory.
        cb: If True, download the cartographic boundary (simplified, ~10x smaller)
            version instead of full TIGER precision. Recommended for web maps.
            CB files use 1:500,000 scale generalization.
        subset_by: Optional spatial filter applied after download:
            - tuple (min_lon, min_lat, max_lon, max_lat) — keep tracts intersecting bbox
            - GeoDataFrame or GeoSeries — keep tracts intersecting the union of geometries

    Returns:
        GeoDataFrame with tract polygons and GEOID column.
    """
    if cb:
        # Cartographic boundary URL: GENZ{year}/shp/cb_{year}_{state}_tract_500k.zip
        filename = f"cb_{year}_{state_fips}_tract_500k"
        url = f"{TIGER_CB_BASE_URL}/GENZ{year}/shp/{filename}.zip"
        ftp_path = None  # CB files are not on FTP
    else:
        pattern = TIGER_LAYERS["tract"].format(year=year, state=state_fips)
        filename = pattern.split("/")[-1]
        url = f"{TIGER_BASE_URL}/TIGER{year}/{pattern}.zip"
        ftp_path = f"/geo/tiger/TIGER{year}/{pattern}.zip"

    shp_dir = _download_shapefile(url, cache_dir, ftp_path=ftp_path)

    shp_files = [f for f in os.listdir(shp_dir) if f.endswith(".shp")]
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in {shp_dir}")

    gdf = gpd.read_file(os.path.join(shp_dir, shp_files[0]))
    console.print(f"[green]Loaded {len(gdf)} tracts for state {state_fips}[/green]")

    if subset_by is not None:
        gdf = _apply_subset(gdf, subset_by)
        console.print(f"[dim]subset_by filter: {len(gdf)} tracts retained[/dim]")

    return gdf


def download_block_groups(state_fips, year=2023, cache_dir=None):
    """Download block group boundaries for a state."""
    pattern = TIGER_LAYERS["bg"].format(year=year, state=state_fips)
    url = f"{TIGER_BASE_URL}/TIGER{year}/{pattern}.zip"
    shp_dir = _download_shapefile(url, cache_dir)

    shp_files = [f for f in os.listdir(shp_dir) if f.endswith(".shp")]
    gdf = gpd.read_file(os.path.join(shp_dir, shp_files[0]))
    console.print(f"[green]Loaded {len(gdf)} block groups for state {state_fips}[/green]")
    return gdf


def download_roads(state_fips, county_fips, year=2023, cache_dir=None, cb=False, subset_by=None):
    """Download road centerlines for a specific county.

    Useful for road frontage measurement against parcels.

    Args:
        state_fips: 2-digit state FIPS (e.g., "48").
        county_fips: 3-digit county FIPS (e.g., "085" for Collin).
        year: TIGER vintage year.
        cache_dir: Override the default shapefile cache directory.
        cb: Ignored for roads — cartographic boundary files do not exist for
            road networks. A warning is printed if cb=True is passed.
        subset_by: Optional spatial filter applied after download:
            - tuple (min_lon, min_lat, max_lon, max_lat) — keep segments intersecting bbox
            - GeoDataFrame or GeoSeries — keep segments intersecting the union of geometries
    """
    if cb:
        console.print(
            "[yellow]cb=True ignored for roads — "
            "cartographic boundary files are not available for road networks.[/yellow]"
        )

    full_fips = f"{state_fips}{county_fips}"
    pattern = TIGER_LAYERS["roads"].format(year=year, county_fips=full_fips)
    url = f"{TIGER_BASE_URL}/TIGER{year}/{pattern}.zip"
    ftp_path = f"/geo/tiger/TIGER{year}/{pattern}.zip"
    shp_dir = _download_shapefile(url, cache_dir, ftp_path=ftp_path)

    shp_files = [f for f in os.listdir(shp_dir) if f.endswith(".shp")]
    gdf = gpd.read_file(os.path.join(shp_dir, shp_files[0]))
    console.print(f"[green]Loaded {len(gdf)} road segments for {full_fips}[/green]")

    if subset_by is not None:
        gdf = _apply_subset(gdf, subset_by)
        console.print(f"[dim]subset_by filter: {len(gdf)} road segments retained[/dim]")

    return gdf


def _apply_subset(gdf, subset_by):
    """Filter a GeoDataFrame by a bounding box tuple or GeoDataFrame mask."""
    from shapely.geometry import box as shapely_box

    gdf_wgs = gdf.to_crs("EPSG:4326") if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf

    if isinstance(subset_by, tuple) and len(subset_by) == 4:
        min_lon, min_lat, max_lon, max_lat = subset_by
        bbox_geom = shapely_box(min_lon, min_lat, max_lon, max_lat)
        mask = gdf_wgs.geometry.intersects(bbox_geom)
    elif isinstance(subset_by, (gpd.GeoDataFrame, gpd.GeoSeries)):
        geoseries = subset_by.geometry if isinstance(subset_by, gpd.GeoDataFrame) else subset_by
        mask_geom = geoseries.to_crs("EPSG:4326").union_all()
        mask = gdf_wgs.geometry.intersects(mask_geom)
    else:
        raise TypeError(
            "subset_by must be a (min_lon, min_lat, max_lon, max_lat) tuple, "
            "GeoDataFrame, or GeoSeries"
        )

    return gdf.loc[mask].reset_index(drop=True)


def load_tracts_to_postgis(gdf, engine):
    """Load tract GeoDataFrame into the canonical census_tracts table.

    Appends only tracts not already present (by GEOID), so it's safe
    to call repeatedly for different counties or states.
    """
    from sqlalchemy import text

    # Normalize column names and CRS to match our schema (EPSG:4326)
    tracts = gdf[["GEOID", "STATEFP", "COUNTYFP", "NAME", "ALAND", "AWATER", "geometry"]].copy()
    tracts.columns = ["geoid", "state_fips", "county_fips", "name", "aland", "awater", "geometry"]
    if tracts.crs and tracts.crs.to_epsg() != 4326:
        tracts = tracts.to_crs(epsg=4326)

    with engine.connect() as conn:
        existing = set(
            row[0]
            for row in conn.execute(text("SELECT geoid FROM census_tracts")).fetchall()
        )

    new_tracts = tracts[~tracts["geoid"].isin(existing)]
    if new_tracts.empty:
        console.print("[dim]All tracts already loaded — skipping.[/dim]")
        return 0

    new_tracts.to_postgis("census_tracts", engine, if_exists="append", index=False)
    console.print(f"[green]Loaded {len(new_tracts)} new tracts → census_tracts[/green]")
    return len(new_tracts)


def load_roads_to_postgis(gdf, state_fips, county_fips, engine):
    """Load road GeoDataFrame into the canonical roads table.

    Replaces roads for the specified county (safe to re-run).
    """
    from sqlalchemy import text

    # Clear existing roads for this county
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM roads WHERE state_fips = :s AND county_fips = :c"),
            {"s": state_fips, "c": county_fips},
        )

    roads = gdf[["LINEARID", "FULLNAME", "RTTYP", "MTFCC", "geometry"]].copy()
    roads.columns = ["linearid", "fullname", "rttyp", "mtfcc", "geometry"]
    if roads.crs and roads.crs.to_epsg() != 4326:
        roads = roads.to_crs(epsg=4326)
    roads["state_fips"] = state_fips
    roads["county_fips"] = county_fips

    roads.to_postgis("roads", engine, if_exists="append", index=False)
    console.print(
        f"[green]Loaded {len(roads)} road segments → roads "
        f"(county {state_fips}{county_fips})[/green]"
    )
    return len(roads)


def load_to_postgis(gdf, table_name, engine, if_exists="replace"):
    """Load a TIGER GeoDataFrame into PostGIS (generic, for custom tables)."""
    gdf.to_postgis(table_name, engine, if_exists=if_exists)
    console.print(f"[green]Loaded {len(gdf)} features → {table_name}[/green]")
