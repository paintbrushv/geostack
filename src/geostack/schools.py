"""School district quality lookup for workforce housing deal analysis.

Data sources:
  - TIGER/Line Unified School District boundaries (Census Bureau)
  - Texas Education Agency (TEA) accountability ratings

Usage:
    from geostack.schools import school_profile

    profile = school_profile(lon=-96.6183, lat=32.8028, year=2023)
    # Returns: {district_name, tea_rating, leaid, domain_scores, ...}
"""

from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from rich.console import Console

console = Console()

_TIGER_URL = (
    "https://www2.census.gov/geo/tiger/TIGER{year}/UNSD/tl_{year}_48_unsd.zip"
)
_TEA_URL = (
    "https://rptsvr1.tea.texas.gov/cgi/sas/broker?"
    "_service=marykay&_program=perfrept.perfmast.sas&_debug=0"
    "&ccyy={year}&dsname=RATE&sumlev=D"
    "&key=RATE&key=D1&key=D2&key=D3"
    "&datafmt=C&prgopt=reports/acct/dd/dd_get_data.sas"
)

# Local cache directory (gitignored via data/)
_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "schools"


def _cache_path(filename: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / filename


def _fetch_tea_ratings(year: int = 2023) -> pd.DataFrame:
    """Download TEA district accountability ratings CSV."""
    cache = _cache_path(f"tea_ratings_{year}.parquet")
    if cache.exists():
        return pd.read_parquet(cache)

    url = _TEA_URL.format(year=year)
    console.print(f"[dim]Fetching TEA ratings from {url}...[/dim]")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    raw = pd.read_csv(io.StringIO(resp.text), dtype=str, low_memory=False)
    df = _parse_tea_ratings(raw)
    df.to_parquet(cache, index=False)
    return df


def _parse_tea_ratings(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw TEA CSV or pre-built DataFrame to canonical column names.

    Accepts either the raw downloaded CSV (with TEA column names) or a
    pre-built DataFrame with canonical names (for tests).
    """
    df = raw.copy()

    # If already has canonical columns (test fixture), just coerce district_id
    if "district_id" in df.columns and "overall_rating" in df.columns:
        df["district_id"] = df["district_id"].astype(object).str.zfill(6)
        return df

    # Map raw TEA CSV column names to canonical names
    col_map = {
        "DISTRICT": "district_id",
        "DISTNAME": "district_name",
        "D_RATING": "overall_rating",
        "DD1G": "student_achievement",
        "DD2G": "school_progress",
        "DD3G": "closing_gaps",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    keep = [v for v in col_map.values() if v in df.columns]
    df = df[keep].copy()
    df["district_id"] = df["district_id"].astype(object).str.zfill(6)
    return df


def _load_tiger_districts(year: int = 2023) -> gpd.GeoDataFrame:
    """Download and cache TIGER unified school district shapefile for Texas."""
    cache = _cache_path(f"tx_school_districts_{year}.parquet")
    if cache.exists():
        return gpd.read_parquet(cache)

    url = _TIGER_URL.format(year=year)
    console.print(f"[dim]Fetching TIGER school districts from {url}...[/dim]")
    gdf = gpd.read_file(url)
    gdf = gdf.to_crs("EPSG:4326")

    # LEAID: TIGER GEOID has 2-char state prefix + 5-char district ID
    if "GEOID" in gdf.columns:
        gdf["LEAID"] = gdf["GEOID"].str[2:]  # strip state FIPS prefix

    gdf.to_parquet(cache, index=False)
    return gdf


def _district_at_point(
    lon: float, lat: float, districts_gdf: gpd.GeoDataFrame
) -> tuple[str | None, str | None]:
    """Find the school district containing a lat/lon point.

    Returns:
        (district_name, leaid) or (None, None) if not found.
    """
    pt = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([lon], [lat]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pt, districts_gdf[["NAME", "LEAID", "geometry"]],
                       how="left", predicate="within")
    matched = joined.dropna(subset=["NAME"])
    if matched.empty:
        return None, None
    row = matched.iloc[0]
    return row.get("NAME"), row.get("LEAID")


def _normalize_district_name(name: str) -> str:
    """Normalize a school district name for matching across TEA and TIGER formats.

    TIGER uses: "Garland Independent School District"
    TEA uses:   "GARLAND ISD"

    Both normalize to: "garland"
    """
    n = name.lower().strip()
    # Expand common TEA abbreviations, then strip suffixes
    for suffix in [
        " independent school district",
        " consolidated independent school district",
        " isd", " cisd", " school district", " sd",
    ]:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
            break
    return n.strip()


def _match_by_name(tiger_name: str, tea_df: pd.DataFrame) -> pd.DataFrame:
    """Match a TIGER district name against TEA DISTNAME using normalized names."""
    target = _normalize_district_name(tiger_name)
    tea_df = tea_df.copy()
    tea_df["_norm"] = tea_df["district_name"].apply(_normalize_district_name)
    match = tea_df[tea_df["_norm"] == target].drop(columns=["_norm"])
    return match


def school_profile(lon: float, lat: float, year: int = 2023) -> dict:
    """Return school district quality profile for a property location.

    Args:
        lon: Property longitude (WGS84).
        lat: Property latitude (WGS84).
        year: TEA accountability year (2023 is most recent with full ratings).

    Returns:
        dict with keys: district_name, leaid, tea_rating, student_achievement,
        school_progress, closing_gaps, year.

    Note:
        Texas-only — TIGER boundaries and TEA ratings cover Texas school districts only.
        For properties outside Texas, all returned fields except `year` will be None.
        TEA letter ratings (A–F) apply to 2022 and later; older years use a different scale.
    """
    districts = _load_tiger_districts(year)
    tea = _fetch_tea_ratings(year)

    district_name, leaid = _district_at_point(lon, lat, districts)
    if district_name is None:
        return {
            "district_name": None, "leaid": None, "tea_rating": None,
            "student_achievement": None, "school_progress": None,
            "closing_gaps": None, "year": year,
        }

    # Try LEAID-based match first (NCES ID → TEA district code)
    leaid_padded = str(leaid).zfill(6)
    match = tea[tea["district_id"] == leaid_padded]

    # Fallback: name-based match (TIGER uses full name, TEA uses abbreviation)
    if match.empty and "district_name" in tea.columns:
        match = _match_by_name(district_name, tea)

    if match.empty:
        console.print(
            f"[yellow]No TEA rating found for {district_name} (LEAID {leaid})[/yellow]"
        )
        return {
            "district_name": district_name, "leaid": leaid, "tea_rating": None,
            "student_achievement": None, "school_progress": None,
            "closing_gaps": None, "year": year,
        }

    row = match.iloc[0]
    result = {
        "district_name": district_name,
        "leaid": leaid,
        "tea_rating": row.get("overall_rating"),
        "student_achievement": row.get("student_achievement"),
        "school_progress": row.get("school_progress"),
        "closing_gaps": row.get("closing_gaps"),
        "year": year,
    }
    console.print(
        f"[green]School district: {district_name} — TEA rating: {result['tea_rating']}[/green]"
    )
    return result
