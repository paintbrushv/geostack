"""Supply pipeline analysis for workforce housing underwriting.

Combines Census Building Permits Survey (BPS) county data with
CoStar/RealPage comparable property CSV exports to quantify:
  - Permitted multifamily pipeline within the trade area (forward-looking)
  - Recently delivered properties and their rent positioning (backward-looking)

Usage:
    from geostack.supply import supply_pipeline, load_costar_comps

    result = supply_pipeline(
        lon=-96.6183, lat=32.8028, minutes=10,
        state_fips="48", county_fips="113",
        bps_year_range=(2022, 2024),
        costar_csv="path/to/costar_export.csv",
    )
"""

from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from rich.console import Console

console = Console()

_BPS_URL = "https://www2.census.gov/econ/bps/County/co{year}a.txt"
_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "bps"

_REQUIRED_COSTAR_COLS = {"lat", "lon", "name", "units", "status", "delivery_year"}


def _cache_path(filename: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / filename


def _safe_int(val: str) -> int:
    """Parse a BPS field value to int, returning 0 for suppressed/invalid values."""
    cleaned = val.strip().replace(",", "")
    return int(cleaned) if cleaned.isdigit() else 0


def _parse_bps_county(text: str, year: int) -> pd.DataFrame:
    """Parse Census BPS county annual text file into a tidy DataFrame.

    The real Census BPS county file (www2.census.gov/econ/bps/County/) has:
      - Two header rows (category grouping + column names)
      - Data columns: Date, FIPS State, FIPS County, Region, Division,
        County Name, then groups of 3 (Bldgs, Units, Value) per structure type:
        1-unit [6,7,8], 2-units [9,10,11], 3-4 units [12,13,14], 5+ units [15,16,17]

    Returns DataFrame with: county_fips, units_1unit, units_5plus, year.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 18:
            continue
        # Data rows have a 2-char state FIPS at col 1 and 3-char county at col 2
        state_fips = parts[1].strip()
        county_fips_part = parts[2].strip()
        if not state_fips.isdigit() or len(state_fips) != 2:
            continue
        if not county_fips_part.isdigit() or len(county_fips_part) != 3:
            continue
        try:
            rows.append({
                "county_fips": state_fips + county_fips_part,
                "units_1unit": _safe_int(parts[7]),
                "units_5plus": _safe_int(parts[16]),
                "year": year,
            })
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows)


def _fetch_bps_county(year: int) -> pd.DataFrame:
    """Download and cache BPS county annual file for a given year."""
    cache = _cache_path(f"bps_county_{year}.parquet")
    if cache.exists():
        return pd.read_parquet(cache)

    url = _BPS_URL.format(year=year)
    console.print(f"[dim]Fetching BPS county data for {year}...[/dim]")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    df = _parse_bps_county(resp.text, year)
    df.to_parquet(cache, index=False)
    console.print(f"[green]BPS {year}: {len(df):,} counties[/green]")
    return df


def load_costar_comps(csv_path: str) -> gpd.GeoDataFrame:
    """Load a CoStar/RealPage CSV export into a GeoDataFrame.

    Required CSV columns: name, lat, lon, units, status, delivery_year.

    Args:
        csv_path: Path to the CoStar export CSV file.

    Returns:
        GeoDataFrame with EPSG:4326 geometry and normalized columns.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(csv_path, dtype={"delivery_year": "Int64"})

    missing = _REQUIRED_COSTAR_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CoStar CSV missing required columns: {missing}")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    return gdf


def supply_pipeline(
    lon: float,
    lat: float,
    minutes: int,
    state_fips: str,
    county_fips: str,
    bps_year_range: tuple[int, int] = (2022, 2024),
    costar_csv: str | None = None,
) -> dict:
    """Quantify multifamily supply pipeline for a property's trade area.

    Args:
        lon, lat: Property coordinates (WGS84).
        minutes: Drive-time isochrone in minutes (context only; BPS is county-level).
        state_fips: 2-char state FIPS (e.g., "48" for Texas).
        county_fips: 3-char county FIPS (e.g., "113" for Dallas).
        bps_year_range: (start_year, end_year) inclusive.
        costar_csv: Optional path to CoStar export CSV.

    Returns:
        dict with:
          - permitted_units_5plus: BPS 5+ unit authorizations summed over years
          - permitted_units_1unit: BPS 1-unit authorizations
          - bps_years: list of years included
          - bps_by_year: DataFrame (year, units_5plus, units_1unit)
          - delivered_properties: GeoDataFrame from CoStar (or empty GDF if not provided)
    """
    full_fips = (state_fips + county_fips).zfill(5)
    start_yr, end_yr = bps_year_range
    years = list(range(start_yr, end_yr + 1))

    frames = []
    for yr in years:
        try:
            df = _fetch_bps_county(yr)
            county_row = df[df["county_fips"] == full_fips]
            if not county_row.empty:
                frames.append(county_row)
        except Exception as e:
            console.print(f"[yellow]BPS {yr} skipped: {e}[/yellow]")

    if frames:
        bps_df = pd.concat(frames, ignore_index=True)
        by_year = bps_df.groupby("year")[["units_5plus", "units_1unit"]].sum().reset_index()
        total_5plus = int(bps_df["units_5plus"].sum())
        total_1unit = int(bps_df["units_1unit"].sum())
    else:
        by_year = pd.DataFrame(columns=["year", "units_5plus", "units_1unit"])
        total_5plus = 0
        total_1unit = 0

    delivered = gpd.GeoDataFrame()
    if costar_csv:
        delivered = load_costar_comps(costar_csv)

    console.print(
        f"\n[bold]Supply pipeline ({full_fips}, {start_yr}–{end_yr}):[/bold]\n"
        f"  5+ unit authorized: {total_5plus:,}\n"
        f"  Delivered comps: {len(delivered)}"
    )

    return {
        "permitted_units_5plus": total_5plus,
        "permitted_units_1unit": total_1unit,
        "bps_years": years,
        "bps_by_year": by_year,
        "delivered_properties": delivered,
    }
