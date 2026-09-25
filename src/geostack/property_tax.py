"""Property tax jurisdiction analysis for Texas workforce housing.

Data source: Texas Comptroller of Public Accounts — annual property tax rate files.

Usage:
    from geostack.property_tax import effective_tax_rate

    rates = effective_tax_rate(lon=-96.6183, lat=32.8028,
                               county_name="DALLAS COUNTY", year=2024)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from rich.console import Console

from geostack.schools import school_profile, _normalize_district_name

console = Console()

_COMPTROLLER_URL = (
    "https://comptroller.texas.gov/taxes/property-tax/docs/"
    "{year}-{entity_type}-rates-levies.xlsx"
)
_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "tax"
_ENTITY_TYPES = ["county", "city", "school-district"]


def _cache_path(filename: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / filename


def _fetch_comptroller_rates(year: int = 2024) -> pd.DataFrame:
    """Download and combine TX Comptroller property tax rate files."""
    cache = _cache_path(f"tx_tax_rates_{year}.parquet")
    if cache.exists():
        return pd.read_parquet(cache)

    frames = []
    for entity_type in _ENTITY_TYPES:
        url = _COMPTROLLER_URL.format(year=year, entity_type=entity_type)
        console.print(f"[dim]Fetching {entity_type} tax rates from {url}...[/dim]")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            import io as _io
            df = pd.read_excel(_io.BytesIO(resp.content), skiprows=2, dtype=str)
            df.columns = [c.strip().upper() for c in df.columns]
            name_col = next((c for c in df.columns if "NAME" in c and "VERSION" not in c
                             and "CAD" not in c and "COUNTY" not in c), None)
            rate_col = next((c for c in df.columns if "TOTAL" in c and "RATE" in c
                             and "TAX" in c), None)
            if name_col and rate_col:
                sub = df[[name_col, rate_col]].copy()
                sub.columns = ["entity_name", "total_rate"]
                sub["entity_type"] = entity_type.replace("-", "_")
                sub["total_rate"] = pd.to_numeric(sub["total_rate"], errors="coerce")
                sub = sub.dropna(subset=["total_rate"])
                sub["entity_name"] = sub["entity_name"].astype(str).str.strip()
                frames.append(sub)
        except Exception as e:
            console.print(f"[yellow]Tax rate {entity_type} {year} skipped: {e}[/yellow]")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(cache, index=False)
        console.print(f"[green]Loaded {len(combined):,} tax entities for {year}[/green]")
        return combined

    return pd.DataFrame(columns=["entity_name", "total_rate", "entity_type"])


def _lookup_rate(entity_name: str, rates_df: pd.DataFrame) -> float | None:
    """Look up a tax rate by entity name (case-insensitive)."""
    target = entity_name.strip().upper()
    match = rates_df[rates_df["entity_name"].str.upper() == target]
    if match.empty:
        return None
    return float(match.iloc[0]["total_rate"])


def effective_tax_rate(
    lon: float,
    lat: float,
    county_name: str,
    year: int = 2024,
) -> dict:
    """Estimate the effective property tax rate for a Texas property.

    Combines county rate + school district rate (via school_profile).
    Rates are per $100 of assessed value.

    Args:
        lon, lat: Property coordinates (WGS84).
        county_name: Full county name as in Comptroller data (e.g., "DALLAS COUNTY").
        year: Comptroller rate year.

    Returns:
        dict with county_name, county_rate, school_district, school_rate,
        total_rate, rate_per_100, year.
    """
    rates_df = _fetch_comptroller_rates(year)
    school = school_profile(lon, lat, year=min(year, 2023))

    county_rate = _lookup_rate(county_name, rates_df) or 0.0
    school_rate = 0.0
    school_district = None
    if school.get("district_name"):
        district_name = school["district_name"]
        school_district = district_name
        normalized = _normalize_district_name(district_name).upper() + " ISD"
        school_rate = _lookup_rate(normalized, rates_df) or 0.0
        if school_rate == 0.0:
            school_rate = _lookup_rate(district_name, rates_df) or 0.0

    total_rate = county_rate + school_rate

    result = {
        "county_name": county_name,
        "county_rate": county_rate,
        "school_district": school_district,
        "school_rate": school_rate,
        "total_rate": total_rate,
        "rate_per_100": total_rate,
        "year": year,
    }

    console.print(
        f"\n[bold]Property Tax Estimate ({county_name}, {year}):[/bold]"
        f"\n  County rate:  ${county_rate:.4f} / $100 AV"
        f"\n  School rate:  ${school_rate:.4f} / $100 AV"
        f"  ({school_district or 'unknown'})"
        f"\n  Total:        ${total_rate:.4f} / $100 AV"
    )

    return result
