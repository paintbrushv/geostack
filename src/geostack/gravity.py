"""MSA Population Center of Gravity analysis.

Calculates weighted geographic centroids for an MSA over time using census
tract-level data. Supports multiple weighting modes:

- **population**: Decennial Census / ACS total population
- **jobs**: LODES WAC workplace jobs (2002–2023)
- **workers**: LODES RAC employed residents (2002–2023)
- **income_mass**: ACS households × median household income (proxy)

Pre-2020 data is normalized to 2020 tract boundaries via Census relationship
files (areal weighting for counts, pass-through for medians): 2010→2020 for
2010-vintage years, and a chained 2000→2010→2020 crosswalk for ≤2009 data
(ACS 2005-2009 and Decennial 2000, which use 2000-vintage tracts).

Usage:
    from geostack.gravity import (
        population_center_of_gravity,
        gravity_time_series,
        multi_weight_time_series,
        MSA_DEFINITIONS,
    )

    # Single year, single weight
    result = population_center_of_gravity("Dallas", year=2020)

    # Full multi-weight time series
    combined = multi_weight_time_series("Dallas", years=[2010, 2013, 2018, 2020, 2023])
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
from dataclasses import dataclass, asdict

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from rich.console import Console
from shapely.geometry import Point

try:
    from dotenv import load_dotenv
    load_dotenv()  # pick up CENSUS_API_KEY — required for ACS 2024+ vintages
except ImportError:
    pass

from geostack.spatial import suggest_crs

console = Console()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_DIR = pathlib.Path.home() / ".cache" / "geostack" / "gravity"


def _cache_path(key: str) -> pathlib.Path:
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return _CACHE_DIR / f"{h}.parquet"


def _read_cache(key: str) -> pd.DataFrame | None:
    path = _cache_path(key)
    if path.exists():
        logger.debug("Cache hit: %s", key)
        return pd.read_parquet(path)
    return None


def _write_cache(key: str, df: pd.DataFrame) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_cache_path(key), index=False)


# ---------------------------------------------------------------------------
# Weight mode enum
# ---------------------------------------------------------------------------

WEIGHT_MODES = ("population", "jobs", "workers", "income_mass")

WEIGHT_LABELS = {
    "population": "Population",
    "jobs": "Workplace Jobs (LODES WAC)",
    "workers": "Resident Workers (LODES RAC)",
    "income_mass": "Income Mass (HH × MHI)",
}


# ---------------------------------------------------------------------------
# MSA Definitions — state FIPS + county FIPS list
# ---------------------------------------------------------------------------

MSA_DEFINITIONS: dict[str, dict] = {
    "Dallas-Fort Worth-Arlington, TX": {
        "state_fips": "48",
        "counties": {
            "085": "Collin",
            "113": "Dallas",
            "121": "Denton",
            "139": "Ellis",
            "221": "Hood",
            "231": "Hunt",
            "251": "Johnson",
            "257": "Kaufman",
            "367": "Parker",
            "397": "Rockwall",
            "439": "Tarrant",
            "497": "Wise",
        },
        "cbsa_code": "19100",
        "crs": "EPSG:2276",  # TX State Plane North Central (ftUS)
    },
    "Houston-The Woodlands-Sugar Land, TX": {
        "state_fips": "48",
        "counties": {
            "015": "Austin",
            "039": "Brazoria",
            "071": "Chambers",
            "157": "Fort Bend",
            "167": "Galveston",
            "201": "Harris",
            "291": "Liberty",
            "339": "Montgomery",
            "473": "Waller",
        },
        "cbsa_code": "26420",
        "crs": "EPSG:2278",  # TX State Plane South Central (ftUS)
    },
    "Austin-Round Rock-Georgetown, TX": {
        "state_fips": "48",
        "counties": {
            "021": "Bastrop",
            "055": "Caldwell",
            "209": "Hays",
            "453": "Travis",
            "491": "Williamson",
        },
        "cbsa_code": "12420",
        "crs": "EPSG:2277",  # TX State Plane Central (ftUS)
    },
    "San Antonio-New Braunfels, TX": {
        "state_fips": "48",
        "counties": {
            "013": "Atascosa",
            "019": "Bandera",
            "029": "Bexar",
            "091": "Comal",
            "187": "Guadalupe",
            "259": "Kendall",
            "325": "Medina",
            "493": "Wilson",
        },
        "cbsa_code": "41700",
        "crs": "EPSG:2278",  # TX State Plane South Central (ftUS)
    },
    "Phoenix-Mesa-Chandler, AZ": {
        "state_fips": "04",
        "counties": {
            "013": "Maricopa",
            "021": "Pinal",
        },
        "cbsa_code": "38060",
        "crs": "EPSG:2223",  # AZ State Plane Central (intl ft)
    },
    "Atlanta-Sandy Springs-Alpharetta, GA": {
        "state_fips": "13",
        "counties": {
            "013": "Barrow",
            "015": "Bartow",
            "035": "Butts",
            "045": "Carroll",
            "057": "Cherokee",
            "063": "Clayton",
            "067": "Cobb",
            "077": "Coweta",
            "085": "Dawson",
            "089": "DeKalb",
            "097": "Douglas",
            "113": "Fayette",
            "117": "Forsyth",
            "121": "Fulton",
            "135": "Gwinnett",
            "143": "Haralson",
            "149": "Heard",
            "151": "Henry",
            "159": "Jasper",
            "169": "Jones",
            "171": "Lamar",
            "199": "Meriwether",
            "211": "Morgan",
            "217": "Newton",
            "223": "Paulding",
            "227": "Pickens",
            "231": "Pike",
            "247": "Rockdale",
            "255": "Spalding",
            "297": "Walton",
        },
        "cbsa_code": "12060",
        "crs": "EPSG:2240",  # GA State Plane West (ftUS)
    },
    "Fayetteville-Springdale-Rogers, AR": {
        "state_fips": "05",
        "counties": {
            "007": "Benton",
            "087": "Madison",
            "143": "Washington",
        },
        "cbsa_code": "22220",
        "crs": "EPSG:3433",  # AR State Plane North (ftUS)
    },
}

# Fallback projected CRS for MSAs without an explicit "crs" entry.
DEFAULT_MSA_CRS = "EPSG:2276"


def msa_crs(msa_name: str) -> str:
    """Return the projected CRS (feet-based State Plane) for an MSA."""
    return resolve_msa(msa_name).get("crs", DEFAULT_MSA_CRS)


# ---------------------------------------------------------------------------
# Landmark registry — recognizable reference points for map anchoring
#
# Center-of-gravity movement is meaningless in raw lat/lon. Anchoring it to
# named places ("drifting from Las Colinas toward North Dallas") gives the
# reader the context to interpret a 1-mile shift. Keyed by MSA name.
# Each landmark: (name, lat, lon, kind) where kind ∈ {cbd, suburb, airport}.
# ---------------------------------------------------------------------------

MSA_LANDMARKS: dict[str, list[tuple]] = {
    "Dallas-Fort Worth-Arlington, TX": [
        # Major employment / business cores
        ("Downtown Dallas", 32.7777, -96.7970, "cbd"),
        ("Fort Worth CBD", 32.7555, -97.3308, "cbd"),
        ("Las Colinas", 32.8596, -96.9426, "suburb"),
        ("Legacy/Frisco (DNT-121)", 33.0850, -96.8230, "suburb"),  # Legacy West / Frisco towers
        ("Richardson Telecom", 32.9760, -96.7050, "suburb"),
        ("Addison", 32.9618, -96.8292, "suburb"),
        ("North Dallas (US-75)", 32.8810, -96.7790, "suburb"),  # NorthPark / Medical City
        ("Fort Worth West", 32.8340, -97.3010, "suburb"),       # Cultural/Medical District
        ("DFW Airport", 32.8998, -97.0403, "airport"),
        # Suburbs / submarkets
        ("Plano", 33.0198, -96.6989, "suburb"),
        ("Allen", 33.1032, -96.6706, "suburb"),
        ("Frisco", 33.1507, -96.8236, "suburb"),
        ("McKinney", 33.1972, -96.6398, "suburb"),
        ("Denton", 33.2148, -97.1331, "suburb"),
        ("Lewisville", 33.0462, -96.9942, "suburb"),
        ("Carrollton", 32.9756, -96.8900, "suburb"),
        ("Irving", 32.8140, -96.9489, "suburb"),
        ("Garland", 32.9126, -96.6389, "suburb"),
        ("Mesquite", 32.7668, -96.5992, "suburb"),
        ("Grand Prairie", 32.7459, -96.9978, "suburb"),
        ("Grapevine", 32.9343, -97.0781, "suburb"),
        ("Southlake", 32.9412, -97.1342, "suburb"),
        ("AllianceTexas", 32.9850, -97.3180, "suburb"),   # N. Fort Worth logistics hub
        ("Westlake (Schwab/Fidelity)", 32.9910, -97.1970, "suburb"),
    ],
    "Fayetteville-Springdale-Rogers, AR": [
        # Major employment / business cores
        ("Downtown Fayetteville", 36.0626, -94.1574, "cbd"),
        ("Downtown Bentonville (Walmart HQ)", 36.3729, -94.2088, "cbd"),
        ("Downtown Springdale", 36.1867, -94.1288, "cbd"),
        ("Downtown Rogers", 36.3320, -94.1185, "cbd"),
        ("Pinnacle Hills (I-49)", 36.2967, -94.1840, "suburb"),  # Rogers retail/office corridor
        ("J.B. Hunt HQ (Lowell)", 36.2554, -94.1308, "suburb"),
        ("University of Arkansas", 36.0687, -94.1749, "suburb"),
        ("XNA Airport", 36.2819, -94.3068, "airport"),
        # Suburbs / submarkets
        ("Bella Vista", 36.4295, -94.2113, "suburb"),
        ("Centerton", 36.3598, -94.2852, "suburb"),
        ("Cave Springs", 36.2634, -94.2319, "suburb"),
        ("Pea Ridge", 36.4540, -94.1149, "suburb"),
        ("Tontitown", 36.1776, -94.2333, "suburb"),
        ("Siloam Springs", 36.1881, -94.5405, "suburb"),
        ("Farmington", 36.0420, -94.2471, "suburb"),
        ("Prairie Grove", 35.9737, -94.3174, "suburb"),
        ("Huntsville", 36.0879, -93.7413, "suburb"),  # Madison County seat
    ],
}


def get_landmarks(msa_name: str) -> list[tuple]:
    """Return landmark reference points for an MSA (empty list if none defined)."""
    msa = resolve_msa(msa_name)
    return MSA_LANDMARKS.get(msa["name"], [])


def list_msas() -> list[str]:
    """Return available MSA names."""
    return list(MSA_DEFINITIONS.keys())


def resolve_msa(name: str) -> dict:
    """Look up an MSA definition by name (case-insensitive partial match)."""
    name_lower = name.lower()
    for msa_name, defn in MSA_DEFINITIONS.items():
        if name_lower in msa_name.lower():
            return {"name": msa_name, **defn}
    raise ValueError(
        f"MSA not found: {name!r}. Available: {list_msas()}"
    )


# ---------------------------------------------------------------------------
# Census API data fetching
# ---------------------------------------------------------------------------

_CENSUS_BASE = "https://api.census.gov/data"


def _get_api_key() -> str | None:
    return os.environ.get("CENSUS_API_KEY")


def _census_api_get(url: str, params: dict) -> pd.DataFrame:
    """Shared Census API request with API key injection."""
    api_key = _get_api_key()
    if api_key:
        params["key"] = api_key
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data[1:], columns=data[0])


def _fetch_decennial_population(
    state_fips: str,
    county_fips: str,
    year: int,
) -> pd.DataFrame:
    """Fetch tract-level population from Decennial Census (2000/2010/2020)."""
    cache_key = f"decennial_pop_{state_fips}_{county_fips}_{year}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    if year == 2020:
        url = f"{_CENSUS_BASE}/2020/dec/dhc"
        pop_var = "P1_001N"
    elif year == 2010:
        url = f"{_CENSUS_BASE}/2010/dec/sf1"
        pop_var = "P001001"
    elif year == 2000:
        url = f"{_CENSUS_BASE}/2000/dec/sf1"
        pop_var = "P001001"
    else:
        raise ValueError(f"Decennial census only for 2000/2010/2020; got {year}")

    df = _census_api_get(url, {
        "get": f"NAME,{pop_var}",
        "for": "tract:*",
        "in": f"state:{state_fips} county:{county_fips}",
    })
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["population"] = pd.to_numeric(df[pop_var], errors="coerce")
    df = df[["GEOID", "population"]].copy()
    df = df[df["population"].notna() & (df["population"] > 0)]

    _write_cache(cache_key, df)
    return df


def _fetch_acs_population(
    state_fips: str,
    county_fips: str,
    year: int,
) -> pd.DataFrame:
    """Fetch tract-level population from ACS 5-year (2009+)."""
    cache_key = f"acs5_pop_{state_fips}_{county_fips}_{year}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    df = _census_api_get(f"{_CENSUS_BASE}/{year}/acs/acs5", {
        "get": "NAME,B01003_001E",
        "for": "tract:*",
        "in": f"state:{state_fips} county:{county_fips}",
    })
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["population"] = pd.to_numeric(df["B01003_001E"], errors="coerce")
    df.loc[df["population"] < 0, "population"] = np.nan
    df = df[["GEOID", "population"]].copy()
    df = df[df["population"].notna() & (df["population"] > 0)]

    _write_cache(cache_key, df)
    return df


def fetch_population(
    state_fips: str,
    county_fips: str,
    year: int,
) -> pd.DataFrame:
    """Fetch tract-level population. Routes Decennial (2000/2010/2020) vs ACS."""
    if year in (2000, 2010, 2020):
        return _fetch_decennial_population(state_fips, county_fips, year)
    if year >= 2009:
        return _fetch_acs_population(state_fips, county_fips, year)
    raise ValueError(f"No tract-level population for year {year}")


# ---------------------------------------------------------------------------
# Income-mass fetching (ACS)
# ---------------------------------------------------------------------------

def _fetch_acs_income_mass(
    state_fips: str,
    county_fips: str,
    year: int,
) -> pd.DataFrame:
    """Fetch tract-level income mass proxy: occupied_units × median_household_income.

    This is an *estimated* aggregate income proxy, not true total income.
    Tracts with suppressed MHI (sentinel -666666666 or null) are dropped.

    Returns DataFrame with columns: GEOID, income_mass, occupied_units, mhi.
    """
    cache_key = f"acs5_income_mass_{state_fips}_{county_fips}_{year}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    # B25002_002E = occupied housing units, B19013_001E = median household income
    df = _census_api_get(f"{_CENSUS_BASE}/{year}/acs/acs5", {
        "get": "NAME,B25002_002E,B19013_001E",
        "for": "tract:*",
        "in": f"state:{state_fips} county:{county_fips}",
    })
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["occupied_units"] = pd.to_numeric(df["B25002_002E"], errors="coerce")
    df["mhi"] = pd.to_numeric(df["B19013_001E"], errors="coerce")

    # Drop suppressed data (Census sentinel or null)
    df.loc[df["mhi"] < 0, "mhi"] = np.nan
    df.loc[df["occupied_units"] < 0, "occupied_units"] = np.nan
    df = df[df["mhi"].notna() & df["occupied_units"].notna()].copy()
    df = df[df["occupied_units"] > 0].copy()

    df["income_mass"] = df["occupied_units"] * df["mhi"]
    df = df[["GEOID", "income_mass", "occupied_units", "mhi"]].copy()

    _write_cache(cache_key, df)
    return df


def _fetch_acs_aggregate_income(
    state_fips: str,
    county_fips: str,
    year: int,
) -> pd.DataFrame:
    """Fetch tract-level TRUE aggregate household income (ACS B19025).

    Unlike the income_mass proxy (occupied_units × median HHI), B19025 is the
    actual sum of household incomes — no median-vs-mean skew bias and no
    $250k median topcode, so high-income tracts are not understated.

    Returns DataFrame with columns: GEOID, agg_income.
    """
    cache_key = f"acs5_agg_income_{state_fips}_{county_fips}_{year}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    # B19025_001E = aggregate household income in the past 12 months
    df = _census_api_get(f"{_CENSUS_BASE}/{year}/acs/acs5", {
        "get": "NAME,B19025_001E",
        "for": "tract:*",
        "in": f"state:{state_fips} county:{county_fips}",
    })
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["agg_income"] = pd.to_numeric(df["B19025_001E"], errors="coerce")
    df.loc[df["agg_income"] < 0, "agg_income"] = np.nan  # Census sentinel
    df = df[["GEOID", "agg_income"]].copy()
    df = df[df["agg_income"].notna() & (df["agg_income"] > 0)]

    _write_cache(cache_key, df)
    return df


# ---------------------------------------------------------------------------
# LODES fetching (jobs / workers at tract level)
# ---------------------------------------------------------------------------

def _fetch_lodes_tract_jobs(
    state_fips: str,
    county_fips: str,
    year: int,
) -> pd.DataFrame:
    """Fetch tract-level workplace jobs from LODES WAC.

    Returns DataFrame with columns: GEOID, weight (= total_jobs).
    LODES available ~2002–2021.
    """
    cache_key = f"lodes_wac_tract_{state_fips}_{county_fips}_{year}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    from geostack.lodes import download_wac

    full_fips = f"{state_fips}{county_fips}"
    wac = download_wac(state_fips, year=year, county_filter=full_fips)

    # Aggregate blocks → tracts
    tract_jobs = (
        wac.groupby("tract_geoid", as_index=False)["total_jobs"]
        .sum()
        .rename(columns={"tract_geoid": "GEOID", "total_jobs": "weight"})
    )
    tract_jobs = tract_jobs[tract_jobs["weight"] > 0].copy()

    _write_cache(cache_key, tract_jobs)
    return tract_jobs


def _fetch_lodes_tract_workers(
    state_fips: str,
    county_fips: str,
    year: int,
) -> pd.DataFrame:
    """Fetch tract-level resident workers from LODES RAC.

    Returns DataFrame with columns: GEOID, weight (= total_workers).
    """
    cache_key = f"lodes_rac_tract_{state_fips}_{county_fips}_{year}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    from geostack.lodes import download_rac

    full_fips = f"{state_fips}{county_fips}"
    rac = download_rac(state_fips, year=year, county_filter=full_fips)

    tract_workers = (
        rac.groupby("tract_geoid", as_index=False)["total_workers"]
        .sum()
        .rename(columns={"tract_geoid": "GEOID", "total_workers": "weight"})
    )
    tract_workers = tract_workers[tract_workers["weight"] > 0].copy()

    _write_cache(cache_key, tract_workers)
    return tract_workers


# ---------------------------------------------------------------------------
# Crosswalk normalization
# ---------------------------------------------------------------------------

def _needs_crosswalk(year: int) -> bool:
    """Whether data from this year uses pre-2020 GEOIDs needing crosswalk."""
    # ACS 2010-2019 + Decennial 2010 use 2010-vintage tracts (2010→2020 file);
    # ACS ≤2009 + Decennial 2000 use 2000-vintage tracts (chained 2000→2020).
    return year < 2020


def _crosswalk_for_year(state_fips: str, year: int | None) -> pd.DataFrame:
    """Return the vintage-appropriate crosswalk as (geoid_src, geoid_2020, areal_weight).

    year ≤ 2009 (ACS 2005-2009 and Decennial 2000 are on 2000-vintage tracts)
    → chained 2000→2020. Otherwise → 2010→2020. Without the chain, 2000
    tracts that were split before 2020 — concentrated in fast-growth areas
    like Frisco/Collin/Denton — silently drop out of pre-2010 layers.
    """
    from geostack.trends import load_crosswalk, load_crosswalk_2000_2020

    if year is not None and year <= 2009:
        xw = load_crosswalk_2000_2020(state_fips)
        return xw.rename(columns={"geoid_2000": "geoid_src"})
    xw = load_crosswalk(state_fips)
    return xw.rename(columns={"geoid_2010": "geoid_src"})


def _apply_population_crosswalk(
    pop_df: pd.DataFrame,
    state_fips: str,
    year: int | None = None,
) -> pd.DataFrame:
    """Crosswalk population counts to 2020 GEOIDs (vintage chosen by year).

    Population is a count variable, so it's scaled by areal_weight when a
    source tract splits into multiple 2020 tracts.
    """
    xw = _crosswalk_for_year(state_fips, year)

    merged = pop_df.merge(xw, left_on="GEOID", right_on="geoid_src", how="left")
    # Tracts with no crosswalk entry: assume 1:1 mapping
    merged["geoid_2020"] = merged["geoid_2020"].fillna(merged["GEOID"])
    merged["areal_weight"] = merged["areal_weight"].fillna(1.0)

    merged["population"] = merged["population"] * merged["areal_weight"]
    merged["GEOID"] = merged["geoid_2020"]

    result = (
        merged.groupby("GEOID", as_index=False)["population"]
        .sum()
    )
    result["population"] = result["population"].round(0).astype(int)
    return result[result["population"] > 0].copy()


def _apply_weight_crosswalk(
    weight_df: pd.DataFrame,
    state_fips: str,
    weight_col: str = "weight",
    year: int | None = None,
) -> pd.DataFrame:
    """Crosswalk any count-based weight to 2020 GEOIDs (vintage chosen by year)."""
    xw = _crosswalk_for_year(state_fips, year)

    merged = weight_df.merge(xw, left_on="GEOID", right_on="geoid_src", how="left")
    merged["geoid_2020"] = merged["geoid_2020"].fillna(merged["GEOID"])
    merged["areal_weight"] = merged["areal_weight"].fillna(1.0)

    merged[weight_col] = merged[weight_col] * merged["areal_weight"]
    merged["GEOID"] = merged["geoid_2020"]

    result = (
        merged.groupby("GEOID", as_index=False)[weight_col]
        .sum()
    )
    return result[result[weight_col] > 0].copy()


def _apply_income_mass_crosswalk(
    income_df: pd.DataFrame,
    state_fips: str,
    year: int | None = None,
) -> pd.DataFrame:
    """Crosswalk income mass to 2020 GEOIDs (vintage chosen by year).

    occupied_units is a count (scaled by areal weight).
    mhi is a median (passed through unchanged for each piece).
    income_mass is recomputed from the crosswalked values.
    """
    xw = _crosswalk_for_year(state_fips, year)

    merged = income_df.merge(xw, left_on="GEOID", right_on="geoid_src", how="left")
    merged["geoid_2020"] = merged["geoid_2020"].fillna(merged["GEOID"])
    merged["areal_weight"] = merged["areal_weight"].fillna(1.0)

    # Scale count; pass through median
    merged["occupied_units"] = merged["occupied_units"] * merged["areal_weight"]
    # mhi stays unchanged per piece
    merged["income_mass"] = merged["occupied_units"] * merged["mhi"]
    merged["GEOID"] = merged["geoid_2020"]

    # Aggregate: sum income_mass and occupied_units; weighted-avg MHI
    result = merged.groupby("GEOID", as_index=False).agg(
        income_mass=("income_mass", "sum"),
        occupied_units=("occupied_units", "sum"),
    )
    result["mhi"] = np.where(
        result["occupied_units"] > 0,
        result["income_mass"] / result["occupied_units"],
        np.nan,
    )
    return result[result["income_mass"] > 0].copy()


# ---------------------------------------------------------------------------
# Tract geometry fetching
# ---------------------------------------------------------------------------

def _tiger_vintage_for_year(year: int) -> int:
    """Always use 2020 TIGER boundaries. Crosswalk handles GEOID translation."""
    return 2020


def fetch_tract_geometries(
    state_fips: str,
    county_fips: str,
    year: int,
    cb: bool = True,
) -> gpd.GeoDataFrame:
    """Download 2020-vintage tract geometries for a county.

    Returns GeoDataFrame with GEOID and geometry columns, filtered to county.
    """
    from geostack.tiger import download_tracts

    tiger_year = _tiger_vintage_for_year(year)
    use_cb = cb and tiger_year >= 2013
    gdf = download_tracts(state_fips, year=tiger_year, cb=use_cb)

    # Normalize GEOID column name
    for col in ("GEOID", "GEOID20", "GEOID10", "GEO_ID"):
        if col in gdf.columns:
            if col != "GEOID":
                gdf = gdf.rename(columns={col: "GEOID"})
            break
    else:
        raise ValueError(f"No GEOID column in TIGER {tiger_year} tracts")

    full_fips = f"{state_fips}{county_fips}"
    gdf = gdf[gdf["GEOID"].str[:5] == full_fips].copy()
    return gdf[["GEOID", "geometry"]].copy()


# ---------------------------------------------------------------------------
# Generic weighted centroid
# ---------------------------------------------------------------------------

@dataclass
class GravityResult:
    """Result of a single weighted center of gravity calculation."""
    msa_name: str
    year: int
    weight_mode: str
    total_weight: float
    center_lat: float
    center_lon: float
    county_count: int
    tract_count: int
    data_source: str
    tiger_vintage: int
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_weighted_centroid(
    tracts: gpd.GeoDataFrame,
    weights: pd.DataFrame,
    weight_col: str = "population",
) -> tuple[float, float, float, int]:
    """Compute weight-based centroid using projected geometry.

    Args:
        tracts: GeoDataFrame with GEOID and geometry (polygons).
        weights: DataFrame with GEOID and a weight column.
        weight_col: Name of the weight column in weights DataFrame.

    Returns:
        (center_lat, center_lon, total_weight, matched_tract_count)
    """
    merged = tracts.merge(weights, on="GEOID", how="inner")
    if merged.empty:
        raise ValueError("No matching tracts between geometry and weight data")

    target_crs = suggest_crs(merged)
    merged_proj = merged.to_crs(target_crs)
    centroids = merged_proj.geometry.centroid

    w = merged[weight_col].values.astype(float)
    total_w = w.sum()

    if total_w == 0:
        raise ValueError("Total weight is zero")

    weighted_x = (w * centroids.x).sum() / total_w
    weighted_y = (w * centroids.y).sum() / total_w

    center_proj = gpd.GeoDataFrame(
        geometry=[Point(weighted_x, weighted_y)], crs=target_crs,
    )
    center_wgs = center_proj.to_crs("EPSG:4326")
    pt = center_wgs.geometry.iloc[0]

    return (round(pt.y, 6), round(pt.x, 6), float(total_w), len(merged))


# ---------------------------------------------------------------------------
# Data source descriptions
# ---------------------------------------------------------------------------

def _data_source_label(weight_mode: str, year: int) -> str:
    if weight_mode == "population":
        if year in (2000, 2010, 2020):
            return "Decennial Census"
        return f"ACS 5-Year ({year})"
    if weight_mode in ("jobs", "workers"):
        return f"LODES WAC/RAC ({year})"
    if weight_mode == "income_mass":
        return f"ACS 5-Year ({year})"
    return f"Unknown ({year})"


# ---------------------------------------------------------------------------
# Fetch weight data for a county
# ---------------------------------------------------------------------------

def _fetch_county_weights(
    state_fips: str,
    county_fips: str,
    year: int,
    weight_mode: str,
    use_crosswalk: bool = True,
) -> pd.DataFrame:
    """Fetch tract-level weight data for a county, with optional crosswalk.

    Returns DataFrame with columns: GEOID, <weight_col>
    where weight_col is "population", "weight", or "income_mass".
    """
    if weight_mode == "population":
        df = fetch_population(state_fips, county_fips, year)
        if use_crosswalk and _needs_crosswalk(year):
            df = _apply_population_crosswalk(df, state_fips, year=year)
        return df

    elif weight_mode == "jobs":
        # NO crosswalk: LODES8 uses 2020 census geography for ALL years
        # (2002-2023), so jobs GEOIDs are already 2020-vintage. Applying the
        # 2010->2020 crosswalk mis-splits border tracts (verified: 2013 vs 2023
        # LODES tracts overlap 100%).
        return _fetch_lodes_tract_jobs(state_fips, county_fips, year)

    elif weight_mode == "workers":
        # NO crosswalk — LODES8 is single-vintage (2020); see jobs note above.
        return _fetch_lodes_tract_workers(state_fips, county_fips, year)

    elif weight_mode == "income_mass":
        df = _fetch_acs_income_mass(state_fips, county_fips, year)
        if use_crosswalk and _needs_crosswalk(year):
            df = _apply_income_mass_crosswalk(df, state_fips, year=year)
        return df

    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode!r}. Use: {WEIGHT_MODES}")


def _weight_column(weight_mode: str) -> str:
    """Return the column name that holds weights for a given mode."""
    if weight_mode == "population":
        return "population"
    if weight_mode == "income_mass":
        return "income_mass"
    return "weight"


# ---------------------------------------------------------------------------
# Year availability per mode
# ---------------------------------------------------------------------------

_LODES_MIN_YEAR = 2002
_LODES_MAX_YEAR = 2023  # LODES8 latest as of 2026; bump as new vintages release


def _available_years_for_mode(
    weight_mode: str,
    requested_years: list[int],
) -> list[int]:
    """Filter requested years to those available for the weight mode."""
    if weight_mode == "population":
        return [y for y in requested_years if y >= 2000]
    if weight_mode in ("jobs", "workers"):
        return [y for y in requested_years if _LODES_MIN_YEAR <= y <= _LODES_MAX_YEAR]
    if weight_mode == "income_mass":
        return [y for y in requested_years if y >= 2009]
    return requested_years


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def population_center_of_gravity(
    msa_name: str,
    year: int = 2020,
    weight_mode: str = "population",
    use_crosswalk: bool = True,
) -> GravityResult:
    """Calculate weighted center of gravity for an MSA in a given year.

    Args:
        msa_name: MSA name or partial match.
        year: Data year.
        weight_mode: One of "population", "jobs", "workers", "income_mass".
        use_crosswalk: If True, normalize pre-2020 GEOIDs via Census crosswalk.

    Returns:
        GravityResult with coordinates and metadata.
    """
    msa = resolve_msa(msa_name)
    state_fips = msa["state_fips"]
    counties = msa["counties"]
    tiger_vintage = _tiger_vintage_for_year(year)
    weight_col = _weight_column(weight_mode)

    console.print(
        f"[bold]Computing {WEIGHT_LABELS[weight_mode]} center:[/bold] "
        f"{msa['name']} ({year})"
    )

    all_tracts = []
    all_weights = []
    for county_fips, county_name in counties.items():
        try:
            weight_df = _fetch_county_weights(
                state_fips, county_fips, year, weight_mode, use_crosswalk,
            )
            tract_gdf = fetch_tract_geometries(state_fips, county_fips, year)
            all_weights.append(weight_df)
            all_tracts.append(tract_gdf)
            console.print(f"  [dim]{county_name}: {len(weight_df)} tracts[/dim]")
        except Exception as e:
            console.print(f"  [yellow]{county_name}: skipped — {e}[/yellow]")
            logger.warning("Skipping %s (%s): %s", county_name, county_fips, e)

    if not all_tracts:
        raise RuntimeError(
            f"No data for {msa['name']} ({year}, {weight_mode})"
        )

    tracts_gdf = pd.concat(all_tracts, ignore_index=True)
    weight_df = pd.concat(all_weights, ignore_index=True)

    center_lat, center_lon, total_w, tract_count = _compute_weighted_centroid(
        tracts_gdf, weight_df, weight_col,
    )

    notes_parts = []
    if use_crosswalk and _needs_crosswalk(year):
        notes_parts.append(
            f"Crosswalked from 2010-era to 2020 GEOIDs (areal weighting)."
        )
    if weight_mode == "income_mass":
        notes_parts.append(
            "Income mass = occupied_units × median_household_income. "
            "This is an estimated proxy, not true aggregate income."
        )

    return GravityResult(
        msa_name=msa["name"],
        year=year,
        weight_mode=weight_mode,
        total_weight=total_w,
        center_lat=center_lat,
        center_lon=center_lon,
        county_count=len(counties),
        tract_count=tract_count,
        data_source=_data_source_label(weight_mode, year),
        tiger_vintage=tiger_vintage,
        notes=" ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def _default_years(start_year: int, end_year: int) -> list[int]:
    """Generate year list: Decennial + ACS 5-year at reasonable intervals."""
    years = []

    for dec_year in (2000, 2010, 2020):
        if start_year <= dec_year <= end_year:
            years.append(dec_year)

    acs_start = max(start_year, 2009)
    for y in range(acs_start, end_year + 1):
        if y == acs_start or y == end_year or y % 5 == 3:
            if y not in years:
                years.append(y)

    return sorted(set(years))


def gravity_time_series(
    msa_name: str,
    start_year: int = 2000,
    end_year: int = 2024,
    years: list[int] | None = None,
    weight_mode: str = "population",
    use_crosswalk: bool = True,
) -> pd.DataFrame:
    """Compute center of gravity across multiple years for a single weight mode.

    Returns DataFrame with one row per year, columns from GravityResult.
    """
    if years is None:
        years = _default_years(start_year, end_year)

    years = _available_years_for_mode(weight_mode, years)

    console.print(
        f"[bold]{WEIGHT_LABELS[weight_mode]} time series:[/bold] {msa_name}\n"
        f"  Years: {years}"
    )

    results = []
    for year in years:
        try:
            result = population_center_of_gravity(
                msa_name, year=year,
                weight_mode=weight_mode,
                use_crosswalk=use_crosswalk,
            )
            results.append(result.to_dict())
            console.print(
                f"  [green]{year}:[/green] ({result.center_lat:.4f}, "
                f"{result.center_lon:.4f}) "
                f"weight={result.total_weight:,.0f}"
            )
        except Exception as e:
            console.print(f"  [red]{year}:[/red] failed — {e}")
            logger.error("Year %d failed for %s: %s", year, weight_mode, e)

    if not results:
        raise RuntimeError(f"No valid results for {weight_mode}")

    return pd.DataFrame(results)


def multi_weight_time_series(
    msa_name: str,
    start_year: int = 2010,
    end_year: int = 2024,
    years: list[int] | None = None,
    weight_modes: list[str] | None = None,
    use_crosswalk: bool = True,
) -> pd.DataFrame:
    """Compute center of gravity for multiple weight modes across years.

    Returns a single DataFrame with a 'weight_mode' column distinguishing
    each series. Adds distance from population center for non-population modes.
    """
    if weight_modes is None:
        weight_modes = list(WEIGHT_MODES)
    if years is None:
        years = _default_years(start_year, end_year)

    all_series = []
    pop_centers: dict[int, tuple[float, float]] = {}

    # Population first (reference series)
    if "population" in weight_modes:
        pop_df = gravity_time_series(
            msa_name, years=years,
            weight_mode="population",
            use_crosswalk=use_crosswalk,
        )
        pop_df = compute_shift_metrics(pop_df)
        all_series.append(pop_df)
        for _, row in pop_df.iterrows():
            pop_centers[int(row["year"])] = (row["center_lat"], row["center_lon"])

    # Other weight modes
    for mode in weight_modes:
        if mode == "population":
            continue
        try:
            mode_df = gravity_time_series(
                msa_name, years=years,
                weight_mode=mode,
                use_crosswalk=use_crosswalk,
            )
            mode_df = compute_shift_metrics(mode_df)

            # Distance from population center for each year
            divergences = []
            for _, row in mode_df.iterrows():
                yr = int(row["year"])
                if yr in pop_centers:
                    plat, plon = pop_centers[yr]
                    d = _haversine_miles(plat, plon, row["center_lat"], row["center_lon"])
                    b = _bearing_degrees(plat, plon, row["center_lat"], row["center_lon"])
                    divergences.append({"dist_from_pop_center_mi": round(d, 3),
                                        "bearing_from_pop_center": round(b, 1)})
                else:
                    divergences.append({"dist_from_pop_center_mi": np.nan,
                                        "bearing_from_pop_center": np.nan})
            div_df = pd.DataFrame(divergences)
            mode_df = pd.concat([mode_df.reset_index(drop=True), div_df], axis=1)

            all_series.append(mode_df)
        except Exception as e:
            console.print(f"[yellow]{mode}: failed — {e}[/yellow]")
            logger.error("Weight mode %s failed: %s", mode, e)

    if not all_series:
        raise RuntimeError("No weight modes produced results")

    return pd.concat(all_series, ignore_index=True)


# ---------------------------------------------------------------------------
# Shift analysis
# ---------------------------------------------------------------------------

def compute_shift_metrics(series_df: pd.DataFrame) -> pd.DataFrame:
    """Add distance/bearing shift columns relative to the first year.

    Adds: shift_miles, shift_bearing_deg, shift_north_miles, shift_east_miles,
    cumulative_path_miles.
    """
    df = series_df.copy()
    if df.empty:
        return df

    base_lat = df.iloc[0]["center_lat"]
    base_lon = df.iloc[0]["center_lon"]

    shifts = []
    prev_lat, prev_lon = base_lat, base_lon
    cumulative = 0.0

    for _, row in df.iterrows():
        lat, lon = row["center_lat"], row["center_lon"]

        dist = _haversine_miles(base_lat, base_lon, lat, lon)
        bearing = _bearing_degrees(base_lat, base_lon, lat, lon)

        north_miles = _haversine_miles(base_lat, base_lon, lat, base_lon)
        if lat < base_lat:
            north_miles = -north_miles
        east_miles = _haversine_miles(base_lat, base_lon, base_lat, lon)
        if lon < base_lon:
            east_miles = -east_miles

        seg = _haversine_miles(prev_lat, prev_lon, lat, lon)
        cumulative += seg
        prev_lat, prev_lon = lat, lon

        shifts.append({
            "shift_miles": round(dist, 3),
            "shift_bearing_deg": round(bearing, 1),
            "shift_north_miles": round(north_miles, 3),
            "shift_east_miles": round(east_miles, 3),
            "cumulative_path_miles": round(cumulative, 3),
        })

    shift_df = pd.DataFrame(shifts)
    return pd.concat([df.reset_index(drop=True), shift_df], axis=1)


# ---------------------------------------------------------------------------
# Geo math
# ---------------------------------------------------------------------------

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles."""
    R = 3958.8
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing in degrees (0=N, 90=E)."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


# ---------------------------------------------------------------------------
# Trajectory projection
# ---------------------------------------------------------------------------

def project_center(
    series_df: pd.DataFrame,
    target_year: int,
    crs: str = "EPSG:2276",
) -> dict:
    """Linearly extrapolate a center-of-gravity series to a future year.

    Fits independent linear trends to the projected easting/northing over time
    and extrapolates. Returns the projected point plus pace and fit quality so
    the caller can judge whether the drift is even linear.

    This is an *illustrative* extrapolation: it assumes the recent directional
    trend continues, which second-order effects (new development, annexation,
    economic shocks) can easily break. Always report r2 alongside the point.

    Args:
        series_df: One weight mode's series with year, center_lat, center_lon.
        target_year: Year to project to (e.g., 2030).
        crs: Projected CRS for the linear fit (feet/metres).

    Returns:
        dict with target_lat, target_lon, pace_mi_per_decade, trend_bearing_deg,
        r2 (0–1 fit quality), n_points, base_year, target_year.
    """
    df = series_df.sort_values("year")
    years = df["year"].values.astype(float)
    if len(years) < 2:
        raise ValueError("Need at least 2 points to project")

    pts = gpd.GeoSeries(
        [Point(lo, la) for lo, la in zip(df["center_lon"], df["center_lat"])],
        crs="EPSG:4326",
    ).to_crs(crs)
    x = pts.x.values
    y = pts.y.values

    ax, bx = np.polyfit(years, x, 1)
    ay, by = np.polyfit(years, y, 1)

    # Combined R²: fraction of total (x+y) variance explained by the linear fit
    x_pred, y_pred = ax * years + bx, ay * years + by
    ss_res = np.sum((x - x_pred) ** 2) + np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((x - x.mean()) ** 2) + np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    tx = ax * target_year + bx
    ty = ay * target_year + by
    proj_pt = gpd.GeoSeries([Point(tx, ty)], crs=crs).to_crs("EPSG:4326").iloc[0]

    # Units per mile in this CRS (feet for State Plane, metres for UTM)
    from pyproj import CRS as _PyprojCRS
    _unit = _PyprojCRS(crs).axis_info[0].unit_name.lower()
    units_per_mi = 5280.0 if "foot" in _unit else 1609.344
    pace_per_year = np.hypot(ax, ay) / units_per_mi
    trend_bearing = (np.degrees(np.arctan2(ax, ay)) + 360) % 360  # easting vs northing rate

    return {
        "target_lat": round(float(proj_pt.y), 6),
        "target_lon": round(float(proj_pt.x), 6),
        "pace_mi_per_decade": round(float(pace_per_year * 10), 2),
        "trend_bearing_deg": round(float(trend_bearing), 1),
        "r2": round(float(r2), 3),
        "n_points": int(len(years)),
        "base_year": int(years[0]),
        "target_year": int(target_year),
    }


# ---------------------------------------------------------------------------
# Growth-share decomposition
# ---------------------------------------------------------------------------

def growth_share(
    msa_name: str,
    start_year: int,
    end_year: int,
    weight_mode: str = "population",
    by: str = "county",
    use_crosswalk: bool = True,
) -> pd.DataFrame:
    """Decompose net growth by county or directional quadrant.

    Answers "where did the net new people / jobs land?" — the IC-actionable
    question a drifting centroid only hints at.

    Args:
        msa_name: MSA name or partial match.
        start_year, end_year: Period bounds (data years).
        weight_mode: "population", "jobs", "workers", "income_mass".
        by: "county" or "quadrant" (N/E/S/W relative to the start-year center).
        use_crosswalk: Normalize pre-2020 GEOIDs.

    Returns:
        DataFrame sorted by net change descending, columns:
        unit, start, end, net_change, pct_change, share_of_growth.
    """
    msa = resolve_msa(msa_name)
    state_fips = msa["state_fips"]
    counties = msa["counties"]
    weight_col = _weight_column(weight_mode)

    def _fetch_year(year):
        frames = []
        for cf in counties:
            try:
                frames.append(_fetch_county_weights(state_fips, cf, year, weight_mode, use_crosswalk))
            except Exception as e:
                logger.warning("growth_share %s %s: %s", cf, year, e)
        if not frames:
            raise RuntimeError(f"No data for {year}")
        return pd.concat(frames, ignore_index=True).groupby("GEOID", as_index=False)[weight_col].sum()

    start_df = _fetch_year(start_year).rename(columns={weight_col: "start"})
    end_df = _fetch_year(end_year).rename(columns={weight_col: "end"})
    merged = start_df.merge(end_df, on="GEOID", how="outer").fillna(0)

    if by == "county":
        merged["unit"] = merged["GEOID"].str[:5].map(
            lambda fp: counties.get(fp[2:], fp)
        )
    elif by == "quadrant":
        # Assign each tract to a quadrant relative to the start-year center
        base = population_center_of_gravity(
            msa_name, year=start_year, weight_mode="population", use_crosswalk=use_crosswalk,
        )
        # Need tract centroids
        geoms = []
        for cf in counties:
            try:
                geoms.append(fetch_tract_geometries(state_fips, cf, end_year))
            except Exception:
                pass
        tracts = pd.concat(geoms, ignore_index=True)
        cen = tracts.to_crs("EPSG:4326").geometry.centroid
        tracts = tracts.assign(tlat=cen.y.values, tlon=cen.x.values)
        merged = merged.merge(tracts[["GEOID", "tlat", "tlon"]], on="GEOID", how="left")

        def _quad(row):
            if pd.isna(row.get("tlat")):
                return "—"
            b = _bearing_degrees(base.center_lat, base.center_lon, row["tlat"], row["tlon"])
            if b >= 315 or b < 45:
                return "North"
            if b < 135:
                return "East"
            if b < 225:
                return "South"
            return "West"

        merged["unit"] = merged.apply(_quad, axis=1)
    else:
        raise ValueError("by must be 'county' or 'quadrant'")

    agg = merged.groupby("unit", as_index=False).agg(start=("start", "sum"), end=("end", "sum"))
    agg["net_change"] = agg["end"] - agg["start"]
    agg["pct_change"] = np.where(agg["start"] > 0, agg["net_change"] / agg["start"] * 100, np.nan)
    total_net = agg["net_change"].sum()
    agg["share_of_growth"] = np.where(total_net != 0, agg["net_change"] / total_net * 100, np.nan)

    agg = agg.sort_values("net_change", ascending=False).reset_index(drop=True)
    agg.attrs["weight_mode"] = weight_mode
    agg.attrs["start_year"] = start_year
    agg.attrs["end_year"] = end_year
    agg.attrs["by"] = by
    return agg


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_results(
    series_df: pd.DataFrame,
    output_dir: str | pathlib.Path,
    formats: list[str] | None = None,
    prefix: str = "gravity_centers",
) -> list[pathlib.Path]:
    """Export time series results to disk.

    Args:
        series_df: DataFrame from gravity_time_series() or multi_weight_time_series().
        output_dir: Directory to write files.
        formats: "csv", "parquet", "geojson". Default: all three.
        prefix: Filename prefix (default "gravity_centers").

    Returns:
        List of written file paths.
    """
    if formats is None:
        formats = ["csv", "parquet", "geojson"]

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    if "csv" in formats:
        path = out / f"{prefix}.csv"
        series_df.to_csv(path, index=False)
        written.append(path)
        console.print(f"  [green]Wrote:[/green] {path}")

    if "parquet" in formats:
        path = out / f"{prefix}.parquet"
        series_df.to_parquet(path, index=False)
        written.append(path)
        console.print(f"  [green]Wrote:[/green] {path}")

    if "geojson" in formats:
        path = out / f"{prefix}.geojson"
        points = [
            Point(row["center_lon"], row["center_lat"])
            for _, row in series_df.iterrows()
        ]
        gdf = gpd.GeoDataFrame(series_df, geometry=points, crs="EPSG:4326")
        gdf.to_file(path, driver="GeoJSON")
        written.append(path)
        console.print(f"  [green]Wrote:[/green] {path}")

    return written
