"""LEHD Origin-Destination Employment Statistics (LODES) data client.

Downloads and processes LODES8 data from the Census Bureau's LEHD program.
Three dataset types:

  WAC — Workplace Area Characteristics: jobs at each census block
  RAC — Residential Area Characteristics: workers living at each census block
  OD  — Origin-Destination: home-block → work-block job flows

LODES8 uses 2020 census block definitions. Data available roughly 2002–2021.
No API key required — files are flat CSV.gz downloads.

Usage:
    from geostack.lodes import download_wac, load_wac_to_postgis

    # Pull workplace employment for Collin County TX, primary jobs, 2021
    df = download_wac(state="tx", year=2021, county_filter="48085")
    engine = get_engine()
    load_wac_to_postgis(df, engine, year=2021)

    # Query employment profile for a county
    profile = county_employment_profile(state_fips="48", county_fips="085", year=2021)
"""

import io
import os

import pandas as pd
import requests
from rich.console import Console

console = Console()

LODES_BASE = "https://lehd.ces.census.gov/data/lodes/LODES8"

# Job type codes
JOB_TYPES = {
    "JT00": "All Jobs",
    "JT01": "Primary Jobs",
    "JT02": "All Private Jobs",
    "JT03": "Private Primary Jobs",
    "JT04": "All Federal Jobs",
    "JT05": "Federal Primary Jobs",
}

# Worker/job segment codes
SEGMENTS = {
    "S000": "All",
    "SA01": "Age ≤ 29",
    "SA02": "Age 30–54",
    "SA03": "Age ≥ 55",
    "SE01": "Earnings ≤ $1,250/mo",
    "SE02": "Earnings $1,251–$3,333/mo",
    "SE03": "Earnings > $3,333/mo",
    "SI01": "Goods Producing",
    "SI02": "Trade/Transport/Utilities",
    "SI03": "Other Services",
}

# NAICS sector labels (CNS columns in WAC/RAC)
NAICS_SECTORS = {
    "CNS01": "agriculture",
    "CNS02": "mining_oil_gas",
    "CNS03": "utilities",
    "CNS04": "construction",
    "CNS05": "manufacturing",
    "CNS06": "wholesale_trade",
    "CNS07": "retail",
    "CNS08": "transport_warehousing",
    "CNS09": "information",
    "CNS10": "finance_insurance",
    "CNS11": "real_estate",
    "CNS12": "professional_scientific",
    "CNS13": "management",
    "CNS14": "admin_support",
    "CNS15": "education",
    "CNS16": "healthcare",
    "CNS17": "arts_entertainment",
    "CNS18": "accommodation_food",
    "CNS19": "other_services",
    "CNS20": "public_admin",
}

# Canonical WAC column mapping: Census code → friendly name (single source of truth).
# Used by download_wac for renaming and by duckdb_analytics for SQL aliases.
WAC_COL_MAP = {
    "C000": "total_jobs",
    "CA01": "jobs_age_29_under", "CA02": "jobs_age_30_54", "CA03": "jobs_age_55_over",
    "CE01": "jobs_earn_low", "CE02": "jobs_earn_mid", "CE03": "jobs_earn_high",
}
WAC_COL_MAP.update({k: f"jobs_{v}" for k, v in NAICS_SECTORS.items()})

# OD column mapping: Census code → friendly name
OD_COL_MAP = {
    "S000": "total_jobs",
    "SA01": "jobs_age_29_under", "SA02": "jobs_age_30_54", "SA03": "jobs_age_55_over",
    "SE01": "jobs_earn_low", "SE02": "jobs_earn_mid", "SE03": "jobs_earn_high",
    "SI01": "jobs_goods_producing", "SI02": "jobs_trade_transport_util", "SI03": "jobs_other_services",
}

# FIPS → lowercase state abbreviation (LODES uses lowercase abbr in URLs)
FIPS_TO_ABBR = {
    "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca",
    "08": "co", "09": "ct", "10": "de", "11": "dc", "12": "fl",
    "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in",
    "19": "ia", "20": "ks", "21": "ky", "22": "la", "23": "me",
    "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms",
    "29": "mo", "30": "mt", "31": "ne", "32": "nv", "33": "nh",
    "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd",
    "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri",
    "45": "sc", "46": "sd", "47": "tn", "48": "tx", "49": "ut",
    "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi",
    "56": "wy",
}


def _resolve_state_abbr(state):
    """Accept FIPS ('48') or abbreviation ('tx') and return lowercase abbr."""
    s = str(state).strip().lower()
    if len(s) == 2 and s.isdigit():
        abbr = FIPS_TO_ABBR.get(s)
        if not abbr:
            raise ValueError(f"Unknown state FIPS: {s}")
        return abbr
    return s  # already an abbreviation


def _lodes_cache_dir():
    cache = os.path.join(os.path.dirname(__file__), "..", "..", "data", "lodes")
    return os.path.normpath(cache)


def _download_lodes_csv(url, cache_path=None):
    """Download a LODES .csv.gz file and return a DataFrame.

    Caches locally to avoid re-downloading large state files.
    """
    if cache_path and os.path.exists(cache_path):
        console.print(f"[dim]Using cached: {os.path.basename(cache_path)}[/dim]")
        return pd.read_parquet(cache_path)

    console.print(f"[dim]Downloading: {url}[/dim]")
    resp = requests.get(url, stream=True, timeout=300)
    if resp.status_code == 404:
        raise FileNotFoundError(
            f"LODES file not found (404). State/year combination may not be available.\n{url}"
        )
    resp.raise_for_status()

    content = b""
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1024 * 256):
        content += chunk
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            console.print(f"[dim]  {pct:.0f}%...[/dim]", end="\r")
    console.print()

    df = pd.read_csv(io.BytesIO(content), compression="gzip", dtype={"w_geocode": str, "h_geocode": str})

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)
        console.print(f"[dim]Cached to {cache_path}[/dim]")

    return df


def download_wac(
    state,
    year=2021,
    segment="S000",
    job_type="JT01",
    county_filter=None,
    cache=True,
):
    """Download LODES8 Workplace Area Characteristics (WAC) data.

    WAC describes where people work: total jobs and breakdown by age, earnings,
    and NAICS sector at the census block level.

    Args:
        state: State FIPS ('48') or abbreviation ('tx').
        year: Data year (LODES8 covers ~2002–2021; most recent available recommended).
        segment: Worker segment — 'S000' (all), 'SE01'–'SE03' (by earnings), etc.
        job_type: 'JT01' (primary jobs, recommended) or 'JT00' (all jobs).
        county_filter: Optional 5-digit county FIPS (e.g., '48085') to filter
                       the full state file down. LODES files are state-level.
        cache: Cache the raw state-level file locally (recommended — TX WAC is ~180 MB).

    Returns:
        DataFrame with one row per census block in the state (or county if filtered).
        Includes renamed columns: total_jobs, jobs_retail, jobs_healthcare, etc.
    """
    abbr = _resolve_state_abbr(state)
    filename = f"{abbr}_wac_{segment}_{job_type}_{year}.csv.gz"
    url = f"{LODES_BASE}/{abbr}/wac/{filename}"

    cache_path = None
    if cache:
        cache_path = os.path.join(_lodes_cache_dir(), abbr, f"wac_{segment}_{job_type}_{year}.parquet")

    df = _download_lodes_csv(url, cache_path)

    # Ensure geocode is zero-padded string
    df["w_geocode"] = df["w_geocode"].astype(str).str.zfill(15)

    if county_filter:
        fips = str(county_filter).strip().zfill(5)
        df = df[df["w_geocode"].str.startswith(fips)].copy()
        console.print(f"[dim]Filtered to county {fips}: {len(df):,} blocks[/dim]")

    # Extract geographic identifiers from block geocode
    df["state_fips"] = df["w_geocode"].str[:2]
    df["county_fips"] = df["w_geocode"].str[2:5]
    df["tract_geoid"] = df["w_geocode"].str[:11]

    # Rename NAICS columns to readable names
    rename = {"C000": "total_jobs", "CA01": "jobs_age_29_under", "CA02": "jobs_age_30_54",
               "CA03": "jobs_age_55_over", "CE01": "jobs_earn_low", "CE02": "jobs_earn_mid",
               "CE03": "jobs_earn_high"}
    rename.update({k: f"jobs_{v}" for k, v in NAICS_SECTORS.items()})
    df = df.rename(columns=rename)

    console.print(f"[green]WAC {year}: {len(df):,} blocks, {df['total_jobs'].sum():,.0f} total jobs[/green]")
    return df


def download_rac(
    state,
    year=2021,
    segment="S000",
    job_type="JT01",
    county_filter=None,
    cache=True,
):
    """Download LODES8 Residential Area Characteristics (RAC) data.

    RAC describes where workers live: total workers and breakdown by age, earnings,
    and industry of employment at the census block level.

    Args:
        state: State FIPS or abbreviation.
        year: Data year.
        segment: Worker segment — 'S000' (all), 'SE01'–'SE03' (by earnings), etc.
        job_type: 'JT01' (primary jobs) or 'JT00' (all).
        county_filter: 5-digit FIPS to filter to a county.
        cache: Cache the raw state file.

    Returns:
        DataFrame with one row per census block. Includes renamed columns:
        total_workers, workers_earn_low, workers_in_healthcare, etc.
    """
    abbr = _resolve_state_abbr(state)
    filename = f"{abbr}_rac_{segment}_{job_type}_{year}.csv.gz"
    url = f"{LODES_BASE}/{abbr}/rac/{filename}"

    cache_path = None
    if cache:
        cache_path = os.path.join(_lodes_cache_dir(), abbr, f"rac_{segment}_{job_type}_{year}.parquet")

    df = _download_lodes_csv(url, cache_path)
    df["h_geocode"] = df["h_geocode"].astype(str).str.zfill(15)

    if county_filter:
        fips = str(county_filter).strip().zfill(5)
        df = df[df["h_geocode"].str.startswith(fips)].copy()
        console.print(f"[dim]Filtered to county {fips}: {len(df):,} blocks[/dim]")

    df["state_fips"] = df["h_geocode"].str[:2]
    df["county_fips"] = df["h_geocode"].str[2:5]
    df["tract_geoid"] = df["h_geocode"].str[:11]

    rename = {"C000": "total_workers", "CA01": "workers_age_29_under", "CA02": "workers_age_30_54",
               "CA03": "workers_age_55_over", "CE01": "workers_earn_low", "CE02": "workers_earn_mid",
               "CE03": "workers_earn_high"}
    rename.update({k: f"workers_in_{v}" for k, v in NAICS_SECTORS.items()})
    df = df.rename(columns=rename)

    console.print(f"[green]RAC {year}: {len(df):,} blocks, {df['total_workers'].sum():,.0f} total workers[/green]")
    return df


def download_od(
    state,
    year=2021,
    job_type="JT01",
    part="main",
    county_filter=None,
    cache=True,
):
    """Download LODES8 Origin-Destination (OD) flow data.

    OD captures home-block → work-block job counts. Useful for commute shed
    analysis: how many workers commute into a submarket, and from where?

    Args:
        state: State FIPS or abbreviation.
        year: Data year.
        job_type: 'JT01' (primary jobs) or 'JT00' (all).
        part: 'main' (workers who live AND work in-state) or
              'aux' (workers who live out-of-state but work in-state).
        county_filter: Filter to jobs in this 5-digit county FIPS (work location).
        cache: Cache the raw state file.

    Returns:
        DataFrame with columns: w_geocode, h_geocode, total_jobs,
        jobs_age_*, jobs_earn_*, jobs_goods/trade/services.
    """
    abbr = _resolve_state_abbr(state)
    filename = f"{abbr}_od_{part}_{job_type}_{year}.csv.gz"
    url = f"{LODES_BASE}/{abbr}/od/{filename}"

    cache_path = None
    if cache:
        cache_path = os.path.join(
            _lodes_cache_dir(), abbr, f"od_{part}_{job_type}_{year}.parquet"
        )

    df = _download_lodes_csv(url, cache_path)
    df["w_geocode"] = df["w_geocode"].astype(str).str.zfill(15)
    df["h_geocode"] = df["h_geocode"].astype(str).str.zfill(15)

    if county_filter:
        fips = str(county_filter).strip().zfill(5)
        df = df[df["w_geocode"].str.startswith(fips)].copy()
        console.print(f"[dim]Filtered to work county {fips}: {len(df):,} flows[/dim]")

    # Geographic fields for work and home locations
    df["w_state_fips"] = df["w_geocode"].str[:2]
    df["w_county_fips"] = df["w_geocode"].str[2:5]
    df["h_state_fips"] = df["h_geocode"].str[:2]
    df["h_county_fips"] = df["h_geocode"].str[2:5]

    rename = {
        "S000": "total_jobs",
        "SA01": "jobs_age_29_under", "SA02": "jobs_age_30_54", "SA03": "jobs_age_55_over",
        "SE01": "jobs_earn_low", "SE02": "jobs_earn_mid", "SE03": "jobs_earn_high",
        "SI01": "jobs_goods_producing", "SI02": "jobs_trade_transport_util",
        "SI03": "jobs_other_services",
    }
    df = df.rename(columns=rename)

    console.print(
        f"[green]OD {year} ({part}): {len(df):,} flows, "
        f"{df['total_jobs'].sum():,.0f} total jobs[/green]"
    )
    return df


def load_wac_to_postgis(df, engine, year, job_type="JT01", segment="S000"):
    """Upsert WAC block-level employment data into lodes_wac.

    Skips rows already present (idempotent on w_geocode + year + job_type + segment).

    Args:
        df: DataFrame from download_wac().
        engine: SQLAlchemy engine pointing to the geostack database.
        year: Data year (stored in the table for time-series queries).
        job_type: Job type code stored with the data.
        segment: Segment code stored with the data.
    """
    from sqlalchemy import text

    df = df.copy()
    df["year"] = year
    df["job_type"] = job_type
    df["segment"] = segment

    keep_cols = [
        "w_geocode", "state_fips", "county_fips", "tract_geoid",
        "year", "job_type", "segment",
        "total_jobs", "jobs_age_29_under", "jobs_age_30_54", "jobs_age_55_over",
        "jobs_earn_low", "jobs_earn_mid", "jobs_earn_high",
    ] + [f"jobs_{v}" for v in NAICS_SECTORS.values()]

    df = df[[c for c in keep_cols if c in df.columns]]

    # Load into a temp table then upsert
    with engine.begin() as conn:
        existing = conn.execute(
            text(
                "SELECT w_geocode FROM lodes_wac "
                "WHERE year = :y AND job_type = :jt AND segment = :seg"
            ),
            {"y": year, "jt": job_type, "seg": segment},
        ).fetchall()
        existing_set = {r[0] for r in existing}

    new_rows = df[~df["w_geocode"].isin(existing_set)]
    if new_rows.empty:
        console.print("[dim]WAC — all blocks already loaded, skipping.[/dim]")
        return 0

    new_rows.to_sql("lodes_wac", engine, if_exists="append", index=False)
    console.print(f"[green]Loaded {len(new_rows):,} blocks → lodes_wac[/green]")
    return len(new_rows)


def load_rac_to_postgis(df, engine, year, job_type="JT01", segment="S000"):
    """Upsert RAC block-level residential worker data into lodes_rac."""
    from sqlalchemy import text

    df = df.copy()
    df["year"] = year
    df["job_type"] = job_type
    df["segment"] = segment

    keep_cols = [
        "h_geocode", "state_fips", "county_fips", "tract_geoid",
        "year", "job_type", "segment",
        "total_workers", "workers_age_29_under", "workers_age_30_54", "workers_age_55_over",
        "workers_earn_low", "workers_earn_mid", "workers_earn_high",
    ] + [f"workers_in_{v}" for v in NAICS_SECTORS.values()]

    df = df[[c for c in keep_cols if c in df.columns]]

    with engine.begin() as conn:
        existing = conn.execute(
            text(
                "SELECT h_geocode FROM lodes_rac "
                "WHERE year = :y AND job_type = :jt AND segment = :seg"
            ),
            {"y": year, "jt": job_type, "seg": segment},
        ).fetchall()
        existing_set = {r[0] for r in existing}

    new_rows = df[~df["h_geocode"].isin(existing_set)]
    if new_rows.empty:
        console.print("[dim]RAC — all blocks already loaded, skipping.[/dim]")
        return 0

    new_rows.to_sql("lodes_rac", engine, if_exists="append", index=False)
    console.print(f"[green]Loaded {len(new_rows):,} blocks → lodes_rac[/green]")
    return len(new_rows)


def load_od_to_postgis(df, engine, year, job_type="JT01", part="main"):
    """Upsert OD flow data into lodes_od."""
    from sqlalchemy import text

    df = df.copy()
    df["year"] = year
    df["job_type"] = job_type

    keep_cols = [
        "w_geocode", "h_geocode",
        "w_state_fips", "w_county_fips", "h_state_fips", "h_county_fips",
        "year", "job_type",
        "total_jobs", "jobs_age_29_under", "jobs_age_30_54", "jobs_age_55_over",
        "jobs_earn_low", "jobs_earn_mid", "jobs_earn_high",
        "jobs_goods_producing", "jobs_trade_transport_util", "jobs_other_services",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Scope the idempotency check to this specific work county, not the whole table.
    # Otherwise, loading county B after county A would find existing > 0 and silently skip B.
    w_state = df["w_state_fips"].iloc[0] if "w_state_fips" in df.columns else None
    w_county = df["w_county_fips"].iloc[0] if "w_county_fips" in df.columns else None

    with engine.begin() as conn:
        if w_state and w_county:
            existing = conn.execute(
                text(
                    "SELECT COUNT(*) FROM lodes_od "
                    "WHERE w_state_fips = :s AND w_county_fips = :c AND year = :y AND job_type = :jt"
                ),
                {"s": w_state, "c": w_county, "y": year, "jt": job_type},
            ).scalar()
        else:
            existing = conn.execute(
                text("SELECT COUNT(*) FROM lodes_od WHERE year = :y AND job_type = :jt"),
                {"y": year, "jt": job_type},
            ).scalar()

    if existing > 0:
        console.print(
            f"[dim]OD {year} {job_type} already loaded for county "
            f"{w_state}{w_county} ({existing:,} rows), skipping.[/dim]"
        )
        return 0

    df.to_sql("lodes_od", engine, if_exists="append", index=False)
    console.print(f"[green]Loaded {len(df):,} OD flows → lodes_od[/green]")
    return len(df)


def county_employment_profile(state_fips, county_fips, year=2021, engine=None):
    """Query aggregated employment profile for a county from lodes_wac.

    Summarizes total jobs, wage tiers, and top sectors across all blocks
    in the county. Requires lodes_wac to be populated first via load-lodes.

    Args:
        state_fips: 2-digit state FIPS.
        county_fips: 3-digit county FIPS.
        year: Data year.
        engine: SQLAlchemy engine (defaults to DATABASE_URL env var).

    Returns:
        DataFrame with one row containing aggregated employment statistics.
    """
    from geostack.db import get_engine, read_sql

    engine = engine or get_engine()

    sector_sums = ", ".join(
        f'SUM(jobs_{v}) AS jobs_{v}' for v in NAICS_SECTORS.values()
    )
    sql = f"""
        SELECT
            COUNT(*) AS blocks_with_jobs,
            SUM(total_jobs) AS total_jobs,
            SUM(jobs_earn_low) AS jobs_earn_low,
            SUM(jobs_earn_mid) AS jobs_earn_mid,
            SUM(jobs_earn_high) AS jobs_earn_high,
            SUM(jobs_age_29_under) AS jobs_age_29_under,
            SUM(jobs_age_30_54) AS jobs_age_30_54,
            SUM(jobs_age_55_over) AS jobs_age_55_over,
            {sector_sums}
        FROM lodes_wac
        WHERE state_fips = :s AND county_fips = :c AND year = :y AND job_type = 'JT01' AND segment = 'S000'
    """
    return read_sql(sql, engine=engine, params={"s": state_fips, "c": county_fips, "y": year})


def tract_employment(state_fips, county_fips, year=2021, engine=None):
    """Aggregate WAC block-level data to census tracts.

    Returns total jobs per tract, joinable to census_tracts geometry by tract_geoid.
    This is the primary way to create choropleth maps of employment density.

    Args:
        state_fips: 2-digit state FIPS.
        county_fips: 3-digit county FIPS.
        year: Data year.
        engine: SQLAlchemy engine.

    Returns:
        DataFrame with tract_geoid + employment columns. Join to census_tracts
        geometry via: gdf.merge(df, on='geoid') after renaming tract_geoid → geoid.
    """
    from geostack.db import get_engine, read_sql

    engine = engine or get_engine()

    sector_sums = ", ".join(
        f'SUM(jobs_{v}) AS jobs_{v}' for v in NAICS_SECTORS.values()
    )
    sql = f"""
        SELECT
            tract_geoid,
            SUM(total_jobs) AS total_jobs,
            SUM(jobs_earn_low) AS jobs_earn_low,
            SUM(jobs_earn_mid) AS jobs_earn_mid,
            SUM(jobs_earn_high) AS jobs_earn_high,
            {sector_sums}
        FROM lodes_wac
        WHERE state_fips = :s AND county_fips = :c AND year = :y
              AND job_type = 'JT01' AND segment = 'S000'
        GROUP BY tract_geoid
        ORDER BY total_jobs DESC
    """
    return read_sql(sql, engine=engine, params={"s": state_fips, "c": county_fips, "y": year})


def commute_shed(w_state_fips, w_county_fips, year=2021, engine=None):
    """Summarize where workers commuting INTO a county come from.

    Aggregates OD data by home county to show the commute shed.
    Useful for understanding workforce catchment of a submarket.

    Args:
        w_state_fips: Work location state FIPS.
        w_county_fips: Work location county FIPS.
        year: Data year.
        engine: SQLAlchemy engine.

    Returns:
        DataFrame with home county, state, and job counts — sorted by volume.
    """
    from geostack.db import get_engine, read_sql

    engine = engine or get_engine()
    sql = """
        SELECT
            h_state_fips,
            h_county_fips,
            SUM(total_jobs) AS total_jobs,
            SUM(jobs_earn_low) AS jobs_earn_low,
            SUM(jobs_earn_mid) AS jobs_earn_mid,
            SUM(jobs_earn_high) AS jobs_earn_high
        FROM lodes_od
        WHERE w_state_fips = :ws AND w_county_fips = :wc AND year = :y
        GROUP BY h_state_fips, h_county_fips
        ORDER BY total_jobs DESC
    """
    return read_sql(sql, engine=engine, params={"ws": w_state_fips, "wc": w_county_fips, "y": year})
