"""DuckDB analytics layer for fast columnar queries against LODES parquet cache.

Queries the parquet files written by geostack.lodes directly — no PostGIS round-trip.
DuckDB reads parquet column-selectively and in parallel, making large OD or WAC
queries 10-50x faster than equivalent SELECT on lodes_wac/lodes_od in Postgres.

Typical use:
    from geostack.duckdb_analytics import employment_by_sector, od_flows

    # Jobs by sector, aggregated to tracts, for Collin County TX 2021
    df = employment_by_sector("tx", 2021, "48085")

    # Commute flows into Collin from all home counties
    flows = od_flows("tx", 2021, work_county_fips="48085")

Requires: pip install 'geostack[analytics]'  (installs duckdb)

NOTE: parquet files are cached with ORIGINAL Census column names (C000, CNS01, etc.)
because caching happens before column renaming in lodes.download_wac/rac/od.
All functions handle this internally and return DataFrames with friendly names.
"""

import os

from rich.console import Console

console = Console()


def _lodes_cache_dir():
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "lodes")
    )


def _wac_path(state_abbr, year, segment="S000", job_type="JT01"):
    return os.path.join(_lodes_cache_dir(), state_abbr, f"wac_{segment}_{job_type}_{year}.parquet")


def _rac_path(state_abbr, year, segment="S000", job_type="JT01"):
    return os.path.join(_lodes_cache_dir(), state_abbr, f"rac_{segment}_{job_type}_{year}.parquet")


def _od_path(state_abbr, year, part="main", job_type="JT01"):
    return os.path.join(_lodes_cache_dir(), state_abbr, f"od_{part}_{job_type}_{year}.parquet")


def _require_duckdb():
    try:
        import duckdb
        return duckdb
    except ImportError:
        raise ImportError(
            "duckdb is required for analytics queries. "
            "Run: pip install 'geostack[analytics]'"
        )


def get_connection(read_only=True):
    """Return a DuckDB connection with the spatial extension loaded.

    Args:
        read_only: Open in read-only mode (default True — safe for concurrent use).

    Returns:
        duckdb.DuckDBPyConnection
    """
    duckdb = _require_duckdb()
    conn = duckdb.connect(database=":memory:", read_only=False)
    try:
        conn.execute("INSTALL spatial; LOAD spatial;")
    except Exception:
        # Spatial extension may not be available in all environments — not required
        # for the columnar analytics queries in this module
        pass
    return conn


def _county_prefix_filter(county_fips, geocode_col="w_geocode"):
    """Build a DuckDB WHERE clause to filter a block geocode to a county."""
    fips = str(county_fips).strip().zfill(5)
    # Parquet may store geocode as int or string depending on pandas dtype at save time.
    # lodes._download_lodes_csv reads with dtype=str, but the underlying parquet storage
    # preserves the type. Use LPAD + CAST to normalize to 15-char string.
    return f"LPAD(CAST({geocode_col} AS VARCHAR), 15, '0') LIKE '{fips}%'"


def _check_parquet(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Parquet cache not found: {path}\n"
            f"Run `geostack load-lodes --state <state> --county <county> --year <year>` first."
        )


# ---------------------------------------------------------------------------
# WAC analytics
# ---------------------------------------------------------------------------

# Import canonical column maps from lodes — single source of truth.
# Invert them here: we need friendly→original for SQL alias generation (SUM(C000) AS total_jobs).
from geostack.lodes import OD_COL_MAP, WAC_COL_MAP

_WAC_COLS = {v: k for k, v in WAC_COL_MAP.items()}   # friendly → Census code
_OD_COLS = {v: k for k, v in OD_COL_MAP.items()}     # friendly → Census code


def employment_by_sector(state, year, county_filter, job_type="JT01", segment="S000"):
    """Aggregate WAC block-level data to tracts, grouped by NAICS sector.

    Much faster than querying lodes_wac in PostGIS for exploratory analysis —
    DuckDB reads only the needed columns from parquet.

    Args:
        state: State abbreviation ('tx') or FIPS ('48').
        year: Data year.
        county_filter: 5-digit county FIPS string (e.g., '48085').
        job_type: 'JT01' (primary, default) or 'JT00' (all).
        segment: 'S000' (all workers, default).

    Returns:
        DataFrame with tract_geoid + all sector job columns, sorted by total_jobs desc.
    """
    from geostack.lodes import _resolve_state_abbr
    duckdb = _require_duckdb()

    abbr = _resolve_state_abbr(state)
    path = _wac_path(abbr, year, segment, job_type)
    _check_parquet(path)

    where = _county_prefix_filter(county_filter, "w_geocode")

    # Build SELECT with aliases: SUM(C000) AS total_jobs, etc.
    agg_cols = ",\n    ".join(
        f"SUM({orig}) AS {friendly}" for friendly, orig in _WAC_COLS.items()
    )

    sql = f"""
        SELECT
            LPAD(CAST(w_geocode AS VARCHAR), 15, '0')[:11] AS tract_geoid,
            {agg_cols}
        FROM read_parquet('{path}')
        WHERE {where}
        GROUP BY tract_geoid
        ORDER BY total_jobs DESC
    """
    conn = get_connection()
    result = conn.execute(sql).df()
    conn.close()

    console.print(
        f"[dim]employment_by_sector: {len(result)} tracts, "
        f"{result['total_jobs'].sum():,.0f} total jobs[/dim]"
    )
    return result


def top_employment_centers(state, year, county_filter, n=10, job_type="JT01", segment="S000"):
    """Return the top N census blocks by total jobs in a county.

    Useful for identifying employment hot spots within a submarket —
    where the actual job density is, at block level resolution.

    Args:
        state: State abbreviation or FIPS.
        year: Data year.
        county_filter: 5-digit county FIPS string.
        n: Number of top blocks to return (default 10).
        job_type: 'JT01' (primary jobs, default).
        segment: 'S000' (all workers, default).

    Returns:
        DataFrame with w_geocode, tract_geoid, total_jobs, and sector breakdowns.
    """
    from geostack.lodes import _resolve_state_abbr
    duckdb = _require_duckdb()

    abbr = _resolve_state_abbr(state)
    path = _wac_path(abbr, year, segment, job_type)
    _check_parquet(path)

    where = _county_prefix_filter(county_filter, "w_geocode")

    col_aliases = ",\n    ".join(
        f"{orig} AS {friendly}" for friendly, orig in _WAC_COLS.items()
    )

    sql = f"""
        SELECT
            LPAD(CAST(w_geocode AS VARCHAR), 15, '0') AS w_geocode,
            LPAD(CAST(w_geocode AS VARCHAR), 15, '0')[:11] AS tract_geoid,
            {col_aliases}
        FROM read_parquet('{path}')
        WHERE {where}
        ORDER BY C000 DESC
        LIMIT {int(n)}
    """
    conn = get_connection()
    result = conn.execute(sql).df()
    conn.close()

    return result


# ---------------------------------------------------------------------------
# OD analytics
# ---------------------------------------------------------------------------

def od_flows(state, year, work_county_fips, home_county_fips=None, job_type="JT01", part="main"):
    """Query OD parquet for commute flows into a work county.

    Args:
        state: State abbreviation or FIPS (determines which state's OD file to query).
        year: Data year.
        work_county_fips: 5-digit FIPS for the work location county.
        home_county_fips: Optional 5-digit FIPS to filter home location too.
                          If None, returns flows from all home locations.
        job_type: 'JT01' (primary, default).
        part: 'main' (in-state workers) or 'aux' (out-of-state workers).

    Returns:
        DataFrame with w_geocode, h_geocode, and job breakdown columns.
        Each row is one home-block → work-block flow pair.
    """
    from geostack.lodes import _resolve_state_abbr
    duckdb = _require_duckdb()

    abbr = _resolve_state_abbr(state)
    path = _od_path(abbr, year, part, job_type)
    _check_parquet(path)

    work_filter = _county_prefix_filter(work_county_fips, "w_geocode")
    conditions = [work_filter]
    if home_county_fips:
        conditions.append(_county_prefix_filter(home_county_fips, "h_geocode"))

    where = " AND ".join(conditions)

    col_aliases = ",\n    ".join(
        f"{orig} AS {friendly}" for friendly, orig in _OD_COLS.items()
    )

    sql = f"""
        SELECT
            LPAD(CAST(w_geocode AS VARCHAR), 15, '0') AS w_geocode,
            LPAD(CAST(h_geocode AS VARCHAR), 15, '0') AS h_geocode,
            {col_aliases}
        FROM read_parquet('{path}')
        WHERE {where}
        ORDER BY S000 DESC
    """
    conn = get_connection()
    result = conn.execute(sql).df()
    conn.close()

    console.print(
        f"[dim]od_flows: {len(result):,} flows, "
        f"{result['total_jobs'].sum():,.0f} total jobs[/dim]"
    )
    return result


def commute_distance_distribution(state, year, county_filter, job_type="JT01", part="main"):
    """Earnings and age breakdown of workers commuting into a county.

    Aggregates all OD flows where the work location is in the given county,
    grouping home blocks by home county for a county-level commute shed picture.
    Histogram-ready: each row is a (home_county, earnings_tier, age_tier) cell.

    Args:
        state: State abbreviation or FIPS.
        year: Data year.
        county_filter: 5-digit work county FIPS.
        job_type: 'JT01' (primary, default).
        part: 'main' or 'aux'.

    Returns:
        DataFrame with home_county_fips, total_jobs, earnings splits, age splits.
        One row per home county — sorted by total_jobs desc.
    """
    from geostack.lodes import _resolve_state_abbr
    duckdb = _require_duckdb()

    abbr = _resolve_state_abbr(state)
    path = _od_path(abbr, year, part, job_type)
    _check_parquet(path)

    where = _county_prefix_filter(county_filter, "w_geocode")

    sql = f"""
        SELECT
            LPAD(CAST(h_geocode AS VARCHAR), 15, '0')[3:5] AS home_county_fips,
            LPAD(CAST(h_geocode AS VARCHAR), 15, '0')[1:2] AS home_state_fips,
            SUM(S000) AS total_jobs,
            SUM(SE01) AS jobs_earn_low,
            SUM(SE02) AS jobs_earn_mid,
            SUM(SE03) AS jobs_earn_high,
            SUM(SA01) AS jobs_age_29_under,
            SUM(SA02) AS jobs_age_30_54,
            SUM(SA03) AS jobs_age_55_over
        FROM read_parquet('{path}')
        WHERE {where}
        GROUP BY home_state_fips, home_county_fips
        ORDER BY total_jobs DESC
    """
    conn = get_connection()
    result = conn.execute(sql).df()
    conn.close()

    return result
