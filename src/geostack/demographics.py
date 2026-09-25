"""Isochrone-based demographic and labor shed analytics.

Higher-level functions that combine routing, Census ACS, LODES, and
TIGER data to produce institutional-quality demographic profiles for
arbitrary locations — centered on drive-time isochrones rather than
simple radius rings.

Usage:
    from geostack.demographics import (
        tracts_within_isochrone,
        isochrone_demographics,
        isochrone_employment,
        isochrone_commute_shed,
        acs_time_series,
    )

    # Get tracts within a 15-minute drive of a property
    tracts = tracts_within_isochrone(-96.6183, 32.8028, minutes=15, state_fips="48")

    # Full demographic profile within that isochrone
    tract_gdf, summary = isochrone_demographics(
        lon=-96.6183, lat=32.8028, minutes=15,
        state_fips="48", county_fips="113", year=2023,
    )

    # Employment profile within the isochrone
    emp_tracts, emp_summary = isochrone_employment(
        lon=-96.6183, lat=32.8028, minutes=15,
        state_fips="48", county_fips="113", year=2021,
    )

    # Where do workers in this catchment commute from?
    shed = isochrone_commute_shed(
        lon=-96.6183, lat=32.8028, minutes=15,
        state_fips="48", county_fips="113", year=2021,
    )
"""

import pandas as pd
from rich.console import Console

from geostack.census import (
    ACS_VARIABLES,
    _AVG_VARIABLES,
    _SUM_VARIABLES,
    get_tract_demographics,
)
from geostack.db import get_engine, read_postgis, read_sql

console = Console()

# Cache for isochrone tract lookups — avoids redundant ORS + PostGIS calls
# when isochrone_demographics, isochrone_employment, and isochrone_commute_shed
# all query the same (lon, lat, minutes) in the same pipeline run.
_isochrone_cache: dict[tuple, "gpd.GeoDataFrame"] = {}


def clear_isochrone_cache():
    """Clear the isochrone tract cache (call between pipeline runs if needed)."""
    _isochrone_cache.clear()

# Approximate drive-time to mile radius conversion for suburban areas.
# Used when OSRM is unavailable or too slow (public server rate-limits).
# Conservative estimates for mixed suburban/urban (DFW-style metro):
#   10 min ≈ 5 mi, 15 min ≈ 8 mi, 20 min ≈ 12 mi, 30 min ≈ 18 mi
_MINUTES_TO_MILES = {5: 3, 10: 5, 15: 8, 20: 12, 25: 15, 30: 18}


def _estimate_radius_miles(minutes):
    """Estimate radius in miles from drive-time minutes (suburban approximation)."""
    if minutes in _MINUTES_TO_MILES:
        return _MINUTES_TO_MILES[minutes]
    # Linear interpolation between known points
    keys = sorted(_MINUTES_TO_MILES.keys())
    if minutes < keys[0]:
        return _MINUTES_TO_MILES[keys[0]] * minutes / keys[0]
    if minutes > keys[-1]:
        return _MINUTES_TO_MILES[keys[-1]] * minutes / keys[-1]
    for i in range(len(keys) - 1):
        if keys[i] <= minutes <= keys[i + 1]:
            frac = (minutes - keys[i]) / (keys[i + 1] - keys[i])
            return _MINUTES_TO_MILES[keys[i]] + frac * (
                _MINUTES_TO_MILES[keys[i + 1]] - _MINUTES_TO_MILES[keys[i]]
            )
    return minutes * 0.6  # fallback: ~36 mph average


def tracts_within_isochrone(lon, lat, minutes, state_fips, engine=None, method="auto"):
    """Find census tracts that intersect a drive-time catchment area.

    Methods (in priority order):
    - "auto": tries ORS isochrone first, falls back to radius buffer
    - "ors": OpenRouteService isochrone (1 API call, requires ORS_API_KEY)
    - "osrm": OSRM ray-casting isochrone (180 API calls, slow with public server)
    - "buffer": calibrated geodesic radius buffer (no API, instant)

    Args:
        lon, lat: Center point (property location) in WGS84.
        minutes: Drive-time radius in minutes.
        state_fips: State FIPS code (e.g., "48" for Texas).
        engine: SQLAlchemy engine. Defaults to get_engine().
        method: "auto", "ors", "osrm", or "buffer".

    Returns:
        GeoDataFrame with columns: geoid, geometry, overlap_fraction, minutes.
        overlap_fraction ranges from 0.0 to 1.0 — the share of each tract's
        area that falls within the catchment.
    """
    # Check cache first
    cache_key = (round(lon, 6), round(lat, 6), minutes, state_fips)
    if cache_key in _isochrone_cache:
        console.print(f"[dim]{minutes}-min catchment: using cached result[/dim]")
        return _isochrone_cache[cache_key]

    engine = engine or get_engine()

    if method == "auto":
        from geostack.routing import ORS_API_KEY
        method = "ors" if ORS_API_KEY else "buffer"

    if method in ("ors", "osrm"):
        from geostack.routing import isochrone

        backend = method
        console.print(f"[dim]Computing {minutes}-min {backend.upper()} isochrone from ({lat:.4f}, {lon:.4f})...[/dim]")
        try:
            iso_gdf = isochrone(lon, lat, minutes=minutes, backend=backend)
            iso_wkt = iso_gdf.geometry.iloc[0].wkt
            catchment_clause = f"ST_GeomFromText('{iso_wkt}', 4326)"
        except Exception as e:
            console.print(f"[yellow]{backend.upper()} isochrone failed: {e} — falling back to buffer[/yellow]")
            method = "buffer"

    if method == "buffer":
        radius_miles = _estimate_radius_miles(minutes)
        radius_meters = radius_miles * 1609.34
        console.print(
            f"[dim]Computing {minutes}-min catchment (~{radius_miles} mi radius) "
            f"from ({lat:.4f}, {lon:.4f})...[/dim]"
        )
        catchment_clause = (
            f"ST_Buffer(ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)::geography, "
            f"{radius_meters})::geometry"
        )

    sql = f"""
        WITH catchment AS (
            SELECT {catchment_clause} AS geom
        )
        SELECT
            t.geoid,
            t.geometry,
            ST_Area(
                ST_Intersection(
                    ST_Transform(t.geometry, 2276),
                    ST_Transform(catchment.geom, 2276)
                )
            ) / NULLIF(ST_Area(ST_Transform(t.geometry, 2276)), 0) AS overlap_fraction
        FROM census_tracts t, catchment
        WHERE ST_Intersects(t.geometry, catchment.geom)
          AND t.state_fips = '{state_fips}'
        ORDER BY overlap_fraction DESC
    """

    gdf = read_postgis(sql, engine=engine, geom_col="geometry")

    gdf["minutes"] = minutes
    gdf["overlap_fraction"] = gdf["overlap_fraction"].clip(0.0, 1.0)

    method_label = {
        "ors": "ORS isochrone",
        "osrm": "OSRM isochrone",
        "buffer": f"~{_estimate_radius_miles(minutes)} mi buffer",
    }.get(method, method)
    console.print(
        f"[green]{minutes}-min catchment ({method_label}): {len(gdf)} tracts "
        f"(avg overlap {gdf['overlap_fraction'].mean():.0%})[/green]"
    )

    # Cache for reuse by other functions in the same pipeline run
    _isochrone_cache[cache_key] = gdf

    return gdf


def isochrone_demographics(lon, lat, minutes, state_fips, county_fips, year, engine=None):
    """Demographic profile within a drive-time isochrone.

    Fetches ACS tract data and applies population-weighted areal
    interpolation using the isochrone overlap fraction.

    For count variables (population, housing units, etc.): multiplies by
    overlap_fraction to estimate the portion within the isochrone.

    For median/rate variables (income, rent, age): computes a
    population-weighted average across tracts.

    Args:
        lon, lat: Center point in WGS84.
        minutes: Drive-time radius.
        state_fips: State FIPS (e.g., "48").
        county_fips: County FIPS (e.g., "113"). Used to scope ACS query.
                     For multi-county isochrones, call with the primary county.
        year: ACS vintage year (e.g., 2023).
        engine: SQLAlchemy engine.

    Returns:
        Tuple of (tract_gdf, summary_df):
        - tract_gdf: GeoDataFrame with per-tract interpolated ACS values
        - summary_df: Single-row DataFrame with aggregate demographic profile
    """
    engine = engine or get_engine()

    # Get tracts within the isochrone
    tracts = tracts_within_isochrone(lon, lat, minutes, state_fips, engine=engine)
    if tracts.empty:
        console.print("[yellow]No tracts found within isochrone[/yellow]")
        return tracts, pd.DataFrame()

    # Get ACS demographics — query all tracts in the state that overlap
    # (handles multi-county isochrones by querying each county separately)
    # Filter to standard 11-digit tract geoids (skip any MSA/non-standard rows)
    tract_geoids = {g for g in tracts["geoid"].tolist() if len(g) == 11 and g.isdigit()}

    # Identify which counties are represented
    county_fips_list = sorted({g[2:5] for g in tract_geoids})

    acs_frames = []
    for cfips in county_fips_list:
        try:
            gdf = get_tract_demographics(state_fips, cfips, year, engine=engine)
            acs_frames.append(gdf)
        except Exception as e:
            console.print(f"[yellow]Skipping county {cfips}: {e}[/yellow]")

    if not acs_frames:
        console.print("[yellow]No ACS data available for isochrone tracts[/yellow]")
        return tracts, pd.DataFrame()

    acs_gdf = pd.concat(acs_frames)
    # Filter to only tracts within the isochrone
    acs_gdf = acs_gdf[acs_gdf.index.isin(tract_geoids)]

    # Merge overlap_fraction onto ACS data via explicit map (safe against index type mismatch)
    tracts_lookup = tracts.set_index("geoid")["overlap_fraction"]
    acs_gdf["overlap_fraction"] = acs_gdf.index.map(tracts_lookup).fillna(0)

    # Interpolate count variables by overlap_fraction
    for var in _SUM_VARIABLES:
        if var in acs_gdf.columns:
            acs_gdf[f"{var}_weighted"] = acs_gdf[var] * acs_gdf["overlap_fraction"]

    # Build summary row
    summary = {}

    # Sum variables: aggregate the weighted values
    for var in _SUM_VARIABLES:
        weighted_col = f"{var}_weighted"
        if weighted_col in acs_gdf.columns:
            summary[var] = acs_gdf[weighted_col].sum()

    # Median/avg variables: population-weighted average
    pop_weighted = acs_gdf["total_population"] * acs_gdf["overlap_fraction"]
    total_weighted_pop = pop_weighted.sum()

    for var in _AVG_VARIABLES:
        if var in acs_gdf.columns:
            valid = acs_gdf[var].notna() & pop_weighted.notna()
            if valid.any():
                summary[var] = (
                    (acs_gdf.loc[valid, var] * pop_weighted[valid]).sum()
                    / pop_weighted[valid].sum()
                )
            else:
                summary[var] = None

    # Derived metrics
    if summary.get("total_housing_units", 0) > 0:
        summary["vacancy_rate"] = (
            summary.get("vacant_units", 0) / summary["total_housing_units"]
        )
        summary["renter_pct"] = (
            summary.get("renter_occupied", 0)
            / max(summary.get("occupied_units", 1), 1)
        )

    summary["minutes"] = minutes
    summary["tract_count"] = len(acs_gdf)
    summary["total_weighted_population"] = total_weighted_pop
    summary_df = pd.DataFrame([summary])

    console.print(
        f"[green]Demographics ({minutes}-min): "
        f"pop={total_weighted_pop:,.0f}, "
        f"tracts={len(acs_gdf)}, "
        f"med_income=${summary.get('median_household_income', 0):,.0f}[/green]"
    )

    return acs_gdf, summary_df


def isochrone_employment(lon, lat, minutes, state_fips, county_fips, year, engine=None):
    """Employment profile within a drive-time isochrone.

    Uses LODES WAC data aggregated to tract level, weighted by the
    isochrone overlap fraction. Shows sector mix, wage tier distribution,
    and total jobs within the catchment.

    Args:
        lon, lat: Center point in WGS84.
        minutes: Drive-time radius.
        state_fips: State FIPS (e.g., "48").
        county_fips: County FIPS (e.g., "113").
        year: LODES data year (e.g., 2021).
        engine: SQLAlchemy engine.

    Returns:
        Tuple of (tract_df, summary_df):
        - tract_df: DataFrame with per-tract employment weighted by overlap
        - summary_df: Single-row DataFrame with aggregate employment profile
    """
    from geostack.lodes import NAICS_SECTORS

    engine = engine or get_engine()

    # Get tracts within the isochrone
    tracts = tracts_within_isochrone(lon, lat, minutes, state_fips, engine=engine)
    if tracts.empty:
        return pd.DataFrame(), pd.DataFrame()

    tract_geoids = tracts["geoid"].tolist()

    # Identify counties represented in the isochrone
    county_fips_list = sorted({g[2:5] for g in tract_geoids})

    # Query LODES WAC at tract level for all relevant counties
    sector_sums = ", ".join(
        f"SUM(jobs_{v}) AS jobs_{v}" for v in NAICS_SECTORS.values()
    )

    # Use parameterized ANY() to avoid SQL injection via geoid interpolation
    sql = f"""
        SELECT
            tract_geoid,
            SUM(total_jobs) AS total_jobs,
            SUM(jobs_earn_low) AS jobs_earn_low,
            SUM(jobs_earn_mid) AS jobs_earn_mid,
            SUM(jobs_earn_high) AS jobs_earn_high,
            SUM(jobs_age_29_under) AS jobs_age_29_under,
            SUM(jobs_age_30_54) AS jobs_age_30_54,
            SUM(jobs_age_55_over) AS jobs_age_55_over,
            {sector_sums}
        FROM lodes_wac
        WHERE tract_geoid = ANY(:tracts)
          AND year = :y AND job_type = 'JT01' AND segment = 'S000'
        GROUP BY tract_geoid
    """
    emp_df = read_sql(sql, engine=engine, params={"tracts": list(tract_geoids), "y": year})

    if emp_df.empty:
        console.print(f"[yellow]No LODES WAC data for isochrone tracts (year={year})[/yellow]")
        return emp_df, pd.DataFrame()

    # Merge overlap_fraction
    tracts_lookup = tracts.set_index("geoid")["overlap_fraction"]
    emp_df["overlap_fraction"] = emp_df["tract_geoid"].map(tracts_lookup).fillna(0)

    # Weight all numeric columns by overlap_fraction
    numeric_cols = emp_df.select_dtypes(include="number").columns.difference(["overlap_fraction"])
    for col in numeric_cols:
        emp_df[f"{col}_weighted"] = emp_df[col] * emp_df["overlap_fraction"]

    # Build summary
    summary = {"minutes": minutes, "tract_count": len(emp_df)}
    for col in numeric_cols:
        summary[col] = emp_df[f"{col}_weighted"].sum()

    # Sector percentages
    total = summary.get("total_jobs", 0)
    if total > 0:
        for sector_name in NAICS_SECTORS.values():
            col_name = f"jobs_{sector_name}"
            if col_name in summary:
                summary[f"pct_{sector_name}"] = summary[col_name] / total

        # Wage tier percentages
        for tier in ["earn_low", "earn_mid", "earn_high"]:
            col_name = f"jobs_{tier}"
            if col_name in summary:
                summary[f"pct_{tier}"] = summary[col_name] / total

    summary_df = pd.DataFrame([summary])

    console.print(
        f"[green]Employment ({minutes}-min): "
        f"total_jobs={total:,.0f}, "
        f"tracts={len(emp_df)}[/green]"
    )

    return emp_df, summary_df


def isochrone_commute_shed(lon, lat, minutes, state_fips, county_fips, year, engine=None):
    """Analyze where workers in a catchment area commute from.

    Uses LODES Origin-Destination data to identify which counties send
    workers into the isochrone area. This is the labor shed — the answer
    to "where do people who work here actually live?"

    Uses tract-level approximation: aggregates OD flows where the work
    tract falls within the isochrone. This is slightly less precise than
    block-level (blocks don't have geometry in PostGIS) but captures
    the dominant commute patterns.

    Args:
        lon, lat: Center point in WGS84.
        minutes: Drive-time radius.
        state_fips: State FIPS (e.g., "48").
        county_fips: County FIPS (e.g., "113").
        year: LODES data year (e.g., 2021).
        engine: SQLAlchemy engine.

    Returns:
        DataFrame with columns: h_state_fips, h_county_fips, total_jobs,
        jobs_earn_low/mid/high, pct_of_total — sorted by total_jobs desc.
        Each row represents a home county that sends workers into the catchment.
    """
    engine = engine or get_engine()

    # Get tracts within the isochrone
    tracts = tracts_within_isochrone(lon, lat, minutes, state_fips, engine=engine)
    if tracts.empty:
        return pd.DataFrame()

    tract_geoids = tracts["geoid"].tolist()

    # Query OD flows where work location is in our isochrone tracts
    # Match w_geocode (15-digit block) to tract (first 11 digits) using LEFT()
    # Use parameterized ANY() to avoid SQL injection
    sql = """
        SELECT
            h_state_fips,
            h_county_fips,
            SUM(total_jobs) AS total_jobs,
            SUM(jobs_earn_low) AS jobs_earn_low,
            SUM(jobs_earn_mid) AS jobs_earn_mid,
            SUM(jobs_earn_high) AS jobs_earn_high,
            SUM(jobs_age_29_under) AS jobs_age_29_under,
            SUM(jobs_age_30_54) AS jobs_age_30_54,
            SUM(jobs_age_55_over) AS jobs_age_55_over
        FROM lodes_od
        WHERE LEFT(w_geocode, 11) = ANY(:tracts)
          AND year = :y
        GROUP BY h_state_fips, h_county_fips
        ORDER BY total_jobs DESC
    """

    shed_df = read_sql(sql, engine=engine, params={"tracts": list(tract_geoids), "y": year})

    if shed_df.empty:
        console.print(f"[yellow]No OD data for isochrone tracts (year={year})[/yellow]")
        return shed_df

    # Add percentage column
    total = shed_df["total_jobs"].sum()
    shed_df["pct_of_total"] = shed_df["total_jobs"] / total if total > 0 else 0

    # Cumulative percentage for "top N counties cover X% of commuters"
    shed_df["cumulative_pct"] = shed_df["pct_of_total"].cumsum()

    console.print(
        f"[green]Commute shed ({minutes}-min): "
        f"{len(shed_df)} home counties, "
        f"{total:,.0f} total worker flows[/green]"
    )

    return shed_df


def acs_time_series(state_fips, county_fips, years, variables=None, engine=None):
    """Fetch ACS demographic time series for trend analysis.

    Queries PostGIS acs_tract_data across multiple years, falling back
    to the Census API for missing years. Returns long-format data suitable
    for trend charts and CAGR calculations.

    Args:
        state_fips: State FIPS (e.g., "48").
        county_fips: County FIPS (e.g., "113").
        years: List of ACS vintage years (e.g., [2018, 2019, 2020, 2021, 2022, 2023]).
        variables: Optional dict of {name: census_code}. Defaults to ACS_VARIABLES.
        engine: SQLAlchemy engine.

    Returns:
        DataFrame with columns: year, variable, value — aggregated to county level.
    """
    from geostack.census import get_county_acs_summary_direct

    engine = engine or get_engine()
    variables = variables or ACS_VARIABLES

    frames = []

    for year in years:
        try:
            summary = get_county_acs_summary_direct(state_fips, county_fips, year, engine=engine)
            if summary.empty:
                raise ValueError(f"No data for {year}")

            row = summary.iloc[0]
            for var_name in variables:
                if var_name in row.index and pd.notna(row[var_name]):
                    frames.append({
                        "year": year,
                        "variable": var_name,
                        "value": float(row[var_name]),
                    })
        except Exception as e:
            console.print(f"[yellow]ACS time series — skipping {year}: {e}[/yellow]")

    if not frames:
        console.print("[yellow]No ACS time series data retrieved[/yellow]")
        return pd.DataFrame(columns=["year", "variable", "value"])

    df = pd.DataFrame(frames)

    # Compute CAGR for each variable
    cagr_rows = []
    for var in df["variable"].unique():
        var_data = df[df["variable"] == var].sort_values("year")
        if len(var_data) >= 2:
            first_val = var_data.iloc[0]["value"]
            last_val = var_data.iloc[-1]["value"]
            n_years = var_data.iloc[-1]["year"] - var_data.iloc[0]["year"]
            if first_val > 0 and n_years > 0:
                cagr = (last_val / first_val) ** (1 / n_years) - 1
                cagr_rows.append({
                    "variable": var,
                    "cagr": cagr,
                    "start_year": int(var_data.iloc[0]["year"]),
                    "end_year": int(var_data.iloc[-1]["year"]),
                    "start_value": first_val,
                    "end_value": last_val,
                })

    cagr_df = pd.DataFrame(cagr_rows) if cagr_rows else pd.DataFrame()

    console.print(
        f"[green]ACS time series: {len(df)} data points across "
        f"{df['year'].nunique()} years[/green]"
    )

    # Attach CAGR as an attribute for convenience
    df.attrs["cagr"] = cagr_df

    return df


def isochrone_income_distribution(lon, lat, minutes, state_fips, county_fips, year, engine=None):
    """Household income distribution within a drive-time isochrone.

    Fetches B19001 (income brackets) from the Census API for tracts within
    the isochrone and aggregates using overlap_fraction weighting. Returns
    income brackets that enable precise affordability analysis — "what % of
    households can afford $X/mo rent?"

    Args:
        lon, lat: Center point in WGS84.
        minutes: Drive-time radius.
        state_fips, county_fips: FIPS codes.
        year: ACS vintage year.
        engine: SQLAlchemy engine.

    Returns:
        DataFrame with income brackets and cumulative affordability percentages.
    """
    from geostack.census import CensusClient

    engine = engine or get_engine()

    # Get tracts within isochrone
    tracts = tracts_within_isochrone(lon, lat, minutes, state_fips, engine=engine)
    if tracts.empty:
        return pd.DataFrame()

    tract_geoids = {g for g in tracts["geoid"].tolist() if len(g) == 11 and g.isdigit()}
    county_fips_list = sorted({g[2:5] for g in tract_geoids})

    # Income bracket variables (B19001)
    income_vars = {
        "B19001_001E": "total",
        "B19001_002E": "under_10k",
        "B19001_003E": "10k_15k",
        "B19001_004E": "15k_20k",
        "B19001_005E": "20k_25k",
        "B19001_006E": "25k_30k",
        "B19001_007E": "30k_35k",
        "B19001_008E": "35k_40k",
        "B19001_009E": "40k_45k",
        "B19001_010E": "45k_50k",
        "B19001_011E": "50k_60k",
        "B19001_012E": "60k_75k",
        "B19001_013E": "75k_100k",
        "B19001_014E": "100k_125k",
        "B19001_015E": "125k_150k",
        "B19001_016E": "150k_200k",
        "B19001_017E": "200k_plus",
    }

    # Rent bracket variables (B25063) — MF-relevant brackets ($800+)
    # B25063_003E–017E are $50 increments <$100 through $750–$799;
    # _018E+ are the brackets useful for multifamily analysis.
    rent_vars = {
        "B25063_002E": "rent_total",
        "B25063_017E": "rent_under_800",
        "B25063_018E": "rent_800_899",
        "B25063_019E": "rent_900_999",
        "B25063_020E": "rent_1000_1249",
        "B25063_021E": "rent_1250_1499",
        "B25063_022E": "rent_1500_1999",
        "B25063_023E": "rent_2000_2499",
        "B25063_024E": "rent_2500_2999",
        "B25063_025E": "rent_3000_3499",
        "B25063_026E": "rent_3500_plus",
    }

    # Education variables (B15003)
    edu_vars = {
        "B15003_001E": "edu_total_25plus",
        "B15003_017E": "edu_hs_diploma",
        "B15003_019E": "edu_some_college",
        "B15003_021E": "edu_associates",
        "B15003_022E": "edu_bachelors",
        "B15003_023E": "edu_masters",
        "B15003_024E": "edu_professional",
        "B15003_025E": "edu_doctorate",
    }

    all_vars = {**income_vars, **rent_vars, **edu_vars}

    # Fetch from Census API
    client = CensusClient()
    frames = []
    for cfips in county_fips_list:
        try:
            df = client.get_acs(
                list(all_vars.keys()),
                geography="tract",
                state=state_fips,
                county=cfips,
                year=year,
            )
            df = client.build_geoid(df, "tract")
            frames.append(df)
        except Exception as e:
            console.print(f"[yellow]Income dist: skipping county {cfips}: {e}[/yellow]")

    if not frames:
        return pd.DataFrame()

    acs_df = pd.concat(frames, ignore_index=True)
    acs_df = acs_df[acs_df["GEOID"].isin(tract_geoids)]

    # Merge overlap_fraction
    tracts_lookup = tracts.set_index("geoid")["overlap_fraction"]
    acs_df["overlap_fraction"] = acs_df["GEOID"].map(tracts_lookup).fillna(0)

    # Rename and convert
    acs_df = acs_df.rename(columns=all_vars)

    # Weight by overlap_fraction and aggregate
    result = {}
    for col in all_vars.values():
        if col in acs_df.columns:
            acs_df[col] = pd.to_numeric(acs_df[col], errors="coerce").fillna(0)
            result[col] = (acs_df[col] * acs_df["overlap_fraction"]).sum()

    # Build income distribution with affordability thresholds
    income_brackets = [
        ("Under $10K", result.get("under_10k", 0), 0, 10000),
        ("$10K–$15K", result.get("10k_15k", 0), 10000, 15000),
        ("$15K–$20K", result.get("15k_20k", 0), 15000, 20000),
        ("$20K–$25K", result.get("20k_25k", 0), 20000, 25000),
        ("$25K–$30K", result.get("25k_30k", 0), 25000, 30000),
        ("$30K–$35K", result.get("30k_35k", 0), 30000, 35000),
        ("$35K–$40K", result.get("35k_40k", 0), 35000, 40000),
        ("$40K–$45K", result.get("40k_45k", 0), 40000, 45000),
        ("$45K–$50K", result.get("45k_50k", 0), 45000, 50000),
        ("$50K–$60K", result.get("50k_60k", 0), 50000, 60000),
        ("$60K–$75K", result.get("60k_75k", 0), 60000, 75000),
        ("$75K–$100K", result.get("75k_100k", 0), 75000, 100000),
        ("$100K–$125K", result.get("100k_125k", 0), 100000, 125000),
        ("$125K–$150K", result.get("125k_150k", 0), 125000, 150000),
        ("$150K–$200K", result.get("150k_200k", 0), 150000, 200000),
        ("$200K+", result.get("200k_plus", 0), 200000, 999999),
    ]

    total_hh = result.get("total", 0) or 1
    income_df = pd.DataFrame(income_brackets, columns=["bracket", "households", "lower_bound", "upper_bound"])
    income_df["pct"] = income_df["households"] / total_hh
    income_df["cumulative_pct"] = income_df["pct"].cumsum()

    # Attach rent distribution and education as attrs
    income_df.attrs["rent_distribution"] = {
        k: result.get(k, 0) for k in rent_vars.values()
    }
    income_df.attrs["education"] = {
        k: result.get(k, 0) for k in edu_vars.values()
    }
    income_df.attrs["total_households"] = total_hh
    income_df.attrs["minutes"] = minutes

    console.print(
        f"[green]Income distribution ({minutes}-min): "
        f"{total_hh:,.0f} households across {len(acs_df)} tracts[/green]"
    )

    return income_df


def affordability_at_rent(income_df, monthly_rent):
    """Calculate what % of households can afford a given rent.

    Uses the 30% rule: rent affordable if rent ≤ 30% of gross monthly income,
    i.e., annual income ≥ rent × 12 / 0.30.

    Interpolates within the straddling bracket for accuracy rather than
    treating each bracket as all-or-nothing.

    Args:
        income_df: DataFrame from isochrone_income_distribution().
            Must have columns: households, lower_bound, upper_bound, pct.
        monthly_rent: Target monthly rent.

    Returns:
        Float: percentage of households that can afford this rent (0.0–1.0).
    """
    if income_df.empty:
        return 0.0

    # Handle parquets that only have upper_bound (no lower_bound)
    if "lower_bound" not in income_df.columns and "upper_bound" in income_df.columns:
        income_df = income_df.copy()
        ubs = income_df["upper_bound"].tolist()
        income_df["lower_bound"] = [0] + ubs[:-1]

    required_annual = monthly_rent * 12 / 0.30

    pct_cannot_afford = 0.0

    for _, row in income_df.iterrows():
        lb = row["lower_bound"]
        ub = row["upper_bound"]
        pct = row["pct"]

        if ub <= required_annual:
            # Entire bracket is below required income — cannot afford
            pct_cannot_afford += pct
        elif lb >= required_annual:
            # Entire bracket is above required income — can afford
            pass
        else:
            # Bracket straddles the threshold — interpolate linearly
            bracket_width = ub - lb
            if bracket_width > 0:
                fraction_below = (required_annual - lb) / bracket_width
                pct_cannot_afford += pct * fraction_below

    return max(0.0, min(1.0, 1.0 - pct_cannot_afford))
