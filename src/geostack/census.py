"""Census Bureau / ACS API client.

Fetches demographic and economic data from the Census Bureau API and
loads it into PostGIS with tract/block-group geometry from TIGER/Line.

Usage:
    from geostack.census import CensusClient, get_tract_demographics

    # Fetch from API and load into DB:
    client = CensusClient(api_key="your-key")  # or set CENSUS_API_KEY env var
    df = client.get_acs(
        variables=["B01003_001E", "B19013_001E"],  # pop, median income
        geography="tract",
        state="48",       # Texas FIPS
        county="085",     # Collin County FIPS
        year=2023,
    )

    # Query from PostGIS (after load-acs has run):
    gdf = get_tract_demographics("48", "085", year=2023)
    summary = get_county_acs_summary("48", "085", year=2023)
"""

import os

import geopandas as gpd
import pandas as pd
import requests
from rich.console import Console

console = Console()

# Common ACS variable lookups for CRE analysis
ACS_VARIABLES = {
    # Population
    "total_population": "B01003_001E",
    "median_age": "B01002_001E",
    # Income
    "median_household_income": "B19013_001E",
    "per_capita_income": "B19301_001E",
    # Housing
    "total_housing_units": "B25001_001E",
    "occupied_units": "B25002_002E",
    "vacant_units": "B25002_003E",
    "median_rent": "B25064_001E",
    "median_home_value": "B25077_001E",
    "renter_occupied": "B25003_003E",
    "owner_occupied": "B25003_002E",
    # Multifamily-relevant
    "units_in_structure_20plus": "B25024_009E",
    "units_in_structure_10_19": "B25024_008E",
    "units_in_structure_5_9": "B25024_007E",
    "units_in_structure_3_4": "B25024_006E",
    "units_in_structure_2": "B25024_005E",
    "moved_in_last_year": "B25038_003E",
    # Household income distribution (B19001) — for affordability analysis
    "hh_income_total": "B19001_001E",
    "hh_income_under_10k": "B19001_002E",
    "hh_income_10k_15k": "B19001_003E",
    "hh_income_15k_20k": "B19001_004E",
    "hh_income_20k_25k": "B19001_005E",
    "hh_income_25k_30k": "B19001_006E",
    "hh_income_30k_35k": "B19001_007E",
    "hh_income_35k_40k": "B19001_008E",
    "hh_income_40k_45k": "B19001_009E",
    "hh_income_45k_50k": "B19001_010E",
    "hh_income_50k_60k": "B19001_011E",
    "hh_income_60k_75k": "B19001_012E",
    "hh_income_75k_100k": "B19001_013E",
    "hh_income_100k_125k": "B19001_014E",
    "hh_income_125k_150k": "B19001_015E",
    "hh_income_150k_200k": "B19001_016E",
    "hh_income_200k_plus": "B19001_017E",
    # Gross rent distribution (B25063) — where does subject rent sit in the curve?
    # B25063_002E = total with cash rent; _003E–_017E = <$100 through $750–$799;
    # _018E+ = $800+ brackets relevant for multifamily analysis.
    "rent_total": "B25063_002E",
    "rent_under_800": "B25063_017E",       # $750-$799 (proxy; sum _003E–_017E for true <$800)
    "rent_800_899": "B25063_018E",
    "rent_900_999": "B25063_019E",
    "rent_1000_1249": "B25063_020E",
    "rent_1250_1499": "B25063_021E",
    "rent_1500_1999": "B25063_022E",
    "rent_2000_2499": "B25063_023E",
    "rent_2500_2999": "B25063_024E",
    "rent_3000_3499": "B25063_025E",
    "rent_3500_plus": "B25063_026E",
    # Education attainment (B15003) — 25+ population
    "edu_total_25plus": "B15003_001E",
    "edu_hs_diploma": "B15003_017E",
    "edu_some_college": "B15003_019E",
    "edu_associates": "B15003_021E",
    "edu_bachelors": "B15003_022E",
    "edu_masters": "B15003_023E",
    "edu_professional": "B15003_024E",
    "edu_doctorate": "B15003_025E",
}

# Texas county FIPS codes for common CRE markets
TX_COUNTIES = {
    "collin": "085",
    "dallas": "113",
    "tarrant": "439",
    "denton": "121",
    "travis": "453",
    "harris": "201",
    "bexar": "029",
    "williamson": "491",
    "fort_bend": "157",
    "rockwall": "397",
    "cooke": "097",
    "grayson": "181",
}


class CensusClient:
    """Client for the Census Bureau API (ACS 5-year estimates)."""

    BASE_URL = "https://api.census.gov/data"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("CENSUS_API_KEY")
        if not self.api_key:
            console.print(
                "[yellow]Warning: No CENSUS_API_KEY set. "
                "API calls will be rate-limited. "
                "Get a free key at https://api.census.gov/data/key_signup.html[/yellow]"
            )

    # Census API allows max 50 columns per request (including NAME + geo columns)
    _MAX_VARS_PER_REQUEST = 48  # leave room for NAME + state + county + tract

    def get_acs(self, variables, geography="tract", state=None, county=None, year=2023):
        """Fetch ACS 5-year estimate data.

        Automatically chunks requests if variables exceed the Census API's
        50-column limit.

        Args:
            variables: List of Census variable codes (e.g., ["B01003_001E"]).
            geography: "tract", "block group", "county", "place", etc.
            state: State FIPS code (e.g., "48" for Texas).
            county: County FIPS code (e.g., "085" for Collin).
            year: ACS vintage year.

        Returns:
            pandas DataFrame with the requested variables.
        """
        if len(variables) <= self._MAX_VARS_PER_REQUEST:
            return self._fetch_acs_chunk(variables, geography, state, county, year)

        # Chunk variables to stay within API limits
        chunks = [
            variables[i:i + self._MAX_VARS_PER_REQUEST]
            for i in range(0, len(variables), self._MAX_VARS_PER_REQUEST)
        ]
        frames = []
        for chunk in chunks:
            df = self._fetch_acs_chunk(chunk, geography, state, county, year)
            frames.append(df)

        # Merge chunks on geography columns
        result = frames[0]
        geo_cols = [c for c in result.columns if c not in variables and c != "NAME"]
        for df in frames[1:]:
            # Drop NAME to avoid duplication, keep only variable columns + geo cols
            merge_cols = geo_cols + [c for c in df.columns if c in variables]
            result = result.merge(df[merge_cols], on=geo_cols, how="outer")

        return result

    def _fetch_acs_chunk(self, variables, geography, state, county, year):
        """Fetch a single chunk of ACS variables (within API limits)."""
        var_str = ",".join(variables)
        url = f"{self.BASE_URL}/{year}/acs/acs5"

        params = {
            "get": f"NAME,{var_str}",
        }

        # Build geography parameters correctly.
        # Census API: "for=county:329&in=state:48" (specific county)
        #             "for=tract:*&in=state:48 county:329" (all tracts in county)
        if geography == "county" and state and county:
            # Specific county query — don't use wildcard
            params["for"] = f"county:{county}"
            params["in"] = f"state:{state}"
        else:
            params["for"] = f"{geography}:*"
            if state:
                params["in"] = f"state:{state}"
                if county:
                    params["in"] += f" county:{county}"

        if self.api_key:
            params["key"] = self.api_key

        console.print(f"[dim]Fetching ACS {year} — {geography} level ({len(variables)} vars)...[/dim]")
        resp = requests.get(url, params=params)
        resp.raise_for_status()

        data = resp.json()
        df = pd.DataFrame(data[1:], columns=data[0])

        # Convert numeric columns
        for var in variables:
            if var in df.columns:
                df[var] = pd.to_numeric(df[var], errors="coerce")

        return df

    def get_acs_time_series(self, variables, years=None, **kwargs):
        """Fetch the same variables across multiple ACS vintages.

        Returns a single DataFrame with a 'year' column for time-series analysis.
        ACS 5-year estimates are available from 2009 onward.
        """
        if years is None:
            years = list(range(2013, 2024))  # decade of 5yr estimates

        frames = []
        for year in years:
            try:
                df = self.get_acs(variables, year=year, **kwargs)
                df["year"] = year
                frames.append(df)
            except requests.HTTPError as e:
                console.print(f"[yellow]Skipping {year}: {e}[/yellow]")

        if not frames:
            raise ValueError("No data retrieved for any year")

        return pd.concat(frames, ignore_index=True)

    def build_geoid(self, df, geography="tract"):
        """Construct a standard GEOID column for joining to TIGER shapefiles.

        TIGER GEOIDs for tracts: state(2) + county(3) + tract(6) = 11 chars
        """
        if geography == "tract":
            df["GEOID"] = df["state"] + df["county"] + df["tract"]
        elif geography == "block group":
            df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]
        elif geography == "county":
            df["GEOID"] = df["state"] + df["county"]
        return df


# ACS variables that should be summed when aggregating to county level
_SUM_VARIABLES = {
    "total_population",
    "total_housing_units",
    "occupied_units",
    "vacant_units",
    "renter_occupied",
    "owner_occupied",
    "units_in_structure_20plus",
    "units_in_structure_10_19",
    "units_in_structure_5_9",
    "units_in_structure_3_4",
    "units_in_structure_2",
    "moved_in_last_year",
    # Income distribution
    "hh_income_total",
    "hh_income_under_10k", "hh_income_10k_15k", "hh_income_15k_20k",
    "hh_income_20k_25k", "hh_income_25k_30k", "hh_income_30k_35k",
    "hh_income_35k_40k", "hh_income_40k_45k", "hh_income_45k_50k",
    "hh_income_50k_60k", "hh_income_60k_75k", "hh_income_75k_100k",
    "hh_income_100k_125k", "hh_income_125k_150k", "hh_income_150k_200k",
    "hh_income_200k_plus",
    # Rent distribution
    "rent_total", "rent_under_800", "rent_800_899", "rent_900_999",
    "rent_1000_1249", "rent_1250_1499", "rent_1500_1999",
    "rent_2000_2499", "rent_2500_2999", "rent_3000_3499", "rent_3500_plus",
    # Education
    "edu_total_25plus", "edu_hs_diploma", "edu_some_college", "edu_associates",
    "edu_bachelors", "edu_masters", "edu_professional", "edu_doctorate",
}

# ACS variables that should be averaged (medians/rates) when aggregating to county level
_AVG_VARIABLES = {
    "median_age",
    "median_household_income",
    "per_capita_income",
    "median_rent",
    "median_home_value",
}


assert _SUM_VARIABLES | _AVG_VARIABLES == set(ACS_VARIABLES), (
    "ACS_VARIABLES has entries missing from _SUM_VARIABLES/_AVG_VARIABLES: "
    + str(set(ACS_VARIABLES) - (_SUM_VARIABLES | _AVG_VARIABLES))
)


def _validate_fips(value, name):
    """Raise ValueError if value is not a numeric FIPS string."""
    if not str(value).isdigit():
        raise ValueError(f"{name} must be a numeric FIPS code, got: {value!r}")


_CENSUS_SENTINEL = -666666666


def _pivot_select_clauses():
    """Build SQL SELECT clauses for the ACS EAV pivot.

    The FILTER clause excludes the Census missing-value sentinel (-666666666)
    so suppressed values become NULL (→ NaN in pandas) rather than large negatives.
    """
    clauses = []
    for name, code in ACS_VARIABLES.items():
        clauses.append(
            f"MAX(a.value) FILTER (WHERE a.variable = '{code}'"
            f" AND a.value != {_CENSUS_SENTINEL}) AS {name}"
        )
    return ",\n        ".join(clauses)


def get_tract_demographics(state_fips, county_fips, year, engine=None):
    """Query ACS tract demographics from PostGIS as a GeoDataFrame.

    JOINs census_tracts (geometry) with acs_tract_data (EAV) and pivots wide.
    Requires `geostack load-acs` to have been run for the given state/county/year.

    Args:
        state_fips: State FIPS code string (e.g., "48" for Texas).
        county_fips: County FIPS code string (e.g., "085" for Collin).
        year: ACS vintage year (e.g., 2023).
        engine: SQLAlchemy engine. Defaults to get_engine().

    Returns:
        GeoDataFrame indexed by geoid with one column per ACS variable + geometry.
        Missing variables are NaN.
    """
    from geostack.db import get_engine

    engine = engine or get_engine()
    _validate_fips(state_fips, "state_fips")
    _validate_fips(county_fips, "county_fips")

    pivot_cols = _pivot_select_clauses()
    sql = f"""
        SELECT
            t.geoid,
            t.geometry,
            {pivot_cols}
        FROM census_tracts t
        LEFT JOIN acs_tract_data a
            ON a.geoid = t.geoid AND a.year = {int(year)}
        WHERE t.state_fips = '{state_fips}'
          AND t.county_fips = '{county_fips}'
        GROUP BY t.geoid, t.geometry
        ORDER BY t.geoid
    """

    with engine.connect() as conn:
        gdf = gpd.read_postgis(sql, conn, geom_col="geometry")

    gdf = gdf.set_index("geoid")

    # Cast all ACS columns to float and replace Census sentinel → NaN
    _SENTINEL = -666666666
    for col in ACS_VARIABLES:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
            gdf.loc[gdf[col] == _SENTINEL, col] = float("nan")

    console.print(
        f"[dim]get_tract_demographics: {len(gdf)} tracts, "
        f"state={state_fips}, county={county_fips}, year={year}[/dim]"
    )
    return gdf


def get_county_acs_summary(state_fips, county_fips, year, engine=None):
    """Return county-level ACS aggregates as a single-row DataFrame.

    Count/total variables are summed across tracts; median/rate variables are
    averaged (simple mean across tract medians).

    Args:
        state_fips: State FIPS code string (e.g., "48").
        county_fips: County FIPS code string (e.g., "085").
        year: ACS vintage year.
        engine: SQLAlchemy engine. Defaults to get_engine().

    Returns:
        Single-row DataFrame with one column per ACS variable.
    """
    from geostack.db import get_engine, read_sql

    engine = engine or get_engine()
    _validate_fips(state_fips, "state_fips")
    _validate_fips(county_fips, "county_fips")

    pivot_cols = _pivot_select_clauses()

    sum_clauses = ", ".join(
        f"SUM(tract.{name}) AS {name}" for name in ACS_VARIABLES if name in _SUM_VARIABLES
    )
    avg_clauses = ", ".join(
        f"AVG(tract.{name}) AS {name}" for name in ACS_VARIABLES if name in _AVG_VARIABLES
    )
    agg_cols = f"{sum_clauses}, {avg_clauses}"

    sql = f"""
        SELECT {agg_cols}
        FROM (
            SELECT
                t.geoid,
                {pivot_cols}
            FROM census_tracts t
            LEFT JOIN acs_tract_data a
                ON a.geoid = t.geoid AND a.year = {int(year)}
            WHERE t.state_fips = '{state_fips}'
              AND t.county_fips = '{county_fips}'
            GROUP BY t.geoid
        ) tract
    """

    df = read_sql(sql, engine=engine, params={})

    # Replace Census sentinel → NaN
    _SENTINEL = -666666666
    for col in ACS_VARIABLES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == _SENTINEL, col] = float("nan")

    console.print(
        f"[dim]get_county_acs_summary: state={state_fips}, county={county_fips}, year={year}[/dim]"
    )
    return df


def get_county_acs_summary_direct(state_fips, county_fips, year, engine=None):
    """County-level ACS summary queried directly from acs_tract_data.

    Unlike get_county_acs_summary(), this bypasses the census_tracts JOIN
    and queries acs_tract_data by geoid prefix. This is critical for
    time-series analysis because census_tracts only holds one vintage of
    geoids (2020), while acs_tract_data has data loaded against both
    2010-vintage (2013-2019) and 2020-vintage (2020+) geoids.

    Sum variables are summed; median variables are population-weighted
    using B01003_001E (total population) as the weight.

    Args:
        state_fips: State FIPS code string (e.g., "48").
        county_fips: County FIPS code string (e.g., "113").
        year: ACS vintage year.
        engine: SQLAlchemy engine.

    Returns:
        Single-row DataFrame with one column per ACS variable.
    """
    from geostack.db import get_engine, read_sql

    engine = engine or get_engine()
    _validate_fips(state_fips, "state_fips")
    _validate_fips(county_fips, "county_fips")

    state_fips = str(state_fips).zfill(2)
    county_fips = str(county_fips).zfill(3)
    prefix = f"{state_fips}{county_fips}"

    # Pivot EAV to wide: one row per geoid with all variables as columns
    # First get population for weighting medians
    pop_code = ACS_VARIABLES["total_population"]

    # Build the pivot query directly on acs_tract_data
    pivot_cols = []
    for name, code in ACS_VARIABLES.items():
        pivot_cols.append(
            f"MAX(CASE WHEN variable = '{code}' "
            f"AND value != {_CENSUS_SENTINEL} THEN value END) AS {name}"
        )
    pivot_select = ",\n        ".join(pivot_cols)

    # Subquery: pivot each tract to wide format
    # Use bind parameter for LIKE pattern to avoid % escaping issues
    tract_sql = f"""
        SELECT
            geoid,
            {pivot_select}
        FROM acs_tract_data
        WHERE geoid LIKE :prefix AND year = :yr
        GROUP BY geoid
    """

    # Aggregate: sum count vars, population-weighted avg for median vars
    sum_clauses = ", ".join(
        f"SUM(tract.{name}) AS {name}" for name in ACS_VARIABLES if name in _SUM_VARIABLES
    )

    # Population-weighted average for median variables
    avg_clauses = []
    for name in ACS_VARIABLES:
        if name in _AVG_VARIABLES:
            avg_clauses.append(
                f"CASE WHEN SUM(tract.total_population) > 0 THEN "
                f"SUM(tract.{name} * tract.total_population) / "
                f"SUM(CASE WHEN tract.{name} IS NOT NULL THEN tract.total_population END) "
                f"END AS {name}"
            )
    avg_select = ", ".join(avg_clauses)

    agg_cols = f"{sum_clauses}, {avg_select}"

    sql = f"""
        SELECT {agg_cols}
        FROM ({tract_sql}) tract
    """

    df = read_sql(sql, engine=engine, params={"prefix": f"{prefix}%", "yr": int(year)})

    # Replace Census sentinel → NaN
    for col in ACS_VARIABLES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == _CENSUS_SENTINEL, col] = float("nan")

    console.print(
        f"[dim]get_county_acs_summary_direct: state={state_fips}, county={county_fips}, year={year}[/dim]"
    )
    return df
