"""Demographic projections for workforce housing underwriting.

Uses CAGR extrapolation from ACS income time series.
Methodology: Hamilton-Perry CAGR extrapolation.

Usage:
    from geostack.projections import county_income_projection
    proj = county_income_projection("48", "113", base_years=(2019, 2023), target_year=2028)
"""

from __future__ import annotations

import pandas as pd
import requests
from rich.console import Console

from geostack.census import CensusClient, ACS_VARIABLES
from geostack.trends import compute_tract_trends, DEFAULT_VINTAGES
from geostack.analysis import aggregate_ilr_to_tracts

console = Console()

_MEDIAN_HH_INCOME = "B19013_001E"
_RENTER_TOTAL = "B25118_014E"

_PEP_2019_URL = "https://api.census.gov/data/2019/pep/population"
_PEP_2023_URL = "https://api.census.gov/data/2023/pep/charv"


from geostack.trends import compute_cagr


def _fetch_pep_population(state_fips: str, county_fips: str) -> dict[int, int]:
    """Fetch county population from Census PEP across both vintage APIs.

    Queries the 2019 vintage (2010-2019 via DATE_CODE) and the 2023 vintage
    (2020-2023 via YEAR). Deduplicates 2020 by preferring the 2023 vintage.

    Args:
        state_fips: 2-digit state FIPS (e.g., "48").
        county_fips: 3-digit county FIPS (e.g., "439").

    Returns:
        Dict of {year: population} with int keys and int values.
    """
    population: dict[int, int] = {}

    # --- 2019 vintage: DATE_CODE 3=2010 through 12=2019 ---
    try:
        console.print("[dim]Fetching PEP 2010-2019 population...[/dim]")
        resp = requests.get(
            _PEP_2019_URL,
            params={
                "get": "POP,DATE_CODE,NAME",
                "for": f"county:{county_fips}",
                "in": f"state:{state_fips}",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        headers = data[0]
        pop_idx = headers.index("POP")
        dc_idx = headers.index("DATE_CODE")

        for row in data[1:]:
            date_code = int(row[dc_idx])
            if date_code < 3 or date_code > 12:
                continue  # skip April 2010 base/estimate rows
            year = 2007 + date_code
            population[year] = int(row[pop_idx])
    except Exception as e:
        console.print(f"[yellow]PEP 2019 vintage failed: {e}[/yellow]")

    # --- 2023 vintage: YEAR is calendar year ---
    try:
        console.print("[dim]Fetching PEP 2020-2023 population...[/dim]")
        resp = requests.get(
            _PEP_2023_URL,
            params={
                "get": "POP,YEAR,NAME",
                "for": f"county:{county_fips}",
                "in": f"state:{state_fips}",
                "SEX": "0",
                "AGE": "999",
                "HISP": "0",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        headers = data[0]
        pop_idx = headers.index("POP")
        yr_idx = headers.index("YEAR")

        for row in data[1:]:
            year = int(row[yr_idx])
            pop = int(row[pop_idx])
            # Overwrite any 2020 from 2019 vintage (prefer newer estimate)
            population[year] = pop
    except Exception as e:
        console.print(f"[yellow]PEP 2023 vintage failed: {e}[/yellow]")

    return population


_TOTAL_HOUSEHOLDS = "B11001_001E"


def _fetch_household_time_series(
    state_fips: str,
    county_fips: str,
    start_year: int = 2015,
    end_year: int = 2023,
) -> dict[int, int]:
    """Fetch county household count from ACS B11001_001E across multiple years.

    Args:
        state_fips: 2-digit state FIPS (e.g., "48").
        county_fips: 3-digit county FIPS (e.g., "439").
        start_year: First ACS vintage to query (default 2015).
        end_year: Last ACS vintage to query (default 2023).

    Returns:
        Dict of {year: households} with int keys and int values.
        Years with missing/suppressed data are omitted.
    """
    client = CensusClient()
    households: dict[int, int] = {}

    for yr in range(start_year, end_year + 1):
        try:
            df = client.get_acs(
                [_TOTAL_HOUSEHOLDS],
                geography="county",
                state=state_fips,
                county=county_fips,
                year=yr,
            )
            if not df.empty:
                val = pd.to_numeric(df.iloc[0].get(_TOTAL_HOUSEHOLDS, 0), errors="coerce")
                if val and val > 0:
                    households[yr] = int(val)
        except Exception as e:
            console.print(f"[yellow]ACS {yr} households skipped: {e}[/yellow]")

    return households


def _project_value(base_value: float, cagr: float, n_years: int) -> float:
    return base_value * (1 + cagr) ** n_years


from geostack.qcew import fetch_qcew


def _build_variable_section(
    time_series: dict[int, int | float],
    target_year: int,
) -> dict:
    """Build a projection section dict from a time series.

    Computes CAGR from first to last year, then projects low/base/high.
    Returns a dict matching the county_demand_forecast schema.
    """
    if len(time_series) < 2:
        return {
            "time_series": time_series,
            "base_value": None,
            "base_year": None,
            "cagr": None,
            "projected": None,
        }

    sorted_years = sorted(time_series.keys())
    start_yr = sorted_years[0]
    end_yr = sorted_years[-1]
    start_val = time_series[start_yr]
    end_val = time_series[end_yr]
    n_years = end_yr - start_yr

    cagr = compute_cagr(start_val, end_val, n_years)
    project_years = target_year - end_yr

    return {
        "time_series": dict(sorted(time_series.items())),
        "base_value": end_val,
        "base_year": end_yr,
        "cagr": round(cagr, 6),
        "projected": {
            "low": round(_project_value(end_val, cagr * 0.5, project_years)),
            "base": round(_project_value(end_val, cagr, project_years)),
            "high": round(_project_value(end_val, cagr * 1.5, project_years)),
        },
    }


def _fetch_employment_time_series(
    state_fips: str,
    county_fips: str,
    start_year: int = 2015,
    end_year: int = 2023,
) -> dict[int, int]:
    """Fetch county total employment from QCEW across multiple years.

    Uses naics_code "10" (Total, All Industries) at agglvl_code 74
    (county-level 2-digit NAICS summary).

    Args:
        state_fips: 2-digit state FIPS.
        county_fips: 3-digit county FIPS.
        start_year: First year to query.
        end_year: Last year to query.

    Returns:
        Dict of {year: employment} with int keys and int values.
    """
    employment: dict[int, int] = {}

    for yr in range(start_year, end_year + 1):
        try:
            df = fetch_qcew(state_fips, county_fips, yr)
            # Filter to total employment (naics "10", agglvl 74)
            total = df[
                (df["naics_code"].astype(str) == "10")
                & (df["agglvl_code"] == 74)
            ]
            if not total.empty:
                emp = total.iloc[0]["annual_avg_employment"]
                if pd.notna(emp) and emp > 0:
                    employment[yr] = int(emp)
        except Exception as e:
            console.print(f"[yellow]QCEW {yr} skipped: {e}[/yellow]")

    return employment


def county_demand_forecast(
    state_fips: str,
    county_fips: str,
    target_year: int = 2030,
) -> dict:
    """County-level macro demand forecast with 3 variables, low/base/high scenarios.

    Fetches population (Census PEP), employment (BLS QCEW), and households
    (ACS B11001_001E) time series, computes CAGR for each, and projects to
    target_year with low (0.5x CAGR), base (1.0x), and high (1.5x) scenarios.

    Args:
        state_fips: 2-digit state FIPS (e.g., "48").
        county_fips: 3-digit county FIPS (e.g., "439").
        target_year: Year to project forward to (default 2030).

    Returns:
        Dict with keys: county, state_fips, county_fips, target_year,
        population, employment, households. Each variable section contains:
        time_series, base_value, base_year, cagr, projected {low, base, high}.
    """
    console.print(
        f"\n[bold]County Demand Forecast[/bold] — "
        f"state={state_fips}, county={county_fips}, target={target_year}"
    )

    # Fetch all three data sources
    pop_ts = _fetch_pep_population(state_fips, county_fips)
    hh_ts = _fetch_household_time_series(state_fips, county_fips)
    emp_ts = _fetch_employment_time_series(state_fips, county_fips)

    # Build projection sections
    pop_section = _build_variable_section(pop_ts, target_year)
    hh_section = _build_variable_section(hh_ts, target_year)
    emp_section = _build_variable_section(emp_ts, target_year)

    # Try to get county name from FIPS lookup
    county_name = ""
    try:
        from geostack.fips import county_name as _county_name
        county_name = _county_name(state_fips, county_fips) or ""
    except Exception:
        pass

    result = {
        "county": county_name,
        "state_fips": state_fips,
        "county_fips": county_fips,
        "target_year": target_year,
        "population": pop_section,
        "employment": emp_section,
        "households": hh_section,
    }

    # Rich console summary
    for label, section in [("Population", pop_section), ("Employment", emp_section),
                           ("Households", hh_section)]:
        if section["cagr"] is not None:
            proj = section["projected"]
            console.print(
                f"  [green]{label}:[/green] base={section['base_value']:,} "
                f"({section['base_year']}), CAGR={section['cagr']:.2%}, "
                f"projected Low={proj['low']:,} | Base={proj['base']:,} | High={proj['high']:,}"
            )
        else:
            console.print(f"  [yellow]{label}:[/yellow] insufficient data")

    return result


def county_income_projection(state_fips: str, county_fips: str,
                              base_years: tuple[int, int] = (2019, 2023),
                              target_year: int = 2028) -> dict:
    start_yr, end_yr = base_years
    years = list(range(start_yr, end_yr + 1))
    client = CensusClient()

    time_series = {}
    for yr in years:
        try:
            df = client.get_acs([_MEDIAN_HH_INCOME, _RENTER_TOTAL],
                                geography="county", state=state_fips,
                                county=county_fips, year=yr)
            if not df.empty:
                val = pd.to_numeric(df.iloc[0].get(_MEDIAN_HH_INCOME, 0), errors="coerce")
                if val and val > 0:
                    time_series[yr] = float(val)
        except Exception as e:
            console.print(f"[yellow]ACS {yr} skipped: {e}[/yellow]")

    if len(time_series) < 2:
        return {"base_median": None, "projected_median": None, "cagr": None,
                "scenarios": {}, "time_series": time_series,
                "base_years": base_years, "target_year": target_year}

    sorted_years = sorted(time_series.keys())
    start_val = time_series[sorted_years[0]]
    end_val = time_series[sorted_years[-1]]
    n_years = sorted_years[-1] - sorted_years[0]

    cagr = compute_cagr(start_val, end_val, n_years)
    project_years = target_year - sorted_years[-1]

    projected_base = _project_value(end_val, cagr, project_years)
    projected_low = _project_value(end_val, cagr * 0.5, project_years)
    projected_high = _project_value(end_val, cagr * 1.5, project_years)

    result = {
        "base_median": end_val,
        "projected_median": projected_base,
        "cagr": cagr,
        "scenarios": {"low": projected_low, "base": projected_base, "high": projected_high},
        "time_series": time_series,
        "base_years": base_years,
        "target_year": target_year,
    }

    console.print(
        f"\n[bold]Income Projection ({sorted_years[0]}→{target_year}):[/bold]"
        f"\n  Base ({sorted_years[-1]}): ${end_val:,.0f}"
        f"\n  CAGR: {cagr:.2%}"
        f"\n  Projected: Low ${projected_low:,.0f} | Base ${projected_base:,.0f} | High ${projected_high:,.0f}"
    )
    return result


def _classify_tract_growth(
    unit_cagr: float,
    income_cagr: float,
    ilr: float | None,
) -> str:
    """Classify a tract's growth trajectory.

    Args:
        unit_cagr: Housing unit 10-year CAGR (from ACS B25001).
        income_cagr: Median household income 10-year CAGR (from ACS B19013).
        ilr: Average improvement-to-land ratio, or None if unavailable.

    Returns:
        One of: new_development, growing, saturated,
        redevelopment_candidate, declining.
    """
    # Order matters: check new_development first, then declining
    if unit_cagr > 0.03:
        return "new_development"

    if income_cagr < 0:
        return "declining"

    if unit_cagr > 0.005:
        return "growing"

    # Flat housing growth (-0.5% to 0.5%), non-negative income
    if ilr is not None and ilr < 1.0:
        return "redevelopment_candidate"

    return "saturated"


def tract_growth_profile(
    state_fips: str,
    county_fips: str,
    include_ilr: bool = False,
    engine=None,
) -> pd.DataFrame:
    """Classify every tract in a county by growth trajectory.

    Uses ACS housing unit (B25001) and median household income (B19013)
    CAGRs from the trends module, optionally cross-referencing ILR from
    CAD parcel data.

    Args:
        state_fips: 2-digit state FIPS.
        county_fips: 3-digit county FIPS.
        include_ilr: If True, fetch ILR from analysis.aggregate_ilr_to_tracts()
                     to distinguish saturated vs redevelopment_candidate tracts.
        engine: SQLAlchemy engine (needed only if include_ilr=True).

    Returns:
        DataFrame with columns: geoid, classification, unit_cagr_10yr,
        income_cagr_10yr, ilr_avg.
    """
    console.print(
        f"\n[bold]Tract Growth Profile[/bold] — "
        f"state={state_fips}, county={county_fips}"
    )

    # Fetch ACS time series for housing units + income across 3 vintages
    client = CensusClient()
    unit_var = ACS_VARIABLES["total_housing_units"]   # B25001_001E
    income_var = ACS_VARIABLES["median_household_income"]  # B19013_001E

    frames = []
    for vintage in DEFAULT_VINTAGES:
        try:
            df = client.get_acs(
                [unit_var, income_var],
                geography="tract",
                state=state_fips,
                county=county_fips,
                year=vintage,
            )
            if not df.empty:
                client.build_geoid(df, geography="tract")
                # Melt to long format: geoid, vintage, variable, value
                for var_name, var_code in [("total_housing_units", unit_var),
                                           ("median_hh_income", income_var)]:
                    if var_code in df.columns:
                        long = df[["GEOID", var_code]].copy()
                        long = long.rename(columns={"GEOID": "geoid", var_code: "value"})
                        long["vintage"] = vintage
                        long["variable"] = var_name
                        long["value"] = pd.to_numeric(long["value"], errors="coerce")
                        frames.append(long)
        except Exception as e:
            console.print(f"[yellow]ACS {vintage} tracts skipped: {e}[/yellow]")

    if frames:
        long_data = pd.concat(frames, ignore_index=True)
        # Filter out Census suppressed values
        long_data = long_data[long_data["value"] > 0].copy()
    else:
        console.print("[yellow]No ACS tract data fetched — computing trends on empty data[/yellow]")
        long_data = pd.DataFrame(columns=["geoid", "vintage", "variable", "value"])

    # Compute trends (always call so mocks in tests can intercept)
    thresholds = {"stagnant_cagr": 0.005, "accel_threshold": 0.005}
    trends = compute_tract_trends(long_data, thresholds)

    if trends.empty:
        console.print("[red]No tract trend data available[/red]")
        return pd.DataFrame(columns=["geoid", "classification", "unit_cagr_10yr",
                                     "income_cagr_10yr", "ilr_avg"])

    # Pivot to get unit CAGR and income CAGR per tract
    unit_trends = trends[trends["variable"] == "total_housing_units"].set_index("geoid")
    income_trends = trends[trends["variable"] == "median_hh_income"].set_index("geoid")

    all_geoids = sorted(set(unit_trends.index) | set(income_trends.index))

    # Optionally fetch ILR
    ilr_by_tract: dict[str, float] = {}
    if include_ilr:
        try:
            ilr_df = aggregate_ilr_to_tracts(
                state_fips=state_fips,
                county_fips=county_fips,
                engine=engine,
            )
            if not ilr_df.empty and "geoid" in ilr_df.columns and "ilr_avg" in ilr_df.columns:
                ilr_by_tract = dict(zip(ilr_df["geoid"], ilr_df["ilr_avg"]))
        except Exception as e:
            console.print(f"[yellow]ILR fetch failed: {e}[/yellow]")

    # Classify each tract
    rows = []
    for geoid in all_geoids:
        unit_cagr = (
            unit_trends.loc[geoid, "cagr_10yr"]
            if geoid in unit_trends.index
            else 0.0
        )
        income_cagr = (
            income_trends.loc[geoid, "cagr_10yr"]
            if geoid in income_trends.index
            else 0.0
        )
        ilr_val = ilr_by_tract.get(geoid)

        classification = _classify_tract_growth(unit_cagr, income_cagr, ilr_val)

        rows.append({
            "geoid": geoid,
            "classification": classification,
            "unit_cagr_10yr": round(unit_cagr, 6),
            "income_cagr_10yr": round(income_cagr, 6),
            "ilr_avg": ilr_val,
        })

    result = pd.DataFrame(rows)

    # Summary
    counts = result["classification"].value_counts()
    console.print(f"\n[bold]Growth Classification ({len(result)} tracts):[/bold]")
    for cls, n in counts.items():
        console.print(f"  {cls}: {n}")

    return result
