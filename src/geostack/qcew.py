"""BLS Quarterly Census of Employment and Wages (QCEW) client.

QCEW provides county-level employment and wage data by 6-digit NAICS sector,
covering ~95% of U.S. jobs. This fills the wage granularity gap in LODES,
which only has 3 earnings tiers (≤$1,250/mo, $1,251–$3,333/mo, >$3,333/mo).

Data is released ~6 months after quarter end. Annual averages provide the
most stable view for CRE market studies.

No API key required. Public BLS QCEW data files.

Usage:
    from geostack.qcew import fetch_qcew, top_employers_by_naics, wage_distribution

    # Fetch all NAICS sectors for Dallas County 2023
    df = fetch_qcew(state_fips="48", county_fips="113", year=2023)

    # Top 10 sectors by employment
    top = top_employers_by_naics("48", "113", 2023, n=10)

    # Wage distribution across sectors
    wages = wage_distribution("48", "113", 2023)
"""

import io

import pandas as pd
import requests
from rich.console import Console

console = Console()

# QCEW data download base URL
QCEW_BASE = "https://data.bls.gov/cew/data/api"

# FIPS → QCEW area code: state(2) + county(3) = 5-digit code
# Ownership codes: 0=All, 5=Private, 1=Federal, 2=State, 3=Local
PRIVATE_OWNERSHIP = "5"

# 2-digit NAICS sector names (BLS CSV doesn't include industry titles)
NAICS_2DIGIT = {
    "11": "Agriculture, Forestry, Fishing",
    "21": "Mining, Oil & Gas",
    "22": "Utilities",
    "23": "Construction",
    "31-33": "Manufacturing",
    "42": "Wholesale Trade",
    "44-45": "Retail Trade",
    "48-49": "Transportation & Warehousing",
    "51": "Information",
    "52": "Finance & Insurance",
    "53": "Real Estate",
    "54": "Professional & Technical Services",
    "55": "Management of Companies",
    "56": "Admin, Support & Waste Mgmt",
    "61": "Educational Services",
    "62": "Health Care & Social Assistance",
    "71": "Arts, Entertainment & Recreation",
    "72": "Accommodation & Food Services",
    "81": "Other Services",
    "92": "Public Administration",
    "10": "Total, All Industries",
    "101": "Goods-Producing",
    "102": "Service-Providing",
    "1011": "Natural Resources & Mining",
    "1012": "Construction",
    "1013": "Manufacturing",
    "1021": "Trade, Transportation & Utilities",
    "1022": "Information",
    "1023": "Financial Activities",
    "1024": "Professional & Business Services",
    "1025": "Education & Health Services",
    "1026": "Leisure & Hospitality",
    "1027": "Other Services",
    "1028": "Unclassified",
    "1029": "Public Administration",
}


def _build_area_code(state_fips, county_fips):
    """Build the 5-digit QCEW area FIPS code."""
    return f"{str(state_fips).zfill(2)}{str(county_fips).zfill(3)}"


def fetch_qcew(state_fips, county_fips, year, quarter="a", ownership=PRIVATE_OWNERSHIP):
    """Fetch QCEW data for a county from the BLS API.

    Args:
        state_fips: 2-digit state FIPS (e.g., "48").
        county_fips: 3-digit county FIPS (e.g., "113").
        year: Data year (e.g., 2023).
        quarter: "1", "2", "3", "4", or "a" (annual average, default).
        ownership: "0" (all), "5" (private, default), "1" (federal),
                   "2" (state govt), "3" (local govt).

    Returns:
        DataFrame with columns: naics_code, naics_title, establishment_count,
        annual_avg_employment, total_annual_wages, avg_weekly_wage, avg_annual_pay.
    """
    area_code = _build_area_code(state_fips, county_fips)

    # BLS QCEW API endpoint for single-county data
    url = f"{QCEW_BASE}/{year}/{quarter}/area/{area_code}.csv"

    console.print(f"[dim]Fetching QCEW {year} for area {area_code}...[/dim]")
    resp = requests.get(url, timeout=60)

    if resp.status_code == 404:
        raise FileNotFoundError(
            f"QCEW data not found for area {area_code}, year {year}. "
            "Data may not be released yet."
        )
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))

    # Filter to the requested ownership type
    df = df[df["own_code"].astype(str) == str(ownership)].copy()

    # Filter to supersector/sector level (2-digit NAICS or higher)
    # agglvl_code 74 = county, 2-digit NAICS, private
    # agglvl_code 75 = county, 3-digit NAICS, private
    # agglvl_code 78 = county, 6-digit NAICS, private
    # We want 2-digit for summary, keep all for detail
    df = df[df["agglvl_code"].isin([74, 75, 76, 77, 78])].copy()

    # Normalize column names
    rename_map = {
        "industry_code": "naics_code",
        "annual_avg_estabs": "establishment_count",
        "annual_avg_emplvl": "annual_avg_employment",
        "total_annual_wages": "total_annual_wages",
        "annual_avg_wkly_wage": "avg_weekly_wage",
        "avg_annual_pay": "avg_annual_pay",
    }

    # Only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Add sector names from lookup (BLS CSV doesn't include them)
    df["naics_code"] = df["naics_code"].astype(str).str.strip()
    df["naics_title"] = df["naics_code"].map(NAICS_2DIGIT).fillna("Other")

    keep_cols = [
        "naics_code", "naics_title", "agglvl_code",
        "establishment_count", "annual_avg_employment",
        "total_annual_wages", "avg_weekly_wage", "avg_annual_pay",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Convert numeric columns
    for col in ["establishment_count", "annual_avg_employment", "total_annual_wages",
                "avg_weekly_wage", "avg_annual_pay"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    console.print(
        f"[green]QCEW {year}: {len(df)} industry records for area {area_code}[/green]"
    )
    return df


def top_employers_by_naics(state_fips, county_fips, year, n=10, ownership=PRIVATE_OWNERSHIP):
    """Top N employment sectors by job count (2-digit NAICS).

    Args:
        state_fips: 2-digit state FIPS.
        county_fips: 3-digit county FIPS.
        year: Data year.
        n: Number of top sectors to return.
        ownership: Ownership code (default "5" = private).

    Returns:
        DataFrame with top N sectors sorted by annual_avg_employment,
        including naics_code, naics_title, employment, avg_weekly_wage,
        avg_annual_pay, pct_of_total.
    """
    df = fetch_qcew(state_fips, county_fips, year, ownership=ownership)

    # Filter to 2-digit NAICS (agglvl_code 74 for private)
    sector_df = df[df["agglvl_code"] == 74].copy()

    # Remove totals rows (NAICS "10" = all industries)
    sector_df = sector_df[~sector_df["naics_code"].astype(str).isin(["10", "101", "102"])].copy()

    sector_df = sector_df.sort_values("annual_avg_employment", ascending=False).head(n)

    total_emp = sector_df["annual_avg_employment"].sum()
    sector_df["pct_of_total"] = (
        sector_df["annual_avg_employment"] / total_emp if total_emp > 0 else 0
    )

    console.print(
        f"[green]Top {n} sectors: {total_emp:,.0f} total employment[/green]"
    )
    return sector_df.reset_index(drop=True)


def wage_distribution(state_fips, county_fips, year, ownership=PRIVATE_OWNERSHIP):
    """Wage distribution across 2-digit NAICS sectors.

    Returns sectors sorted by average annual pay, useful for understanding
    the income profile of a county's workforce.

    Args:
        state_fips: 2-digit state FIPS.
        county_fips: 3-digit county FIPS.
        year: Data year.
        ownership: Ownership code.

    Returns:
        DataFrame with sectors sorted by avg_annual_pay descending,
        including cumulative employment percentage.
    """
    df = fetch_qcew(state_fips, county_fips, year, ownership=ownership)

    # Filter to 2-digit NAICS
    sector_df = df[df["agglvl_code"] == 74].copy()
    sector_df = sector_df[~sector_df["naics_code"].astype(str).isin(["10", "101", "102"])].copy()

    sector_df = sector_df.sort_values("avg_annual_pay", ascending=False)

    total_emp = sector_df["annual_avg_employment"].sum()
    sector_df["pct_employment"] = (
        sector_df["annual_avg_employment"] / total_emp if total_emp > 0 else 0
    )
    sector_df["cumulative_emp_pct"] = sector_df["pct_employment"].cumsum()

    # Compute county-wide weighted average pay
    if total_emp > 0:
        weighted_avg_pay = (
            (sector_df["avg_annual_pay"] * sector_df["annual_avg_employment"]).sum()
            / total_emp
        )
    else:
        weighted_avg_pay = 0

    sector_df.attrs["weighted_avg_annual_pay"] = weighted_avg_pay
    sector_df.attrs["total_employment"] = total_emp

    console.print(
        f"[green]Wage distribution: "
        f"weighted avg pay ${weighted_avg_pay:,.0f}/yr, "
        f"{len(sector_df)} sectors[/green]"
    )
    return sector_df.reset_index(drop=True)


def load_qcew_to_postgis(df, state_fips, county_fips, year, engine=None):
    """Load QCEW data into PostGIS for caching.

    Args:
        df: DataFrame from fetch_qcew().
        state_fips: 2-digit state FIPS.
        county_fips: 3-digit county FIPS.
        year: Data year.
        engine: SQLAlchemy engine.
    """
    from sqlalchemy import text

    engine = engine or get_engine()

    df = df.copy()
    df["state_fips"] = str(state_fips).zfill(2)
    df["county_fips"] = str(county_fips).zfill(3)
    df["year"] = year

    # Check for existing data
    with engine.begin() as conn:
        existing = conn.execute(
            text(
                "SELECT COUNT(*) FROM qcew_annual "
                "WHERE state_fips = :s AND county_fips = :c AND year = :y"
            ),
            {"s": str(state_fips).zfill(2), "c": str(county_fips).zfill(3), "y": year},
        ).scalar()

    if existing > 0:
        console.print(f"[dim]QCEW {year} already loaded ({existing} rows), skipping.[/dim]")
        return 0

    keep_cols = [
        "state_fips", "county_fips", "year",
        "naics_code", "naics_title",
        "establishment_count", "annual_avg_employment",
        "total_annual_wages", "avg_weekly_wage", "avg_annual_pay",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    df.to_sql("qcew_annual", engine, if_exists="append", index=False)
    console.print(f"[green]Loaded {len(df)} QCEW records → qcew_annual[/green]")
    return len(df)
