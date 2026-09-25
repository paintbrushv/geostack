"""IRS Statistics of Income (SOI) county-to-county migration flow analysis.

Downloads and parses IRS SOI inflow/outflow CSV files to quantify household
migration into and out of a target county. Data is in thousands of dollars
for AGI fields.

Data source:
  https://www.irs.gov/statistics/soi-tax-stats-migration-data

Usage:
    from geostack.migration import soi_migration

    result = soi_migration("48", "113", year=2022)
    # Returns dict with: inflow, outflow DataFrames, net_returns,
    #                    in_avg_agi, out_avg_agi, income_delta
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests
from rich.console import Console

from geostack.census import CensusClient

console = Console()

# IRS SOI migration CSV URL templates
# Inflow: people who moved INTO y2 county FROM y1 county
_SOI_INFLOW_URL = (
    "https://www.irs.gov/pub/irs-soi/countyinflow{year1}{year2}.csv"
)
# Outflow: people who moved OUT OF y1 county TO y2 county
_SOI_OUTFLOW_URL = (
    "https://www.irs.gov/pub/irs-soi/countyoutflow{year1}{year2}.csv"
)

# IRS uses special state FIPS codes for summary rows
_EXCLUDED_FIPS_PREFIXES = {"96", "97", "98"}

# Local cache directory (gitignored via data/)
_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "migration"


def _cache_path(filename: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / filename


def _parse_soi_csv(text: str, direction: str) -> pd.DataFrame:
    """Parse an IRS SOI county migration CSV into a tidy DataFrame.

    Args:
        text: Raw CSV text content from IRS SOI file.
        direction: "inflow" or "outflow".

    Returns:
        DataFrame with columns: origin_fips or dest_fips, county_name, returns,
        exemptions, agi, avg_agi_per_return. Summary rows (state FIPS 96/97/98)
        and same-county rows are excluded. Zero-return rows are excluded.
    """
    df = pd.read_csv(io.StringIO(text), dtype=str)

    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    if direction == "inflow":
        state_col = "y1_statefips"
        county_col = "y1_countyfips"
        name_col = "y1_countyname"
        partner_key = "origin_fips"
        subject_state_col = "y2_statefips"
        subject_county_col = "y2_countyfips"
    else:  # outflow
        state_col = "y2_statefips"
        county_col = "y2_countyfips"
        name_col = "y2_countyname"
        partner_key = "dest_fips"
        subject_state_col = "y1_statefips"
        subject_county_col = "y1_countyfips"

    # Build partner FIPS
    df[partner_key] = (
        df[state_col].str.strip().str.zfill(2)
        + df[county_col].str.strip().str.zfill(3)
    )

    # Build subject FIPS for same-county exclusion
    subject_fips = (
        df[subject_state_col].str.strip().str.zfill(2)
        + df[subject_county_col].str.strip().str.zfill(3)
    )

    # Exclude summary rows: partner state FIPS in excluded prefixes
    partner_state = df[state_col].str.strip().str.zfill(2)
    mask_summary = partner_state.isin(_EXCLUDED_FIPS_PREFIXES)

    # Exclude same-county rows
    mask_same = df[partner_key] == subject_fips

    # Coerce numeric columns
    df["n1"] = pd.to_numeric(df["n1"].str.strip(), errors="coerce").fillna(0)
    df["n2"] = pd.to_numeric(df["n2"].str.strip(), errors="coerce").fillna(0)
    df["agi"] = pd.to_numeric(df["agi"].str.strip(), errors="coerce").fillna(0)

    # Exclude zero-return rows
    mask_zero = df["n1"] == 0

    df = df[~mask_summary & ~mask_same & ~mask_zero].copy()

    # Rename to canonical names
    df = df.rename(columns={
        name_col: "county_name",
        "n1": "returns",
        "n2": "exemptions",
    })

    # Compute avg AGI per return (AGI is in thousands)
    df["avg_agi_per_return"] = df["agi"] / df["returns"]

    cols = [partner_key, "county_name", "returns", "exemptions", "agi",
            "avg_agi_per_return"]
    return df[cols].reset_index(drop=True)


def _download_soi(url: str, cache_file: str) -> str:
    """Download an IRS SOI CSV, returning raw text. Uses file cache."""
    cache = _cache_path(cache_file)
    if cache.exists():
        return cache.read_text()

    console.print(f"[dim]Downloading IRS SOI data from {url}...[/dim]")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cache.write_text(resp.text)
    return resp.text


def soi_migration(
    state_fips: str,
    county_fips: str,
    year: int = 2022,
) -> dict:
    """Download and parse IRS SOI migration flows for a target county.

    Args:
        state_fips: 2-digit state FIPS code (e.g. "48" for Texas).
        county_fips: 3-digit county FIPS code (e.g. "113" for Dallas).
        year: Tax year for migration data (default 2022). IRS publishes
              year Y data as the Y→Y+1 file pair.

    Returns:
        dict with keys:
          inflow      — DataFrame of counties people moved FROM into target
          outflow     — DataFrame of counties people moved TO from target
          net_returns — inflow returns minus outflow returns
          in_avg_agi  — weighted avg AGI per return for inflow (thousands)
          out_avg_agi — weighted avg AGI per return for outflow (thousands)
          income_delta — in_avg_agi minus out_avg_agi (positive = richer arrivals)
    """
    from geostack.fips import county_name

    state_fips = state_fips.zfill(2)
    county_fips = county_fips.zfill(3)
    subject_fips = state_fips + county_fips

    year1 = str(year)[-2:]
    year2 = str(year + 1)[-2:]

    label = county_name(state_fips, county_fips)
    console.print(
        f"[bold]SOI migration:[/bold] {label} ({subject_fips}) — tax year {year}"
    )

    # Check per-county parquet cache first
    inflow_cache = _cache_path(f"inflow_{subject_fips}_{year}.parquet")
    outflow_cache = _cache_path(f"outflow_{subject_fips}_{year}.parquet")

    if inflow_cache.exists() and outflow_cache.exists():
        inflow_df = pd.read_parquet(inflow_cache)
        outflow_df = pd.read_parquet(outflow_cache)
    else:
        # Download full national CSVs, then filter to subject county
        inflow_url = _SOI_INFLOW_URL.format(year1=year1, year2=year2)
        outflow_url = _SOI_OUTFLOW_URL.format(year1=year1, year2=year2)

        inflow_raw = _download_soi(
            inflow_url, f"national_inflow_{year}.csv"
        )
        outflow_raw = _download_soi(
            outflow_url, f"national_outflow_{year}.csv"
        )

        # Filter to subject county rows
        # For inflow: subject county is the destination (y2_statefips/y2_countyfips)
        # After parsing, we need to re-read raw to filter by subject county cols.
        # Since _parse_soi_csv drops the subject cols, we filter from raw instead.
        inflow_df = _filter_to_subject(
            inflow_raw, "inflow", state_fips, county_fips
        )
        outflow_df = _filter_to_subject(
            outflow_raw, "outflow", state_fips, county_fips
        )

        inflow_df.to_parquet(inflow_cache, index=False)
        outflow_df.to_parquet(outflow_cache, index=False)

    # Summary metrics
    in_returns = inflow_df["returns"].sum()
    out_returns = outflow_df["returns"].sum()
    net_returns = int(in_returns - out_returns)

    in_avg_agi = (
        (inflow_df["agi"] * inflow_df["returns"]).sum() / in_returns
        if in_returns > 0 else 0.0
    )
    out_avg_agi = (
        (outflow_df["agi"] * outflow_df["returns"]).sum() / out_returns
        if out_returns > 0 else 0.0
    )
    income_delta = round(in_avg_agi - out_avg_agi, 2)

    return {
        "inflow": inflow_df,
        "outflow": outflow_df,
        "net_returns": net_returns,
        "in_avg_agi": round(in_avg_agi, 2),
        "out_avg_agi": round(out_avg_agi, 2),
        "income_delta": income_delta,
        "year": year,
    }


def _filter_to_subject(
    text: str,
    direction: str,
    state_fips: str,
    county_fips: str,
) -> pd.DataFrame:
    """Filter a full national SOI CSV to rows for the subject county,
    then parse with _parse_soi_csv logic.

    For inflow:  subject county is y2 (destination)
    For outflow: subject county is y1 (origin)
    """
    df_raw = pd.read_csv(io.StringIO(text), dtype=str)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    if direction == "inflow":
        mask = (
            df_raw["y2_statefips"].str.strip().str.zfill(2) == state_fips
        ) & (
            df_raw["y2_countyfips"].str.strip().str.zfill(3) == county_fips
        )
    else:
        mask = (
            df_raw["y1_statefips"].str.strip().str.zfill(2) == state_fips
        ) & (
            df_raw["y1_countyfips"].str.strip().str.zfill(3) == county_fips
        )

    subset_text = df_raw[mask].to_csv(index=False)
    return _parse_soi_csv(subset_text, direction=direction)


# ── ACS B07 Geographic Mobility ──

_ACS_MOBILITY_VARS = [
    "B07010_001E",  # total pop 1yr+
    "B07010_002E",  # same house
    "B07010_017E",  # moved within same county
    "B07010_033E",  # moved from different county, same state
    "B07010_049E",  # moved from different state
    "B07011_001E",  # median income — all
    "B07011_017E",  # median income — same-county movers
    "B07011_033E",  # median income — diff-county movers
    "B07011_049E",  # median income — diff-state movers
]


def acs_mobility_profile(state_fips: str, county_fips: str,
                         year: int = 2023) -> dict:
    """ACS Geographic Mobility profile for a county."""
    client = CensusClient()
    df = client.get_acs(_ACS_MOBILITY_VARS, geography="county",
                        state=state_fips, county=county_fips, year=year)
    if df.empty:
        return {}

    row = df.iloc[0]

    def _int(var):
        return int(pd.to_numeric(row.get(var, 0), errors="coerce") or 0)

    total = max(_int("B07010_001E"), 1)
    same_house = _int("B07010_002E")
    in_county = _int("B07010_017E")
    diff_county = _int("B07010_033E")
    diff_state = _int("B07010_049E")

    mover_income_diff_county = pd.to_numeric(
        row.get("B07011_033E", 0), errors="coerce") or 0
    mover_income_diff_state = pd.to_numeric(
        row.get("B07011_049E", 0), errors="coerce") or 0
    mover_median = max(mover_income_diff_county, mover_income_diff_state)

    result = {
        "total_pop": total,
        "same_house_pct": same_house / total,
        "in_county_movers": in_county,
        "in_county_movers_pct": in_county / total,
        "diff_county_movers": diff_county,
        "diff_county_movers_pct": diff_county / total,
        "diff_state_movers": diff_state,
        "diff_state_movers_pct": diff_state / total,
        "mover_median_income": mover_median,
    }

    console.print(
        f"\n[bold]ACS Mobility (B07010, {year}):[/bold]"
        f"\n  Same house: {result['same_house_pct']:.1%}"
        f"\n  Within county: {in_county:,} ({result['in_county_movers_pct']:.1%})"
        f"\n  Diff county: {diff_county:,} ({result['diff_county_movers_pct']:.1%})"
        f"\n  Diff state: {diff_state:,} ({result['diff_state_movers_pct']:.1%})"
        f"\n  In-migrant median income: ${mover_median:,.0f}"
    )
    return result


def migration_profile(state_fips: str, county_fips: str,
                      soi_year: int = 2022, acs_year: int = 2023) -> dict:
    """Combined migration profile: IRS SOI flows + ACS mobility."""
    soi = soi_migration(state_fips, county_fips, year=soi_year)
    acs = acs_mobility_profile(state_fips, county_fips, year=acs_year)
    return {"soi": soi, "acs": acs}
