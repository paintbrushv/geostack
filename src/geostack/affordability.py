"""Rent-to-income analysis for workforce housing.

Replaces traditional metro/submarket averages with tract-level, renter-only
income distributions and LODES-based employment destination analysis.

Traditional underwriting overstates the rent ceiling by:
  1. Including homeowners (who earn more than renters on average)
  2. Using geographic boundaries (MSA, submarket, 5mi radius) that don't
     reflect the property's actual renter population
  3. Ignoring where residents actually work and what those jobs pay

This module fixes these problems:
  - B25118 (renter-only household income) instead of B19001 (all HH)
  - LODES OD in the correct direction: home→work (where do residents
    of this property commute TO, and what do those jobs pay?)
  - QCEW employer wages at the employment destinations

Usage:
    from geostack.affordability import (
        rent_ceiling_comparison,
        resident_commute_profile,
        plot_income_comparison,
        plot_commute_profile,
    )

    # Full comparison of income methods
    comparison = rent_ceiling_comparison(
        lon=-96.6183, lat=32.8028, minutes=10,
        state_fips="48", county_fips="113",
        target_rents=[900, 1100, 1300, 1500],
    )

    # Where do residents commute and what do they earn?
    commute = resident_commute_profile(
        lon=-96.6183, lat=32.8028, minutes=10,
        state_fips="48", county_fips="113",
    )
"""

from __future__ import annotations

import pandas as pd
from rich.console import Console

from geostack.census import CensusClient
from geostack.db import get_engine, read_sql
from geostack.demographics import tracts_within_isochrone

console = Console()

# ── B25118: Tenure by Household Income (renter-occupied subset) ──
# B25118_001–013 = owner-occupied; B25118_014 = renter total; 015–025 = renter brackets
RENTER_INCOME_VARS: dict[str, tuple[str, int, int]] = {
    "B25118_015E": ("under_5k", 0, 5_000),
    "B25118_016E": ("5k_10k", 5_000, 10_000),
    "B25118_017E": ("10k_15k", 10_000, 15_000),
    "B25118_018E": ("15k_20k", 15_000, 20_000),
    "B25118_019E": ("20k_25k", 20_000, 25_000),
    "B25118_020E": ("25k_35k", 25_000, 35_000),
    "B25118_021E": ("35k_50k", 35_000, 50_000),
    "B25118_022E": ("50k_75k", 50_000, 75_000),
    "B25118_023E": ("75k_100k", 75_000, 100_000),
    "B25118_024E": ("100k_150k", 100_000, 150_000),
    "B25118_025E": ("150k_plus", 150_000, 300_000),
}
_RENTER_TOTAL_VAR = "B25118_014E"

# All-household income brackets (B19001) — for the "traditional" comparison.
ALL_HH_INCOME_VARS: dict[str, tuple[str, int, int]] = {
    "B19001_002E": ("under_10k", 0, 10_000),
    "B19001_003E": ("10k_15k", 10_000, 15_000),
    "B19001_004E": ("15k_20k", 15_000, 20_000),
    "B19001_005E": ("20k_25k", 20_000, 25_000),
    "B19001_006E": ("25k_30k", 25_000, 30_000),
    "B19001_007E": ("30k_35k", 30_000, 35_000),
    "B19001_008E": ("35k_40k", 35_000, 40_000),
    "B19001_009E": ("40k_45k", 40_000, 45_000),
    "B19001_010E": ("45k_50k", 45_000, 50_000),
    "B19001_011E": ("50k_60k", 50_000, 60_000),
    "B19001_012E": ("60k_75k", 60_000, 75_000),
    "B19001_013E": ("75k_100k", 75_000, 100_000),
    "B19001_014E": ("100k_125k", 100_000, 125_000),
    "B19001_015E": ("125k_150k", 125_000, 150_000),
    "B19001_016E": ("150k_200k", 150_000, 200_000),
    "B19001_017E": ("200k_plus", 200_000, 400_000),
}
_ALL_HH_TOTAL_VAR = "B19001_001E"

# LODES earning segments (annual, approximate)
_EARN_SEGMENTS = {
    "jobs_earn_low": ("Under $15K", 0, 15_000),
    "jobs_earn_mid": ("$15K–$40K", 15_000, 40_000),
    "jobs_earn_high": ("Over $40K", 40_000, 120_000),
}

# B25070: Gross Rent as a Percentage of Household Income
# Brackets tell us what % of renters are already rent-burdened
RENT_BURDEN_VARS = {
    "B25070_001E": "total",
    "B25070_002E": "under_10pct",
    "B25070_003E": "10_15pct",
    "B25070_004E": "15_20pct",
    "B25070_005E": "20_25pct",
    "B25070_006E": "25_30pct",
    "B25070_007E": "30_35pct",
    "B25070_008E": "35_40pct",
    "B25070_009E": "40_50pct",
    "B25070_010E": "50pct_plus",
    "B25070_011E": "not_computed",
}


# ────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────

def _build_distribution(var_spec, totals, total_hh, label):
    """Build a standard income distribution DataFrame from aggregated counts."""
    rows = []
    for census_var, (name, lb, ub) in var_spec.items():
        rows.append({
            "bracket": name,
            "lower_bound": lb,
            "upper_bound": ub,
            "households": totals.get(census_var, 0),
        })
    df = pd.DataFrame(rows)
    total_hh = max(total_hh, 1)
    df["pct"] = df["households"] / total_hh
    df["cumulative_pct"] = df["pct"].cumsum()
    df["midpoint"] = (df["lower_bound"] + df["upper_bound"]) / 2
    df.attrs["total_households"] = total_hh
    df.attrs["label"] = label
    df.attrs["median_income"] = _estimate_median(df)
    return df


def _estimate_median(dist):
    """Estimate median income from bracket distribution via linear interpolation."""
    for i, row in dist.iterrows():
        if row["cumulative_pct"] >= 0.5:
            if i == 0:
                return row["midpoint"]
            prev = dist.iloc[i - 1]
            frac = (0.5 - prev["cumulative_pct"]) / max(row["pct"], 1e-9)
            return row["lower_bound"] + frac * (row["upper_bound"] - row["lower_bound"])
    return dist["midpoint"].iloc[-1]


def _affordability_at_rent(dist, monthly_rent):
    """Fraction of households that can afford a given monthly rent (30% rule)."""
    required_annual = monthly_rent * 12 / 0.30
    pct_cannot = 0.0
    for _, row in dist.iterrows():
        lb, ub, pct = row["lower_bound"], row["upper_bound"], row["pct"]
        if ub <= required_annual:
            pct_cannot += pct
        elif lb < required_annual:
            bw = ub - lb
            if bw > 0:
                pct_cannot += pct * (required_annual - lb) / bw
    return max(0.0, min(1.0, 1.0 - pct_cannot))


def affordability_capture_curve(
    income_dist: pd.DataFrame,
    rent_range=range(800, 2001, 50),
) -> pd.DataFrame:
    """Percent of renter households that can afford each rent level at 30% RTI.

    Args:
        income_dist: DataFrame from isochrone_renter_income() or county_renter_income().
        rent_range: Iterable of monthly rent levels to evaluate (dollars).

    Returns:
        DataFrame with columns: monthly_rent, pct_can_afford, hh_count_can_afford.
    """
    rows = []
    total_hh = income_dist.attrs.get("total_households", income_dist["households"].sum())
    for rent in rent_range:
        pct = _affordability_at_rent(income_dist, rent)
        rows.append({
            "monthly_rent": int(rent),
            "pct_can_afford": pct,
            "hh_count_can_afford": int(round(pct * total_hh)),
        })
    return pd.DataFrame(rows)


def _fetch_acs_for_tracts(tract_geoids, var_spec, total_var, year):
    """Fetch ACS variables for a set of tract GEOIDs from the Census API."""
    county_groups: dict[str, list[str]] = {}
    for g in tract_geoids:
        county_groups.setdefault(g[2:5], []).append(g)

    all_vars = list(var_spec.keys()) + [total_var]
    client = CensusClient()
    frames = []
    for cfips, geoids in county_groups.items():
        state_fips = geoids[0][:2]
        try:
            df = client.get_acs(all_vars, geography="tract",
                                state=state_fips, county=cfips, year=year)
            df = client.build_geoid(df, "tract")
            df = df[df["GEOID"].isin(tract_geoids)]
            frames.append(df)
        except Exception as e:
            console.print(f"[yellow]ACS skipped county {cfips}: {e}[/yellow]")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fetch_county_acs(state_fips, county_fips, var_list, year):
    """County-level ACS via tract aggregation."""
    client = CensusClient()
    df = client.get_acs(var_list, geography="tract", state=state_fips,
                        county=county_fips, year=year)
    if df.empty:
        return {}
    totals = {}
    for var in var_list:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors="coerce").fillna(0)
            totals[var] = df[var].sum()
    return totals


def _weighted_acs(tract_geoids, weights, var_spec, total_var, year):
    """Fetch ACS for tracts and weight by a lookup dict {geoid: weight}."""
    acs_df = _fetch_acs_for_tracts(set(tract_geoids), var_spec, total_var, year)
    if acs_df.empty:
        return {}
    acs_df["w"] = acs_df["GEOID"].map(weights).fillna(0)
    all_vars = list(var_spec.keys()) + [total_var]
    totals = {}
    for var in all_vars:
        if var in acs_df.columns:
            acs_df[var] = pd.to_numeric(acs_df[var], errors="coerce").fillna(0)
            totals[var] = (acs_df[var] * acs_df["w"]).sum()
    return totals


# ────────────────────────────────────────────────────────────────────
# Income distribution functions
# ────────────────────────────────────────────────────────────────────

def isochrone_renter_income(lon, lat, minutes, state_fips, county_fips,
                            year=2023, engine=None):
    """Renter income distribution for tracts around the property.

    This IS the property's renter population — B25118 from the tracts
    where the renters live, weighted by isochrone overlap.
    """
    engine = engine or get_engine()
    tracts = tracts_within_isochrone(lon, lat, minutes, state_fips, engine=engine)
    if tracts.empty:
        return pd.DataFrame()

    geoids = {g for g in tracts["geoid"].tolist() if len(g) == 11}
    weights = tracts.set_index("geoid")["overlap_fraction"].to_dict()
    totals = _weighted_acs(list(geoids), weights, RENTER_INCOME_VARS, _RENTER_TOTAL_VAR, year)
    if not totals:
        return pd.DataFrame()

    total_hh = totals.get(_RENTER_TOTAL_VAR, 0)
    dist = _build_distribution(RENTER_INCOME_VARS, totals, total_hh,
                               label=f"Property Renter ({minutes}-min)")
    dist.attrs["method"] = "isochrone_renter"
    console.print(
        f"[green]Property renter income: median ~ ${dist.attrs['median_income']:,.0f}, "
        f"{len(geoids)} tracts[/green]")
    return dist


def isochrone_all_hh_income(lon, lat, minutes, state_fips, county_fips,
                            year=2023, engine=None):
    """All-household income (owners + renters) — the traditional baseline."""
    engine = engine or get_engine()
    tracts = tracts_within_isochrone(lon, lat, minutes, state_fips, engine=engine)
    if tracts.empty:
        return pd.DataFrame()

    geoids = {g for g in tracts["geoid"].tolist() if len(g) == 11}
    weights = tracts.set_index("geoid")["overlap_fraction"].to_dict()
    totals = _weighted_acs(list(geoids), weights, ALL_HH_INCOME_VARS, _ALL_HH_TOTAL_VAR, year)
    if not totals:
        return pd.DataFrame()

    total_hh = totals.get(_ALL_HH_TOTAL_VAR, 0)
    dist = _build_distribution(ALL_HH_INCOME_VARS, totals, total_hh,
                               label=f"All Households ({minutes}-min)")
    dist.attrs["method"] = "all_hh_isochrone"
    console.print(
        f"[green]All-HH isochrone income: median ~ ${dist.attrs['median_income']:,.0f}, "
        f"{len(geoids)} tracts[/green]")
    return dist


def county_renter_income(state_fips, county_fips, year=2023):
    """County-level renter income distribution (the "metro average" proxy)."""
    all_vars = list(RENTER_INCOME_VARS.keys()) + [_RENTER_TOTAL_VAR]
    totals = _fetch_county_acs(state_fips, county_fips, all_vars, year)
    if not totals:
        return pd.DataFrame()
    total_hh = totals.get(_RENTER_TOTAL_VAR, 0)
    from geostack.fips import county_name
    cname = county_name(state_fips, county_fips)
    dist = _build_distribution(RENTER_INCOME_VARS, totals, total_hh,
                               label=f"{cname} County Renter")
    dist.attrs["method"] = "county_renter"
    console.print(f"[green]County renter ({cname}): median ~ ${dist.attrs['median_income']:,.0f}[/green]")
    return dist


def county_all_hh_income(state_fips, county_fips, year=2023):
    """County-level ALL household income — the classic underwriting baseline."""
    all_vars = list(ALL_HH_INCOME_VARS.keys()) + [_ALL_HH_TOTAL_VAR]
    totals = _fetch_county_acs(state_fips, county_fips, all_vars, year)
    if not totals:
        return pd.DataFrame()
    total_hh = totals.get(_ALL_HH_TOTAL_VAR, 0)
    from geostack.fips import county_name
    cname = county_name(state_fips, county_fips)
    dist = _build_distribution(ALL_HH_INCOME_VARS, totals, total_hh,
                               label=f"{cname} County All HH")
    dist.attrs["method"] = "county_all_hh"
    console.print(f"[green]County all-HH ({cname}): median ~ ${dist.attrs['median_income']:,.0f}[/green]")
    return dist


# ────────────────────────────────────────────────────────────────────
# B25070 rent burden
# ────────────────────────────────────────────────────────────────────

def rent_burden_profile(lon, lat, minutes, state_fips, county_fips,
                        year=2023, engine=None):
    """What % of renters near the property are already rent-burdened?

    Uses B25070 (gross rent as % of household income) for tracts in the
    isochrone. Returns a dict with:
      - affordable_pct: paying < 30% of income on rent
      - burdened_pct: paying 30–49.9%
      - severely_burdened_pct: paying 50%+
      - total_renters: weighted count
    """
    engine = engine or get_engine()
    tracts = tracts_within_isochrone(lon, lat, minutes, state_fips, engine=engine)
    if tracts.empty:
        return {}

    geoids = {g for g in tracts["geoid"].tolist() if len(g) == 11}
    county_groups: dict[str, list[str]] = {}
    for g in geoids:
        county_groups.setdefault(g[2:5], []).append(g)

    client = CensusClient()
    frames = []
    for cfips, glist in county_groups.items():
        sfips = glist[0][:2]
        try:
            df = client.get_acs(list(RENT_BURDEN_VARS.keys()), geography="tract",
                                state=sfips, county=cfips, year=year)
            df = client.build_geoid(df, "tract")
            df = df[df["GEOID"].isin(geoids)]
            frames.append(df)
        except Exception as e:
            console.print(f"[yellow]B25070 skipped county {cfips}: {e}[/yellow]")

    if not frames:
        return {}

    acs_df = pd.concat(frames, ignore_index=True)
    overlap = tracts.set_index("geoid")["overlap_fraction"]
    acs_df["w"] = acs_df["GEOID"].map(overlap).fillna(0)

    # Convert and weight
    for var in RENT_BURDEN_VARS:
        if var in acs_df.columns:
            acs_df[var] = pd.to_numeric(acs_df[var], errors="coerce").fillna(0)

    totals = {}
    for var, name in RENT_BURDEN_VARS.items():
        if var in acs_df.columns:
            totals[name] = (acs_df[var] * acs_df["w"]).sum()

    total = max(totals.get("total", 1), 1)
    not_computed = totals.get("not_computed", 0)
    computable = max(total - not_computed, 1)

    affordable = sum(totals.get(k, 0) for k in
                     ["under_10pct", "10_15pct", "15_20pct", "20_25pct", "25_30pct"])
    burdened = sum(totals.get(k, 0) for k in ["30_35pct", "35_40pct", "40_50pct"])
    severely = totals.get("50pct_plus", 0)

    result = {
        "affordable_pct": affordable / computable,
        "burdened_pct": burdened / computable,
        "severely_burdened_pct": severely / computable,
        "total_renters": total,
        "not_computed": not_computed,
    }

    console.print(
        f"\n[bold]Rent burden (B25070, {minutes}-min):[/bold]"
        f"\n  Affordable (<30%):     {result['affordable_pct']:5.1%}"
        f"\n  Burdened (30–50%):     {result['burdened_pct']:5.1%}"
        f"\n  Severely burdened (50%+): {result['severely_burdened_pct']:5.1%}"
        f"\n  ({computable:,.0f} renter HH computed)"
    )
    return result


# ────────────────────────────────────────────────────────────────────
# LODES commute destination profile (HOME → WORK)
# ────────────────────────────────────────────────────────────────────

def resident_commute_profile(
    lon: float,
    lat: float,
    minutes: int,
    state_fips: str,
    county_fips: str,
    lodes_year: int = 2021,
    engine=None,
) -> dict:
    """Where do residents near this property commute TO for work?

    The correct LODES direction for analyzing a residential property:
    home tracts = isochrone around property → OD → work tracts.

    Returns dict with:
      - destinations_by_county: DataFrame (work county, total jobs, pct, earn segments)
      - destinations_by_tract: DataFrame (work tract, jobs, earnings breakdown)
      - earn_distribution: {segment: pct} — what share of resident workers
        fall into each LODES earnings tier
      - summary: top employment counties, total outbound flows, avg commute
    """
    engine = engine or get_engine()

    # Step 1: tracts around the property (where renters LIVE)
    home_tracts = tracts_within_isochrone(lon, lat, minutes, state_fips, engine=engine)
    if home_tracts.empty:
        return {}

    home_geoids = home_tracts["geoid"].tolist()

    # Step 2: LODES OD — where do residents GO to work?
    # h_geocode = home (near property), w_geocode = work (destination)
    sql_county = """
        SELECT
            w_state_fips,
            w_county_fips,
            SUM(total_jobs)      AS total_jobs,
            SUM(jobs_earn_low)   AS jobs_earn_low,
            SUM(jobs_earn_mid)   AS jobs_earn_mid,
            SUM(jobs_earn_high)  AS jobs_earn_high
        FROM lodes_od
        WHERE LEFT(h_geocode, 11) = ANY(:home_tracts)
          AND year = :y
        GROUP BY w_state_fips, w_county_fips
        ORDER BY total_jobs DESC
    """
    by_county = read_sql(sql_county, engine=engine,
                         params={"home_tracts": home_geoids, "y": lodes_year})

    sql_tract = """
        SELECT
            LEFT(w_geocode, 11)  AS work_tract,
            SUM(total_jobs)      AS total_jobs,
            SUM(jobs_earn_low)   AS jobs_earn_low,
            SUM(jobs_earn_mid)   AS jobs_earn_mid,
            SUM(jobs_earn_high)  AS jobs_earn_high
        FROM lodes_od
        WHERE LEFT(h_geocode, 11) = ANY(:home_tracts)
          AND year = :y
        GROUP BY LEFT(w_geocode, 11)
        ORDER BY total_jobs DESC
    """
    by_tract = read_sql(sql_tract, engine=engine,
                        params={"home_tracts": home_geoids, "y": lodes_year})

    if by_county.empty:
        console.print(f"[yellow]No outbound OD flows (year={lodes_year})[/yellow]")
        return {}

    total_outbound = by_county["total_jobs"].sum()
    by_county["pct"] = by_county["total_jobs"] / total_outbound
    by_county["cumulative_pct"] = by_county["pct"].cumsum()

    # Resolve county names
    from geostack.fips import county_name
    by_county["county_name"] = by_county.apply(
        lambda r: county_name(r["w_state_fips"], r["w_county_fips"]), axis=1)

    # Earnings distribution across ALL outbound workers
    earn_total = max(total_outbound, 1)
    earn_dist = {
        "earn_low_pct": by_county["jobs_earn_low"].sum() / earn_total,
        "earn_mid_pct": by_county["jobs_earn_mid"].sum() / earn_total,
        "earn_high_pct": by_county["jobs_earn_high"].sum() / earn_total,
        "earn_low_label": "Under $15K/yr ($1,250/mo)",
        "earn_mid_label": "$15K–$40K/yr ($1,250–$3,333/mo)",
        "earn_high_label": "Over $40K/yr ($3,333+/mo)",
    }

    # Implied rent ceiling from each earnings tier (30% rule)
    earn_dist["ceiling_low"] = 15_000 * 0.30 / 12     # ~$375/mo
    earn_dist["ceiling_mid"] = 27_500 * 0.30 / 12     # ~$688/mo (midpoint)
    earn_dist["ceiling_high"] = 80_000 * 0.30 / 12    # ~$2,000/mo (conservative)

    console.print(
        f"\n[bold]Resident commute profile ({minutes}-min around property):[/bold]"
    )
    console.print(f"  Total outbound worker flows: {total_outbound:,}")
    console.print(f"  Destination counties: {len(by_county)}")
    console.print(f"\n  [bold]Top employment destinations:[/bold]")
    for _, r in by_county.head(10).iterrows():
        console.print(
            f"    {r['county_name'] + ' County':.<35s} "
            f"{r['total_jobs']:>7,} jobs ({r['pct']:5.1%})"
        )

    console.print(f"\n  [bold]LODES earnings tiers (outbound workers):[/bold]")
    console.print(f"    Under $15K/yr ............ {earn_dist['earn_low_pct']:5.1%}")
    console.print(f"    $15K–$40K/yr ............. {earn_dist['earn_mid_pct']:5.1%}")
    console.print(f"    Over $40K/yr ............. {earn_dist['earn_high_pct']:5.1%}")

    return {
        "destinations_by_county": by_county,
        "destinations_by_tract": by_tract,
        "earn_distribution": earn_dist,
        "total_outbound": total_outbound,
        "home_tracts": len(home_geoids),
    }


# ────────────────────────────────────────────────────────────────────
# Comparison & visualization
# ────────────────────────────────────────────────────────────────────

def rent_ceiling_comparison(
    lon: float,
    lat: float,
    minutes: int,
    state_fips: str,
    county_fips: str,
    target_rents: list[int],
    current_rent: int | None = None,
    target_rent: int | None = None,
    comp_rents: list[int] | None = None,
    acs_year: int = 2023,
    lodes_year: int = 2021,
    engine=None,
) -> dict:
    """Compare rent affordability: county vs isochrone vs renter-only.

    Also runs resident_commute_profile to show where residents work
    and what LODES earnings tiers imply for the rent ceiling.

    Returns dict with:
      - distributions: {method: DataFrame}
      - affordability: DataFrame (rent x method x pct_can_afford)
      - medians / rent_ceilings: by method
      - commute: resident_commute_profile output
      - rent_burden: B25070 rent-burden breakdown dict
      - capture_curve: DataFrame (monthly_rent, pct_can_afford) from property renter dist
      - current_rent / target_rent / comp_rents: passed-through rent params
    """
    engine = engine or get_engine()
    distributions = {}

    console.print("\n[bold]Computing income distributions...[/bold]")

    console.print("[dim]  County all-household...[/dim]")
    distributions["county_all_hh"] = county_all_hh_income(state_fips, county_fips, acs_year)

    console.print("[dim]  County renter-only...[/dim]")
    distributions["county_renter"] = county_renter_income(state_fips, county_fips, acs_year)

    console.print("[dim]  Isochrone all-household...[/dim]")
    distributions["isochrone_all_hh"] = isochrone_all_hh_income(
        lon, lat, minutes, state_fips, county_fips, acs_year, engine)

    console.print("[dim]  Property renter (B25118)...[/dim]")
    distributions["property_renter"] = isochrone_renter_income(
        lon, lat, minutes, state_fips, county_fips, acs_year, engine)

    # B25070 rent burden
    console.print("[dim]  Rent burden (B25070)...[/dim]")
    burden = rent_burden_profile(
        lon, lat, minutes, state_fips, county_fips, acs_year, engine)

    # LODES outbound commute profile
    console.print("[dim]  Resident commute destinations (LODES)...[/dim]")
    commute = resident_commute_profile(
        lon, lat, minutes, state_fips, county_fips, lodes_year, engine)

    # Affordability at each rent level
    rows = []
    for rent in target_rents:
        for method, dist in distributions.items():
            if dist.empty:
                continue
            rows.append({
                "monthly_rent": rent,
                "method": method,
                "label": dist.attrs.get("label", method),
                "pct_can_afford": _affordability_at_rent(dist, rent),
            })
    affordability = pd.DataFrame(rows)

    medians = {}
    rent_ceilings = {}
    for method, dist in distributions.items():
        if dist.empty:
            continue
        med = dist.attrs.get("median_income", 0)
        medians[method] = med
        rent_ceilings[method] = med * 0.30 / 12

    console.print("\n[bold]Rent ceilings (30% of median):[/bold]")
    for method, ceiling in sorted(rent_ceilings.items(), key=lambda x: -x[1]):
        label = distributions[method].attrs.get("label", method)
        console.print(f"  {label:.<45s} ${ceiling:>7,.0f}/mo")

    # Capture curve: % of property renter households that can afford each rent level
    prop_dist = distributions.get("property_renter", pd.DataFrame())
    capture_curve = pd.DataFrame()
    if not prop_dist.empty:
        _low  = min((v for v in [current_rent, target_rent] if v is not None), default=800)
        _high = max((v for v in [current_rent, target_rent] if v is not None), default=2000)
        rent_min = _low - 100
        rent_max = _high + 300
        capture_curve = affordability_capture_curve(
            prop_dist, rent_range=range(max(500, rent_min), rent_max + 1, 50)
        )

    return {
        "distributions": distributions,
        "affordability": affordability,
        "medians": medians,
        "rent_ceilings": rent_ceilings,
        "commute": commute,
        "rent_burden": burden,
        "capture_curve": capture_curve,
        "current_rent": current_rent,
        "target_rent": target_rent,
        "comp_rents": comp_rents or [],
    }


def plot_income_comparison(
    comparison: dict,
    property_name: str | None = None,
    target_rent: int | None = None,
    figsize: tuple[float, float] = (14, 14),
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """Publication-quality four-panel figure.

    Panel 1: Overlaid income distributions (step histogram)
    Panel 2: Rent affordability curves
    Panel 3: LODES earnings tiers + commute destinations
    Panel 4: B25070 rent burden (% already paying >30% of income)
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    distributions = comparison["distributions"]
    affordability = comparison["affordability"]
    commute = comparison.get("commute", {})
    burden = comparison.get("rent_burden", {})

    STYLES = {
        "county_all_hh":    {"color": "#c4c4c4", "lw": 1.5, "ls": "--", "alpha": 0.7},
        "county_renter":    {"color": "#9e9e9e", "lw": 1.5, "ls": "-.",  "alpha": 0.8},
        "isochrone_all_hh": {"color": "#5b9bd5", "lw": 2.0, "ls": "--", "alpha": 0.8},
        "property_renter":  {"color": "#c0392b", "lw": 2.5, "ls": "-",  "alpha": 1.0},
    }

    has_commute = bool(commute)
    has_burden = bool(burden)
    capture_curve = comparison.get("capture_curve", pd.DataFrame())
    current_rent_plot = comparison.get("current_rent")
    target_rent_plot = comparison.get("target_rent") or target_rent
    comp_rents_plot = comparison.get("comp_rents", [])
    has_capture = not capture_curve.empty

    nrows = 2 + int(has_commute) + int(has_burden) + int(has_capture)
    ratios = [1.2, 1] + ([0.7] if has_commute else []) + ([0.5] if has_burden else []) + ([0.8] if has_capture else [])
    fig, axes = plt.subplots(nrows, 1, figsize=(figsize[0], figsize[1] + 3 * has_burden),
                             height_ratios=ratios)
    fig.subplots_adjust(hspace=0.45)
    ax1, ax2 = axes[0], axes[1]

    # ── Panel 1: Income distributions ──
    for method, dist in distributions.items():
        if dist.empty:
            continue
        style = STYLES.get(method, {"color": "gray", "lw": 1, "ls": "-", "alpha": 0.5})
        midpoints = dist["midpoint"].values / 1000
        ax1.step(midpoints, dist["pct"].values * 100, where="mid",
                 label=dist.attrs.get("label", method), **style)

    ax1.set_xlabel("Annual Household Income ($K)", fontsize=11)
    ax1.set_ylabel("% of Households", fontsize=11)
    ax1.set_xlim(0, 200)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Income Distribution — Who Can Actually Rent Here?",
                   fontsize=13, fontweight="bold", pad=12)

    if target_rent:
        req = target_rent * 12 / 0.30 / 1000
        ax1.axvline(req, color="#c0392b", lw=1, ls=":", alpha=0.6, zorder=0)
        ax1.annotate(f"Income needed\nfor ${target_rent:,}/mo",
                     xy=(req + 1, ax1.get_ylim()[1] * 0.85),
                     fontsize=8, ha="left", color="#c0392b")

    # ── Panel 2: Affordability curves ──
    if not affordability.empty:
        for method in STYLES:
            sub = affordability[affordability["method"] == method]
            if sub.empty:
                continue
            style = STYLES[method]
            ax2.plot(sub["monthly_rent"], sub["pct_can_afford"] * 100,
                     label=sub["label"].iloc[0], **style)
        if target_rent:
            ax2.axvline(target_rent, color="#c0392b", lw=1, ls=":", alpha=0.6)
            ax2.annotate(f"Subject rent: ${target_rent:,}/mo",
                         xy=(target_rent + 20, 5), fontsize=8, color="#c0392b")

    ax2.set_xlabel("Monthly Rent", fontsize=11)
    ax2.set_ylabel("% Who Can Afford", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("Rent Affordability at 30% Rent-to-Income",
                   fontsize=13, fontweight="bold", pad=12)

    # ── Panel 3: LODES commute destinations + earnings ──
    if has_commute:
        ax3 = axes[2]  # always index 2 since commute comes before burden
        earn = commute["earn_distribution"]
        by_county = commute["destinations_by_county"]

        # Left half: earnings tier bar
        tiers = ["Under $15K/yr", "$15K–$40K/yr", "Over $40K/yr"]
        pcts = [earn["earn_low_pct"] * 100, earn["earn_mid_pct"] * 100,
                earn["earn_high_pct"] * 100]
        colors = ["#e74c3c", "#f39c12", "#27ae60"]
        bars = ax3.barh(tiers, pcts, color=colors, height=0.5, alpha=0.85)

        # Annotate with implied rent ceiling
        ceilings = [earn["ceiling_low"], earn["ceiling_mid"], earn["ceiling_high"]]
        for bar, pct, ceil in zip(bars, pcts, ceilings):
            ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{pct:.1f}%  (ceiling: ${ceil:,.0f}/mo)",
                     va="center", fontsize=9, fontweight="bold")

        ax3.set_xlim(0, max(pcts) * 1.5)
        ax3.set_xlabel("% of Outbound Workers", fontsize=11)
        ax3.set_title(
            f"LODES Earnings Tiers — Where Residents Work "
            f"({commute['total_outbound']:,} worker flows)",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax3.grid(axis="x", alpha=0.3)

        # Top 5 destination counties as text annotation
        top5 = by_county.head(5)
        dest_text = "Top destinations: " + " | ".join(
            f"{r['county_name']} Co. ({r['pct']:.0%})"
            for _, r in top5.iterrows()
        )
        ax3.text(0.5, -0.18, dest_text, transform=ax3.transAxes,
                 ha="center", fontsize=9, color="gray")

    # ── Panel 4: Rent burden (B25070) ──
    if has_burden:
        ax4 = axes[2 + int(has_commute)]
        labels = ["Affordable\n(<30%)", "Burdened\n(30–50%)", "Severely Burdened\n(50%+)"]
        vals = [burden["affordable_pct"] * 100, burden["burdened_pct"] * 100,
                burden["severely_burdened_pct"] * 100]
        colors_b = ["#27ae60", "#f39c12", "#e74c3c"]
        bars = ax4.barh(labels, vals, color=colors_b, height=0.5, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
        ax4.set_xlim(0, max(vals) * 1.4)
        ax4.set_xlabel("% of Renter Households", fontsize=11)
        burdened_total = burden["burdened_pct"] + burden["severely_burdened_pct"]
        ax4.set_title(
            f"Current Rent Burden — {burdened_total:.0%} of Renters Already Pay >30% of Income",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax4.grid(axis="x", alpha=0.3)

    # ── Panel 5: Affordability capture curve ──
    if has_capture:
        ax5 = axes[-1]
        ax5.plot(
            capture_curve["monthly_rent"],
            capture_curve["pct_can_afford"] * 100,
            color="#c0392b", lw=2.5, label="Property renter pool",
        )

        # Shade the "deep pool" zone: 50–70% capture rate
        ax5.axhspan(50, 70, color="#27ae60", alpha=0.08, label="Deep pool (50–70%)")
        ax5.axhline(50, color="#27ae60", lw=0.8, ls="--", alpha=0.5)

        # Annotate current and target rent
        for rent, label, color in [
            (current_rent_plot, "Current", "#5b9bd5"),
            (target_rent_plot, "Target", "#c0392b"),
        ]:
            if rent:
                ax5.axvline(rent, color=color, lw=1.5, ls=":", alpha=0.8)
                pct_row = capture_curve[capture_curve["monthly_rent"] <= rent].tail(1)
                pct_val = pct_row["pct_can_afford"].iloc[0] * 100 if not pct_row.empty else None
                ann = f"{label}: ${rent:,}/mo"
                if pct_val is not None:
                    ann += f"\n({pct_val:.0f}% can afford)"
                ax5.annotate(ann, xy=(rent + 20, pct_val or 50),
                             fontsize=8, color=color, va="center")

        # Annotate comp rents
        for comp_rent in (comp_rents_plot or []):
            ax5.axvline(comp_rent, color="#9e9e9e", lw=1, ls=":", alpha=0.5)

        ax5.set_xlabel("Monthly Rent", fontsize=11)
        ax5.set_ylabel("% Who Can Afford", fontsize=11)
        ax5.set_ylim(0, 105)
        ax5.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax5.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax5.grid(axis="y", alpha=0.3)
        ax5.set_title(
            "Affordability Capture Curve — % of Renter Pool at Each Rent Level",
            fontsize=13, fontweight="bold", pad=12,
        )

    # ── Suptitle ──
    title = "Rent-to-Income Analysis"
    if property_name:
        title = f"{property_name}\n{title}"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)

    fig.text(0.99, 0.005,
             "Source: ACS 5-Year (B25118, B19001) + LODES OD | geostack",
             ha="right", fontsize=7, color="gray", style="italic")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        console.print(f"[green]Saved → {save_path}[/green]")

    return fig
