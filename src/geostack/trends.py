"""Tract-level ACS time series with CPI deflation and provenance tracking.

Provides utilities for building multi-vintage ACS time series at the census
tract level, deflating income/rent variables to constant dollars using the
BLS CPI, and attaching provenance metadata to DataFrames.

Usage:
    from geostack.trends import load_cpi_deflators, deflate_to_constant_dollars

    deflators = load_cpi_deflators(base_year=2023)
    result = deflate_to_constant_dollars(df, deflators, DEFLATABLE_VARIABLES)
"""

from __future__ import annotations

import hashlib
import datetime
import pathlib

import numpy as np
import pandas as pd
import requests
from rich.console import Console

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = pathlib.Path.home() / ".cache" / "geostack" / "trends"
_BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
_CPI_SERIES_ID = "CUUR0000SA0"

DEFAULT_VINTAGES = [2013, 2018, 2023]

DEFLATABLE_VARIABLES = {
    "median_rent",
    "median_hh_income",
    "median_family_income",
    "per_capita_income",
    "median_home_value",
    "median_gross_rent",
}

DEFAULT_THRESHOLDS = {
    "rent_growth_high": 0.30,
    "rent_growth_low": -0.05,
    "income_growth_high": 0.20,
    "vacancy_high": 0.12,
    "vacancy_low": 0.04,
}

VARIABLE_TYPES = {
    "median_rent": "income",
    "median_hh_income": "income",
    "median_family_income": "income",
    "per_capita_income": "income",
    "median_home_value": "income",
    "median_gross_rent": "income",
    "renter_hh": "count",
    "owner_hh": "count",
    "total_hh": "count",
    "total_pop": "count",
    "vacancy_rate": "rate",
    "renter_rate": "rate",
}

_console = Console()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str, suffix: str = ".parquet") -> pathlib.Path:
    """Return a deterministic cache file path for the given key."""
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{digest}{suffix}"


# ---------------------------------------------------------------------------
# CPI deflation
# ---------------------------------------------------------------------------

def load_cpi_deflators(
    base_year: int = 2023,
    vintages: list[int] | None = None,
) -> dict[int, float]:
    """Fetch annual CPI values from BLS and return deflators relative to base_year.

    Deflators are expressed as cpi_year / cpi_base_year, so values < 1.0
    indicate years with lower price levels than the base year (i.e., earlier years
    when base_year is recent).

    Parameters
    ----------
    base_year:
        The reference year. Deflators for this year will equal 1.0.
    vintages:
        List of years to include. Defaults to DEFAULT_VINTAGES plus base_year.

    Returns
    -------
    dict mapping year (int) -> deflator (float)
    """
    if vintages is None:
        vintages = DEFAULT_VINTAGES

    all_years = sorted(set(vintages) | {base_year})
    min_yr = min(all_years)
    max_yr = max(all_years)

    cache_key = f"cpi_{_CPI_SERIES_ID}_{min_yr}_{max_yr}"
    cache_file = _cache_path(cache_key)

    if cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        resp = requests.get(
            _BLS_API_URL,
            json={
                "seriesid": [_CPI_SERIES_ID],
                "startyear": str(min_yr),
                "endyear": str(max_yr),
                "annualaverage": True,
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()

        series_data = payload["Results"]["series"][0]["data"]
        df = pd.DataFrame(series_data)
        df["year"] = df["year"].astype(int)
        df["value"] = df["value"].astype(float)

        # Cache for future calls
        df.to_parquet(cache_file, index=False)

    # Filter to annual averages (period M13)
    annual = df[df["period"] == "M13"].copy()
    annual = annual.set_index("year")["value"]

    base_cpi = annual[base_year]
    deflators: dict[int, float] = {}
    for yr in all_years:
        if yr in annual.index:
            deflators[yr] = float(annual[yr] / base_cpi)

    return deflators


def deflate_to_constant_dollars(
    data: pd.DataFrame,
    cpi_deflators: dict[int, float],
    deflatable_variables: set[str],
) -> pd.DataFrame:
    """Divide deflatable variable values by their vintage's CPI deflator.

    Non-deflatable variables (counts, rates) are left unchanged. Provenance
    metadata is attached to the result DataFrame's .attrs dict.

    Parameters
    ----------
    data:
        Long-format DataFrame with columns: geoid, vintage, variable, value.
    cpi_deflators:
        Mapping of year -> deflator (from load_cpi_deflators).
    deflatable_variables:
        Set of variable names that should be deflated.

    Returns
    -------
    DataFrame with the same shape; deflatable rows have adjusted values.
    Provenance stored in result.attrs["cpi_base_year"] and
    result.attrs["cpi_deflators"].
    """
    result = data.copy()

    mask = result["variable"].isin(deflatable_variables)

    def _apply_deflator(row):
        deflator = cpi_deflators.get(row["vintage"])
        if deflator is None or deflator == 0:
            return row["value"]
        return row["value"] / deflator

    result.loc[mask, "value"] = result[mask].apply(_apply_deflator, axis=1)

    # Determine base year from the deflator equal to 1.0
    base_year = next(
        (yr for yr, d in cpi_deflators.items() if abs(d - 1.0) < 1e-9),
        None,
    )

    result.attrs["cpi_base_year"] = base_year
    result.attrs["cpi_deflators"] = dict(cpi_deflators)

    return result


# ---------------------------------------------------------------------------
# Census 2010→2020 tract crosswalk
# ---------------------------------------------------------------------------

_CROSSWALK_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/tract/"
    "tab20_tract20_tract10_st{state_fips}.txt"
)
_VINTAGES_NEEDING_CROSSWALK = {2013, 2018}


def load_crosswalk(
    state_fips: str = "48",
    *,
    _skip_cache: bool = False,
) -> pd.DataFrame:
    """Download the Census 2010→2020 tract relationship file for a state.

    Returns a DataFrame with columns:
        geoid_2010 (str), geoid_2020 (str), areal_weight (float)

    areal_weight = AREALAND_PART / AREALAND_TRACT_10, so weights for all
    2020 tracts that overlap a given 2010 tract sum to 1.0.

    Parameters
    ----------
    state_fips:
        Two-digit state FIPS code (zero-padded string, e.g. "48").
    _skip_cache:
        If True, bypass the parquet cache (used in tests).
    """
    cache_key = f"crosswalk_{state_fips}"
    cache_file = _cache_path(cache_key)

    if not _skip_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    url = _CROSSWALK_URL.format(state_fips=state_fips)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    from io import StringIO
    raw = pd.read_csv(StringIO(resp.text), sep="|", dtype=str)

    # Compute areal weight
    raw["AREALAND_PART"] = raw["AREALAND_PART"].astype(float)
    raw["AREALAND_TRACT_10"] = raw["AREALAND_TRACT_10"].astype(float)

    # Avoid division by zero for water-only tracts
    raw["areal_weight"] = np.where(
        raw["AREALAND_TRACT_10"] > 0,
        raw["AREALAND_PART"] / raw["AREALAND_TRACT_10"],
        1.0,
    )

    df = raw[["GEOID_TRACT_10", "GEOID_TRACT_20", "areal_weight"]].copy()
    df = df.rename(columns={
        "GEOID_TRACT_10": "geoid_2010",
        "GEOID_TRACT_20": "geoid_2020",
    })

    if not _skip_cache:
        df.to_parquet(cache_file, index=False)

    return df


# 2000→2010 tract relationship files (comma-delimited, no header, 30 columns).
# Filename pattern: {state_abbr}{state_fips}trf.txt (e.g. tx48trf.txt).
_CROSSWALK_2000_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel/trf_txt/{abbr}{state_fips}trf.txt"
)

_TRF_2000_COLUMNS = [
    "STATE00", "COUNTY00", "TRACT00", "GEOID00", "POP00", "HU00", "PART00",
    "AREA00", "AREALAND00",
    "STATE10", "COUNTY10", "TRACT10", "GEOID10", "POP10", "HU10", "PART10",
    "AREA10", "AREALAND10",
    "AREAPT", "AREALANDPT", "AREAPCT00PT", "AREALANDPCT00PT",
    "AREAPCT10PT", "AREALANDPCT10PT",
    "POP10PT", "POPPCT00", "POPPCT10", "HU10PT", "HUPCT00", "HUPCT10",
]


def load_crosswalk_2000(
    state_fips: str = "48",
    *,
    _skip_cache: bool = False,
) -> pd.DataFrame:
    """Download the Census 2000→2010 tract relationship file for a state.

    Returns columns: geoid_2000, geoid_2010, areal_weight — where areal_weight
    is the fraction of the 2000 tract's land area falling in each 2010 tract
    (computed from raw AREALANDPT/AREALAND00 for precision; the file's
    2-decimal percent columns round tiny slivers to 0.00).

    Cross-state border slivers in the file are filtered to in-state rows.
    """
    cache_key = f"crosswalk2000_{state_fips}"
    cache_file = _cache_path(cache_key)
    if not _skip_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    from geostack.lodes import FIPS_TO_ABBR
    abbr = FIPS_TO_ABBR[str(state_fips).zfill(2)]
    url = _CROSSWALK_2000_URL.format(abbr=abbr, state_fips=state_fips)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    from io import StringIO
    raw = pd.read_csv(StringIO(resp.text), header=None, names=_TRF_2000_COLUMNS,
                      dtype=str)
    raw = raw[raw["STATE00"] == str(state_fips).zfill(2)].copy()

    raw["AREALANDPT"] = pd.to_numeric(raw["AREALANDPT"], errors="coerce").fillna(0)
    raw["AREALAND00"] = pd.to_numeric(raw["AREALAND00"], errors="coerce").fillna(0)
    raw["areal_weight"] = np.where(
        raw["AREALAND00"] > 0,
        raw["AREALANDPT"] / raw["AREALAND00"],
        1.0,
    )

    df = raw[["GEOID00", "GEOID10", "areal_weight"]].rename(
        columns={"GEOID00": "geoid_2000", "GEOID10": "geoid_2010"})
    # Drop pure zero-overlap slivers; renormalize so each 2000 tract sums to 1
    df = df[df["areal_weight"] > 0].copy()
    totals = df.groupby("geoid_2000")["areal_weight"].transform("sum")
    df["areal_weight"] = df["areal_weight"] / totals.replace(0, np.nan)
    df = df.dropna(subset=["areal_weight"])

    if not _skip_cache:
        df.to_parquet(cache_file, index=False)
    return df


def load_crosswalk_2000_2020(
    state_fips: str = "48",
    *,
    _skip_cache: bool = False,
) -> pd.DataFrame:
    """Chained 2000→2020 tract crosswalk (2000→2010 ∘ 2010→2020).

    Needed because pre-2010 data (Decennial 2000, ACS 5-year ≤2009) is
    published on 2000-vintage tracts; the 2010→2020 file alone cannot map
    them — which silently drops exactly the fast-growing areas whose tracts
    were split (e.g. Frisco/Collin/Denton).

    Returns columns: geoid_2000, geoid_2020, areal_weight — chained weights
    (w₀₀→₁₀ × w₁₀→₂₀) summed per (2000, 2020) pair; each 2000 tract's weights
    sum to ~1.0.
    """
    cache_key = f"crosswalk2000_2020_{state_fips}"
    cache_file = _cache_path(cache_key)
    if not _skip_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    a = load_crosswalk_2000(state_fips, _skip_cache=_skip_cache)      # 2000→2010
    b = load_crosswalk(state_fips, _skip_cache=_skip_cache)           # 2010→2020
    chained = a.merge(b, on="geoid_2010", suffixes=("_a", "_b"))
    chained["areal_weight"] = chained["areal_weight_a"] * chained["areal_weight_b"]
    df = (chained.groupby(["geoid_2000", "geoid_2020"], as_index=False)
                 ["areal_weight"].sum())
    df = df[df["areal_weight"] > 0]

    if not _skip_cache:
        df.to_parquet(cache_file, index=False)
    return df


def apply_crosswalk(
    raw_data: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    variable_types: dict[str, str],
) -> pd.DataFrame:
    """Translate 2010-vintage tract GEOIDs to 2020 GEOIDs using areal weighting.

    For vintages in _VINTAGES_NEEDING_CROSSWALK (2013, 2018):
      - Count variables are multiplied by areal_weight.
      - Rate/median/income variables are passed through unchanged for each piece.
      - 2010 tracts with no crosswalk match are kept with weight=1.0.

    For vintage 2023 (already uses 2020 GEOIDs), the data passes through
    untouched.

    Provenance is recorded in result.attrs["crosswalk"].

    Parameters
    ----------
    raw_data:
        Long-format DataFrame with columns: geoid, vintage, variable, value.
    crosswalk_df:
        DataFrame from load_crosswalk() with geoid_2010, geoid_2020,
        areal_weight.
    variable_types:
        Mapping of variable name -> type ("count", "rate", "income", etc.).

    Returns
    -------
    DataFrame with 2020-era GEOIDs in the geoid column and appropriately
    scaled values.
    """
    vintages_present = set(raw_data["vintage"].unique())
    needs_crosswalk = vintages_present & _VINTAGES_NEEDING_CROSSWALK
    passthrough_vintages = vintages_present - _VINTAGES_NEEDING_CROSSWALK

    parts: list[pd.DataFrame] = []

    # Pass 2023 (and any other non-crosswalk vintages) through unchanged
    if passthrough_vintages:
        parts.append(raw_data[raw_data["vintage"].isin(passthrough_vintages)].copy())

    # Apply crosswalk to vintages that need it
    if needs_crosswalk:
        xw_data = raw_data[raw_data["vintage"].isin(needs_crosswalk)].copy()

        # Left-join so unmatched 2010 tracts are preserved
        merged = xw_data.merge(
            crosswalk_df,
            left_on="geoid",
            right_on="geoid_2010",
            how="left",
        )

        # Fill unmatched rows (tract unchanged, no crosswalk entry)
        merged["geoid_2020"] = merged["geoid_2020"].fillna(merged["geoid"])
        merged["areal_weight"] = merged["areal_weight"].fillna(1.0)

        # Scale count variables; pass others through
        def _scale_value(row):
            vtype = variable_types.get(row["variable"], "count")
            if vtype == "count":
                return row["value"] * row["areal_weight"]
            return row["value"]

        merged["value"] = merged.apply(_scale_value, axis=1)

        # Replace geoid with 2020 GEOID and drop crosswalk columns
        merged["geoid"] = merged["geoid_2020"]
        result_cw = merged.drop(columns=["geoid_2010", "geoid_2020", "areal_weight"])
        parts.append(result_cw)

    result = pd.concat(parts, ignore_index=True) if parts else raw_data.copy()

    result.attrs["crosswalk"] = {
        "vintages_crosswalked": sorted(needs_crosswalk),
        "vintages_passthrough": sorted(passthrough_vintages),
    }

    return result


# ---------------------------------------------------------------------------
# Trend computation, classification, and calibration
# ---------------------------------------------------------------------------

def compute_cagr(start_val: float, end_val: float, n_years: int) -> float:
    """Compute compound annual growth rate.

    Returns 0.0 for invalid inputs (zero/negative start, zero n_years).
    """
    if not n_years or start_val <= 0 or end_val <= 0:
        return 0.0
    return (end_val / start_val) ** (1.0 / n_years) - 1.0


def _classify_trend(
    cagr_10yr: float,
    acceleration: float,
    thresholds: dict,
) -> str:
    """Classify a trend into one of five categories.

    Categories (in order of evaluation):
      - "Stagnant"            — |cagr_10yr| <= stagnant_cagr
      - "Accelerating Growth" — cagr_10yr > stagnant AND acceleration > accel_threshold
      - "Steady Growth"       — cagr_10yr > stagnant AND acceleration <= accel_threshold
      - "Accelerating Decline"— cagr_10yr < -stagnant AND acceleration < -accel_threshold
      - "Declining"           — cagr_10yr < -stagnant AND acceleration >= -accel_threshold
    """
    stagnant = thresholds.get("stagnant_cagr", 0.005)
    accel = thresholds.get("accel_threshold", 0.005)

    if abs(cagr_10yr) <= stagnant:
        return "Stagnant"
    if cagr_10yr > stagnant:
        if acceleration > accel:
            return "Accelerating Growth"
        return "Steady Growth"
    # cagr_10yr < -stagnant
    if acceleration < -accel:
        return "Accelerating Decline"
    return "Declining"


def compute_tract_trends(
    data: pd.DataFrame,
    thresholds: dict,
    vintages: list[int] | None = None,
) -> pd.DataFrame:
    """Compute per-(geoid, variable) CAGR, acceleration, and trend classification.

    Parameters
    ----------
    data:
        Long-format DataFrame with columns: geoid, vintage, variable, value.
        Must contain rows for vintages 2013, 2018, and 2023 (or as specified).
    thresholds:
        Dict with at least ``stagnant_cagr`` and ``accel_threshold`` keys.
    vintages:
        Three-element sorted list [early, mid, recent]. Defaults to [2013, 2018, 2023].

    Returns
    -------
    DataFrame with columns: geoid, variable, value_<yr>, cagr_10yr, cagr_5yr,
    acceleration, trend_class.  Provenance stored in .attrs.
    """
    if vintages is None:
        vintages = DEFAULT_VINTAGES

    v_early, v_mid, v_recent = vintages[0], vintages[1], vintages[2]
    n_full = v_recent - v_early   # e.g. 10
    n_half = v_recent - v_mid     # e.g.  5

    rows = []
    for (geoid, variable), group in data.groupby(["geoid", "variable"]):
        val_map = dict(zip(group["vintage"], group["value"]))

        early_val = val_map.get(v_early)
        mid_val   = val_map.get(v_mid)
        recent_val = val_map.get(v_recent)

        if early_val is None or recent_val is None:
            continue

        cagr_10yr = compute_cagr(early_val, recent_val, n_full)

        if mid_val is not None:
            cagr_5yr = compute_cagr(mid_val, recent_val, n_half)
            cagr_early = compute_cagr(early_val, mid_val, v_mid - v_early)
            acceleration = cagr_5yr - cagr_early
        else:
            cagr_5yr = 0.0
            acceleration = 0.0

        trend_class = _classify_trend(cagr_10yr, acceleration, thresholds)

        row = {
            "geoid": geoid,
            "variable": variable,
            f"value_{v_early}": early_val,
            f"value_{v_mid}": mid_val,
            f"value_{v_recent}": recent_val,
            "cagr_10yr": cagr_10yr,
            "cagr_5yr": cagr_5yr,
            "acceleration": acceleration,
            "trend_class": trend_class,
        }
        rows.append(row)

    result = pd.DataFrame(rows)

    # --- Provenance / reproducibility hash ---
    variables_sorted = sorted(data["variable"].unique().tolist())
    hash_payload = (
        str(sorted(vintages))
        + str(variables_sorted)
        + str(sorted(thresholds.items()))
    )
    repro_hash = hashlib.sha256(hash_payload.encode()).hexdigest()

    result.attrs["reproducibility_hash"] = repro_hash
    result.attrs["vintages"] = vintages
    result.attrs["thresholds"] = dict(thresholds)
    result.attrs["generated_at"] = datetime.datetime.utcnow().isoformat()

    return result


console = _console  # alias for use in compute_catchment_trends


def calibrate_thresholds(
    tract_trends: pd.DataFrame,
    variable: str,
    method: str = "percentile",
) -> dict:
    """Derive classification thresholds from the empirical distribution of trends.

    Parameters
    ----------
    tract_trends:
        DataFrame produced by ``compute_tract_trends``, containing at minimum
        ``variable``, ``cagr_10yr``, and ``acceleration`` columns.
    variable:
        Which variable to calibrate against (e.g. "median_rent").
    method:
        "percentile" — stagnant_cagr = median(|cagr_10yr|) * 0.25,
                       accel_threshold = IQR(acceleration) * 0.5.

    Returns
    -------
    dict with keys ``stagnant_cagr`` and ``accel_threshold``.
    """
    subset = tract_trends[tract_trends["variable"] == variable]

    if method == "percentile":
        cagr_vals = subset["cagr_10yr"].abs()
        accel_vals = subset["acceleration"]

        stagnant_cagr = float(np.median(cagr_vals)) * 0.25
        iqr = float(np.percentile(accel_vals, 75) - np.percentile(accel_vals, 25))
        accel_threshold = iqr * 0.5

        return {
            "stagnant_cagr": stagnant_cagr,
            "accel_threshold": accel_threshold,
        }

    raise ValueError(f"Unknown calibration method: {method!r}")


# ---------------------------------------------------------------------------
# Catchment aggregation
# ---------------------------------------------------------------------------

def compute_catchment_trends(
    lon: float, lat: float, minutes: int,
    tract_trends: pd.DataFrame,
    renter_hh_by_tract: dict[str, float],
    state_fips: str = "48",
    engine=None,
) -> dict:
    """Aggregate tract-level trends for a property's isochrone catchment.

    Uses renter-household-weighted averages of CAGR and acceleration metrics.
    """
    from geostack.demographics import tracts_within_isochrone
    from geostack.db import get_engine as _get_engine

    engine = engine or _get_engine()
    iso_tracts = tracts_within_isochrone(lon, lat, minutes, state_fips, engine=engine)

    if iso_tracts.empty:
        return {"n_tracts": 0, "total_renter_hh": 0}

    iso_geoids = set(iso_tracts["geoid"].tolist())
    local = tract_trends[tract_trends["geoid"].isin(iso_geoids)].copy()

    if local.empty:
        return {"n_tracts": 0, "total_renter_hh": 0}

    local["weight"] = local["geoid"].map(renter_hh_by_tract).fillna(0)

    result = {"n_tracts": local["geoid"].nunique()}
    result["total_renter_hh"] = sum(renter_hh_by_tract.get(g, 0) for g in iso_geoids)

    for variable, group in local.groupby("variable"):
        w = group["weight"]
        var_result = {}
        for metric in ["cagr_10yr", "cagr_5yr", "acceleration"]:
            vals = group[metric].dropna()
            weights = w.loc[vals.index]
            if len(vals) > 0 and weights.sum() > 0:
                var_result[metric] = (vals * weights).sum() / weights.sum()
            else:
                var_result[metric] = None

        if var_result.get("cagr_10yr") is not None and var_result.get("acceleration") is not None:
            thresholds = tract_trends.attrs.get("thresholds", DEFAULT_THRESHOLDS)
            var_result["trend_class"] = _classify_trend(
                var_result["cagr_10yr"], var_result["acceleration"], thresholds
            )
        result[variable] = var_result

    console.print(
        f"\n[bold]Catchment trends ({result['n_tracts']} tracts, "
        f"{result['total_renter_hh']:,} renter HH):[/bold]"
    )
    for var, metrics in result.items():
        if isinstance(metrics, dict) and "cagr_10yr" in metrics:
            cagr = metrics["cagr_10yr"]
            cls = metrics.get("trend_class", "")
            console.print(f"  {var}: {cagr:.2%} CAGR (10yr) — {cls}")

    return result


# ---------------------------------------------------------------------------
# Trend map visualization
# ---------------------------------------------------------------------------

def plot_trend_map(
    tract_trends: pd.DataFrame,
    tract_geometries: "gpd.GeoDataFrame",
    variable: str,
    metric: str = "cagr_10yr",
    title: str | None = None,
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """Choropleth trend map with basemap tiles."""
    import contextily as cx
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    sub = tract_trends[tract_trends["variable"] == variable].copy()
    merged = tract_geometries.merge(sub, on="geoid", how="inner")
    merged_wm = merged.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=(18, 15))

    if metric == "trend_class":
        class_colors = {
            "Accelerating Growth": "#1a8c3a",
            "Steady Growth": "#82d47e",
            "Stagnant": "#d4d4d4",
            "Declining": "#e8a838",
            "Accelerating Decline": "#c0392b",
        }
        merged_wm["_color"] = merged_wm["trend_class"].map(class_colors).fillna("#d4d4d4")
        merged_wm.plot(color=merged_wm["_color"], edgecolor="none", ax=ax, alpha=0.75, zorder=1)
        patches = [mpatches.Patch(color=c, label=l) for l, c in class_colors.items()]
        ax.legend(handles=patches, loc="lower left", fontsize=10, framealpha=0.95,
                  title="Trend Classification", title_fontsize=11)
    else:
        vmax = max(abs(merged_wm[metric].quantile(0.05)), abs(merged_wm[metric].quantile(0.95)))
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "trend", ["#c0392b", "#e8a838", "#f5f5f5", "#82d47e", "#1a8c3a"], N=256
        )
        merged_wm.plot(column=metric, cmap=cmap, vmin=-vmax, vmax=vmax,
                       edgecolor="none", ax=ax, alpha=0.75, zorder=1,
                       legend=True, legend_kwds={
                           "label": f"{metric} ({variable})",
                           "orientation": "horizontal",
                           "fraction": 0.03, "pad": 0.02, "shrink": 0.6,
                       })

    cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels, zoom=11, zorder=0)
    cx.add_basemap(ax, source=cx.providers.CartoDB.VoyagerOnlyLabels, zoom=11, zorder=3)

    xmin, ymin, xmax, ymax = merged_wm.total_bounds
    ax.set_xlim(xmin - 2000, xmax + 2000)
    ax.set_ylim(ymin - 2000, ymax + 2000)

    map_title = title or f"{variable.replace('_', ' ').title()} — {metric}"
    fig.suptitle(map_title, fontsize=20, fontweight="bold", y=0.97)

    provenance = tract_trends.attrs
    fig.text(0.98, 0.015,
             f"Vintages: {provenance.get('vintages', '?')}  |  "
             f"CPI base: {provenance.get('cpi_base_year', '?')}  |  geostack",
             ha="right", fontsize=8, color="#999", style="italic")

    ax.set_axis_off()
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        console.print(f"[green]Saved → {save_path}[/green]")

    return fig
