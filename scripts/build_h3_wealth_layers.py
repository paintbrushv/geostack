"""Build DFW res-8 H3 wealth-proxy layers from ACS tract tables.

Net worth is not published at any small-area geography, so this builds the
strongest public *proxies* for accumulated wealth (stock), complementing the
agg_income map (flow):

    B19062_001E  aggregate interest, dividends & net rental income ($)
    B19054_001/2 households total / with investment income  -> share
    B25079_001E  aggregate value of owner-occupied housing units ($)
    B25081_001/9 owner-occupied total / owned free and clear -> share

All four are tract-level ACS 2024 5-yr, area-interpolated to H3 res 8 with the
same dasymetric pipeline as the agg_income layer, saved alongside it, and the
B19062 map is rendered here (the other three render from the same parquet).

Run:
    python scripts/build_h3_wealth_layers.py
"""

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd

from geostack.gravity import (
    MSA_DEFINITIONS, _CENSUS_BASE, _census_api_get,
    _read_cache, _write_cache, fetch_tract_geometries,
)
from geostack.h3grid import h3_choropleth_static, interpolate_polygons_to_h3

DFW = pathlib.Path("outputs/msa_gravity/dfw_h3")
PARQUET = DFW / "dfw_h3_res8_wealth.parquet"
FMT = "{:,.0f}"
YEAR = 2024

# ACS variable -> output column (all extensive counts/sums; shares derived after)
WEALTH_VARS = {
    "B19062_001E": "agg_inv_income",   # aggregate interest/dividends/net rental income
    "B19054_001E": "hh_total",         # households (universe for inv-income share)
    "B19054_002E": "inv_hh",           # households with interest/div/rental income
    "B25079_001E": "agg_home_value",   # aggregate value of owner-occupied units
    "B25081_001E": "owner_occ",        # owner-occupied housing units
    "B25081_009E": "free_clear",       # ...without a mortgage (free and clear)
}


def fetch_county_wealth(sf: str, cf: str, year: int = YEAR) -> pd.DataFrame:
    """Tract-level wealth-proxy variables for one county (cached)."""
    cache_key = f"acs5_wealth_{sf}_{cf}_{year}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached
    df = _census_api_get(f"{_CENSUS_BASE}/{year}/acs/acs5", {
        "get": "NAME," + ",".join(WEALTH_VARS),
        "for": "tract:*",
        "in": f"state:{sf} county:{cf}",
    })
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    for var, col in WEALTH_VARS.items():
        df[col] = pd.to_numeric(df[var], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan  # Census sentinels (-666666666 etc.)
    df = df[["GEOID", *WEALTH_VARS.values()]].copy()
    _write_cache(cache_key, df)
    return df


def build_wealth_parquet() -> gpd.GeoDataFrame:
    msa = MSA_DEFINITIONS["Dallas-Fort Worth-Arlington, TX"]
    sf = msa["state_fips"]
    tracts, vals = [], []
    for cf in msa["counties"]:
        tracts.append(fetch_tract_geometries(sf, cf, YEAR))
        vals.append(fetch_county_wealth(sf, cf))
    merged = pd.concat(tracts, ignore_index=True).merge(
        pd.concat(vals, ignore_index=True), on="GEOID", how="left")
    cols = list(WEALTH_VARS.values())
    merged[cols] = merged[cols].fillna(0)

    hexes = interpolate_polygons_to_h3(merged, resolution=8, count_cols=cols)
    hexes["inv_income_share"] = 100 * hexes["inv_hh"] / hexes["hh_total"].replace(0, np.nan)
    hexes["free_clear_share"] = 100 * hexes["free_clear"] / hexes["owner_occ"].replace(0, np.nan)

    # Bring over occupied_units from the canonical res-8 layer for filtering
    base = gpd.read_parquet(DFW / "dfw_h3_res8.parquet")[["h3", "occupied_units"]]
    hexes = hexes.merge(base, on="h3", how="left")

    hexes.attrs.update(resolution=8, msa_name="Dallas-Fort Worth-Arlington, TX", year=YEAR)
    hexes.to_parquet(PARQUET)
    print(f"WEALTH PARQUET saved: {PARQUET} ({len(hexes)} hexes)")
    return hexes


def map_inv_income(hexes: gpd.GeoDataFrame) -> None:
    m = hexes[hexes["occupied_units"] >= 50].copy()
    m["agg_inv_income_m"] = m["agg_inv_income"] / 1e6
    m.attrs.update(hexes.attrs)
    h3_choropleth_static(m, "agg_inv_income_m", scheme="quantiles", k=7, cmap="BuGn",
        legend_fmt=FMT, title="DFW — Aggregate investment income ($M/hex, H3 res 8, ACS 2024)",
        subtitle="ACS B19062: interest, dividends & net rental income — income that only exists where households hold assets · hexes with >=50 households",
        output_path=DFW / "dfw_h3_agg_inv_income.png")
    # High-end variant: same style as the standard map, but the top quantile —
    # which holds ~p98+ values from ~$105M up to $596M — is split into
    # continuing dollar steps so the wealth concentration in old-money
    # submarkets (Highland Park, Preston Hollow, Westover Hills) stays legible.
    # Bin edges chosen against the empirical distribution: p50≈18.6, p70≈33.4,
    # p85≈51.2, p90≈61.4, p95≈77.9, p98≈104.8, p99≈126.9, p99.5≈165.7.
    h3_choropleth_static(m, "agg_inv_income_m",
        bins=[18, 28, 40, 55, 70, 90, 115, 145, 180, 230],
        cmap="BuGn", legend_fmt=FMT,
        title="DFW — Aggregate investment income ($M/hex, H3 res 8, ACS 2024)",
        subtitle="ACS B19062: interest, dividends & net rental income — income that only exists where households hold assets · hexes with >=50 households · extended top-end bins",
        output_path=DFW / "dfw_h3_agg_inv_income_highend.png")


def map_inv_income_share(hexes: gpd.GeoDataFrame) -> None:
    m = hexes[hexes["occupied_units"] >= 50].copy()
    m.attrs.update(hexes.attrs)
    h3_choropleth_static(m, "inv_income_share", scheme="quantiles", k=7, cmap="PuBu",
        legend_fmt=FMT,
        title="DFW — Households with investment income (% of households, H3 res 8, ACS 2024)",
        subtitle="ACS B19054: share of households reporting interest, dividends or net rental income — breadth of asset ownership · hexes with >=50 households",
        output_path=DFW / "dfw_h3_inv_income_share.png")
    # High-end variant: same style as the standard map, with the top quantile
    # split into continuing percentage-point steps so the highest-participation
    # submarkets stay distinguishable. Bin edges against the empirical
    # distribution: p50≈14.9%, p70≈20.9%, p85≈28.0%, p90≈31.1%, p95≈36.1%,
    # p97≈40.9%, p99≈46.9%, p99.5≈52.4%, p99.9≈58.5%, max≈62.9%.
    h3_choropleth_static(m, "inv_income_share",
        bins=[15, 20, 25, 28, 31, 34, 38, 43, 48, 55],
        cmap="PuBu", legend_fmt=FMT,
        title="DFW — Households with investment income (% of households, H3 res 8, ACS 2024)",
        subtitle="ACS B19054: share of households reporting interest, dividends or net rental income — breadth of asset ownership · hexes with >=50 households · extended top-end bins",
        output_path=DFW / "dfw_h3_inv_income_share_highend.png")


def map_home_value(hexes: gpd.GeoDataFrame) -> None:
    m = hexes[hexes["occupied_units"] >= 50].copy()
    m["agg_home_value_m"] = m["agg_home_value"] / 1e6
    m.attrs.update(hexes.attrs)
    h3_choropleth_static(m, "agg_home_value_m", scheme="quantiles", k=7, cmap="YlOrBr",
        legend_fmt=FMT,
        title="DFW — Aggregate owner-occupied home value ($M/hex, H3 res 8, ACS 2024)",
        subtitle="ACS B25079: total self-reported value of owner-occupied homes — housing-wealth stock per hex · hexes with >=50 households",
        output_path=DFW / "dfw_h3_agg_home_value.png")


def map_free_clear(hexes: gpd.GeoDataFrame) -> None:
    m = hexes[hexes["owner_occ"] >= 30].copy()
    m.attrs.update(hexes.attrs)
    h3_choropleth_static(m, "free_clear_share", scheme="quantiles", k=7, cmap="cividis",
        legend_fmt=FMT,
        title="DFW — Homes owned free and clear (% of owner-occupied, H3 res 8, ACS 2024)",
        subtitle="ACS B25081: owner-occupied homes without a mortgage — accumulated-equity proxy, high where wealth is paid-in rather than borrowed · hexes with >=30 owner-occupied homes",
        output_path=DFW / "dfw_h3_free_clear_share.png")


def load_or_build() -> gpd.GeoDataFrame:
    if PARQUET.exists():
        hexes = gpd.read_parquet(PARQUET)
        hexes.attrs.update(resolution=8, msa_name="Dallas-Fort Worth-Arlington, TX", year=YEAR)
        return hexes
    return build_wealth_parquet()


def main():
    hexes = load_or_build()
    map_inv_income(hexes)
    map_inv_income_share(hexes)
    map_home_value(hexes)
    map_free_clear(hexes)


if __name__ == "__main__":
    main()
