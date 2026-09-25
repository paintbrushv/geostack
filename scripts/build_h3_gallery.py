"""Rebuild the canonical H3 showcase gallery with consistent comma legends.

Regenerates the scheme-based choropleths (density / income / value-per-acre)
for DFW (res 8 + res 7) and Cooke County / Gainesville (res 9/10) using
legend_fmt="{:,.0f}" so all legends read with thousands separators and no
decimals. Reuses cached hex layers and the PostGIS dcad_parcels table.

Continuous/diverging maps (jobs growth, jobs-housing balance) and categorical
maps (Gi* hot-spots, bivariate demand-gap) use their own colorbars/legends and
are intentionally not part of this pass.

Run:
    python scripts/build_h3_gallery.py
"""

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import json
import pathlib

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from geostack.db import get_engine
from geostack.h3grid import h3_choropleth_static, parcels_to_h3_value, points_to_h3

FMT = "{:,.0f}"
DFW = pathlib.Path("outputs/msa_gravity/dfw_h3")
DFW7 = pathlib.Path("outputs/msa_gravity/dfw_h3_res7")
COOKE = pathlib.Path("outputs/msa_gravity/cooke_h3")


def _load(p, res, name):
    hx = gpd.read_parquet(p)
    hx.attrs["resolution"] = res
    hx.attrs["msa_name"] = name
    return hx


def dfw_res8():
    hx = _load(DFW / "dfw_h3_res8.parquet", 8, "Dallas-Fort Worth-Arlington, TX")
    jobs = hx[hx["jobs"] >= 100].copy(); jobs.attrs.update(hx.attrs)
    h3_choropleth_static(jobs, "jobs_density", scheme="naturalbreaks", k=7, cmap="inferno",
        legend_fmt=FMT, title="DFW — Jobs density (H3 res 8, 2023)",
        subtitle="Hexes with >=100 jobs · OpenStreetMap full detail",
        output_path=DFW / "dfw_h3_jobs_density.png")
    pop = hx[hx["population"] >= 200].copy(); pop.attrs.update(hx.attrs)
    h3_choropleth_static(pop, "pop_density", scheme="naturalbreaks", k=7, cmap="viridis",
        legend_fmt=FMT, title="DFW — Population density (H3 res 8, 2024)",
        subtitle="Hexes with >=200 residents · OpenStreetMap full detail",
        output_path=DFW / "dfw_h3_pop_density.png")
    inc = hx[hx["occupied_units"] >= 50].copy(); inc.attrs.update(hx.attrs)
    h3_choropleth_static(inc, "median_income", scheme="quantiles", k=7, cmap="YlGnBu",
        legend_fmt=FMT, title="DFW — Median household income (H3 res 8, 2024)",
        subtitle="Hexes with >=50 households · OpenStreetMap full detail",
        output_path=DFW / "dfw_h3_median_income.png")
    agg = hx[hx["occupied_units"] >= 50].copy()
    agg["agg_income_m"] = agg["agg_income"] / 1e6
    agg.attrs.update(hx.attrs)
    h3_choropleth_static(agg, "agg_income_m", scheme="quantiles", k=7, cmap="RdPu",
        legend_fmt=FMT, title="DFW — Aggregate household income ($M/hex, H3 res 8, ACS 2024)",
        subtitle="True aggregate income (ACS B19025) — total household purchasing power per hex · hexes with >=50 households",
        output_path=DFW / "dfw_h3_agg_income.png")
    # High-end variant: same style/lower breaks as dfw_h3_agg_income.png, but
    # the $66M-$683M top quantile (which one class flattens) is split into
    # continuing dollar steps so the wealth gradient stays readable.
    h3_choropleth_static(agg, "agg_income_m",
        bins=[8.5, 12.3, 18.2, 29.2, 44.3, 65.6, 90, 120, 160, 210],
        cmap="RdPu", legend_fmt=FMT,
        title="DFW — Aggregate household income ($M/hex, H3 res 8, ACS 2024)",
        subtitle="True aggregate income (ACS B19025) — total household purchasing power per hex · hexes with >=50 households · extended top-end bins",
        output_path=DFW / "dfw_h3_agg_income_highend.png")


def dfw_parcels():
    eng = get_engine()
    parc = gpd.read_postgis(
        "select gis_acct, tot_val, geometry from dcad_parcels where is_commercial and tot_val>0",
        eng, geom_col="geometry")
    vpa = parcels_to_h3_value(parc, value_col="tot_val", resolution=9)
    vpa["vpa_k"] = vpa["value_per_acre"] / 1000
    vpa.attrs.update(resolution=9, msa_name="Dallas County")
    h3_choropleth_static(vpa[vpa["parcel_count"] >= 2], "vpa_k", cmap="inferno",
        scheme="quantiles", k=7, legend_fmt=FMT,
        title="Dallas County — Commercial value per acre ($K/acre, H3 res 9, DCAD 2024)",
        subtitle="Fiscal productivity: appraised value per acre of land · OpenStreetMap",
        output_path=DFW / "dfw_h3_value_per_acre.png")
    pts = parc.copy(); pts["geometry"] = pts.geometry.centroid
    cv = points_to_h3(pts, resolution=8, value_cols=["tot_val"], agg="sum")
    cv["value_musd"] = cv["tot_val"] / 1e6
    cv.attrs.update(resolution=8, msa_name="Dallas County")
    h3_choropleth_static(cv[cv["tot_val"] >= 1e6], "value_musd", cmap="inferno",
        scheme="quantiles", k=7, legend_fmt=FMT,
        title="Dallas County — Commercial appraised value ($M/hex, H3 res 8, DCAD 2024)",
        subtitle="From real DCAD parcel geometry (~100% coverage) · OpenStreetMap",
        output_path=DFW / "dfw_h3_commercial_value_geometry.png")


def dfw_res7():
    hx = _load(DFW7 / "dfw_h3_res7.parquet", 7, "Dallas-Fort Worth-Arlington, TX")
    j = hx[hx["jobs"] >= 300].copy(); j.attrs.update(hx.attrs)
    h3_choropleth_static(j, "jobs_density", scheme="naturalbreaks", k=7, cmap="inferno",
        legend_fmt=FMT, zoom=10, title="DFW metro — Jobs density (H3 res 7, 2023)",
        subtitle="Metro overview · OpenStreetMap full detail",
        output_path=DFW7 / "res7_jobs_density.png")
    p = hx[hx["population"] >= 500].copy(); p.attrs.update(hx.attrs)
    h3_choropleth_static(p, "pop_density", scheme="naturalbreaks", k=7, cmap="viridis",
        legend_fmt=FMT, zoom=10, title="DFW metro — Population density (H3 res 7, 2024)",
        subtitle="Metro overview · OpenStreetMap full detail",
        output_path=DFW7 / "res7_pop_density.png")
    agg = hx[hx["occupied_units"] >= 150].copy()
    agg["agg_income_m"] = agg["agg_income"] / 1e6
    agg.attrs.update(hx.attrs)
    h3_choropleth_static(agg, "agg_income_m", scheme="quantiles", k=7, cmap="RdPu",
        legend_fmt=FMT, zoom=10, title="DFW metro — Aggregate household income ($M/hex, H3 res 7, ACS 2024)",
        subtitle="True aggregate income (ACS B19025) — total household purchasing power per hex · metro overview",
        output_path=DFW7 / "res7_agg_income.png")


def cooke():
    hx = _load(COOKE / "cooke_h3_res9.parquet", 9, "Cooke County, TX (Gainesville)")
    p = hx[hx["population"] >= 15].copy(); p.attrs.update(hx.attrs)
    h3_choropleth_static(p, "pop_density", scheme="naturalbreaks", k=6, cmap="viridis",
        legend_fmt=FMT, zoom=12, title="Cooke County / Gainesville TX — Population density (H3 res 9, 2024)",
        subtitle="Rural county · res 9 · populated hexes · OpenStreetMap full detail",
        output_path=COOKE / "cooke_h3_pop_density.png")
    j = hx[hx["jobs"] >= 10].copy(); j.attrs.update(hx.attrs)
    h3_choropleth_static(j, "jobs_density", scheme="naturalbreaks", k=6, cmap="inferno",
        legend_fmt=FMT, zoom=12, title="Cooke County / Gainesville TX — Jobs density (H3 res 9, 2023)",
        subtitle="Rural county · res 9 · employed hexes · OpenStreetMap full detail",
        output_path=COOKE / "cooke_h3_jobs_density.png")
    inc = hx[hx["occupied_units"] >= 8].copy(); inc.attrs.update(hx.attrs)
    h3_choropleth_static(inc, "median_income", scheme="quantiles", k=6, cmap="YlGnBu",
        legend_fmt=FMT, zoom=12, title="Cooke County / Gainesville TX — Median household income (H3 res 9, 2024)",
        subtitle="Rural county · res 9 · OpenStreetMap full detail",
        output_path=COOKE / "cooke_h3_median_income.png")
    # Value-per-acre from the full CCAD roll (source coordinates)
    df = pd.DataFrame(json.load(open("data/cooke/ccad_full_roll.json")))
    for c in ["latitude", "longitude", "appraisedValue", "legalAcreage"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["latitude"].between(33.4, 34.0)) & (df["longitude"].between(-97.7, -96.7))
            & (df["appraisedValue"] > 0)]
    allpts = gpd.GeoDataFrame(df, geometry=[Point(lo, la) for lo, la in zip(df.longitude, df.latitude)],
                              crs="EPSG:4326")
    v9 = parcels_to_h3_value(allpts, "appraisedValue", resolution=9, acreage_col="legalAcreage")
    v9["vpa_k"] = v9["value_per_acre"] / 1000; v9.attrs.update(resolution=9, msa_name="Cooke County, TX")
    h3_choropleth_static(v9[v9["parcel_count"] >= 2], "vpa_k", cmap="inferno", scheme="quantiles",
        k=6, legend_fmt=FMT, zoom=11,
        title="Cooke County — Value per acre ($K/acre, H3 res 9, full CCAD roll 2026)",
        subtitle=f"{len(allpts):,} parcels, full roll w/ source coordinates · OpenStreetMap",
        output_path=COOKE / "cooke_h3_value_per_acre_county.png")
    g = df[(df["latitude"].between(33.60, 33.68)) & (df["longitude"].between(-97.18, -97.08))]
    gp = gpd.GeoDataFrame(g, geometry=[Point(lo, la) for lo, la in zip(g.longitude, g.latitude)],
                          crs="EPSG:4326")
    v10 = parcels_to_h3_value(gp, "appraisedValue", resolution=10, acreage_col="legalAcreage")
    v10["vpa_k"] = v10["value_per_acre"] / 1000; v10.attrs.update(resolution=10, msa_name="Gainesville, TX")
    h3_choropleth_static(v10[v10["parcel_count"] >= 1], "vpa_k", cmap="inferno", scheme="quantiles",
        k=6, legend_fmt=FMT, zoom=14,
        title="Gainesville TX — Value per acre ($K/acre, H3 res 10, full CCAD roll 2026)",
        subtitle=f"{len(gp):,} Gainesville parcels · res 10 (~70m) · OpenStreetMap",
        output_path=COOKE / "cooke_h3_value_per_acre.png")


def dfw_wealth():
    # ACS wealth-proxy maps (B19062/B19054/B25079/B25081) — canonical builder
    # lives in build_h3_wealth_layers.py; reuses its cached parquet.
    import build_h3_wealth_layers as wealth
    wealth.main()


def main():
    dfw_res8()
    dfw_parcels()
    dfw_res7()
    dfw_wealth()
    cooke()
    print("GALLERY REBUILT with comma legends")


if __name__ == "__main__":
    main()
