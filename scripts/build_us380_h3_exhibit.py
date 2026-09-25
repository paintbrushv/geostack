"""Build a public-safe US 380 H3 corridor exhibit from existing geostack patterns.

The corridor geometry is a transparent public proxy: 2024 Census TIGER US 380 /
University Drive centerlines plus a coarse TxDOT McKinney Blue Alternative
control line based on FEIS/ROD endpoints. It is not a parcel screen, owner list,
or private score. Outputs are aggregate H3/corridor layers for corridor study.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import unary_union

from geostack.h3grid import county_h3_layer
from geostack.tiger import download_roads

STATE = "48"
COUNTIES = ["085", "121"]  # Collin, Denton
YEARS = [(2010, 2010), (2013, 2013), (2020, 2020), (2023, 2023)]  # ACS/Decennial year, LODES year
RESOLUTION = 8
BUFFER_MILES = 3.0
PROJECT_CRS = "EPSG:2276"  # Texas North Central, feet
OUT = Path("outputs/msa_gravity/us380_h3")
CACHE = Path("data/tiger_cache")
ROUTE_NAMES = {
    "US Hwy 380",
    "E US Hwy 380",
    "W US Hwy 380",
    "University Dr",
    "E University Dr",
    "W University Dr",
}
TIMEPOINTS = [year for year, _ in YEARS]
BASE_YEAR = 2010
NOW_YEAR = 2023

# Coarse public-control line for TxDOT's selected McKinney Blue Alternative.
# The FEIS/ROD confirms Coit Road → FM 1827/New Hope Road and selection of the
# northern Blue Alternative; replace this hand-built control line with official
# TxDOT GIS when a shapefile/feature service is available.
MCKINNEY_BLUE_ALT_WAYPOINTS = [
    (-96.7720, 33.2220),  # existing US 380 near Coit Road
    (-96.7240, 33.2470),
    (-96.6550, 33.2620),
    (-96.5850, 33.2550),
    (-96.5150, 33.2350),  # near FM 1827 / New Hope Road tie-in
]


def load_us380_route() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load public US 380 route plus coarse TxDOT McKinney Blue Alternative proxy."""
    roads = []
    for county in COUNTIES:
        gdf = download_roads(STATE, county, year=2024, cache_dir=CACHE)
        names = gdf["FULLNAME"].fillna("")
        mask = names.isin(ROUTE_NAMES) & gdf["MTFCC"].eq("S1200")
        roads.append(gdf.loc[mask].copy())

    tiger_route = pd.concat(roads, ignore_index=True)
    if tiger_route.empty:
        raise RuntimeError("No TIGER US 380 route segments matched")

    tiger_route = gpd.GeoDataFrame(tiger_route, geometry="geometry", crs="EPSG:4269")
    blue_alt = gpd.GeoDataFrame(
        {
            "LINEARID": ["txdot-blue-alt-proxy"],
            "FULLNAME": ["TxDOT McKinney Blue Alternative proxy"],
            "RTTYP": ["TXDOT"],
            "MTFCC": ["FEIS_ROD_PROXY"],
        },
        geometry=[LineString(MCKINNEY_BLUE_ALT_WAYPOINTS)],
        crs="EPSG:4326",
    ).to_crs(tiger_route.crs)

    route = pd.concat([tiger_route, blue_alt], ignore_index=True)
    route = gpd.GeoDataFrame(route, geometry="geometry", crs=tiger_route.crs).to_crs(PROJECT_CRS)
    route_geom = unary_union(route.geometry)
    buffer_geom = route_geom.buffer(BUFFER_MILES * 5280)
    buffer_gdf = gpd.GeoDataFrame(
        {"name": [f"US 380 / TxDOT segment {BUFFER_MILES:g}-mile corridor proxy"]},
        geometry=[buffer_geom],
        crs=PROJECT_CRS,
    )
    return route, buffer_gdf


def load_or_build_h3(year: int, jobs_year: int) -> gpd.GeoDataFrame:
    """Build/cache Collin + Denton res-8 H3 layer for a time point."""
    path = OUT / f"collin_denton_h3_res{RESOLUTION}_{year}_jobs{jobs_year}.parquet"
    if path.exists():
        return gpd.read_parquet(path)

    hexes = county_h3_layer(
        STATE,
        COUNTIES,
        name="Collin + Denton Counties, TX",
        resolution=RESOLUTION,
        year=year,
        jobs_year=jobs_year,
        use_crosswalk=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    hexes.to_parquet(path, index=False)
    return hexes


def corridor_slice(hexes: gpd.GeoDataFrame, buffer_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return H3 cells intersecting the corridor buffer."""
    h = hexes.to_crs(PROJECT_CRS).copy()
    mask = h.geometry.intersects(buffer_gdf.geometry.iloc[0])
    return h.loc[mask].copy()


def build_panel(route: gpd.GeoDataFrame, buffer_gdf: gpd.GeoDataFrame) -> dict[int, gpd.GeoDataFrame]:
    slices: dict[int, gpd.GeoDataFrame] = {}
    summaries = []

    for year, jobs_year in YEARS:
        h3 = load_or_build_h3(year, jobs_year)
        sl = corridor_slice(h3, buffer_gdf)
        sl["year"] = year
        sl["jobs_year"] = jobs_year
        sl["income_per_sqmi"] = sl["agg_income"] / sl["hex_area_sqmi"].replace(0, pd.NA)
        slices[year] = sl
        summaries.append(
            {
                "acs_year": year,
                "jobs_year": jobs_year,
                "h3_resolution": RESOLUTION,
                "buffer_miles": BUFFER_MILES,
                "hexes": len(sl),
                "population": sl["population"].sum(),
                "jobs": sl["jobs"].sum(),
                "workers": sl["workers"].sum(),
                "occupied_units": sl["occupied_units"].sum(),
                "agg_income": sl["agg_income"].sum(),
                "median_income_weighted": sl["income_mass"].sum() / sl["occupied_units"].sum(),
                "jobs_per_resident_worker": sl["jobs"].sum() / sl["workers"].sum(),
            }
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "us380_corridor_h3_timeseries.csv", index=False)

    base = slices[BASE_YEAR][["h3", "population", "jobs", "agg_income"]].rename(
        columns={"population": "population_base", "jobs": "jobs_base", "agg_income": "agg_income_base"}
    )
    now = slices[NOW_YEAR][["h3", "population", "jobs", "agg_income", "geometry"]].rename(
        columns={"population": "population_now", "jobs": "jobs_now", "agg_income": "agg_income_now"}
    )
    change = now.merge(base, on="h3", how="left")
    change["population_change"] = change["population_now"] - change["population_base"].fillna(0)
    change["jobs_change"] = change["jobs_now"] - change["jobs_base"].fillna(0)
    change["agg_income_change"] = change["agg_income_now"] - change["agg_income_base"].fillna(0)
    change["agg_income_change_millions"] = change["agg_income_change"] / 1_000_000
    gpd.GeoDataFrame(change, geometry="geometry", crs=PROJECT_CRS).to_parquet(
        OUT / f"us380_corridor_h3_change_{BASE_YEAR}_{NOW_YEAR}.parquet", index=False
    )
    make_maps(slices, change, route, buffer_gdf)
    return slices


def plot_hex_map(ax, gdf, column, title, label, cmap="viridis", vmin=None, vmax=None) -> None:
    gdf.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        linewidth=0,
        legend=False,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    ax.set_axis_off()


def _shared_limits(frames: list[gpd.GeoDataFrame], column: str) -> tuple[float, float]:
    values = pd.concat([f[column] for f in frames]).dropna()
    return float(values.quantile(0.02)), float(values.quantile(0.98))


def make_maps(slices: dict[int, gpd.GeoDataFrame], change: pd.DataFrame, route, buffer_gdf) -> None:
    bounds = buffer_gdf.total_bounds
    xpad = (bounds[2] - bounds[0]) * 0.04
    ypad = (bounds[3] - bounds[1]) * 0.04

    fig, axes = plt.subplots(3, len(TIMEPOINTS), figsize=(22, 12), constrained_layout=False)
    year_frames = [slices[y] for y in TIMEPOINTS]
    pop_min, pop_max = _shared_limits(year_frames, "pop_density")
    jobs_min, jobs_max = _shared_limits(year_frames, "jobs_density")
    for ax, year in zip(axes[0], TIMEPOINTS):
        plot_hex_map(ax, slices[year], "pop_density", f"Population density, {year}", "people / sq mi", vmin=pop_min, vmax=pop_max)
    for ax, year in zip(axes[1], TIMEPOINTS):
        plot_hex_map(ax, slices[year], "jobs_density", f"Workplace jobs density, LODES {slices[year]['jobs_year'].iloc[0]}", "jobs / sq mi", "magma", jobs_min, jobs_max)

    change_gdf = gpd.GeoDataFrame(change, geometry="geometry", crs=PROJECT_CRS)
    plot_hex_map(axes[2, 0], change_gdf, "population_change", f"Population change, {BASE_YEAR}→{NOW_YEAR}", "people", "YlGnBu")
    plot_hex_map(axes[2, 1], change_gdf, "jobs_change", f"Jobs change, {BASE_YEAR}→{NOW_YEAR}", "jobs", "PuRd")
    plot_hex_map(axes[2, 2], change_gdf, "agg_income_change_millions", f"Aggregate HH income change, {BASE_YEAR}→{NOW_YEAR}", "$ millions", "Greens")
    axes[2, 3].axis("off")
    axes[2, 3].text(
        0.02,
        0.88,
        "Lookback frame\n\n"
        "Baseline: 2010 / 2013 public layers\n"
        "Current: 2020 / 2023 public layers\n\n"
        "Geometry includes TIGER US 380 /\n"
        "University Drive plus a coarse\n"
        "TxDOT McKinney Blue Alternative\n"
        "control line from FEIS/ROD endpoints.",
        transform=axes[2, 3].transAxes,
        va="top",
        fontsize=12,
    )

    for ax in axes.ravel()[:11]:
        buffer_gdf.boundary.plot(ax=ax, color="#333333", linewidth=0.8, alpha=0.7)
        route.plot(ax=ax, color="#111111", linewidth=1.1)
        ax.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
        ax.set_ylim(bounds[1] - ypad, bounds[3] + ypad)

    fig.subplots_adjust(left=0.02, right=0.99, top=0.91, bottom=0.06, wspace=0.03, hspace=0.12)
    fig.suptitle("US 380 corridor proxy: public H3 lookback before clean filings", fontsize=17, fontweight="bold")
    fig.text(
        0.01,
        0.01,
        "Public-safe exhibit. Geometry: TIGER US 380 / University Drive plus coarse TxDOT McKinney Blue Alternative proxy, 3-mile buffer. "
        "Data: geostack H3 res-8 ACS/Decennial population/income + LODES WAC jobs. Corridor proxy, not parcel advice.",
        fontsize=8,
    )
    fig.savefig(OUT / "us380_h3_evolution_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    route, buffer_gdf = load_us380_route()
    route.to_crs("EPSG:4326").to_file(OUT / "us380_public_route_txdot_proxy.geojson", driver="GeoJSON")
    buffer_gdf.to_crs("EPSG:4326").to_file(OUT / "us380_corridor_3mi_proxy.geojson", driver="GeoJSON")
    build_panel(route, buffer_gdf)
    print(f"Wrote US 380 H3 exhibit outputs to {OUT}")


if __name__ == "__main__":
    main()
