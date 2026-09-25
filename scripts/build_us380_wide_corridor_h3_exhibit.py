"""Build a wider public-safe H3 corridor exhibit for corridor study.

US 380 itself is mostly a retail/commercial corridor. The household growth in
Collin County concentrates on the north-south arteries that cross it — Dallas
North Tollway, Preston Road, and US 75 / Central Expressway — which is where
Prosper/Celina/north McKinney rooftops actually sit. This script reuses the
cached Collin+Denton H3 layers from build_us380_h3_exhibit.py and re-slices
them against a wider corridor buffer that adds those three north-south roads
to the existing US 380 / University Drive route.

Public-safe: TIGER road centerlines + the same coarse TxDOT McKinney Blue
Alternative control line as the narrow exhibit. No parcel, owner, or private
score data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

from geostack.tiger import download_roads

from build_us380_h3_exhibit import (
    BASE_YEAR,
    NOW_YEAR,
    PROJECT_CRS,
    RESOLUTION,
    STATE,
    COUNTIES,
    load_us380_route,
    load_or_build_h3,
    build_panel,
)

BUFFER_MILES = 3.0
OUT = Path("outputs/msa_gravity/us380_h3")
CACHE = Path("data/tiger_cache")

# North-south household-growth arteries that cross US 380 in Collin County.
NS_ROUTE_NAMES = {
    "Dallas North Tollway",
    "Preston Rd",
    "N Preston Rd",
    "S Preston Rd",
    "US Hwy 75",
    "N Central Expy",
    "S Central Expy",
    "Central Expy",
}


def load_ns_routes() -> gpd.GeoDataFrame:
    roads = []
    for county in COUNTIES:
        gdf = download_roads(STATE, county, year=2024, cache_dir=CACHE)
        mask = gdf["FULLNAME"].fillna("").isin(NS_ROUTE_NAMES)
        roads.append(gdf.loc[mask].copy())
    ns = pd.concat(roads, ignore_index=True)
    if ns.empty:
        raise RuntimeError("No TIGER north-south route segments matched")
    return gpd.GeoDataFrame(ns, geometry="geometry", crs="EPSG:4269")


def load_wide_corridor() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    us380_route, _ = load_us380_route()
    ns_route = load_ns_routes().to_crs(us380_route.crs)

    route = pd.concat([us380_route, ns_route], ignore_index=True)
    route = gpd.GeoDataFrame(route, geometry="geometry", crs=us380_route.crs).to_crs(PROJECT_CRS)
    route_geom = unary_union(route.geometry)
    buffer_geom = route_geom.buffer(BUFFER_MILES * 5280)
    buffer_gdf = gpd.GeoDataFrame(
        {"name": [f"US 380 + Dallas North Tollway/Preston Rd/US 75 {BUFFER_MILES:g}-mile corridor proxy"]},
        geometry=[buffer_geom],
        crs=PROJECT_CRS,
    )
    return route, buffer_gdf


NARROW_OUTPUTS = [
    "us380_corridor_h3_timeseries.csv",
    f"us380_corridor_h3_change_{BASE_YEAR}_{NOW_YEAR}.parquet",
    "us380_h3_evolution_panel.png",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    route, buffer_gdf = load_wide_corridor()
    route.to_crs("EPSG:4326").to_file(OUT / "us380_wide_route_proxy.geojson", driver="GeoJSON")
    buffer_gdf.to_crs("EPSG:4326").to_file(OUT / "us380_wide_corridor_3mi_proxy.geojson", driver="GeoJSON")

    # build_panel/make_maps write to the same fixed filenames as the narrow
    # exhibit. Back those up first so this run doesn't clobber the corridor study v0.1-0.4
    # provenance, then rename the freshly-built wide files afterward and
    # restore the narrow originals.
    backups = {}
    for name in NARROW_OUTPUTS:
        path = OUT / name
        if path.exists():
            backup = path.with_suffix(path.suffix + ".narrow-bak")
            path.rename(backup)
            backups[name] = backup

    try:
        build_panel(route, buffer_gdf)

        (OUT / "us380_corridor_h3_timeseries.csv").rename(OUT / "us380_wide_corridor_h3_timeseries.csv")
        (OUT / f"us380_corridor_h3_change_{BASE_YEAR}_{NOW_YEAR}.parquet").rename(
            OUT / f"us380_wide_corridor_h3_change_{BASE_YEAR}_{NOW_YEAR}.parquet"
        )
        (OUT / "us380_h3_evolution_panel.png").rename(OUT / "us380_wide_h3_evolution_panel.png")
    finally:
        for name, backup in backups.items():
            backup.rename(OUT / name)

    print(f"Wrote wide US 380 corridor H3 exhibit outputs to {OUT}")


if __name__ == "__main__":
    main()
