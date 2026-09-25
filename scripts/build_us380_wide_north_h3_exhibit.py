"""Build the north-of-380 growth-frontier H3 corridor exhibit for corridor study.

US 380 itself reads mostly as retail/commercial. The household growth in
Collin County concentrates on the north-south arteries that cross it — Dallas
North Tollway, Preston Road, and US 75 / Central Expressway — but only the
segments north of US 380 (Prosper / Celina / north McKinney) are the actual
growth frontier. South of 380 those same roads run through already-mature
Frisco/Plano, which dilutes the percentage growth story even though it adds
to the absolute totals.

This script reuses build_us380_wide_corridor_h3_exhibit.py's route loaders,
clips each north-south road to the portion north of its own crossing point
with the US 380 corridor, then buffers the combined 380 + clipped-N-S lines
by the same 3 miles used elsewhere in the study. It reuses the cached
Collin+Denton H3 layers — only the buffer geometry changes.

Public-safe: TIGER road centerlines + the same coarse TxDOT McKinney Blue
Alternative control line as the other corridor exhibits. No parcel, owner, or
private score data.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import nearest_points, unary_union

from build_us380_h3_exhibit import (
    BASE_YEAR,
    NOW_YEAR,
    PROJECT_CRS,
    load_us380_route,
    build_panel,
)
from build_us380_wide_corridor_h3_exhibit import load_ns_routes

BUFFER_MILES = 3.0
CLIP_MARGIN_FT = 50_000  # ~9.5 miles, generous so the clip box never clips x-extent
OUT = Path("outputs/msa_gravity/us380_h3")

# Group TIGER FULLNAME variants into the three named north-south corridors.
NS_GROUPS = {
    "Dallas North Tollway": {"Dallas North Tollway"},
    "Preston Rd": {"Preston Rd", "N Preston Rd", "S Preston Rd"},
    "US 75 / Central Expy": {"US Hwy 75", "N Central Expy", "S Central Expy", "Central Expy"},
}


def clip_north_of_380(ns_route: gpd.GeoDataFrame, line_380: object) -> gpd.GeoDataFrame:
    """Keep only the portion of each named N-S road north of its own crossing with US 380."""
    bounds = ns_route.total_bounds  # xmin, ymin, xmax, ymax
    xmin, xmax = bounds[0] - CLIP_MARGIN_FT, bounds[2] + CLIP_MARGIN_FT
    ymax = bounds[3] + CLIP_MARGIN_FT

    clipped_rows = []
    for label, names in NS_GROUPS.items():
        sub = ns_route[ns_route["FULLNAME"].isin(names)]
        if sub.empty:
            print(f"  warning: no segments found for {label!r}")
            continue
        group_geom = unary_union(sub.geometry)
        crossing = group_geom.intersection(line_380)
        NEAR_TOLERANCE_FT = 5_280  # 1 mile — covers TIGER segment gaps at interchanges

        if crossing.is_empty:
            # TIGER road segments often have small named-segment gaps right at an
            # interchange (ramps/frontage roads carry a different FULLNAME/MTFCC).
            # Fall back to the nearest point between the two lines if it's close
            # enough to plausibly be the same interchange.
            p_ns, p_380 = nearest_points(group_geom, line_380)
            gap = p_ns.distance(p_380)
            if gap > NEAR_TOLERANCE_FT:
                print(f"  warning: {label!r} does not cross the US 380 route (nearest gap {gap:.0f} ft) — keeping unclipped")
                clipped_rows.append(sub.assign(ns_group=label))
                continue
            y_threshold = max(p_ns.y, p_380.y)
            print(f"  {label}: no exact intersection, nearest-point gap {gap:.0f} ft — using y={y_threshold:.0f} ft as crossing")
        else:
            # Southernmost crossing point sets the north-of-380 threshold for this road.
            pts = getattr(crossing, "geoms", [crossing])
            y_threshold = min(p.centroid.y if hasattr(p, "centroid") else p.y for p in pts)
        clip_box = box(xmin, y_threshold, xmax, ymax)
        clipped_geom = group_geom.intersection(clip_box)
        if clipped_geom.is_empty:
            print(f"  warning: {label!r} clipped to nothing north of 380 — check geometry")
            continue
        clipped_rows.append(
            gpd.GeoDataFrame(
                {"FULLNAME": [label], "ns_group": [label]},
                geometry=[clipped_geom],
                crs=ns_route.crs,
            )
        )
        print(f"  {label}: crossed 380 at y={y_threshold:.0f} ft (proj), kept segment north of it")

    return gpd.GeoDataFrame(pd.concat(clipped_rows, ignore_index=True), crs=ns_route.crs)


def load_north_corridor() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    us380_route, _ = load_us380_route()  # already in PROJECT_CRS
    line_380 = unary_union(us380_route.geometry)

    ns_route = load_ns_routes().to_crs(PROJECT_CRS)
    ns_clipped = clip_north_of_380(ns_route, line_380)

    route = pd.concat(
        [us380_route[["geometry"]], ns_clipped[["geometry"]]], ignore_index=True
    )
    route = gpd.GeoDataFrame(route, geometry="geometry", crs=PROJECT_CRS)
    route_geom = unary_union(route.geometry)
    buffer_geom = route_geom.buffer(BUFFER_MILES * 5280)
    buffer_gdf = gpd.GeoDataFrame(
        {"name": [f"US 380 + north-of-380 DNT/Preston/US-75 growth frontier, {BUFFER_MILES:g}-mile proxy"]},
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
    route, buffer_gdf = load_north_corridor()
    route.to_crs("EPSG:4326").to_file(OUT / "us380_wide_north_route_proxy.geojson", driver="GeoJSON")
    buffer_gdf.to_crs("EPSG:4326").to_file(OUT / "us380_wide_north_corridor_3mi_proxy.geojson", driver="GeoJSON")

    backups = {}
    for name in NARROW_OUTPUTS:
        path = OUT / name
        if path.exists():
            backup = path.with_suffix(path.suffix + ".narrow-bak")
            path.rename(backup)
            backups[name] = backup

    try:
        build_panel(route, buffer_gdf)
        (OUT / "us380_corridor_h3_timeseries.csv").rename(OUT / "us380_wide_north_corridor_h3_timeseries.csv")
        (OUT / f"us380_corridor_h3_change_{BASE_YEAR}_{NOW_YEAR}.parquet").rename(
            OUT / f"us380_wide_north_corridor_h3_change_{BASE_YEAR}_{NOW_YEAR}.parquet"
        )
        (OUT / "us380_h3_evolution_panel.png").rename(OUT / "us380_wide_north_h3_evolution_panel.png")
    finally:
        for name, backup in backups.items():
            backup.rename(OUT / name)

    print(f"Wrote north-of-380 wide corridor H3 exhibit outputs to {OUT}")


if __name__ == "__main__":
    main()
