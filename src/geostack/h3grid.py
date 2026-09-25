"""H3 hexagonal grid layers — a reusable spatial-fusion primitive.

Uber's H3 is a hexagonal hierarchical spatial index. Aggregating heterogeneous
sources (census tracts, points, parcels, mobility) into a common H3 grid gives:

- **Equal-area, uniform units** — fixes the unequal-tract problem (a 40-sq-mi
  rural tract and a 0.2-sq-mi downtown tract are no longer both "one unit").
- **A common join key** — tracts, ZIPs, and block groups don't nest cleanly;
  a fixed hex grid does, so you can fuse demographics + jobs + POIs + parcels.
- **Stable-over-time, cross-metro comparable** geography (no decennial redraws).

This module converts polygon (tract) and point data into H3 cells via areal
interpolation, and renders interactive (Folium) and static (matplotlib)
choropleths. Other geostack consumers import from here.

Resolution guide (edge length): res 7 ≈ 1.2 km (metro overview),
res 8 ≈ 0.53 km (neighborhood / trade area — default), res 9 ≈ 0.20 km
(block-group scale).

Install:  pip install 'geostack[h3]'

Usage:
    from geostack.h3grid import interpolate_polygons_to_h3, h3_choropleth_static

    hexes = interpolate_polygons_to_h3(
        tracts_gdf, resolution=8,
        count_cols=["population", "jobs"],   # allocated by area share
        mean_cols=["median_income"],         # area-weighted average
    )
    h3_choropleth_static(hexes, "population", output_path="dfw_pop_h3.png")

    # Convenience: build a full DFW hex layer straight from gravity data
    from geostack.h3grid import msa_h3_layer
    hexes = msa_h3_layer("Dallas", resolution=8)  # pop/jobs/workers/income + derived
"""

from __future__ import annotations

import logging
import math
import pathlib
from typing import TYPE_CHECKING, Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from rich.console import Console
from shapely.geometry import Polygon

if TYPE_CHECKING:
    import folium
    import matplotlib.figure

console = Console()
logger = logging.getLogger(__name__)


def _require_h3():
    try:
        import h3
        return h3
    except ImportError:
        raise ImportError("h3 is required. Run: pip install 'geostack[h3]'  (or pip install h3)")


# ---------------------------------------------------------------------------
# Resolution metadata
# ---------------------------------------------------------------------------

def h3_edge_km(resolution: int) -> float:
    """Average hexagon edge length in km for a resolution."""
    return _require_h3().average_hexagon_edge_length(resolution, unit="km")


def h3_area_sqmi(resolution: int) -> float:
    """Average hexagon area in square miles for a resolution."""
    h3 = _require_h3()
    return h3.average_hexagon_area(resolution, unit="km^2") / 2.589988


# ---------------------------------------------------------------------------
# Cell ↔ geometry
# ---------------------------------------------------------------------------

def cell_to_polygon(cell: str) -> Polygon:
    """Convert an H3 cell id to a shapely polygon (EPSG:4326, lon/lat)."""
    h3 = _require_h3()
    boundary = h3.cell_to_boundary(cell)  # tuple of (lat, lng)
    return Polygon([(lng, lat) for lat, lng in boundary])


def cells_covering(geom, resolution: int) -> set[str]:
    """Return the set of H3 cells whose interior intersects a polygon geometry.

    Buffers slightly so edge-overlapping cells (whose center is outside) are
    included — important for honest areal interpolation at the boundary.

    Args:
        geom: shapely Polygon/MultiPolygon in EPSG:4326 (lon/lat).
        resolution: H3 resolution.

    Returns:
        Set of H3 cell id strings.
    """
    h3 = _require_h3()
    # Buffer by ~one edge length so partial-coverage cells are caught
    buf_deg = h3.average_hexagon_edge_length(resolution, unit="km") / 111.0
    g = geom.buffer(buf_deg)

    polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
    cells: set[str] = set()
    for poly in polys:
        ring = [(lat, lng) for lng, lat in poly.exterior.coords]
        holes = [[(lat, lng) for lng, lat in interior.coords] for interior in poly.interiors]
        lpoly = h3.LatLngPoly(ring, *holes)
        cells.update(h3.polygon_to_cells(lpoly, resolution))
    return cells


def cells_to_gdf(cells: Iterable[str], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of hex polygons from H3 cell ids."""
    cells = list(cells)
    geoms = [cell_to_polygon(c) for c in cells]
    return gpd.GeoDataFrame({"h3": cells}, geometry=geoms, crs="EPSG:4326").to_crs(crs)


# ---------------------------------------------------------------------------
# Areal interpolation: polygons → H3
# ---------------------------------------------------------------------------

def interpolate_polygons_to_h3(
    gdf: gpd.GeoDataFrame,
    resolution: int = 8,
    count_cols: Sequence[str] = (),
    mean_cols: Sequence[str] = (),
    area_crs: str | None = None,
    min_coverage: float = 0.0,
) -> gpd.GeoDataFrame:
    """Areal-interpolate polygon (e.g. tract) values onto an H3 grid.

    Count variables (population, jobs) are allocated to each hex in proportion
    to the overlap area with each source polygon (area-weighted dasymetric).
    Mean/intensive variables (median income, rates) are area-weighted-averaged.

    NOTE: area weighting assumes values are uniformly distributed within each
    source polygon. For finer allocation, weight by building footprints or land
    use upstream. This is the honest standard baseline.

    Args:
        gdf: Source polygons with the value columns. Any CRS (reprojected).
        resolution: H3 resolution (default 8).
        count_cols: Columns allocated proportionally (extensive quantities).
        mean_cols: Columns area-weighted-averaged (intensive quantities).
        area_crs: Projected CRS for area math. Auto-suggested if None.
        min_coverage: Drop hexes covered less than this fraction by source
            polygons (0 keeps all touched hexes).

    Returns:
        GeoDataFrame of hex cells (EPSG:4326) with h3, the allocated columns,
        and a 'covered_frac' column.
    """
    from geostack.spatial import suggest_crs

    src = gdf.to_crs("EPSG:4326").copy()
    union = src.geometry.union_all()
    cells = cells_covering(union, resolution)
    hexes = cells_to_gdf(cells)

    acrs = area_crs or suggest_crs(src)
    src_p = src.to_crs(acrs)
    hex_p = hexes.to_crs(acrs)
    src_p["_src_area"] = src_p.geometry.area
    hex_p["_hex_area"] = hex_p.geometry.area

    # Intersection pieces carry both source value and hex id
    pieces = gpd.overlay(
        src_p[["_src_area", *count_cols, *mean_cols, "geometry"]],
        hex_p[["h3", "_hex_area", "geometry"]],
        how="intersection", keep_geom_type=True,
    )
    if pieces.empty:
        return hexes.iloc[0:0]

    pieces["_piece_area"] = pieces.geometry.area
    pieces["_frac_of_src"] = pieces["_piece_area"] / pieces["_src_area"].replace(0, np.nan)

    out = {}
    # Counts: value * (piece / source area), summed per hex
    for col in count_cols:
        pieces[f"_alloc_{col}"] = pieces[col].astype(float) * pieces["_frac_of_src"]
        out[col] = pieces.groupby("h3")[f"_alloc_{col}"].sum()
    # Means: area-weighted average per hex (weight only where value is present)
    for col in mean_cols:
        valid = pieces[pieces[col].notna()].copy()
        valid["_wv"] = valid[col].astype(float) * valid["_piece_area"]
        num = valid.groupby("h3")["_wv"].sum()
        den = valid.groupby("h3")["_piece_area"].sum()
        out[col] = num / den.replace(0, np.nan)

    coverage = pieces.groupby("h3")["_piece_area"].sum() / hex_p.set_index("h3")["_hex_area"]

    agg = pd.DataFrame(out)
    agg["covered_frac"] = coverage
    agg = agg.reset_index()

    result = hexes.merge(agg, on="h3", how="inner")
    if min_coverage > 0:
        result = result[result["covered_frac"] >= min_coverage]

    # Density helpers
    result["hex_area_sqmi"] = result.to_crs(acrs).geometry.area / (5280 ** 2)
    result.attrs["resolution"] = resolution
    return result


# ---------------------------------------------------------------------------
# Point binning → H3
# ---------------------------------------------------------------------------

def points_to_h3(
    gdf: gpd.GeoDataFrame,
    resolution: int = 9,
    value_cols: Sequence[str] = (),
    agg: str = "sum",
) -> gpd.GeoDataFrame:
    """Bin point data into H3 cells (counts, plus optional value aggregates).

    Args:
        gdf: Point GeoDataFrame (any CRS; reprojected to 4326).
        resolution: H3 resolution (points are finer — default 9).
        value_cols: Columns to aggregate per cell.
        agg: Aggregation for value_cols ("sum", "mean", "median", "max").

    Returns:
        Hex GeoDataFrame with 'h3', 'point_count', aggregated value cols, geometry.
    """
    h3 = _require_h3()
    pts = gdf.to_crs("EPSG:4326").copy()
    pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty]
    pts["h3"] = [h3.latlng_to_cell(geom.y, geom.x, resolution) for geom in pts.geometry]

    grp = pts.groupby("h3")
    agg_df = grp.size().rename("point_count").to_frame()
    for col in value_cols:
        agg_df[col] = grp[col].agg(agg)
    agg_df = agg_df.reset_index()

    hexes = cells_to_gdf(agg_df["h3"])
    result = hexes.merge(agg_df, on="h3", how="inner")
    result.attrs["resolution"] = resolution
    return result


# ---------------------------------------------------------------------------
# Dasymetric (footprint-weighted) interpolation
# ---------------------------------------------------------------------------

def dasymetric_polygons_to_h3(
    gdf: gpd.GeoDataFrame,
    weight_points: gpd.GeoDataFrame,
    resolution: int = 8,
    count_cols: Sequence[str] = (),
    weight_col: str | None = None,
    id_col: str = "GEOID",
    fallback_area: bool = True,
) -> gpd.GeoDataFrame:
    """Distribute polygon counts to H3 cells weighted by a point layer.

    Area-weighting (interpolate_polygons_to_h3) assumes values spread uniformly
    within each source polygon. Dasymetric weighting instead allocates each
    source polygon's count to hexes in proportion to where a *weight layer*
    actually sits — e.g. parcels, building footprints, or address points — so
    population/jobs land where development is, not across empty acreage.

    For each source polygon, a hex receives:
        count * (weight in that hex ∩ polygon) / (total weight in the polygon)

    Args:
        gdf: Source polygons (e.g. tracts) with count columns and id_col.
        weight_points: Point layer used as the allocation surface (parcels,
            footprint centroids, address points).
        resolution: H3 resolution (default 8).
        count_cols: Columns to allocate (extensive quantities).
        weight_col: Per-point weight (e.g. parcel value, footprint sqft). If
            None, each point counts as 1 (pure density of points).
        id_col: Source polygon id column.
        fallback_area: For source polygons containing no weight points, fall
            back to area-weighted allocation (so their counts aren't dropped).

    Returns:
        Hex GeoDataFrame (EPSG:4326) with h3, allocated count columns,
        'weight_total' (allocation weight per hex), and geometry.
    """
    h3 = _require_h3()
    from geostack.spatial import suggest_crs

    src = gdf.to_crs("EPSG:4326").copy()
    pts = weight_points.to_crs("EPSG:4326").copy()
    pts = pts[pts.geometry.notna() & ~pts.geometry.is_empty].copy()
    pts["_w"] = pts[weight_col].astype(float) if weight_col else 1.0

    # Each point → its H3 cell and its source polygon
    pts["h3"] = [h3.latlng_to_cell(g.y, g.x, resolution) for g in pts.geometry]
    joined = gpd.sjoin(
        pts[["_w", "h3", "geometry"]], src[[id_col, *count_cols, "geometry"]],
        how="inner", predicate="within",
    )

    # Weight totals per (polygon, hex) and per polygon
    pw = joined.groupby([id_col, "h3"])["_w"].sum().reset_index(name="_w_ph")
    tot = joined.groupby(id_col)["_w"].sum().rename("_w_tot")
    pw = pw.merge(tot, on=id_col)
    pw["_frac"] = pw["_w_ph"] / pw["_w_tot"].replace(0, np.nan)

    src_vals = src[[id_col, *count_cols]].drop_duplicates(id_col)
    pw = pw.merge(src_vals, on=id_col)
    for c in count_cols:
        pw[c] = pw[c].astype(float) * pw["_frac"]

    alloc = pw.groupby("h3").agg({**{c: "sum" for c in count_cols}, "_w_ph": "sum"})
    alloc = alloc.rename(columns={"_w_ph": "weight_total"}).reset_index()

    # Fallback: polygons with no weight points keep their counts via area-weighting
    covered_ids = set(joined[id_col].unique())
    missing = src[~src[id_col].isin(covered_ids)]
    if fallback_area and not missing.empty and count_cols:
        area_hexes = interpolate_polygons_to_h3(
            missing, resolution=resolution, count_cols=list(count_cols),
        )
        merged = pd.concat([alloc, area_hexes[["h3", *count_cols]]], ignore_index=True)
        alloc = merged.groupby("h3", as_index=False).sum(numeric_only=True)

    hexes = cells_to_gdf(alloc["h3"])
    result = hexes.merge(alloc, on="h3", how="inner")
    acrs = suggest_crs(result)
    result["hex_area_sqmi"] = result.to_crs(acrs).geometry.area / (5280 ** 2)
    result.attrs["resolution"] = resolution
    result.attrs["method"] = "dasymetric"
    return result


# ---------------------------------------------------------------------------
# Convenience: MSA hex layer from gravity data
# ---------------------------------------------------------------------------

def msa_h3_layer(
    msa_name: str,
    resolution: int = 8,
    year: int = 2024,
    jobs_year: int = 2023,
    use_crosswalk: bool = True,
) -> gpd.GeoDataFrame:
    """Build a multi-metric H3 layer for an MSA from gravity/census data.

    Allocates tract population, jobs, resident workers, and income onto a hex
    grid, then derives density and balance metrics that are actionable for CRE
    site selection.

    Args:
        msa_name: MSA name or partial match.
        resolution: H3 resolution (default 8).
        year: ACS year for population / income (default 2024).
        jobs_year: LODES year for jobs / workers (default 2023).
        use_crosswalk: Normalize pre-2020 GEOIDs.

    Returns:
        Hex GeoDataFrame with: population, jobs, workers, occupied_units,
        income_mass (HH × MHI proxy), agg_income (true ACS B19025 aggregate
        household income), plus derived pop_density, jobs_density,
        median_income, mean_income, jobs_per_capita, jobs_housing_balance
        (jobs per resident worker).
    """
    from geostack.gravity import resolve_msa

    msa = resolve_msa(msa_name)
    return _build_h3_layer(
        msa["state_fips"], list(msa["counties"]), msa["name"],
        resolution=resolution, year=year, jobs_year=jobs_year, use_crosswalk=use_crosswalk,
    )


def county_h3_layer(
    state_fips: str,
    county_fips: str | list[str],
    name: str | None = None,
    resolution: int = 9,
    year: int = 2024,
    jobs_year: int = 2023,
    use_crosswalk: bool = True,
) -> gpd.GeoDataFrame:
    """Build a multi-metric H3 layer for a single county (or list of counties).

    Same metrics as msa_h3_layer, for non-MSA geographies (e.g. rural Cooke
    County, TX). Default resolution is finer (9) since a small county needs
    finer hexes to resolve its town(s).

    Args:
        state_fips: 2-digit state FIPS (e.g. "48").
        county_fips: 3-digit county FIPS string, or list of them.
        name: Display name (stored in attrs). Defaults to the FIPS.
        resolution: H3 resolution (default 9 for fine rural detail).
        year / jobs_year: ACS / LODES years.
        use_crosswalk: Normalize pre-2020 GEOIDs.

    Returns:
        Hex GeoDataFrame (same columns as msa_h3_layer).
    """
    cfs = [county_fips] if isinstance(county_fips, str) else list(county_fips)
    return _build_h3_layer(
        state_fips, cfs, name or f"County {state_fips}{cfs[0]}",
        resolution=resolution, year=year, jobs_year=jobs_year, use_crosswalk=use_crosswalk,
    )


def _build_h3_layer(
    sf: str,
    counties: list[str],
    name: str,
    resolution: int = 8,
    year: int = 2024,
    jobs_year: int = 2023,
    use_crosswalk: bool = True,
) -> gpd.GeoDataFrame:
    """Shared builder behind msa_h3_layer and county_h3_layer."""
    from geostack.gravity import (
        fetch_tract_geometries, fetch_population,
        _fetch_lodes_tract_jobs, _fetch_lodes_tract_workers,
        _fetch_acs_income_mass, _fetch_acs_aggregate_income,
        _apply_population_crosswalk,
        _apply_weight_crosswalk, _apply_income_mass_crosswalk, _needs_crosswalk,
    )

    console.print(f"[bold]Building H3 res-{resolution} layer:[/bold] {name}")

    tracts, pops, jobs, workers, incomes, agg_incomes = [], [], [], [], [], []
    for cf in counties:
        try:
            tracts.append(fetch_tract_geometries(sf, cf, year))
            p = fetch_population(sf, cf, year)
            j = _fetch_lodes_tract_jobs(sf, cf, jobs_year)
            w = _fetch_lodes_tract_workers(sf, cf, jobs_year)
            inc = _fetch_acs_income_mass(sf, cf, year)
            agg = _fetch_acs_aggregate_income(sf, cf, year)
            if use_crosswalk and _needs_crosswalk(year):
                p = _apply_population_crosswalk(p, sf, year=year)
                inc = _apply_income_mass_crosswalk(inc, sf, year=year)
                agg = _apply_weight_crosswalk(agg, sf, weight_col="agg_income", year=year)
            # NO crosswalk for LODES jobs/workers: LODES8 is 2020-vintage for
            # all years, so crosswalking mis-splits border tracts.
            pops.append(p); jobs.append(j); workers.append(w)
            incomes.append(inc); agg_incomes.append(agg)
        except Exception as e:
            logger.warning("Skipping county %s: %s", cf, e)

    tract_gdf = pd.concat(tracts, ignore_index=True)
    pop_df = pd.concat(pops, ignore_index=True)
    job_df = pd.concat(jobs, ignore_index=True).rename(columns={"weight": "jobs"})
    wrk_df = pd.concat(workers, ignore_index=True).rename(columns={"weight": "workers"})
    inc_df = pd.concat(incomes, ignore_index=True)[["GEOID", "income_mass", "occupied_units"]]
    agg_df = pd.concat(agg_incomes, ignore_index=True)[["GEOID", "agg_income"]]

    merged = (
        tract_gdf.merge(pop_df, on="GEOID", how="left")
        .merge(job_df, on="GEOID", how="left")
        .merge(wrk_df, on="GEOID", how="left")
        .merge(inc_df, on="GEOID", how="left")
        .merge(agg_df, on="GEOID", how="left")
    )
    for c in ["population", "jobs", "workers", "income_mass", "occupied_units", "agg_income"]:
        merged[c] = merged[c].fillna(0)

    hexes = interpolate_polygons_to_h3(
        merged, resolution=resolution,
        count_cols=["population", "jobs", "workers", "income_mass", "occupied_units", "agg_income"],
    )

    # Derived, actionable metrics
    a = hexes["hex_area_sqmi"].replace(0, np.nan)
    hexes["pop_density"] = hexes["population"] / a
    hexes["jobs_density"] = hexes["jobs"] / a
    hexes["median_income"] = (hexes["income_mass"] / hexes["occupied_units"].replace(0, np.nan))
    # True aggregate household income (B19025): mean income per household
    hexes["mean_income"] = (hexes["agg_income"] / hexes["occupied_units"].replace(0, np.nan))
    hexes["jobs_per_capita"] = hexes["jobs"] / hexes["population"].replace(0, np.nan)
    hexes["jobs_housing_balance"] = hexes["jobs"] / hexes["workers"].replace(0, np.nan)
    hexes.attrs["msa_name"] = name
    hexes.attrs["year"] = year
    hexes.attrs["jobs_year"] = jobs_year

    console.print(f"  [green]{len(hexes)} hexes[/green] "
                  f"(pop {hexes['population'].sum():,.0f}, jobs {hexes['jobs'].sum():,.0f})")
    return hexes


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_HEX_CMAPS = {
    "population": "viridis", "pop_density": "viridis",
    "jobs": "inferno", "jobs_density": "inferno",
    "median_income": "YlGnBu", "income_mass": "YlGnBu",
    "agg_income": "RdPu", "mean_income": "YlGnBu",
    "jobs_housing_balance": "RdBu_r", "jobs_per_capita": "RdBu_r",
}


def h3_choropleth_static(
    hex_gdf: gpd.GeoDataFrame,
    column: str,
    output_path: str | pathlib.Path | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    cmap: str | None = None,
    scheme: str = "quantiles",
    k: int = 7,
    bins: Sequence[float] | None = None,
    dpi: int = 220,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    legend_fmt: str | None = None,
    basemap: bool = True,
    basemap_provider: str = "OpenStreetMap.Mapnik",
    alpha: float = 0.55,
    edgecolor: str = "#ffffff",
    edge_width: float = 0.15,
    zoom: int = 11,
) -> "matplotlib.figure.Figure":
    """Static H3 choropleth on a light basemap (publication quality).

    Args:
        hex_gdf: Hex GeoDataFrame (from interpolate_polygons_to_h3 / msa_h3_layer).
        column: Column to map.
        output_path: Save PNG/PDF/SVG if provided.
        title / subtitle: Headline text. Auto if None.
        cmap: Matplotlib colormap. Auto by column if None.
        scheme: mapclassify scheme ("quantiles", "naturalbreaks", "equalinterval").
        k: Number of classes.
        bins: Manual class upper edges (e.g. [25, 50, 100, 200]). Overrides
            scheme/k with mapclassify "userdefined" — use to gradient out a
            long-tailed high end that quantiles would collapse into one class.
        dpi: Resolution.
        vmin / vmax: Manual color bounds (overrides scheme).
        basemap: Add contextily basemap.

    Returns:
        Matplotlib Figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. pip install 'geostack[print-maps]'")

    data = hex_gdf[hex_gdf[column].notna() & np.isfinite(hex_gdf[column])].to_crs("EPSG:3857")
    if data.empty:
        raise ValueError(f"No finite values in column {column!r}")

    cmap = cmap or _HEX_CMAPS.get(column, "viridis")
    res = hex_gdf.attrs.get("resolution", "?")
    msa = hex_gdf.attrs.get("msa_name", "")
    if title is None:
        title = f"{msa} — {column.replace('_', ' ')}".strip(" —")
    if subtitle is None:
        subtitle = f"H3 resolution {res} (~{h3_edge_km(res):.2f} km edge) · area-interpolated from census tracts"

    fig, ax = plt.subplots(figsize=(14, 11), facecolor="white")
    # Hexes semi-transparent and on top (zorder 2); basemap underneath (zorder 1)
    # so roads, street labels, and POIs read through the fill.
    plot_kwargs = dict(
        column=column, ax=ax, cmap=cmap,
        edgecolor=edgecolor, linewidth=edge_width, alpha=alpha, zorder=2,
        legend=True,
        legend_kwds={"shrink": 0.5, "label": column.replace("_", " ")},
    )
    if center is not None:
        # Diverging map centered at `center` (e.g. 0 for change/balance).
        # Robust symmetric bounds from the 95th pct of |value-center| so a few
        # outliers don't wash out the scale.
        from matplotlib.colors import TwoSlopeNorm
        dev = (data[column] - center).abs()
        span = vmax if vmax is not None else float(np.nanpercentile(dev, 95)) or 1.0
        norm = TwoSlopeNorm(vcenter=center, vmin=center - span, vmax=center + span)
        plot_kwargs.update(norm=norm)
    elif vmin is not None or vmax is not None:
        plot_kwargs.update(vmin=vmin, vmax=vmax)
    else:
        lk = {"loc": "lower right", "fontsize": 8,
              "title": column.replace("_", " "), "frameon": True}
        if legend_fmt:
            lk["fmt"] = legend_fmt  # e.g. "{:,.0f}" → commas, no decimals
        if bins is not None:
            # Manual breaks: make the last edge swallow the max so the top
            # class is "<last bin>+" rather than dropping outliers.
            edges = sorted(float(b) for b in bins)
            top = float(data[column].max())
            if edges[-1] < top:
                edges.append(top)
            plot_kwargs.update(scheme="userdefined",
                               classification_kwds={"bins": edges},
                               legend_kwds=lk)
        else:
            plot_kwargs.update(scheme=scheme, k=k, legend_kwds=lk)
    data.plot(**plot_kwargs)

    if basemap:
        try:
            import contextily as ctx
            prov = ctx.providers
            for part in basemap_provider.split("."):
                prov = getattr(prov, part)
            ctx.add_basemap(ax, source=prov, zoom=zoom, attribution=False, zorder=1)
        except Exception as e:
            logger.warning("basemap skipped: %s", e)

    ax.set_title(title, fontsize=15, fontfamily="Helvetica Neue",
                 fontweight="bold", color="#222", loc="left", pad=10)
    ax.set_axis_off()
    fig.text(0.02, 0.92, subtitle, fontsize=9, fontfamily="Helvetica Neue",
             color="#777", va="top")

    plt.tight_layout()
    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        console.print(f"[green]H3 map saved:[/green] {out}")
    return fig


def h3_choropleth(
    hex_gdf: gpd.GeoDataFrame,
    column: str,
    output_path: str | pathlib.Path | None = None,
    cmap: str | None = None,
    scheme: str = "quantiles",
    k: int = 7,
    zoom_start: int = 10,
    tooltip_cols: Sequence[str] | None = None,
) -> "folium.Map":
    """Interactive H3 choropleth (Folium).

    Args:
        hex_gdf: Hex GeoDataFrame.
        column: Column to map.
        output_path: Save HTML if provided.
        cmap: Matplotlib colormap name. Auto by column if None.
        scheme / k: Classification.
        zoom_start: Initial zoom.
        tooltip_cols: Columns shown on hover. Defaults to a sensible set.

    Returns:
        Folium Map.
    """
    try:
        import folium
        import branca.colormap as cm
        import mapclassify
    except ImportError:
        raise ImportError("folium + mapclassify required. pip install 'geostack[viz]'")
    import matplotlib

    data = hex_gdf[hex_gdf[column].notna() & np.isfinite(hex_gdf[column])].to_crs("EPSG:4326")
    if data.empty:
        raise ValueError(f"No finite values in column {column!r}")

    cmap_name = cmap or _HEX_CMAPS.get(column, "viridis")
    mpl_cmap = matplotlib.colormaps[cmap_name]
    vals = data[column].values
    binner = mapclassify.classify(vals, scheme, k=k)
    edges = [float(vals.min())] + list(binner.bins)
    colors = [matplotlib.colors.to_hex(mpl_cmap(i / (len(edges) - 2))) for i in range(len(edges) - 1)]
    colormap = cm.StepColormap(colors, index=edges, vmin=float(vals.min()), vmax=float(vals.max()),
                               caption=column.replace("_", " "))

    center = [data.geometry.centroid.y.mean(), data.geometry.centroid.x.mean()]
    # OpenStreetMap default (full detail: roads, POIs, buildings); lighter
    # alternates via the layer control.
    m = folium.Map(location=center, zoom_start=zoom_start,
                   tiles="OpenStreetMap", name="OpenStreetMap (full detail)")
    folium.TileLayer("CartoDB Voyager", name="Streets & POIs (Voyager)").add_to(m)
    folium.TileLayer("CartoDB positron", name="Light (Positron)").add_to(m)

    if tooltip_cols is None:
        tooltip_cols = [c for c in ["population", "jobs", "median_income",
                                    "jobs_housing_balance", column] if c in data.columns]
        tooltip_cols = list(dict.fromkeys(tooltip_cols))

    # Semi-transparent fill so the basemap's roads/POIs read through the hexes
    folium.GeoJson(
        data[["h3", column, *[c for c in tooltip_cols if c != column], "geometry"]].__geo_interface__,
        style_function=lambda f: {
            "fillColor": colormap(f["properties"][column]) if f["properties"].get(column) is not None else "#00000000",
            "color": "#00000022", "weight": 0.2, "fillOpacity": 0.55,
        },
        tooltip=folium.GeoJsonTooltip(fields=["h3", *tooltip_cols],
                                      aliases=["H3:", *[c.replace("_", " ") + ":" for c in tooltip_cols]],
                                      localize=True),
        name=column.replace("_", " "),
    ).add_to(m)
    colormap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out))
        console.print(f"[green]H3 interactive map saved:[/green] {out}")
    return m


# ═══════════════════════════════════════════════════════════════════════════
# Derived analyses: value-per-acre, growth, hot-spots, bivariate
# ═══════════════════════════════════════════════════════════════════════════

def parcels_to_h3_value(
    parcels: gpd.GeoDataFrame,
    value_col: str,
    resolution: int = 9,
    acreage_col: str | None = None,
) -> gpd.GeoDataFrame:
    """Value-per-acre hex layer from parcels (the "Strong Towns" analysis).

    Bins parcels (points or polygon centroids) into H3 and computes total
    appraised value, land acres, and value per acre — a fiscal-productivity
    surface showing which development patterns generate the most value per
    unit of land.

    Args:
        parcels: GeoDataFrame with a value column. Points or polygons.
        value_col: Appraised/market value column.
        resolution: H3 resolution (parcels are fine — default 9).
        acreage_col: Land acres per parcel. If None, derived from polygon area
            (parcels must be polygons) or falls back to hex area.

    Returns:
        Hex GeoDataFrame with total_value, land_acres, value_per_acre,
        parcel_count, geometry.
    """
    h3 = _require_h3()
    g = parcels.to_crs("EPSG:4326").copy()
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    g["_val"] = pd.to_numeric(g[value_col], errors="coerce")
    g = g[g["_val"] > 0]

    # Land acres per parcel
    if acreage_col and acreage_col in g.columns:
        g["_acres"] = pd.to_numeric(g[acreage_col], errors="coerce")
    elif (g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])).all():
        from geostack.spatial import suggest_crs
        g["_acres"] = g.to_crs(suggest_crs(g)).geometry.area / 43_560.0
    else:
        g["_acres"] = np.nan  # fall back to hex area below

    pts = g.copy()
    pts["geometry"] = pts.geometry.centroid if pts.geometry.geom_type.iloc[0] != "Point" else pts.geometry
    pts["h3"] = [h3.latlng_to_cell(p.y, p.x, resolution) for p in pts.geometry]

    grp = pts.groupby("h3")
    agg = grp.agg(total_value=("_val", "sum"), land_acres=("_acres", "sum"),
                  parcel_count=("_val", "size")).reset_index()

    hexes = cells_to_gdf(agg["h3"])
    result = hexes.merge(agg, on="h3", how="inner")
    from geostack.spatial import suggest_crs
    result["hex_area_acres"] = result.to_crs(suggest_crs(result)).geometry.area / 43_560.0
    # Use summed parcel acres where available, else hex area
    denom = result["land_acres"].where(result["land_acres"] > 0, result["hex_area_acres"])
    result["value_per_acre"] = result["total_value"] / denom.replace(0, np.nan)
    result.attrs["resolution"] = resolution
    return result


def h3_growth(
    hex_start: gpd.GeoDataFrame,
    hex_end: gpd.GeoDataFrame,
    cols: Sequence[str],
    start_year: int | None = None,
    end_year: int | None = None,
) -> gpd.GeoDataFrame:
    """Change between two H3 layers (same resolution/area) — the growth frontier.

    Joins on h3 and computes delta_<col> and pct_<col> per hex, surfacing where
    a metric actually grew (e.g. greenfield jobs the density maps understate).

    Args:
        hex_start, hex_end: Hex layers for two periods (same resolution).
        cols: Columns to difference.
        start_year, end_year: Stamped into result.attrs for labeling.

    Returns:
        Hex GeoDataFrame with delta_<col>, pct_<col>, and the end-period values.
    """
    a = pd.DataFrame(hex_start.drop(columns="geometry"))
    b = hex_end.copy()
    keep = ["h3", *[c for c in cols if c in a.columns]]
    merged = b.merge(a[keep], on="h3", how="outer", suffixes=("", "_start"))
    for c in cols:
        s = merged.get(f"{c}_start")
        e = merged.get(c)
        if s is None or e is None:
            continue
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        e = pd.to_numeric(e, errors="coerce").fillna(0)
        merged[f"delta_{c}"] = e - s
        merged[f"pct_{c}"] = np.where(s > 0, (e - s) / s * 100, np.nan)
    # Ensure geometry for hexes only present in start
    if merged["geometry"].isna().any():
        miss = merged[merged["geometry"].isna()]["h3"]
        fill = cells_to_gdf(miss).set_index("h3")["geometry"]
        merged.loc[merged["geometry"].isna(), "geometry"] = merged.loc[
            merged["geometry"].isna(), "h3"].map(fill)
    out = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    out.attrs["resolution"] = hex_end.attrs.get("resolution")
    out.attrs["start_year"] = start_year
    out.attrs["end_year"] = end_year
    return out


def h3_hotspots(
    hex_gdf: gpd.GeoDataFrame,
    column: str,
    k: int = 2,
) -> gpd.GeoDataFrame:
    """Getis-Ord Gi* hot-spot / cold-spot z-scores over H3 neighbors.

    Identifies *statistically significant* spatial clusters of high (hot) or
    low (cold) values — rigor beyond "looks dense". Uses H3 k-ring neighbors
    (including self) as binary spatial weights.

    Args:
        hex_gdf: Hex layer.
        column: Value column to test.
        k: k-ring radius for the neighborhood (default 2).

    Returns:
        Hex GeoDataFrame with <column>_gi_z (z-score) and gi_class
        (Hot 99/95/90, Not significant, Cold 90/95/99).
    """
    h3 = _require_h3()
    g = hex_gdf[hex_gdf[column].notna() & np.isfinite(hex_gdf[column])].copy()
    x = g[column].to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        g[f"{column}_gi_z"] = np.nan
        g["gi_class"] = "n/a"
        return g

    xbar = x.mean()
    S = np.sqrt((x ** 2).mean() - xbar ** 2)
    idx = {c: i for i, c in enumerate(g["h3"].tolist())}
    present = set(idx)

    z = np.full(n, np.nan)
    for c, i in idx.items():
        nbrs = [c2 for c2 in h3.grid_disk(c, k) if c2 in present]  # includes self
        m = len(nbrs)
        if m == 0:
            continue
        ssum = sum(x[idx[c2]] for c2 in nbrs)
        denom = S * np.sqrt((n * m - m * m) / (n - 1))
        z[i] = (ssum - xbar * m) / denom if denom > 0 else 0.0

    g[f"{column}_gi_z"] = z

    def _cls(v):
        if not np.isfinite(v):
            return "Not significant"
        if v >= 2.58: return "Hot 99%"
        if v >= 1.96: return "Hot 95%"
        if v >= 1.65: return "Hot 90%"
        if v <= -2.58: return "Cold 99%"
        if v <= -1.96: return "Cold 95%"
        if v <= -1.65: return "Cold 90%"
        return "Not significant"

    g["gi_class"] = [_cls(v) for v in z]
    g.attrs["resolution"] = hex_gdf.attrs.get("resolution")
    return g


_GI_COLORS = {
    "Hot 99%": "#b2182b", "Hot 95%": "#ef8a62", "Hot 90%": "#fddbc7",
    "Not significant": "#f7f7f7",
    "Cold 90%": "#d1e5f0", "Cold 95%": "#67a9cf", "Cold 99%": "#2166ac", "n/a": "#eeeeee",
}


def h3_hotspot_map(
    hex_gdf: gpd.GeoDataFrame,
    title: str | None = None,
    subtitle: str | None = None,
    output_path: str | pathlib.Path | None = None,
    dpi: int = 220,
    basemap_provider: str = "OpenStreetMap.Mapnik",
    zoom: int = 11,
) -> "matplotlib.figure.Figure":
    """Render a Gi* hot-spot map (output of h3_hotspots) with a diverging legend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        raise ImportError("matplotlib required. pip install 'geostack[print-maps]'")

    data = hex_gdf[hex_gdf.get("gi_class").notna()].to_crs("EPSG:3857")
    order = ["Cold 99%", "Cold 95%", "Cold 90%", "Not significant",
             "Hot 90%", "Hot 95%", "Hot 99%"]
    fig, ax = plt.subplots(figsize=(14, 11), facecolor="white")
    for cls in order:
        sub = data[data["gi_class"] == cls]
        if not sub.empty:
            sub.plot(ax=ax, color=_GI_COLORS[cls], edgecolor="#ffffff",
                     linewidth=0.12, alpha=0.7, zorder=2)
    if basemap_provider:
        try:
            import contextily as ctx
            prov = ctx.providers
            for part in basemap_provider.split("."):
                prov = getattr(prov, part)
            ctx.add_basemap(ax, source=prov, zoom=zoom, attribution=False, zorder=1)
        except Exception as e:
            logger.warning("basemap skipped: %s", e)
    ax.set_axis_off()
    ax.set_title(title or "Hot-spot analysis (Getis-Ord Gi*)", fontsize=15,
                 fontfamily="Helvetica Neue", fontweight="bold", color="#222",
                 loc="left", pad=10)
    if subtitle:
        fig.text(0.02, 0.92, subtitle, fontsize=9, fontfamily="Helvetica Neue",
                 color="#777", va="top")
    legend = [Patch(facecolor=_GI_COLORS[c], edgecolor="#ccc", label=c) for c in order]
    ax.legend(handles=legend, loc="lower right", fontsize=8, frameon=True,
              title="Gi* cluster")
    plt.tight_layout()
    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        console.print(f"[green]Hot-spot map saved:[/green] {out}")
    return fig


def h3_bivariate(
    hex_gdf: gpd.GeoDataFrame,
    x_col: str,
    y_col: str,
    k: int = 3,
) -> gpd.GeoDataFrame:
    """Assign a k×k bivariate class from two columns (e.g. demand vs supply).

    Each hex gets x and y classified into k quantile bins; bivar_class encodes
    the pair. Use with h3_bivariate_map. Great for demand-gap (high demand +
    low supply) detection.

    Returns hex GeoDataFrame with x_class, y_class, bivar_class (0..k*k-1).
    """
    import mapclassify
    g = hex_gdf[hex_gdf[x_col].notna() & hex_gdf[y_col].notna()
               & np.isfinite(hex_gdf[x_col]) & np.isfinite(hex_gdf[y_col])].copy()
    g["x_class"] = mapclassify.Quantiles(g[x_col].to_numpy(), k=k).yb
    g["y_class"] = mapclassify.Quantiles(g[y_col].to_numpy(), k=k).yb
    g["bivar_class"] = g["x_class"] * k + g["y_class"]
    g.attrs["resolution"] = hex_gdf.attrs.get("resolution")
    g.attrs["k"] = k
    g.attrs["x_col"] = x_col
    g.attrs["y_col"] = y_col
    return g


def _bivariate_palette(k: int = 3) -> dict:
    """k×k bivariate color matrix (x = low→high left→right, y = low→high bottom→top)."""
    # Corners: low-low grey, high-x blue, high-y pink, high-high purple
    import matplotlib.colors as mc
    ll = np.array(mc.to_rgb("#e8e8e8"))
    hx = np.array(mc.to_rgb("#5ac8c8"))  # high x
    hy = np.array(mc.to_rgb("#be64ac"))  # high y
    hh = np.array(mc.to_rgb("#3b4994"))  # high both
    pal = {}
    for xi in range(k):
        for yi in range(k):
            fx = xi / (k - 1) if k > 1 else 0
            fy = yi / (k - 1) if k > 1 else 0
            c = (ll * (1 - fx) * (1 - fy) + hx * fx * (1 - fy)
                 + hy * (1 - fx) * fy + hh * fx * fy)
            pal[xi * k + yi] = mc.to_hex(c)
    return pal


def h3_bivariate_map(
    hex_gdf: gpd.GeoDataFrame,
    title: str | None = None,
    subtitle: str | None = None,
    x_label: str = "x",
    y_label: str = "y",
    output_path: str | pathlib.Path | None = None,
    dpi: int = 220,
    basemap_provider: str = "OpenStreetMap.Mapnik",
    zoom: int = 11,
) -> "matplotlib.figure.Figure":
    """Render a bivariate H3 map (output of h3_bivariate) with a k×k legend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. pip install 'geostack[print-maps]'")

    k = hex_gdf.attrs.get("k", 3)
    pal = _bivariate_palette(k)
    data = hex_gdf.to_crs("EPSG:3857")

    fig, ax = plt.subplots(figsize=(14, 11), facecolor="white")
    for cls, color in pal.items():
        sub = data[data["bivar_class"] == cls]
        if not sub.empty:
            sub.plot(ax=ax, color=color, edgecolor="#ffffff", linewidth=0.12,
                     alpha=0.72, zorder=2)
    if basemap_provider:
        try:
            import contextily as ctx
            prov = ctx.providers
            for part in basemap_provider.split("."):
                prov = getattr(prov, part)
            ctx.add_basemap(ax, source=prov, zoom=zoom, attribution=False, zorder=1)
        except Exception as e:
            logger.warning("basemap skipped: %s", e)
    ax.set_axis_off()
    ax.set_title(title or "Bivariate map", fontsize=15, fontfamily="Helvetica Neue",
                 fontweight="bold", color="#222", loc="left", pad=10)
    if subtitle:
        fig.text(0.02, 0.92, subtitle, fontsize=9, fontfamily="Helvetica Neue",
                 color="#777", va="top")

    # Inset k×k legend
    iax = fig.add_axes([0.80, 0.12, 0.14, 0.14])
    for xi in range(k):
        for yi in range(k):
            iax.add_patch(plt.Rectangle((xi, yi), 1, 1, color=pal[xi * k + yi]))
    iax.set_xlim(0, k); iax.set_ylim(0, k)
    iax.set_xticks([]); iax.set_yticks([])
    iax.set_xlabel(f"{x_label} →", fontsize=7)
    iax.set_ylabel(f"{y_label} →", fontsize=7)
    iax.set_aspect("equal")

    plt.tight_layout()
    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        console.print(f"[green]Bivariate map saved:[/green] {out}")
    return fig
