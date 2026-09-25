"""Polycentric gravity analysis: node detection, UA comparison, ring/quadrant decomposition.

Extends the core gravity module with spatial decomposition tools for
polycentric metros like DFW, where multiple employment/population nodes
compete for dominance.

Usage:
    from geostack.gravity_polycentric import (
        detect_nodes,
        node_time_series,
        ua_comparison,
        ring_quadrant_decomposition,
    )

    # Detect employment sub-centers
    nodes = detect_nodes("Dallas", year=2021, weight_mode="jobs")

    # Track nodes across time
    evolution = node_time_series("Dallas", years=[2010, 2015, 2020, 2021])

    # UA vs MSA comparison
    ua_vs_msa = ua_comparison("Dallas", years=[2010, 2020])

    # Ring × quadrant decomposition
    sectors = ring_quadrant_decomposition("Dallas", years=[2010, 2020])
"""

from __future__ import annotations

import logging
import os
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
from rich.console import Console
from shapely.geometry import Point

from geostack.gravity import (
    MSA_DEFINITIONS,
    _CACHE_DIR,
    _compute_weighted_centroid,
    _fetch_county_weights,
    _haversine_miles,
    _bearing_degrees,
    _read_cache,
    _weight_column,
    _write_cache,
    fetch_tract_geometries,
    get_landmarks,
    resolve_msa,
)
from geostack.spatial import suggest_crs

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Urbanized Area boundary download
# ---------------------------------------------------------------------------

_TIGER_BASE = "https://www2.census.gov/geo/tiger"


def download_urbanized_areas(year: int = 2020) -> gpd.GeoDataFrame:
    """Download Census Urbanized Area boundaries (national file).

    Args:
        year: TIGER vintage (2020 recommended).

    Returns:
        GeoDataFrame with UACE20 (UA code), NAME20, geometry.
    """
    from geostack.tiger import _download_shapefile

    # CB version is much smaller (~5 MB vs ~200 MB)
    filename = f"cb_{year}_us_ua10_500k"
    if year >= 2020:
        filename = f"cb_{year}_us_ua20_500k"

    url = f"{_TIGER_BASE}/GENZ{year}/shp/{filename}.zip"

    cache_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "shapefiles"
    )

    shp_dir = _download_shapefile(url, cache_dir)
    shp_files = [f for f in os.listdir(shp_dir) if f.endswith(".shp")]
    if not shp_files:
        raise FileNotFoundError(f"No .shp file in {shp_dir}")

    gdf = gpd.read_file(os.path.join(shp_dir, shp_files[0]))
    console.print(f"[green]Loaded {len(gdf)} urbanized areas[/green]")
    return gdf


def get_msa_urbanized_area(
    msa_name: str,
    year: int = 2020,
) -> gpd.GeoDataFrame:
    """Get the Urbanized Area boundary that overlaps an MSA's counties.

    For polycentric metros like DFW, there may be multiple UAs
    (Dallas-Fort Worth-Arlington UA, Denton-Lewisville UA, etc.).
    Returns all UAs that intersect the MSA extent.
    """
    msa = resolve_msa(msa_name)
    state_fips = msa["state_fips"]
    counties = msa["counties"]

    # Get MSA tract extent
    all_tracts = []
    for county_fips in counties:
        tracts = fetch_tract_geometries(state_fips, county_fips, year)
        all_tracts.append(tracts)
    msa_tracts = pd.concat(all_tracts, ignore_index=True)
    msa_extent = msa_tracts.to_crs("EPSG:4326").union_all()

    # Download UAs and filter to those intersecting MSA
    ua_gdf = download_urbanized_areas(year)
    ua_gdf = ua_gdf.to_crs("EPSG:4326")
    ua_gdf = ua_gdf[ua_gdf.geometry.intersects(msa_extent)].copy()

    console.print(f"[green]Found {len(ua_gdf)} UAs overlapping {msa['name']}[/green]")
    return ua_gdf


# ---------------------------------------------------------------------------
# UA vs MSA gravity comparison
# ---------------------------------------------------------------------------

def ua_comparison(
    msa_name: str,
    years: list[int],
    weight_mode: str = "population",
    use_crosswalk: bool = True,
) -> pd.DataFrame:
    """Compare gravity centers computed over full MSA vs UA-only tracts.

    Returns DataFrame with rows for each (year, scope) combination where
    scope is "MSA" or "UA".
    """
    from geostack.gravity import population_center_of_gravity, compute_shift_metrics

    msa = resolve_msa(msa_name)
    state_fips = msa["state_fips"]
    counties = msa["counties"]

    console.print(f"[bold]UA vs MSA comparison:[/bold] {msa['name']}")

    # Get UA boundary
    ua_gdf = get_msa_urbanized_area(msa_name)
    ua_union = ua_gdf.to_crs("EPSG:4326").union_all()

    results = []
    for year in years:
        console.print(f"\n  [bold]{year}:[/bold]")

        # Full MSA
        try:
            msa_result = population_center_of_gravity(
                msa_name, year=year, weight_mode=weight_mode,
                use_crosswalk=use_crosswalk,
            )
            row = msa_result.to_dict()
            row["scope"] = "MSA"
            results.append(row)
        except Exception as e:
            console.print(f"  [red]MSA failed: {e}[/red]")
            continue

        # UA-only: filter tracts to those within UA boundary
        try:
            all_tracts = []
            all_weights = []
            for county_fips, county_name in counties.items():
                try:
                    weight_df = _fetch_county_weights(
                        state_fips, county_fips, year, weight_mode, use_crosswalk,
                    )
                    tract_gdf = fetch_tract_geometries(state_fips, county_fips, year)
                    all_weights.append(weight_df)
                    all_tracts.append(tract_gdf)
                except Exception:
                    pass

            if not all_tracts:
                continue

            tracts_gdf = pd.concat(all_tracts, ignore_index=True)
            weight_df = pd.concat(all_weights, ignore_index=True)

            # Filter to tracts intersecting UA
            tracts_wgs = tracts_gdf.to_crs("EPSG:4326")
            ua_mask = tracts_wgs.geometry.intersects(ua_union)
            ua_tracts = tracts_gdf[ua_mask.values].copy()

            weight_col = _weight_column(weight_mode)
            center_lat, center_lon, total_w, tract_count = _compute_weighted_centroid(
                ua_tracts, weight_df, weight_col,
            )

            ua_row = {
                "msa_name": msa["name"],
                "year": year,
                "weight_mode": weight_mode,
                "total_weight": total_w,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "county_count": len(counties),
                "tract_count": tract_count,
                "data_source": f"UA-filtered ({weight_mode})",
                "tiger_vintage": 2020,
                "scope": "UA",
                "notes": f"Tracts within urbanized area boundary only ({tract_count} of {len(tracts_gdf)})",
            }
            results.append(ua_row)

            gap = _haversine_miles(
                msa_result.center_lat, msa_result.center_lon,
                center_lat, center_lon,
            )
            console.print(
                f"  UA center: ({center_lat:.4f}, {center_lon:.4f}) "
                f"| MSA→UA gap: {gap:.2f} mi"
            )

        except Exception as e:
            console.print(f"  [red]UA failed: {e}[/red]")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Polycentric node detection
# ---------------------------------------------------------------------------

def _compute_tract_density(
    tracts_gdf: gpd.GeoDataFrame,
    weight_df: pd.DataFrame,
    weight_col: str = "weight",
) -> gpd.GeoDataFrame:
    """Join weights to tracts and compute density (weight per sq mi)."""
    merged = tracts_gdf.merge(weight_df, on="GEOID", how="inner")
    if merged.empty:
        return merged

    # Compute area in sq mi using projected CRS
    crs = suggest_crs(merged)
    merged_proj = merged.to_crs(crs)

    # CRS units: EPSG:2276 is feet, UTM is meters
    if "2276" in str(crs) or "2275" in str(crs) or "2278" in str(crs):
        merged["area_sqmi"] = merged_proj.geometry.area / (5280 ** 2)
    else:
        merged["area_sqmi"] = merged_proj.geometry.area / (1609.344 ** 2)

    merged["density"] = merged[weight_col] / merged["area_sqmi"].replace(0, np.nan)
    merged["centroid_x"] = merged_proj.geometry.centroid.x
    merged["centroid_y"] = merged_proj.geometry.centroid.y

    return merged


def _assign_to_peaks(
    high_density: gpd.GeoDataFrame,
    weight_col: str,
    min_separation_mi: float,
    peak_rel_threshold: float,
) -> np.ndarray:
    """Find density-surface peaks and assign each tract to its nearest peak.

    Returns an int label array (peak index per tract). Separates sub-centers
    on a continuous corridor that DBSCAN would chain into one cluster.
    """
    from scipy.stats import gaussian_kde
    from scipy.ndimage import maximum_filter

    hx = high_density["centroid_x"].values.astype(float)
    hy = high_density["centroid_y"].values.astype(float)
    hw = high_density[weight_col].values.astype(float)

    # Weighted KDE: repeat each centroid proportional to its weight
    reps = np.clip(hw / hw.max() * 200, 1, 500).astype(int)
    ex = np.repeat(hx, reps)
    ey = np.repeat(hy, reps)
    kde = gaussian_kde(np.vstack([ex, ey]), bw_method=0.10)

    pad = 0.05
    xr, yr = hx.max() - hx.min(), hy.max() - hy.min()
    gx = np.linspace(hx.min() - xr * pad, hx.max() + xr * pad, 300)
    gy = np.linspace(hy.min() - yr * pad, hy.max() + yr * pad, 300)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    Zn = Z / Z.max()

    # Local maxima separated by >= min_separation_mi
    cell_ft = gx[1] - gx[0]
    fp = max(int((min_separation_mi * 5280) / cell_ft), 3)
    mx = maximum_filter(Zn, size=2 * fp + 1, mode="nearest")
    peak_mask = (Zn == mx) & (Zn >= peak_rel_threshold)
    pidx = np.argwhere(peak_mask)
    peaks = np.array([[gx[j], gy[i]] for i, j in pidx])

    if len(peaks) == 0:
        return np.full(len(high_density), -1, dtype=int)

    # Assign each tract to nearest peak
    tracts = np.column_stack([hx, hy])
    d2 = ((tracts[:, None, :] - peaks[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).astype(int)


def detect_nodes(
    msa_name: str,
    year: int = 2021,
    weight_mode: str = "jobs",
    density_percentile: float = 85,
    min_cluster_tracts: int = 3,
    cluster_distance_ft: float = 15000,
    use_crosswalk: bool = True,
    method: str = "peaks",
    min_separation_mi: float = 3.0,
    peak_rel_threshold: float = 0.07,
) -> gpd.GeoDataFrame:
    """Algorithmically detect employment/population sub-centers in an MSA.

    Two methods:

    - **peaks** (default): finds local maxima of the weighted density surface,
      then watershed-assigns high-density tracts to the nearest peak. Separates
      sub-centers that sit on a *continuous* dense corridor — e.g. the Dallas
      North Tollway ribbon from Downtown through Legacy/Frisco — which the
      DBSCAN method chains into one mega-node.

    - **dbscan**: density-based spatial clustering (single-linkage). Faster but
      merges any sub-centers connected by a chain of above-threshold tracts.

    Args:
        msa_name: MSA name or partial match.
        year: Data year.
        weight_mode: "jobs" (recommended), "population", "workers", "income_mass".
        density_percentile: Only consider tracts above this density percentile.
        min_cluster_tracts: Minimum tracts assigned for a peak/cluster to count.
        cluster_distance_ft: DBSCAN linkage distance (dbscan method only).
        use_crosswalk: Apply crosswalk normalization for pre-2020 data.
        method: "peaks" (default) or "dbscan".
        min_separation_mi: Minimum spacing between peaks (peaks method).
        peak_rel_threshold: Drop peaks below this fraction of the max density
            (peaks method) — filters noise.

    Returns:
        GeoDataFrame with one row per detected node: node_id, center_lat,
        center_lon, total_weight, tract_count, mean_density, place, geometry.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise ImportError("scikit-learn is required for node detection. Run: pip install scikit-learn")

    msa = resolve_msa(msa_name)
    state_fips = msa["state_fips"]
    counties = msa["counties"]
    weight_col = _weight_column(weight_mode)

    console.print(
        f"[bold]Detecting nodes:[/bold] {msa['name']} ({year}, {weight_mode})"
    )

    # Fetch all tracts + weights
    all_tracts = []
    all_weights = []
    for county_fips, county_name in counties.items():
        try:
            weight_df = _fetch_county_weights(
                state_fips, county_fips, year, weight_mode, use_crosswalk,
            )
            tract_gdf = fetch_tract_geometries(state_fips, county_fips, year)
            all_weights.append(weight_df)
            all_tracts.append(tract_gdf)
        except Exception as e:
            logger.warning("Skipping %s: %s", county_name, e)

    tracts_gdf = pd.concat(all_tracts, ignore_index=True)
    weight_df = pd.concat(all_weights, ignore_index=True)

    # Compute density
    density_gdf = _compute_tract_density(tracts_gdf, weight_df, weight_col)
    if density_gdf.empty:
        raise ValueError("No density data available")

    # Filter to high-density tracts
    threshold = density_gdf["density"].quantile(density_percentile / 100)
    high_density = density_gdf[density_gdf["density"] >= threshold].copy()
    console.print(
        f"  Density threshold (p{density_percentile:.0f}): "
        f"{threshold:,.0f} {weight_mode}/sq mi → "
        f"{len(high_density)} tracts above threshold"
    )

    if len(high_density) < min_cluster_tracts:
        console.print("[yellow]  Too few high-density tracts for clustering[/yellow]")
        return gpd.GeoDataFrame()

    high_density = high_density.copy()

    if method == "peaks":
        labels = _assign_to_peaks(
            high_density, weight_col, min_separation_mi, peak_rel_threshold,
        )
        high_density["cluster"] = labels
    elif method == "dbscan":
        coords = high_density[["centroid_x", "centroid_y"]].values
        high_density["cluster"] = DBSCAN(
            eps=cluster_distance_ft, min_samples=min_cluster_tracts,
            metric="euclidean",
        ).fit(coords).labels_
    else:
        raise ValueError(f"method must be 'peaks' or 'dbscan', got {method!r}")

    # Drop noise (-1) and clusters below the minimum size
    clustered = high_density[high_density["cluster"] >= 0]
    sizes = clustered.groupby("cluster").size()
    keep = sizes[sizes >= min_cluster_tracts].index
    clustered = clustered[clustered["cluster"].isin(keep)]
    n_clusters = clustered["cluster"].nunique()
    console.print(f"  Detected {n_clusters} nodes ({method})")

    # Compute per-node metrics
    nodes = []
    crs = suggest_crs(tracts_gdf)
    for cluster_id in sorted(clustered["cluster"].unique()):
        cluster_tracts = clustered[clustered["cluster"] == cluster_id]

        # Weighted centroid of this cluster
        weights = cluster_tracts[weight_col].values.astype(float)
        total_w = weights.sum()
        cx = (weights * cluster_tracts["centroid_x"].values).sum() / total_w
        cy = (weights * cluster_tracts["centroid_y"].values).sum() / total_w

        # Convert to WGS84
        center_proj = gpd.GeoDataFrame(
            geometry=[Point(cx, cy)], crs=crs,
        )
        center_wgs = center_proj.to_crs("EPSG:4326")
        pt = center_wgs.geometry.iloc[0]

        nodes.append({
            "node_id": int(cluster_id),
            "center_lat": round(pt.y, 6),
            "center_lon": round(pt.x, 6),
            "total_weight": float(total_w),
            "tract_count": len(cluster_tracts),
            "mean_density": float(cluster_tracts["density"].mean()),
            "year": year,
            "weight_mode": weight_mode,
            "geometry": pt,
        })

    nodes_gdf = gpd.GeoDataFrame(nodes, crs="EPSG:4326")

    # Sort by total_weight descending and assign rank-based names
    nodes_gdf = nodes_gdf.sort_values("total_weight", ascending=False).reset_index(drop=True)
    nodes_gdf["node_id"] = range(len(nodes_gdf))
    nodes_gdf["node_rank"] = [f"Node {i+1}" for i in range(len(nodes_gdf))]

    # Name each node by its nearest landmark (algorithmic → recognizable place)
    nodes_gdf = name_nodes_by_landmark(nodes_gdf, msa_name)

    for _, node in nodes_gdf.iterrows():
        place = node.get("place", "")
        place_str = f" [{place}]" if place else ""
        console.print(
            f"  [green]{node['node_rank']}{place_str}:[/green] "
            f"({node['center_lat']:.4f}, {node['center_lon']:.4f}) "
            f"weight={node['total_weight']:,.0f} "
            f"({node['tract_count']} tracts, "
            f"density={node['mean_density']:,.0f}/sq mi)"
        )

    return nodes_gdf


def name_nodes_by_landmark(
    nodes_gdf: gpd.GeoDataFrame,
    msa_name: str,
    max_distance_mi: float = 8.0,
) -> gpd.GeoDataFrame:
    """Add a 'place' column naming each node by its nearest landmark.

    Turns abstract "Node 1" into recognizable "Dallas core" / "Fort Worth".
    If no landmark is within max_distance_mi, place is the empty string.

    Args:
        nodes_gdf: GeoDataFrame with center_lat / center_lon columns.
        msa_name: MSA name (to fetch its landmark registry).
        max_distance_mi: Cap beyond which a node stays unnamed.

    Returns:
        nodes_gdf with an added 'place' column.
    """
    landmarks = get_landmarks(msa_name)
    if nodes_gdf.empty or not landmarks:
        nodes_gdf = nodes_gdf.copy()
        nodes_gdf["place"] = ""
        return nodes_gdf

    nodes_gdf = nodes_gdf.copy()
    used = set()
    places = []
    # Assign in weight order (already sorted) so the biggest node claims the CBD
    for _, node in nodes_gdf.iterrows():
        best_name, best_d = "", float("inf")
        for name, la, lo, kind in landmarks:
            d = _haversine_miles(node["center_lat"], node["center_lon"], la, lo)
            # Prefer unused landmarks; allow reuse only if nothing else is close
            penalty = 100 if name in used else 0
            if d + penalty < best_d:
                best_d = d + penalty
                best_name = name
                best_raw_d = d
        if best_name and best_raw_d <= max_distance_mi:
            places.append(best_name)
            used.add(best_name)
        else:
            places.append("")
    nodes_gdf["place"] = places
    return nodes_gdf


# ---------------------------------------------------------------------------
# Node time series — track nodes across years
# ---------------------------------------------------------------------------

def node_time_series(
    msa_name: str,
    years: list[int],
    weight_mode: str = "jobs",
    use_crosswalk: bool = True,
    top_n: int | None = 12,
    **detect_kwargs,
) -> pd.DataFrame:
    """Detect nodes for each year and match them across time.

    Nodes are matched across years by spatial proximity (nearest-neighbor
    matching from the base year's node positions). With peak detection a metro
    can yield 15-20 sub-centers; top_n keeps the largest by reference-year
    weight for legibility (set None to keep all).

    Returns DataFrame with columns: node_id, node_rank, year, center_lat,
    center_lon, total_weight, tract_count, dist_from_msa_center_mi,
    bearing_from_msa_center.
    """
    from geostack.gravity import population_center_of_gravity

    msa = resolve_msa(msa_name)
    console.print(f"[bold]Node time series:[/bold] {msa['name']}")

    # Get MSA population center for reference
    msa_centers = {}
    for year in years:
        try:
            result = population_center_of_gravity(
                msa_name, year=year, weight_mode="population",
                use_crosswalk=use_crosswalk,
            )
            msa_centers[year] = (result.center_lat, result.center_lon)
        except Exception:
            pass

    # Detect nodes for each year
    all_nodes = {}
    for year in years:
        try:
            nodes = detect_nodes(
                msa_name, year=year, weight_mode=weight_mode,
                use_crosswalk=use_crosswalk, **detect_kwargs,
            )
            if not nodes.empty:
                all_nodes[year] = nodes
        except Exception as e:
            console.print(f"  [red]{year}: node detection failed — {e}[/red]")

    if not all_nodes:
        raise RuntimeError("No nodes detected for any year")

    # Use the year with the most nodes as the reference for matching
    ref_year = max(all_nodes, key=lambda y: len(all_nodes[y]))
    ref_nodes = all_nodes[ref_year]

    # Keep only the top_n largest sub-centers (by reference-year weight)
    if top_n is not None and len(ref_nodes) > top_n:
        ref_nodes = ref_nodes.nlargest(top_n, "total_weight").reset_index(drop=True)
        console.print(f"  Keeping top {top_n} of {len(all_nodes[ref_year])} sub-centers")

    console.print(
        f"\n  Reference year: {ref_year} ({len(ref_nodes)} nodes)"
    )

    # Map reference node_id → place name (consistent labels across years)
    ref_place = {}
    if "place" in ref_nodes.columns:
        ref_place = dict(zip(ref_nodes["node_id"], ref_nodes["place"]))

    # For each reference node, find its nearest counterpart in each year
    # (one row per ref node per year — no duplicates, no fold-ins).
    results = []
    match_cap_mi = 4.0  # a sub-center shouldn't move more than this between samples
    for year, nodes_gdf in sorted(all_nodes.items()):
        used_targets = set()
        for _, ref in ref_nodes.iterrows():
            best_d, best = float("inf"), None
            for ti, tgt in nodes_gdf.iterrows():
                if ti in used_targets:
                    continue
                d = _haversine_miles(ref["center_lat"], ref["center_lon"],
                                     tgt["center_lat"], tgt["center_lon"])
                if d < best_d:
                    best_d, best, best_ti = d, tgt, ti
            if best is None or best_d > match_cap_mi:
                continue  # node not present this year
            used_targets.add(best_ti)
            nid = int(ref["node_id"])
            row = {
                "node_id": nid,
                "node_rank": f"Node {nid + 1}",
                "place": ref_place.get(nid, ""),
                "year": year,
                "center_lat": best["center_lat"],
                "center_lon": best["center_lon"],
                "total_weight": best["total_weight"],
                "tract_count": best["tract_count"],
                "mean_density": best["mean_density"],
                "weight_mode": weight_mode,
            }
            if year in msa_centers:
                mlat, mlon = msa_centers[year]
                row["dist_from_msa_center_mi"] = round(
                    _haversine_miles(mlat, mlon, best["center_lat"], best["center_lon"]), 3)
                row["bearing_from_msa_center"] = round(
                    _bearing_degrees(mlat, mlon, best["center_lat"], best["center_lon"]), 1)
            else:
                row["dist_from_msa_center_mi"] = np.nan
                row["bearing_from_msa_center"] = np.nan
            results.append(row)

    return pd.DataFrame(results)


def _match_nodes(
    ref_nodes: gpd.GeoDataFrame,
    target_nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Match target nodes to reference nodes by nearest centroid."""
    target = target_nodes.copy()
    target["matched_to"] = -1

    for idx, tgt in target.iterrows():
        best_dist = float("inf")
        best_id = 0
        for _, ref in ref_nodes.iterrows():
            d = _haversine_miles(
                ref["center_lat"], ref["center_lon"],
                tgt["center_lat"], tgt["center_lon"],
            )
            if d < best_dist:
                best_dist = d
                best_id = ref["node_id"]
        target.at[idx, "matched_to"] = best_id

    return target


# ---------------------------------------------------------------------------
# Ring × Quadrant decomposition
# ---------------------------------------------------------------------------

def ring_quadrant_decomposition(
    msa_name: str,
    years: list[int],
    weight_mode: str = "population",
    ring_radii_mi: list[float] | None = None,
    use_crosswalk: bool = True,
) -> pd.DataFrame:
    """Decompose MSA into concentric rings × directional quadrants.

    Uses the MSA's population center as the origin. Each tract is assigned
    to a ring (by distance) and quadrant (by bearing), then the weighted
    centroid is computed for each sector.

    Args:
        msa_name: MSA name.
        years: Years to analyze.
        weight_mode: Weight column to use.
        ring_radii_mi: Ring boundaries in miles. Default: [10, 20, 30].
        use_crosswalk: Apply crosswalk for pre-2020 data.

    Returns:
        DataFrame with one row per (year, ring, quadrant) combination.
    """
    from geostack.gravity import population_center_of_gravity

    if ring_radii_mi is None:
        ring_radii_mi = [10.0, 20.0, 30.0]

    msa = resolve_msa(msa_name)
    state_fips = msa["state_fips"]
    counties = msa["counties"]
    weight_col = _weight_column(weight_mode)

    console.print(
        f"[bold]Ring × Quadrant decomposition:[/bold] {msa['name']}\n"
        f"  Rings: {ring_radii_mi} mi | Quadrants: N/E/S/W"
    )

    quadrant_names = {0: "N", 1: "E", 2: "S", 3: "W"}

    results = []
    for year in years:
        console.print(f"\n  [bold]{year}:[/bold]")

        # Get MSA center as origin
        try:
            msa_result = population_center_of_gravity(
                msa_name, year=year, weight_mode="population",
                use_crosswalk=use_crosswalk,
            )
            origin_lat = msa_result.center_lat
            origin_lon = msa_result.center_lon
        except Exception as e:
            console.print(f"  [red]Skipping {year}: {e}[/red]")
            continue

        # Fetch all tracts + weights
        all_tracts = []
        all_weights = []
        for county_fips in counties:
            try:
                weight_df = _fetch_county_weights(
                    state_fips, county_fips, year, weight_mode, use_crosswalk,
                )
                tract_gdf = fetch_tract_geometries(state_fips, county_fips, year)
                all_weights.append(weight_df)
                all_tracts.append(tract_gdf)
            except Exception:
                pass

        if not all_tracts:
            continue

        tracts_gdf = pd.concat(all_tracts, ignore_index=True)
        weight_df = pd.concat(all_weights, ignore_index=True)

        # Join weights to tracts
        merged = tracts_gdf.merge(weight_df, on="GEOID", how="inner")
        if merged.empty:
            continue

        # Compute distance and bearing from origin for each tract centroid
        crs = suggest_crs(merged)
        merged_proj = merged.to_crs(crs)
        centroids = merged_proj.geometry.centroid
        centroids_wgs = gpd.GeoDataFrame(
            geometry=centroids, crs=crs,
        ).to_crs("EPSG:4326")

        merged["tract_lat"] = centroids_wgs.geometry.y.values
        merged["tract_lon"] = centroids_wgs.geometry.x.values

        dists = []
        bearings = []
        for _, row in merged.iterrows():
            d = _haversine_miles(origin_lat, origin_lon, row["tract_lat"], row["tract_lon"])
            b = _bearing_degrees(origin_lat, origin_lon, row["tract_lat"], row["tract_lon"])
            dists.append(d)
            bearings.append(b)

        merged["dist_mi"] = dists
        merged["bearing"] = bearings

        # Assign ring
        ring_labels = []
        ring_bounds = [0] + ring_radii_mi + [999]
        for d in merged["dist_mi"]:
            for i in range(len(ring_bounds) - 1):
                if ring_bounds[i] <= d < ring_bounds[i + 1]:
                    if i < len(ring_radii_mi):
                        ring_labels.append(f"{int(ring_bounds[i])}–{int(ring_bounds[i+1])} mi")
                    else:
                        ring_labels.append(f"{int(ring_radii_mi[-1])}+ mi")
                    break
            else:
                ring_labels.append(f"{int(ring_radii_mi[-1])}+ mi")
        merged["ring"] = ring_labels

        # Assign quadrant (N: 315-45, E: 45-135, S: 135-225, W: 225-315)
        def _quadrant(bearing):
            if bearing >= 315 or bearing < 45:
                return "N"
            elif 45 <= bearing < 135:
                return "E"
            elif 135 <= bearing < 225:
                return "S"
            else:
                return "W"

        merged["quadrant"] = merged["bearing"].apply(_quadrant)

        # Compute weighted centroid for each sector
        for ring in merged["ring"].unique():
            for quad in ["N", "E", "S", "W"]:
                sector = merged[(merged["ring"] == ring) & (merged["quadrant"] == quad)]
                if sector.empty or sector[weight_col].sum() == 0:
                    continue

                sector_tracts = tracts_gdf[tracts_gdf["GEOID"].isin(sector["GEOID"])]
                sector_weights = weight_df[weight_df["GEOID"].isin(sector["GEOID"])]

                try:
                    clat, clon, tw, tc = _compute_weighted_centroid(
                        sector_tracts, sector_weights, weight_col,
                    )
                except Exception:
                    continue

                results.append({
                    "year": year,
                    "ring": ring,
                    "quadrant": quad,
                    "sector": f"{ring} {quad}",
                    "total_weight": tw,
                    "tract_count": tc,
                    "center_lat": clat,
                    "center_lon": clon,
                    "weight_mode": weight_mode,
                    "origin_lat": origin_lat,
                    "origin_lon": origin_lon,
                })

        # Summary
        ring_summary = merged.groupby("ring")[weight_col].sum()
        for ring, total in ring_summary.items():
            console.print(f"    {ring}: {total:,.0f} ({weight_mode})")

    return pd.DataFrame(results)
