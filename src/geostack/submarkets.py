"""Data-driven submarket delineation via tract classification + basemap visualization.

Two approaches:
  1. K-Means clustering on economic features (unsupervised)
  2. Rule-based classification: economic type thresholds per tract

The basemap version uses contextily tile layers for roads, labels, and terrain —
works for any metro with no manual highway download or label placement.

Usage:
    from geostack.submarkets import (
        cluster_tracts, submarket_profile, property_submarket,
        classify_tracts, plot_submarket_map,
    )

    # K-Means clustering
    clustered = cluster_tracts(tract_features_gdf, n_clusters=8)

    # Rule-based classification + basemap map
    classified = classify_tracts(tract_features_gdf)
    plot_submarket_map(classified, title="DFW Renter Submarkets",
                       isd_gdf=districts, save_path="map.png")
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

console = Console()

_FEATURE_COLS = [
    "renter_median_income",
    "rent_burden_pct",
    "median_rent",
    "pct_built_pre1990",
]

# ── Economic classification rules ──
ECON_CLASSES = {
    "Burdened Low-Income": {
        "rule": lambda r: r["renter_median_income"] < 50_000 and r["pct_built_pre1990"] <= 0.35,
        "color": "#a8322d",
    },
    "Workforce Corridor": {
        "rule": lambda r: r["renter_median_income"] < 50_000 and r["pct_built_pre1990"] > 0.35,
        "color": "#4a6274",
    },
    "Established Affluent": {
        "rule": lambda r: r["renter_median_income"] >= 100_000 and r["pct_built_pre1990"] > 0.35,
        "color": "#7b4f9e",
    },
    "Transitional / Gentrifying": {
        "rule": lambda r: 50_000 <= r["renter_median_income"] < 100_000 and r["pct_built_pre1990"] > 0.35,
        "color": "#daa520",
    },
    "Suburban Moderate": {
        "rule": lambda r: 50_000 <= r["renter_median_income"] < 80_000 and r["pct_built_pre1990"] <= 0.35,
        "color": "#4c72b0",
    },
    "Suburban Professional": {
        "rule": lambda r: r["renter_median_income"] >= 80_000 and r["pct_built_pre1990"] <= 0.35,
        "color": "#27a35b",
    },
}


# ═══════════════════════════════════════════════════════════════════
# K-Means clustering
# ═══════════════════════════════════════════════════════════════════

def cluster_tracts(
    tract_features: gpd.GeoDataFrame,
    n_clusters: int = 8,
    feature_cols: list[str] | None = None,
    random_state: int = 42,
) -> gpd.GeoDataFrame:
    """Cluster census tracts by economic similarity using K-Means."""
    cols = feature_cols or _FEATURE_COLS
    available_cols = [c for c in cols if c in tract_features.columns]
    if not available_cols:
        raise ValueError(f"No feature columns found. Expected: {cols}")

    df = tract_features.copy()
    X = df[available_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df["cluster_id"] = labels
    sil = silhouette_score(X_scaled, labels)
    df.attrs["silhouette_score"] = sil

    console.print(
        f"\n[bold]Submarket Clustering:[/bold]"
        f"\n  Tracts: {len(df):,}  |  Clusters: {n_clusters}"
        f"  |  Silhouette: {sil:.3f}"
    )
    return df


def submarket_profile(clustered: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute summary statistics per cluster."""
    feature_cols = [c for c in _FEATURE_COLS if c in clustered.columns]
    profile = clustered.groupby("cluster_id")[feature_cols].mean()
    profile["tract_count"] = clustered.groupby("cluster_id").size()
    return profile.reset_index()


def property_submarket(
    lon: float, lat: float, clustered: gpd.GeoDataFrame
) -> dict:
    """Find which submarket cluster a property falls in."""
    pt = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326",
    )
    joined = gpd.sjoin(pt, clustered[["cluster_id", "geometry"]],
                       how="left", predicate="within")
    if joined.empty or joined["cluster_id"].isna().all():
        return {"cluster_id": None}
    return {"cluster_id": int(joined.iloc[0]["cluster_id"])}


# ═══════════════════════════════════════════════════════════════════
# Rule-based classification
# ═══════════════════════════════════════════════════════════════════

def classify_tracts(
    tract_features: gpd.GeoDataFrame,
    econ_classes: dict | None = None,
) -> gpd.GeoDataFrame:
    """Classify each tract into an economic type using threshold rules.

    Args:
        tract_features: GeoDataFrame with renter_median_income and pct_built_pre1990.
        econ_classes: Optional custom classification rules. Defaults to ECON_CLASSES.

    Returns:
        Copy with 'econ_class' column added.
    """
    classes = econ_classes or ECON_CLASSES

    def _classify(row):
        for name, spec in classes.items():
            if spec["rule"](row):
                return name
        return "Unclassified"

    df = tract_features.copy()
    df["econ_class"] = df.apply(_classify, axis=1)

    counts = df["econ_class"].value_counts()
    console.print("\n[bold]Economic classification:[/bold]")
    for cls, n in counts.items():
        console.print(f"  {cls}: {n} tracts")

    return df


# ═══════════════════════════════════════════════════════════════════
# Basemap visualization (works for any metro)
# ═══════════════════════════════════════════════════════════════════

def plot_submarket_map(
    classified: gpd.GeoDataFrame,
    title: str = "Renter Submarkets",
    subtitle: str | None = None,
    isd_gdf: gpd.GeoDataFrame | None = None,
    econ_classes: dict | None = None,
    tile_source=None,
    tile_zoom: int = 11,
    tract_alpha: float = 0.50,
    figsize: tuple[float, float] = (20, 17),
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """Publication-quality submarket map with contextily basemap tiles.

    Roads, city labels, water, and terrain come from the tile provider.
    No manual highway downloads needed. Works for any metro.

    Args:
        classified: GeoDataFrame with 'econ_class' column (from classify_tracts).
        title: Map title.
        subtitle: Optional subtitle text.
        isd_gdf: Optional school district boundaries to overlay.
        econ_classes: Classification dict for colors. Defaults to ECON_CLASSES.
        tile_source: contextily tile provider. Defaults to OpenStreetMap.Mapnik.
        tile_zoom: Tile zoom level (10=metro, 11=county, 12=city).
        tract_alpha: Tract fill transparency (lower = more basemap visible).
        figsize: Figure size in inches.
        save_path: If provided, save to this path.

    Returns:
        matplotlib Figure.
    """
    import contextily as cx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    classes = econ_classes or ECON_CLASSES

    # Reproject to Web Mercator for tile alignment
    tracts_wm = classified.to_crs(epsg=3857)

    # Color mapping
    color_map = {name: spec["color"] for name, spec in classes.items()}
    color_map["Unclassified"] = "#d9d9d9"
    tracts_wm = tracts_wm.copy()
    tracts_wm["_color"] = tracts_wm["econ_class"].map(color_map)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Basemap tiles FIRST (zorder=0) — roads, labels, terrain
    source = tile_source or cx.providers.OpenStreetMap.Mapnik
    cx.add_basemap(ax, source=source, zoom=tile_zoom, zorder=0)

    # Tract fills on top of basemap with transparency
    tracts_wm.plot(
        color=tracts_wm["_color"], edgecolor="none",
        ax=ax, alpha=tract_alpha, zorder=1,
    )

    # ISD boundaries
    if isd_gdf is not None:
        from shapely.geometry import box
        data_bounds = classified.total_bounds
        pad_deg = 0.05
        data_box = box(
            data_bounds[0] - pad_deg, data_bounds[1] - pad_deg,
            data_bounds[2] + pad_deg, data_bounds[3] + pad_deg,
        )
        local_isds = isd_gdf[isd_gdf.geometry.representative_point().within(data_box)]
        if not local_isds.empty:
            local_isds.to_crs(epsg=3857).boundary.plot(
                ax=ax, color="#222", linewidth=1.2, alpha=0.50, zorder=2,
            )

    # Clip to data extent
    xmin, ymin, xmax, ymax = tracts_wm.total_bounds
    pad = 2000
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    # Legend
    legend_order = list(classes.keys())
    patches = [mpatches.Patch(color=color_map[n], label=n, alpha=0.80)
               for n in legend_order]
    legend = ax.legend(
        handles=patches, loc="lower left", fontsize=10, framealpha=0.95,
        title="Renter Economic Classification", title_fontsize=11,
        edgecolor="#999", fancybox=False, borderpad=1.0, labelspacing=0.7,
    )
    legend.get_title().set_fontweight("bold")

    # Title + subtitle
    fig.suptitle(title, fontsize=22, fontweight="bold", color="#1a1a1a", y=0.97)
    sub = subtitle or (
        "Census tract economic classification  |  "
        "ISD boundaries in black  |  Basemap: OpenStreetMap"
    )
    fig.text(0.5, 0.945, sub, ha="center", fontsize=11, color="#555")

    fig.text(
        0.98, 0.015,
        "Data: ACS 5-Year 2023  |  geostack",
        ha="right", fontsize=8, color="#999", style="italic",
    )

    ax.set_axis_off()
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
        console.print(f"[green]Saved → {save_path}[/green]")

    return fig
