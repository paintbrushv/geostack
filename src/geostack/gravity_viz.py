"""Visualization for MSA Population Center of Gravity analysis.

Six canonical visualizations (older heatmap/KDE/Tufte variants were pruned in
favor of these):

- gravity_contour_map  — topographic density surface; distinct labeled sub-centers
- gravity_hero_map     — landmark-anchored static: where weight is, how centers
                         diverge, where it's heading (context + zoomed inset)
- gravity_hero_interactive — Folium companion to the hero map
- divergence_chart     — multi-weight divergence over time (small multiples)
- node_map             — interactive submarket/node map
- node_evolution_chart — submarket weight & distance evolution

Usage:
    from geostack.gravity import multi_weight_time_series
    from geostack.gravity_viz import gravity_contour_map, gravity_hero_map

    gravity_contour_map("Dallas", weight_mode="jobs", display_year=2023,
                        output_path="outputs/dfw_jobs_structure.png")

    combined = multi_weight_time_series("Dallas", years=[2013, 2018, 2024])
    gravity_hero_map(combined, output_path="outputs/dfw_hero.png")
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
from rich.console import Console
from shapely.geometry import LineString, Point

if TYPE_CHECKING:
    import folium
    import matplotlib.figure

console = Console()

import logging
logger = logging.getLogger(__name__)

_TUFTE_FONT = "Helvetica Neue"


def _node_label(row, fallback_id=None):
    """Prefer a node's landmark place name; fall back to its rank label."""
    place = row.get("place", "") if hasattr(row, "get") else ""
    if place:
        return place
    rank = row.get("node_rank", "") if hasattr(row, "get") else ""
    if rank:
        return rank
    return f"Node {(fallback_id or 0) + 1}"


# ═══════════════════════════════════════════════════════════════════════════
# Hero map — landmark-anchored, context + detail, the consolidated answer
#
# Solves the scale problem (a 1-mile shift on an 80-mile metro is invisible)
# with a two-panel design:
#   • Main panel: metro-scale geography, labeled landmarks, population-density
#     backdrop, a scale bar — so the reader knows WHERE the weight is.
#   • Inset panel: zoomed to the center cluster, all weight-mode centers spread
#     apart with direct labels, plus the drift arrow — so DIVERGENCE and
#     HEADING become legible against the nearest landmark.
# ═══════════════════════════════════════════════════════════════════════════

_LANDMARK_MARKERS = {
    "cbd": ("s", "#333", 45),       # square — central business district
    "suburb": ("o", "#888", 22),    # circle — suburb
    "airport": ("^", "#888", 34),   # triangle — airport
}


def _project_lonlat(lons, lats, crs="EPSG:2276"):
    """Project lon/lat arrays to a projected CRS; return (x, y) in CRS units (feet for 2276)."""
    pts = gpd.GeoSeries([Point(lo, la) for lo, la in zip(lons, lats)], crs="EPSG:4326")
    proj = pts.to_crs(crs)
    return proj.x.values, proj.y.values


def _scale_bar(ax, length_mi, x0, y0, units_per_mile, color="#333", label=None):
    """Draw a simple horizontal scale bar at (x0, y0) in projected units."""
    length = length_mi * units_per_mile
    ax.plot([x0, x0 + length], [y0, y0], color=color, linewidth=1.6, zorder=20,
            solid_capstyle="butt")
    # End ticks
    tick = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.008
    for xt in (x0, x0 + length):
        ax.plot([xt, xt], [y0 - tick, y0 + tick], color=color, linewidth=1.2, zorder=20)
    ax.annotate(
        label or f"{length_mi:.0f} mi",
        (x0 + length / 2, y0 + tick * 1.8),
        fontsize=7.5, fontfamily=_TUFTE_FONT, color=color,
        ha="center", va="bottom", zorder=20,
    )


def gravity_hero_map(
    combined_df: pd.DataFrame,
    msa_name: str | None = None,
    density_mode: str = "population",
    display_year: int | None = None,
    output_path: str | pathlib.Path | None = None,
    dpi: int = 300,
    use_crosswalk: bool = True,
) -> "matplotlib.figure.Figure":
    """Consolidated hero map: density + center divergence + drift, landmark-anchored.

    Two panels share a projected CRS (true distances). The main panel gives
    metro context (landmarks, density, scale bar); the inset zooms into the
    center cluster so the spread between weight-mode centers and the drift
    direction are legible.

    Args:
        combined_df: From multi_weight_time_series() — needs all desired weight
            modes and a 'weight_mode' column. Population is used for the drift arrow.
        msa_name: MSA name (auto-detected from combined_df if None).
        density_mode: Which weight drives the density backdrop (default population).
        display_year: Year for the center cluster + density. Defaults to latest.
        output_path: Save if provided.
        dpi: Resolution.
        use_crosswalk: Crosswalk for pre-2020 data.

    Returns:
        Matplotlib Figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        from matplotlib.patches import Rectangle, ConnectionPatch
    except ImportError:
        raise ImportError("matplotlib required.")

    from geostack.gravity import (
        _fetch_county_weights,
        _weight_column,
        fetch_tract_geometries,
        get_landmarks,
        resolve_msa,
    )

    if combined_df.empty:
        raise ValueError("combined_df is empty")

    if msa_name is None:
        msa_name = combined_df.iloc[0].get("msa_name", "MSA")
    msa = resolve_msa(msa_name)
    state_fips = msa["state_fips"]
    counties = msa["counties"]

    modes = combined_df["weight_mode"].unique().tolist()
    all_years = sorted(combined_df["year"].unique())
    if display_year is None:
        display_year = int(all_years[-1])
    first_yr, last_yr = int(all_years[0]), int(all_years[-1])

    proj = msa.get("crs", "EPSG:2276")  # per-MSA feet-based State Plane
    FT_PER_MI = 5280.0

    # ── Density backdrop ──
    console.print(f"[bold]Building hero map:[/bold] {msa['name']} ({display_year})")
    weight_col = _weight_column(density_mode)
    all_t, all_w = [], []
    for cf in counties:
        try:
            all_w.append(_fetch_county_weights(state_fips, cf, display_year, density_mode, use_crosswalk))
            all_t.append(fetch_tract_geometries(state_fips, cf, display_year))
        except Exception:
            pass
    tracts = pd.concat(all_t, ignore_index=True)
    wdf = pd.concat(all_w, ignore_index=True)
    merged = tracts.merge(wdf, on="GEOID", how="inner")
    merged = merged[merged[weight_col] > 0].copy()

    cents = merged.to_crs(proj).geometry.centroid
    dens_x = cents.x.values
    dens_y = cents.y.values
    dens_w = merged[weight_col].values.astype(float)

    # ── Centers for display_year (divergence) ──
    year_centers = combined_df[combined_df["year"] == display_year]
    cen_x, cen_y, cen_mode = {}, {}, {}
    for _, row in year_centers.iterrows():
        mode = row["weight_mode"]
        x, y = _project_lonlat([row["center_lon"]], [row["center_lat"]], proj)
        cen_x[mode] = x[0]
        cen_y[mode] = y[0]

    # ── Population drift path ──
    pop_series = combined_df[combined_df["weight_mode"] == "population"].sort_values("year")
    drift_x, drift_y = _project_lonlat(
        pop_series["center_lon"].values, pop_series["center_lat"].values, proj,
    )

    # ── Landmarks ──
    landmarks = get_landmarks(msa_name)
    lm_x, lm_y = (_project_lonlat(
        [l[2] for l in landmarks], [l[1] for l in landmarks], proj,
    ) if landmarks else (np.array([]), np.array([])))

    # ── Figure: main (left) + inset (right) ──
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    axi = fig.add_subplot(gs[0, 1])

    # County outlines
    merged["_cty"] = merged["GEOID"].str[:5]
    cty = merged.to_crs(proj).dissolve(by="_cty")

    # ─────────────────────── MAIN PANEL ───────────────────────
    cty.boundary.plot(ax=ax, color="#ddd", linewidth=0.5, zorder=1)

    # Density dots (area-proportional, sqrt; light)
    sizes = np.sqrt(dens_w / dens_w.max()) * 38
    ax.scatter(dens_x, dens_y, s=np.clip(sizes, 0.3, 50),
               c="#c9b39a", alpha=0.45, edgecolors="none", zorder=2)

    # Landmarks
    for (name, la, lo, kind), x, y in zip(landmarks, lm_x, lm_y):
        mk, col, sz = _LANDMARK_MARKERS.get(kind, ("o", "#888", 22))
        ax.scatter(x, y, marker=mk, s=sz, c=col, edgecolors="white",
                   linewidths=0.6, zorder=6)
        ax.annotate(
            name, (x, y), fontsize=7.5, fontfamily=_TUFTE_FONT,
            color="#555", ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points", zorder=7,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    # Center cluster location box (links to inset)
    cluster_cx = np.mean(list(cen_x.values()))
    cluster_cy = np.mean(list(cen_y.values()))
    box_half = 3.2 * FT_PER_MI  # 6.4 mi window
    box = Rectangle(
        (cluster_cx - box_half, cluster_cy - box_half),
        box_half * 2, box_half * 2,
        fill=False, edgecolor="#1B3A5C", linewidth=1.2, linestyle="-", zorder=9,
    )
    ax.add_patch(box)

    ax.set_aspect("equal")
    ax.set_axis_off()

    # Scale bar — bottom-left of main
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    _scale_bar(ax, 10, xmin + (xmax - xmin) * 0.05, ymin + (ymax - ymin) * 0.05, FT_PER_MI)

    ax.set_title(
        f"Where {msa['name'].split('-')[0]}'s {density_mode} sits, {display_year}",
        fontsize=11, fontfamily=_TUFTE_FONT, color="#333",
        fontweight="bold", loc="left", pad=8,
    )

    # ─────────────────────── INSET PANEL ───────────────────────
    # Zoom window around the center cluster
    win = 3.2 * FT_PER_MI
    ix0, ix1 = cluster_cx - win, cluster_cx + win
    iy0, iy1 = cluster_cy - win, cluster_cy + win

    # Light density in inset too
    inset_mask = (dens_x >= ix0) & (dens_x <= ix1) & (dens_y >= iy0) & (dens_y <= iy1)
    axi.scatter(dens_x[inset_mask], dens_y[inset_mask],
                s=np.clip(sizes[inset_mask], 1, 60),
                c="#e3d6c2", alpha=0.5, edgecolors="none", zorder=1)

    # Landmarks within window
    for (name, la, lo, kind), x, y in zip(landmarks, lm_x, lm_y):
        if ix0 <= x <= ix1 and iy0 <= y <= iy1:
            mk, col, sz = _LANDMARK_MARKERS.get(kind, ("o", "#888", 22))
            axi.scatter(x, y, marker=mk, s=sz * 1.4, c=col, edgecolors="white",
                        linewidths=0.7, zorder=6)
            axi.annotate(
                name, (x, y), fontsize=8.5, fontfamily=_TUFTE_FONT,
                color="#555", ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            )

    # Population drift arrow (first → last), with year label
    if len(drift_x) >= 2:
        axi.annotate(
            "", xy=(drift_x[-1], drift_y[-1]), xytext=(drift_x[0], drift_y[0]),
            arrowprops=dict(arrowstyle="-|>", color="#1B3A5C", linewidth=2.2,
                            mutation_scale=20, shrinkA=0, shrinkB=4),
            zorder=8,
        )
        # Small marker + label at the drift origin (first year)
        axi.scatter(drift_x[0], drift_y[0], s=24, facecolors="white",
                    edgecolors="#1B3A5C", linewidths=1.2, zorder=8)
        axi.annotate(
            str(first_yr), (drift_x[0], drift_y[0]),
            fontsize=7.5, fontfamily=_TUFTE_FONT, color="#1B3A5C",
            ha="right", va="center", xytext=(-5, 0), textcoords="offset points",
            zorder=8,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    # Weight-mode centers — markers at true position, labels in a de-conflicted
    # right-side column with leader lines (clustered points → stacked labels).
    present = [m for m in modes if m in cen_x]
    for mode in present:
        style = _MODE_STYLES.get(mode, _MODE_STYLES["population"])
        axi.scatter(cen_x[mode], cen_y[mode], s=130, c=style["color"],
                    marker=style["marker"], edgecolors="white",
                    linewidths=1.4, zorder=10)

    # Label column placement: to the right of the cluster, stacked by y-rank
    label_x = ix1 - win * 0.42
    order = sorted(present, key=lambda m: cen_y[m], reverse=True)
    n = len(order)
    span = win * 1.1
    top = cluster_cy + span / 2
    step = span / max(n - 1, 1)
    for i, mode in enumerate(order):
        style = _MODE_STYLES.get(mode, _MODE_STYLES["population"])
        ly = top - i * step
        # Leader line from marker to label
        axi.plot([cen_x[mode], label_x - win * 0.02], [cen_y[mode], ly],
                 color=style["color"], linewidth=0.7, alpha=0.6, zorder=9)
        axi.annotate(
            f"{style['label']} ({display_year})",
            (label_x, ly),
            fontsize=8.5, fontfamily=_TUFTE_FONT, color=style["color"],
            fontweight="bold", ha="left", va="center", zorder=11,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )

    axi.set_xlim(ix0, ix1)
    axi.set_ylim(iy0, iy1)
    axi.set_aspect("equal")
    axi.set_axis_off()
    # Inset border to echo the locator box
    for spine_xy in [(ix0, iy0, ix1 - ix0, iy1 - iy0)]:
        axi.add_patch(Rectangle((spine_xy[0], spine_xy[1]), spine_xy[2], spine_xy[3],
                                 fill=False, edgecolor="#1B3A5C", linewidth=1.2, zorder=12))

    _scale_bar(axi, 1, ix0 + win * 0.12, iy0 + win * 0.12, FT_PER_MI)

    axi.set_title(
        f"How the centers diverge, {display_year}",
        fontsize=11, fontfamily=_TUFTE_FONT, color="#333",
        fontweight="bold", loc="left", pad=8,
    )

    # ── Connector lines from locator box to inset ──
    for corner in [(1, 1), (1, 0)]:
        bx = cluster_cx + box_half * (1 if corner[0] else -1)
        by = cluster_cy + box_half * (1 if corner[1] else -1)
        ix = ix0 if corner[0] else ix0
        iy = iy1 if corner[1] else iy0
        con = ConnectionPatch(
            xyA=(bx, by), coordsA=ax.transData,
            xyB=(ix0, iy if corner[1] else iy0), coordsB=axi.transData,
            color="#1B3A5C", linewidth=0.6, alpha=0.5, linestyle="--",
        )
        fig.add_artist(con)

    # ── Title + takeaway ──
    fig.text(0.02, 0.97, msa["name"], fontsize=15, fontfamily=_TUFTE_FONT,
             fontweight="bold", color="#333", va="top")

    # Build the plain-English takeaway from the data
    takeaway = _hero_takeaway(combined_df, display_year, msa_name)
    fig.text(0.02, 0.045, takeaway, fontsize=9.5, fontfamily=_TUFTE_FONT,
             color="#555", va="bottom", wrap=True)

    fig.text(0.98, 0.02, f"U.S. Census Bureau & LODES, {display_year}",
             fontsize=7, fontfamily=_TUFTE_FONT, color="#aaa", ha="right")

    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        console.print(f"[green]Hero map saved:[/green] {out}")

    return fig


def _hero_takeaway(combined_df, display_year, msa_name) -> str:
    """Compose a one-sentence plain-English takeaway from the centers."""
    from geostack.gravity import _haversine_miles, _bearing_degrees

    yc = combined_df[combined_df["year"] == display_year]
    centers = {r["weight_mode"]: (r["center_lat"], r["center_lon"]) for _, r in yc.iterrows()}
    if "population" not in centers:
        return ""
    plat, plon = centers["population"]

    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

    parts = []
    for mode, label in (("jobs", "jobs"), ("income_mass", "income"), ("workers", "workers")):
        if mode in centers:
            la, lo = centers[mode]
            d = _haversine_miles(plat, plon, la, lo)
            b = _bearing_degrees(plat, plon, la, lo)
            comp = dirs[int((b + 11.25) / 22.5) % 16]
            parts.append(f"{label} sit {d:.1f} mi {comp}")

    # Drift
    pop = combined_df[combined_df["weight_mode"] == "population"].sort_values("year")
    drift = ""
    if len(pop) >= 2 and "shift_miles" in pop.columns:
        first, last = pop.iloc[0], pop.iloc[-1]
        d = last.get("shift_miles", 0)
        b = last.get("shift_bearing_deg", 0)
        comp = dirs[int((b + 11.25) / 22.5) % 16]
        drift = (f"  The population center has drifted {d:.1f} mi {comp} "
                 f"since {int(first['year'])}.")

    if parts:
        return (f"Relative to where people live, {', '.join(parts)}." + drift).strip()
    return drift.strip()


# ---------------------------------------------------------------------------
# Interactive hero map (Folium companion)
# ---------------------------------------------------------------------------

def gravity_hero_interactive(
    combined_df: pd.DataFrame,
    msa_name: str | None = None,
    density_mode: str = "population",
    display_year: int | None = None,
    output_path: str | pathlib.Path | None = None,
    zoom_start: int = 10,
    use_crosswalk: bool = True,
) -> "folium.Map":
    """Interactive companion to gravity_hero_map: landmark-anchored, drill-in.

    Density choropleth backdrop, labeled landmark markers, the weight-mode
    centers for the display year (the divergence), and the population drift
    path. Pan/zoom resolves the scale problem the static map solves with an inset.

    Args:
        combined_df: From multi_weight_time_series().
        msa_name: MSA name (auto-detected if None).
        density_mode: Weight for the choropleth backdrop.
        display_year: Year for centers + density. Defaults to latest.
        output_path: Save HTML if provided.
        zoom_start: Initial zoom.
        use_crosswalk: Crosswalk for pre-2020 data.

    Returns:
        Folium Map.
    """
    try:
        import folium
        from folium.plugins import AntPath
        import branca.colormap as cm
    except ImportError:
        raise ImportError("folium is required. Run: pip install 'geostack[viz]'")

    from geostack.gravity import (
        _fetch_county_weights,
        _weight_column,
        fetch_tract_geometries,
        get_landmarks,
        resolve_msa,
    )
    from geostack.spatial import suggest_crs

    if combined_df.empty:
        raise ValueError("combined_df is empty")

    if msa_name is None:
        msa_name = combined_df.iloc[0].get("msa_name", "MSA")
    msa = resolve_msa(msa_name)
    state_fips = msa["state_fips"]
    counties = msa["counties"]
    weight_col = _weight_column(density_mode)

    modes = combined_df["weight_mode"].unique().tolist()
    all_years = sorted(combined_df["year"].unique())
    if display_year is None:
        display_year = int(all_years[-1])
    first_yr, last_yr = int(all_years[0]), int(all_years[-1])

    # Density backdrop
    console.print(f"[bold]Building interactive hero:[/bold] {msa['name']} ({display_year})")
    all_t, all_w = [], []
    for cf in counties:
        try:
            all_w.append(_fetch_county_weights(state_fips, cf, display_year, density_mode, use_crosswalk))
            all_t.append(fetch_tract_geometries(state_fips, cf, display_year))
        except Exception:
            pass
    tracts = pd.concat(all_t, ignore_index=True)
    wdf = pd.concat(all_w, ignore_index=True)
    merged = tracts.merge(wdf, on="GEOID", how="inner")
    merged = merged[merged[weight_col] > 0].copy()

    # Density per sq mi
    proj_crs = suggest_crs(merged)
    mp = merged.to_crs(proj_crs)
    if "227" in str(proj_crs):
        merged["area_sqmi"] = mp.geometry.area / (5280 ** 2)
    else:
        merged["area_sqmi"] = mp.geometry.area / (1609.344 ** 2)
    merged["density"] = merged[weight_col] / merged["area_sqmi"].replace(0, np.nan)
    merged = merged[merged["density"].notna()].copy()
    merged_wgs = merged.to_crs("EPSG:4326")
    merged_wgs["geometry"] = merged_wgs.geometry.simplify(0.001, preserve_topology=True)

    # Map centered on the center cluster
    yc = combined_df[combined_df["year"] == display_year]
    center_lat = yc["center_lat"].mean()
    center_lon = yc["center_lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start,
                   tiles="CartoDB positron")

    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: rgba(255,255,255,0.94); padding: 9px 18px;
                border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);
                font-family: 'Helvetica Neue', Arial, sans-serif; max-width: 640px;">
        <h3 style="margin: 0; color: #1B3A5C; font-size: 14px;">
            {msa['name']} — Centers of Gravity vs. {density_mode.title()} Density, {display_year}</h3>
        <p style="margin: 2px 0 0; color: #666; font-size: 11px;">
            {_hero_takeaway(combined_df, display_year, msa_name)}</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Choropleth
    vmin = merged["density"].quantile(0.05)
    vmax = merged["density"].quantile(0.95)
    colormap = cm.LinearColormap(
        ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
        vmin=vmin, vmax=vmax, caption=f"{density_mode.title()} per sq mi",
    )
    folium.GeoJson(
        merged_wgs[["GEOID", weight_col, "density", "geometry"]].__geo_interface__,
        style_function=lambda f: {
            "fillColor": colormap(min(max(f["properties"].get("density", vmin), vmin), vmax)),
            "fillOpacity": 0.55, "color": "#ccc", "weight": 0.2,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["GEOID", weight_col, "density"],
            aliases=["Tract:", f"{density_mode}:", "per sq mi:"], localize=True),
        name=f"{density_mode.title()} Density",
    ).add_to(m)
    colormap.add_to(m)

    # Landmarks
    lm_fg = folium.FeatureGroup(name="Landmarks")
    for name, la, lo, kind in get_landmarks(msa_name):
        icon_color = "#333" if kind == "cbd" else "#888"
        folium.CircleMarker(
            [la, lo], radius=4, color=icon_color, fill=True,
            fill_color=icon_color, fill_opacity=0.7, tooltip=name,
        ).add_to(lm_fg)
        folium.Marker(
            [la, lo],
            icon=folium.DivIcon(html=f"""<div style="font-family:'Helvetica Neue';
                font-size:10px;color:#555;font-weight:{'bold' if kind=='cbd' else 'normal'};
                text-shadow:1px 1px 2px white,-1px -1px 2px white;white-space:nowrap;
                transform:translate(8px,-7px);">{name}</div>""",
                icon_size=(0, 0)),
        ).add_to(lm_fg)
    lm_fg.add_to(m)

    # Population drift path
    pop = combined_df[combined_df["weight_mode"] == "population"].sort_values("year")
    drift_coords = [[r["center_lat"], r["center_lon"]] for _, r in pop.iterrows()]
    if len(drift_coords) >= 2:
        AntPath(drift_coords, color="#1B3A5C", weight=3, opacity=0.85,
                delay=1400, dash_array=[12, 24]).add_to(m)

    # Weight-mode centers for display year
    centers_fg = folium.FeatureGroup(name=f"Centers ({display_year})")
    for mode in modes:
        mrow = yc[yc["weight_mode"] == mode]
        if mrow.empty:
            continue
        r = mrow.iloc[0]
        style = _MODE_STYLES.get(mode, _MODE_STYLES["population"])
        div = r.get("dist_from_pop_center_mi", 0)
        div_txt = f"<br>{div:.2f} mi from population center" if mode != "population" and pd.notna(div) else ""
        popup = (f"<div style='font-family:Helvetica Neue;font-size:12px;'>"
                 f"<b>{style['label']} center, {display_year}</b><br>"
                 f"Total: {r.get('total_weight', 0):,.0f}<br>"
                 f"({r['center_lat']:.4f}, {r['center_lon']:.4f}){div_txt}</div>")
        folium.CircleMarker(
            [r["center_lat"], r["center_lon"]], radius=10,
            color="white", weight=2, fill=True, fill_color=style["color"],
            fill_opacity=0.95, popup=folium.Popup(popup, max_width=240),
            tooltip=f"{style['label']} center",
        ).add_to(centers_fg)
        folium.Marker(
            [r["center_lat"], r["center_lon"]],
            icon=folium.DivIcon(html=f"""<div style="font-family:'Helvetica Neue';
                font-size:11px;color:{style['color']};font-weight:bold;
                text-shadow:1px 1px 2px white,-1px -1px 2px white;white-space:nowrap;
                transform:translate(12px,-6px);">{style['label']}</div>""",
                icon_size=(0, 0)),
        ).add_to(centers_fg)
    centers_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out))
        console.print(f"[green]Interactive hero saved:[/green] {out}")

    return m


# ═══════════════════════════════════════════════════════════════════════════
# Contour ("topographic") density map — distinct, labeled nodes + CoG
#
# Replaces the single smeared KDE blob. Filled contours band the density so
# each sub-center reads as its own nested peak; landmark-named labels sit on
# the peaks; the center of gravity and (optional) drift vector overlay it.
# Built to survive an IC review: headline takeaway, growth context, honest
# precision, source/methodology footer.
# ═══════════════════════════════════════════════════════════════════════════

def gravity_contour_map(
    msa_name: str,
    weight_mode: str = "population",
    display_year: int | None = None,
    drift_from_year: int | None = None,
    output_path: str | pathlib.Path | None = None,
    dpi: int = 300,
    use_crosswalk: bool = True,
    label_nodes: bool = True,
    headline: str | None = None,
    basemap_provider: str = "OpenStreetMap.Mapnik",
) -> "matplotlib.figure.Figure":
    """Topographic density map: distinct labeled nodes + center of gravity.

    Args:
        msa_name: MSA name or partial match.
        weight_mode: "population", "jobs", "workers", "income_mass".
        display_year: Year for the surface. Defaults to the latest available.
        drift_from_year: If set, draws a drift arrow for the center of gravity
            from this year to display_year.
        output_path: Save PNG/PDF/SVG if provided.
        dpi: Resolution.
        use_crosswalk: Crosswalk pre-2020 GEOIDs.
        label_nodes: Run node detection and label peaks by nearest landmark.
        headline: Optional headline string; auto-generated if None.

    Returns:
        Matplotlib Figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        from matplotlib.colors import LinearSegmentedColormap
        from scipy.stats import gaussian_kde
        import contextily as ctx
    except ImportError:
        raise ImportError(
            "matplotlib, scipy, contextily required. "
            "Run: pip install 'geostack[print-maps]' scipy"
        )

    from geostack.gravity import (
        _fetch_county_weights,
        _weight_column,
        _available_years_for_mode,
        fetch_tract_geometries,
        get_landmarks,
        resolve_msa,
        population_center_of_gravity,
    )

    msa = resolve_msa(msa_name)
    state_fips = msa["state_fips"]
    counties = msa["counties"]
    weight_col = _weight_column(weight_mode)

    # Resolve latest available year for this mode if not given
    if display_year is None:
        candidate = list(range(2024, 2009, -1))
        avail = _available_years_for_mode(weight_mode, candidate)
        display_year = avail[0] if avail else 2023

    weight_label = {
        "population": "Population", "jobs": "Jobs",
        "workers": "Resident workers", "income_mass": "Income mass",
    }.get(weight_mode, weight_mode)

    console.print(f"[bold]Building contour map:[/bold] {msa['name']} ({weight_mode}, {display_year})")

    # ── Fetch tract weights ──
    all_t, all_w = [], []
    for cf in counties:
        try:
            all_w.append(_fetch_county_weights(state_fips, cf, display_year, weight_mode, use_crosswalk))
            all_t.append(fetch_tract_geometries(state_fips, cf, display_year))
        except Exception:
            pass
    tracts = pd.concat(all_t, ignore_index=True)
    wdf = pd.concat(all_w, ignore_index=True)
    merged = tracts.merge(wdf, on="GEOID", how="inner")
    merged = merged[merged[weight_col] > 0].copy()

    merged_web = merged.to_crs("EPSG:3857")
    cents = merged_web.geometry.centroid
    cx, cy = cents.x.values, cents.y.values
    w = merged[weight_col].values.astype(float)

    # ── Weighted KDE ──
    scale = np.clip(w / w.max() * 220, 1, 600).astype(int)
    cx_e = np.repeat(cx, scale)
    cy_e = np.repeat(cy, scale)
    kde = gaussian_kde(np.vstack([cx_e, cy_e]), bw_method=0.14)

    pad = 0.06
    xr, yr = cx.max() - cx.min(), cy.max() - cy.min()
    xmin, xmax = cx.min() - xr * pad, cx.max() + xr * pad
    ymin, ymax = cy.min() - yr * pad, cy.max() + yr * pad
    gx = np.linspace(xmin, xmax, 420)
    gy = np.linspace(ymin, ymax, 420)
    GX, GY = np.meshgrid(gx, gy)
    Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    Z = Z / Z.max()  # normalize 0..1

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(15, 11), facecolor="white")

    # Contour levels focused on the peaks (lower band transparent)
    levels = [0.06, 0.12, 0.20, 0.32, 0.46, 0.62, 0.80, 1.0]
    heat = LinearSegmentedColormap.from_list("heat", [
        "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
        "#fc4e2a", "#e31a1c", "#bd0026", "#800026",
    ], N=256)

    cf = ax.contourf(GX, GY, Z, levels=levels, cmap=heat, alpha=0.62,
                     extend="max", zorder=2)
    # Crisp contour lines so each node reads as nested rings
    ax.contour(GX, GY, Z, levels=levels, colors="#7a0a16",
               linewidths=0.5, alpha=0.55, zorder=3)

    # County outlines
    merged["_cty"] = merged["GEOID"].str[:5]
    merged.to_crs("EPSG:3857").dissolve(by="_cty").boundary.plot(
        ax=ax, color="#bbb", linewidth=0.5, zorder=1)

    # Basemap (suppress built-in attribution; we add our own source line)
    _prov = ctx.providers
    for _part in basemap_provider.split("."):
        _prov = getattr(_prov, _part)
    ctx.add_basemap(ax, source=_prov, zoom=10, zorder=0, attribution=False)

    # ── True landmark reference points (plotted at real coordinates) ──
    # Earlier versions plotted detected-node *centroids* labeled by nearest
    # landmark, which put "Downtown Dallas" 1–6 mi from downtown. We now plot
    # the landmark registry at true coords so every label sits on its place;
    # the contour surface shows the density and the viewer maps peaks → places.
    from geostack.gravity import get_landmarks
    landmarks = get_landmarks(msa_name)
    if landmarks:
        lx, ly = _project_lonlat([l[2] for l in landmarks], [l[1] for l in landmarks], "EPSG:3857")
        # Greedy thinning: always keep CBD/airport; for suburbs, drop the whole
        # marker (dot AND label together — never a naked dot) when an already-
        # placed label is within min_sep. De-clutters the core while keeping
        # peripheral centers (Southlake, Alliance) labeled.
        min_sep = (xmax - xmin) * 0.05  # ~ a few miles in display units
        order = sorted(range(len(landmarks)),
                       key=lambda i: 0 if landmarks[i][3] in ("cbd", "airport") else 1)
        placed = []
        for i in order:
            name, la, lo, kind = landmarks[i]
            x, y = lx[i], ly[i]
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue
            crowded = any((x - px) ** 2 + (y - py) ** 2 < min_sep ** 2 for px, py in placed)
            if kind not in ("cbd", "airport") and crowded:
                continue  # skip entirely — no orphan dots
            placed.append((x, y))
            mk = "s" if kind == "cbd" else ("^" if kind == "airport" else "o")
            sz = 36 if kind == "cbd" else (26 if kind == "airport" else 18)
            ax.scatter(x, y, marker=mk, s=sz, c="#222", edgecolors="white",
                       linewidths=0.7, zorder=10)
            ax.annotate(
                name, (x, y),
                fontsize=8.5 if kind == "cbd" else 7.5,
                fontfamily=_TUFTE_FONT, color="#1a1a1a",
                fontweight="bold" if kind == "cbd" else "normal",
                ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points", zorder=11,
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
            )

    # ── Honest sub-center share summary (for headline + footer) ──
    # Names come from detect_nodes' nearest-landmark assignment, kept only when
    # the cluster centroid is within a sanity cap of the named place.
    place_summary = []
    if label_nodes:
        try:
            from geostack.gravity_polycentric import detect_nodes
            from geostack.gravity import _haversine_miles
            lm_coords = {n: (la, lo) for n, la, lo, k in landmarks}
            nodes = detect_nodes(msa_name, year=display_year, weight_mode=weight_mode,
                                 use_crosswalk=use_crosswalk)
            if not nodes.empty:
                total = nodes["total_weight"].sum()
                for _, node in nodes.iterrows():
                    place = node.get("place", "")
                    if not place or place not in lm_coords:
                        continue
                    la, lo = lm_coords[place]
                    if _haversine_miles(node["center_lat"], node["center_lon"], la, lo) > 4.0:
                        continue  # name not trustworthy — skip from the summary
                    share = node["total_weight"] / total * 100 if total else 0
                    place_summary.append((place, node["total_weight"], share))
                    if len(place_summary) >= 5:
                        break
        except Exception as e:
            logger.warning("Sub-center summary skipped: %s", e)

    # ── Center of gravity + optional drift ──
    cog = population_center_of_gravity(msa_name, year=display_year,
                                       weight_mode=weight_mode, use_crosswalk=use_crosswalk)
    gx0, gy0 = _project_lonlat([cog.center_lon], [cog.center_lat], "EPSG:3857")

    if drift_from_year is not None:
        try:
            cog0 = population_center_of_gravity(msa_name, year=drift_from_year,
                                                weight_mode=weight_mode, use_crosswalk=use_crosswalk)
            dx0, dy0 = _project_lonlat([cog0.center_lon], [cog0.center_lat], "EPSG:3857")
            ax.annotate("", xy=(gx0[0], gy0[0]), xytext=(dx0[0], dy0[0]),
                        arrowprops=dict(arrowstyle="-|>", color="#08306b",
                                        linewidth=2.4, mutation_scale=20,
                                        shrinkA=0, shrinkB=4), zorder=12)
            ax.scatter(dx0[0], dy0[0], s=30, facecolors="white",
                       edgecolors="#08306b", linewidths=1.4, zorder=12)
            # Drift-origin year to the LEFT of its dot (CoG label goes right → no clash)
            ax.annotate(f"{drift_from_year} center", (dx0[0], dy0[0]), fontsize=7.5,
                        fontfamily=_TUFTE_FONT, color="#08306b", ha="right", va="center",
                        xytext=(-7, 0), textcoords="offset points", zorder=12,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        except Exception:
            pass

    # Center-of-gravity star, short label below-right (clear of drift label
    # which sits left, and of landmark labels which sit above their dots).
    ax.scatter(gx0[0], gy0[0], s=240, marker="*", c="#08306b",
               edgecolors="white", linewidths=1.4, zorder=14)
    ax.annotate(
        f"center of gravity {display_year}",
        (gx0[0], gy0[0]),
        fontsize=9, fontfamily=_TUFTE_FONT, color="#08306b", fontweight="bold",
        ha="left", va="top", xytext=(8, -6), textcoords="offset points", zorder=14,
        path_effects=[pe.withStroke(linewidth=2.6, foreground="white")],
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Scale bar (web-mercator metres at this latitude → miles, corrected)
    import math
    lat0 = cog.center_lat
    m_per_mi = 1609.344 / math.cos(math.radians(lat0))  # mercator stretch correction
    _scale_bar(ax, 5, xmin + xr * 0.04, ymin + yr * 0.05, m_per_mi, color="#444")

    # ── Headline + takeaway ──
    if headline is None:
        if place_summary:
            top = place_summary[0]
            headline = (f"{msa['name'].split('-')[0]}'s {weight_label.lower()} concentrates near "
                        f"{top[0]} ({top[2]:.0f}% of the dense core)")
        else:
            headline = f"{msa['name']} {weight_label.lower()} structure, {display_year}"

    fig.text(0.02, 0.975, headline, fontsize=15, fontfamily=_TUFTE_FONT,
             fontweight="bold", color="#222", va="top")
    fig.text(0.02, 0.945,
             f"{weight_label} density by census tract, {display_year} — "
             f"distinct peaks are the metro's sub-centers; black markers are reference landmarks",
             fontsize=10, fontfamily=_TUFTE_FONT, color="#777", va="top")

    # Node share strip (IC context)
    if place_summary:
        strip = "   ".join(
            f"{p}: {wv:,.0f} ({sh:.0f}%)" for p, wv, sh in place_summary
        )
        fig.text(0.02, 0.045, f"Largest dense sub-centers — {strip}",
                 fontsize=8.5, fontfamily=_TUFTE_FONT, color="#555", va="bottom")

    src = "U.S. Census ACS 5-Year" if weight_mode in ("population", "income_mass") else "U.S. Census LEHD LODES8"
    caveat = ""
    if weight_mode in ("jobs", "workers"):
        caveat = (" Density-based detection under-represents low-footprint-per-acre "
                  "industrial/logistics hubs (e.g. AllianceTexas): real jobs, low jobs/sq-mi.")
    fig.text(0.98, 0.02,
             f"Source: {src}, {display_year}. Density via weighted KDE on tract centroids; "
             f"sub-centers via density peaks.{caveat} Estimates carry survey margins of error.",
             fontsize=6.5, fontfamily=_TUFTE_FONT, color="#aaa", ha="right", va="bottom")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        console.print(f"[green]Contour map saved:[/green] {out}")

    return fig


def gravity_contour_panels(
    msa_name: str,
    weight_mode: str = "population",
    years: list[int] | None = None,
    output_path: str | pathlib.Path | None = None,
    dpi: int = 220,
    use_crosswalk: bool = True,
) -> "matplotlib.figure.Figure":
    """Small-multiples contour comparison: one panel per year, shared scales.

    A single contour map can't show a ~1-mile center shift; this renders N
    years side by side with identical extent and a fixed base-year reference
    marker on every panel, so the center of gravity visibly pulls away from it.

    Args:
        msa_name: MSA name or partial match.
        weight_mode: "population", "jobs", "workers", "income_mass".
        years: Years to compare (2-4 recommended). Defaults by mode.
        output_path: Save if provided.
        dpi: Resolution.
        use_crosswalk: Normalize pre-2020 GEOIDs.

    Returns:
        Matplotlib Figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        from matplotlib.colors import LinearSegmentedColormap
        from scipy.stats import gaussian_kde
        import contextily as ctx
    except ImportError:
        raise ImportError("matplotlib, scipy, contextily required. pip install 'geostack[gravity]'")

    from geostack.gravity import (
        _fetch_county_weights, _weight_column, fetch_tract_geometries,
        get_landmarks, resolve_msa, population_center_of_gravity, _haversine_miles,
    )

    msa = resolve_msa(msa_name)
    state_fips, counties = msa["state_fips"], msa["counties"]
    weight_col = _weight_column(weight_mode)

    if years is None:
        # Earliest available year per series → latest, for the longest view of
        # each. Ranges differ by source: population reaches 2000 (Decennial),
        # LODES jobs/workers start 2002, ACS income starts 2009. Pass explicit
        # years for a common comparable window (e.g. [2013, 2018, 2023]).
        years = {
            "population": [2000, 2012, 2024],
            "jobs": [2002, 2013, 2023],
            "workers": [2002, 2013, 2023],
            "income_mass": [2009, 2016, 2024],
        }.get(weight_mode, [2010, 2024])

    wlabel = {"population": "Population", "jobs": "Jobs",
              "workers": "Resident workers", "income_mass": "Income mass"}.get(weight_mode, weight_mode)

    console.print(f"[bold]Building contour panels:[/bold] {msa['name']} {weight_mode} {years}")

    # Tract geometry once (2020 vintage); only weights change across years
    geoms = []
    for cf in counties:
        try:
            geoms.append(fetch_tract_geometries(state_fips, cf, max(years)))
        except Exception:
            pass
    tracts = pd.concat(geoms, ignore_index=True).to_crs("EPSG:3857")
    cent = tracts.geometry.centroid
    tracts["cx"], tracts["cy"] = cent.x.values, cent.y.values
    tracts["area_sqmi"] = tracts.geometry.area / (5280 ** 2)

    # Shared grid extent
    cx, cy = tracts["cx"].values, tracts["cy"].values
    pad = 0.06
    xr, yr = cx.max() - cx.min(), cy.max() - cy.min()
    xmin, xmax = cx.min() - xr * pad, cx.max() + xr * pad
    ymin, ymax = cy.min() - yr * pad, cy.max() + yr * pad
    gx = np.linspace(xmin, xmax, 320)
    gy = np.linspace(ymin, ymax, 320)
    GX, GY = np.meshgrid(gx, gy)

    # Per-year density grid (each normalized to its own max for clean shapes)
    grids, centers = {}, {}
    for y in years:
        wf = []
        for cf in counties:
            try:
                wf.append(_fetch_county_weights(state_fips, cf, y, weight_mode, use_crosswalk))
            except Exception:
                pass
        if not wf:
            continue
        wdf = pd.concat(wf, ignore_index=True)
        m = tracts.merge(wdf, on="GEOID", how="inner")
        m = m[m[weight_col] > 0]
        if m.empty:
            continue
        reps = np.clip(m[weight_col].values / m[weight_col].max() * 200, 1, 500).astype(int)
        ex = np.repeat(m["cx"].values, reps)
        ey = np.repeat(m["cy"].values, reps)
        Z = gaussian_kde(np.vstack([ex, ey]), bw_method=0.13)(
            np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        grids[y] = Z / Z.max()
        try:
            c = population_center_of_gravity(msa_name, year=y, weight_mode=weight_mode,
                                             use_crosswalk=use_crosswalk)
            cxp, cyp = _project_lonlat([c.center_lon], [c.center_lat], "EPSG:3857")
            centers[y] = (cxp[0], cyp[0], c.center_lat, c.center_lon)
        except Exception:
            pass

    valid_years = [y for y in years if y in grids and y in centers]
    base_y = valid_years[0]
    bx, by, blat, blon = centers[base_y]

    levels = [0.06, 0.12, 0.20, 0.32, 0.46, 0.62, 0.80, 1.0]
    heat = LinearSegmentedColormap.from_list("heat", [
        "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
        "#fc4e2a", "#e31a1c", "#bd0026", "#800026"], N=256)

    n = len(valid_years)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 6.2), facecolor="white")
    if n == 1:
        axes = [axes]

    cbds = [(nm, la, lo) for nm, la, lo, k in get_landmarks(msa_name) if k == "cbd"]
    clx, cly = (_project_lonlat([c[2] for c in cbds], [c[1] for c in cbds], "EPSG:3857")
                if cbds else ([], []))

    for ax, y in zip(axes, valid_years):
        ax.contourf(GX, GY, grids[y], levels=levels, cmap=heat, alpha=0.62, extend="max", zorder=2)
        ax.contour(GX, GY, grids[y], levels=levels, colors="#7a0a16", linewidths=0.4, alpha=0.5, zorder=3)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom=9,
                        zorder=0, attribution=False)
        # CBD anchors
        for (nm, la, lo), x, yy in zip(cbds, clx, cly):
            ax.scatter(x, yy, marker="s", s=22, c="#222", edgecolors="white", linewidths=0.6, zorder=9)
            ax.annotate(nm, (x, yy), fontsize=6.5, fontfamily=_TUFTE_FONT, color="#333",
                        ha="center", va="bottom", xytext=(0, 4), textcoords="offset points",
                        zorder=9, path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
        # Fixed base-year reference (hollow) on every panel
        ax.scatter(bx, by, s=110, facecolors="none", edgecolors="#08306b",
                   linewidths=1.4, zorder=11)
        # Faint trail of all centers
        tx = [centers[yy][0] for yy in valid_years]
        ty = [centers[yy][1] for yy in valid_years]
        ax.plot(tx, ty, color="#08306b", linewidth=0.8, alpha=0.4, zorder=10)
        # This year's star
        cxp, cyp, clat, clon = centers[y]
        ax.scatter(cxp, cyp, s=240, marker="*", c="#08306b", edgecolors="white",
                   linewidths=1.2, zorder=13)
        drift = _haversine_miles(blat, blon, clat, clon)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_axis_off()
        sub = f"{y}" if y == base_y else f"{y}  ·  {drift:.2f} mi from {base_y}"
        ax.set_title(sub, fontsize=11, fontfamily=_TUFTE_FONT, color="#333",
                     fontweight="bold", pad=6)

    # Scale bar on first panel
    import math
    m_per_mi = 1609.344 / math.cos(math.radians(blat))
    _scale_bar(axes[0], 10, xmin + xr * 0.05, ymin + yr * 0.04, m_per_mi, color="#444")

    fig.suptitle(
        f"{msa['name']} — {wlabel} density & center of gravity, {valid_years[0]}–{valid_years[-1]}",
        fontsize=14, fontfamily=_TUFTE_FONT, fontweight="bold", color="#222", y=0.99)
    fig.text(0.5, 0.045,
             f"Hollow ring = {base_y} center (fixed reference on every panel); star = that year's center. "
             f"Same extent and per-year contour scaling across panels.",
             fontsize=8, fontfamily=_TUFTE_FONT, color="#777", ha="center")

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        console.print(f"[green]Contour panels saved:[/green] {out}")

    return fig


# ---------------------------------------------------------------------------
# Multi-weight comparison styles
# ---------------------------------------------------------------------------

_MODE_STYLES = {
    "population": {
        "color": "#1B3A5C",
        "marker": "o",
        "label": "Population",
        "dash": None,
    },
    "jobs": {
        "color": "#D32F2F",
        "marker": "s",
        "label": "Workplace Jobs",
        "dash": "8 4",
    },
    "workers": {
        "color": "#4CAF50",
        "marker": "^",
        "label": "Resident Workers",
        "dash": "6 3 2 3",
    },
    "income_mass": {
        "color": "#F57C00",
        "marker": "D",
        "label": "Income Mass",
        "dash": "4 4",
    },
}


# ---------------------------------------------------------------------------
# Divergence chart (pop center vs other centers over time)
# ---------------------------------------------------------------------------

def divergence_chart(
    combined_df: pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    dpi: int = 200,
) -> "matplotlib.figure.Figure":
    """Chart showing how non-population centers diverge from the population center.

    Produces a 2x2 panel:
    - Top-left: spatial scatter of all center paths
    - Top-right: distance of each center from population center over time
    - Bottom-left: per-mode shift from own base year
    - Bottom-right: summary table with all modes

    Args:
        combined_df: DataFrame from multi_weight_time_series().
        output_path: If provided, save PNG to this path.
        dpi: Resolution.

    Returns:
        Matplotlib Figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required. Run: pip install 'geostack[print-maps]'")

    if combined_df.empty:
        raise ValueError("combined_df is empty")

    modes = combined_df["weight_mode"].unique().tolist()
    msa_name = combined_df.iloc[0].get("msa_name", "MSA")
    all_years = sorted(combined_df["year"].unique())
    first_year, last_year = int(all_years[0]), int(all_years[-1])

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f"{msa_name}\nMulti-Weight Centers of Gravity, {first_year}–{last_year}",
        fontsize=14, fontweight="bold", color="#1B3A5C", y=0.97,
    )

    # --- Panel 1: Spatial scatter of all paths ---
    ax1 = axes[0, 0]
    for mode in modes:
        style = _MODE_STYLES.get(mode, _MODE_STYLES["population"])
        mode_df = combined_df[combined_df["weight_mode"] == mode].sort_values("year")
        lons = mode_df["center_lon"].values
        lats = mode_df["center_lat"].values
        ax1.plot(lons, lats, marker=style["marker"], color=style["color"],
                 linewidth=1.5, markersize=5, alpha=0.8, label=style["label"])
        # Label first and last
        if len(lons) > 0:
            yrs = mode_df["year"].values
            ax1.annotate(str(int(yrs[0])), (lons[0], lats[0]),
                         fontsize=7, color=style["color"], ha="right")
            if len(lons) > 1:
                ax1.annotate(str(int(yrs[-1])), (lons[-1], lats[-1]),
                             fontsize=7, color=style["color"], ha="left")

    ax1.legend(fontsize=8, loc="best")
    ax1.set_xlabel("Longitude", fontsize=10)
    ax1.set_ylabel("Latitude", fontsize=10)
    ax1.set_title("Spatial Paths by Weight Mode", fontsize=11, color="#333")
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(useOffset=False)

    # --- Panel 2: Distance from population center ---
    ax2 = axes[0, 1]
    for mode in modes:
        if mode == "population":
            continue
        style = _MODE_STYLES.get(mode, _MODE_STYLES["population"])
        mode_df = combined_df[combined_df["weight_mode"] == mode].sort_values("year")
        if "dist_from_pop_center_mi" in mode_df.columns:
            years_arr = mode_df["year"].values
            dists = mode_df["dist_from_pop_center_mi"].values
            ax2.plot(years_arr, dists, marker=style["marker"], color=style["color"],
                     linewidth=2, markersize=6, label=style["label"])

    ax2.legend(fontsize=9)
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel("Distance from Pop Center (miles)", fontsize=10)
    ax2.set_title("Divergence from Population Center", fontsize=11, color="#333")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(first_year - 1, last_year + 1)

    # --- Panel 3: Each mode's own shift from base year ---
    ax3 = axes[1, 0]
    for mode in modes:
        style = _MODE_STYLES.get(mode, _MODE_STYLES["population"])
        mode_df = combined_df[combined_df["weight_mode"] == mode].sort_values("year")
        if "shift_miles" in mode_df.columns:
            ax3.plot(mode_df["year"].values, mode_df["shift_miles"].values,
                     marker=style["marker"], color=style["color"],
                     linewidth=2, markersize=5, label=style["label"])

    ax3.legend(fontsize=9)
    ax3.set_xlabel("Year", fontsize=10)
    ax3.set_ylabel("Shift from Own Base (miles)", fontsize=10)
    ax3.set_title("Shift from Base Year (Each Mode)", fontsize=11, color="#333")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(first_year - 1, last_year + 1)

    # --- Panel 4: Summary table ---
    ax4 = axes[1, 1]
    ax4.axis("off")

    table_data = []
    for mode in modes:
        style = _MODE_STYLES.get(mode, _MODE_STYLES["population"])
        mode_df = combined_df[combined_df["weight_mode"] == mode].sort_values("year")
        if mode_df.empty:
            continue
        first = mode_df.iloc[0]
        last = mode_df.iloc[-1]
        net_shift = last.get("shift_miles", 0)
        div_dist = last.get("dist_from_pop_center_mi", 0)
        if pd.isna(div_dist):
            div_dist = 0

        table_data.append([
            style["label"],
            f"{int(first['year'])}–{int(last['year'])}",
            f"({last['center_lat']:.4f}, {last['center_lon']:.4f})",
            f"{net_shift:.2f} mi",
            f"{div_dist:.2f} mi" if mode != "population" else "—",
        ])

    col_labels = ["Mode", "Period", "Latest Center", "Net Shift", "Div. from Pop"]
    if table_data:
        table = ax4.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.6)
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor("#1B3A5C")
            cell.set_text_props(color="white", fontweight="bold")
        # Color-code mode names
        for i, mode in enumerate(modes):
            if i < len(table_data):
                style = _MODE_STYLES.get(mode, _MODE_STYLES["population"])
                table[i + 1, 0].set_text_props(color=style["color"], fontweight="bold")

    ax4.set_title("Comparison Summary", fontsize=11, color="#333", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        console.print(f"[green]Divergence chart saved:[/green] {out}")

    return fig


# ---------------------------------------------------------------------------
# Polycentric node map
# ---------------------------------------------------------------------------

_NODE_COLORS = [
    "#D32F2F", "#1B3A5C", "#4CAF50", "#F57C00", "#7B1FA2",
    "#00838F", "#C62828", "#283593", "#2E7D32", "#E65100",
    "#6A1B9A", "#00695C",
]


def node_map(
    node_df: pd.DataFrame,
    msa_center: tuple[float, float] | None = None,
    ua_gdf: gpd.GeoDataFrame | None = None,
    output_path: str | pathlib.Path | None = None,
    title: str | None = None,
    zoom_start: int = 10,
) -> "folium.Map":
    """Interactive map showing detected sub-center nodes and their evolution.

    Args:
        node_df: DataFrame from node_time_series() with node_id, year,
                 center_lat, center_lon, total_weight.
        msa_center: (lat, lon) of the overall MSA gravity center.
        ua_gdf: Optional Urbanized Area boundary for overlay.
        output_path: Save HTML if provided.
        title: Map title.
        zoom_start: Initial zoom level.

    Returns:
        Folium Map object.
    """
    try:
        import folium
        from folium.plugins import AntPath
    except ImportError:
        raise ImportError("folium is required. Run: pip install 'geostack[viz]'")

    if node_df.empty:
        raise ValueError("node_df is empty")

    center_lat = node_df["center_lat"].mean()
    center_lon = node_df["center_lon"].mean()
    years = sorted(node_df["year"].unique())
    first_yr, last_yr = int(years[0]), int(years[-1])

    if title is None:
        msa_name = node_df.iloc[0].get("msa_name", "MSA")
        weight_mode = node_df.iloc[0].get("weight_mode", "jobs")
        title = f"Polycentric Nodes — {weight_mode.title()}-Weighted, {first_yr}–{last_yr}"

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )

    # Title
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background: rgba(255,255,255,0.92); padding: 10px 20px;
                border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);
                font-family: 'Helvetica Neue', Arial, sans-serif; max-width: 600px;">
        <h3 style="margin: 0; color: #1B3A5C; font-size: 15px;">{title}</h3>
        <p style="margin: 2px 0 0; color: #666; font-size: 11px;">
            DBSCAN clustering on high-density tracts &bull; {len(years)} periods
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Optional UA boundary overlay
    if ua_gdf is not None and not ua_gdf.empty:
        folium.GeoJson(
            ua_gdf.to_crs("EPSG:4326").__geo_interface__,
            style_function=lambda _: {
                "fillColor": "#E3F2FD",
                "fillOpacity": 0.15,
                "color": "#1565C0",
                "weight": 2,
                "dashArray": "6 3",
            },
            name="Urbanized Area",
        ).add_to(m)

    # MSA overall center
    if msa_center:
        folium.CircleMarker(
            location=list(msa_center),
            radius=12,
            color="#333",
            fill=True,
            fill_color="#333",
            fill_opacity=0.6,
            tooltip="MSA Population Center",
        ).add_to(m)
        folium.Marker(
            location=[msa_center[0] + 0.01, msa_center[1]],
            icon=folium.DivIcon(
                html="""<div style="font-family: 'Helvetica Neue'; font-size: 11px;
                         font-weight: bold; color: #333;
                         text-shadow: 1px 1px 2px white;">MSA Center</div>""",
                icon_size=(70, 15), icon_anchor=(35, 7),
            ),
        ).add_to(m)

    # Draw each node's path over time
    node_ids = sorted(node_df["node_id"].unique())
    for node_id in node_ids:
        color = _NODE_COLORS[int(node_id) % len(_NODE_COLORS)]
        ndf = node_df[node_df["node_id"] == node_id].sort_values("year")
        if ndf.empty:
            continue

        rank = _node_label(ndf.iloc[0], node_id)
        fg = folium.FeatureGroup(name=rank)

        # Path
        path_coords = [
            [row["center_lat"], row["center_lon"]]
            for _, row in ndf.iterrows()
        ]
        if len(path_coords) >= 2:
            folium.PolyLine(
                locations=path_coords,
                color=color,
                weight=2.5,
                opacity=0.7,
            ).add_to(fg)

        # Markers per year
        for i, (_, row) in enumerate(ndf.iterrows()):
            yr = int(row["year"])
            lat, lon = row["center_lat"], row["center_lon"]
            tw = row.get("total_weight", 0)
            is_last = i == len(ndf) - 1

            radius = 9 if is_last else 5
            opacity = 0.9 if is_last else 0.6

            dist_info = ""
            if pd.notna(row.get("dist_from_msa_center_mi")):
                dist_info = (
                    f"<br>Distance from MSA center: "
                    f"{row['dist_from_msa_center_mi']:.2f} mi"
                )

            popup = (
                f"<div style='font-family: Helvetica Neue; font-size: 12px;'>"
                f"<b>{rank} — {yr}</b><br>"
                f"Weight: {tw:,.0f}<br>"
                f"Tracts: {int(row.get('tract_count', 0))}<br>"
                f"Density: {row.get('mean_density', 0):,.0f}/sq mi"
                f"{dist_info}</div>"
            )

            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=opacity,
                popup=folium.Popup(popup, max_width=250),
                tooltip=f"{rank} ({yr})",
            ).add_to(fg)

            if is_last:
                folium.Marker(
                    location=[lat + 0.008, lon],
                    icon=folium.DivIcon(
                        html=f"""<div style="font-family: 'Helvetica Neue';
                                 font-size: 11px; font-weight: bold;
                                 color: {color};
                                 text-shadow: 1px 1px 2px white, -1px -1px 2px white;
                                 white-space: nowrap;">{rank}</div>""",
                        icon_size=(60, 15), icon_anchor=(30, 7),
                    ),
                ).add_to(fg)

        fg.add_to(m)

    # Legend
    legend_items = "".join(
        f'<span style="color: {_NODE_COLORS[int(nid) % len(_NODE_COLORS)]};">●</span> '
        f'{_node_label(node_df[node_df["node_id"] == nid].iloc[0], nid)}<br>'
        for nid in node_ids
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; right: 20px; z-index: 1000;
                background: rgba(255,255,255,0.92); padding: 12px 16px;
                border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);
                font-family: 'Helvetica Neue'; font-size: 11px;">
        <b style="color: #1B3A5C;">Sub-Centers</b><br>
        <span style="color: #333;">●</span> MSA Center<br>
        {legend_items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out))
        console.print(f"[green]Node map saved:[/green] {out}")

    return m


# ---------------------------------------------------------------------------
# Node evolution chart
# ---------------------------------------------------------------------------

def node_evolution_chart(
    node_df: pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    dpi: int = 200,
) -> "matplotlib.figure.Figure":
    """Chart showing how each detected node evolves over time.

    Produces a 2x2 panel:
    - Top-left: spatial scatter of all node paths
    - Top-right: each node's distance from MSA center over time
    - Bottom-left: each node's total weight over time
    - Bottom-right: summary table

    Args:
        node_df: DataFrame from node_time_series().
        output_path: Save PNG if provided.
        dpi: Resolution.

    Returns:
        Matplotlib Figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required.")

    if node_df.empty:
        raise ValueError("node_df is empty")

    node_ids = sorted(node_df["node_id"].unique())
    years = sorted(node_df["year"].unique())
    first_yr, last_yr = int(years[0]), int(years[-1])
    weight_mode = node_df.iloc[0].get("weight_mode", "")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f"Polycentric Node Evolution ({weight_mode.title()}-Weighted), "
        f"{first_yr}–{last_yr}",
        fontsize=14, fontweight="bold", color="#1B3A5C", y=0.97,
    )

    # --- Panel 1: Spatial paths ---
    ax1 = axes[0, 0]
    for nid in node_ids:
        color = _NODE_COLORS[int(nid) % len(_NODE_COLORS)]
        ndf = node_df[node_df["node_id"] == nid].sort_values("year")
        rank = _node_label(ndf.iloc[0], nid)
        ax1.plot(ndf["center_lon"], ndf["center_lat"], "o-",
                 color=color, linewidth=1.5, markersize=5, label=rank)
        if len(ndf) > 0:
            ax1.annotate(rank, (ndf.iloc[-1]["center_lon"], ndf.iloc[-1]["center_lat"]),
                         fontsize=7, color=color, ha="left")

    ax1.legend(fontsize=7, loc="best")
    ax1.set_xlabel("Longitude", fontsize=10)
    ax1.set_ylabel("Latitude", fontsize=10)
    ax1.set_title("Node Spatial Paths", fontsize=11, color="#333")
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(useOffset=False)

    # --- Panel 2: Distance from MSA center ---
    ax2 = axes[0, 1]
    if "dist_from_msa_center_mi" in node_df.columns:
        for nid in node_ids:
            color = _NODE_COLORS[int(nid) % len(_NODE_COLORS)]
            ndf = node_df[node_df["node_id"] == nid].sort_values("year")
            rank = _node_label(ndf.iloc[0], nid)
            ax2.plot(ndf["year"], ndf["dist_from_msa_center_mi"], "o-",
                     color=color, linewidth=2, markersize=5, label=rank)
    ax2.legend(fontsize=8)
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel("Distance from MSA Center (mi)", fontsize=10)
    ax2.set_title("Node Distance from MSA Center", fontsize=11, color="#333")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Weight over time ---
    ax3 = axes[1, 0]
    for nid in node_ids:
        color = _NODE_COLORS[int(nid) % len(_NODE_COLORS)]
        ndf = node_df[node_df["node_id"] == nid].sort_values("year")
        rank = _node_label(ndf.iloc[0], nid)
        ax3.plot(ndf["year"], ndf["total_weight"] / 1000, "o-",
                 color=color, linewidth=2, markersize=5, label=rank)
    ax3.legend(fontsize=8)
    ax3.set_xlabel("Year", fontsize=10)
    ax3.set_ylabel(f"Total {weight_mode.title()} (thousands)", fontsize=10)
    ax3.set_title(f"Node Weight Over Time", fontsize=11, color="#333")
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Summary table ---
    ax4 = axes[1, 1]
    ax4.axis("off")

    table_data = []
    for nid in node_ids:
        ndf = node_df[node_df["node_id"] == nid].sort_values("year")
        if ndf.empty:
            continue
        rank = _node_label(ndf.iloc[0], nid)
        last = ndf.iloc[-1]
        first = ndf.iloc[0]
        weight_change = last["total_weight"] - first["total_weight"]
        pct_change = (weight_change / first["total_weight"] * 100) if first["total_weight"] > 0 else 0

        table_data.append([
            rank,
            f"{last['total_weight']:,.0f}",
            f"{last.get('dist_from_msa_center_mi', 0):.1f} mi",
            f"{weight_change:+,.0f} ({pct_change:+.1f}%)",
            f"{last.get('tract_count', 0):.0f}",
        ])

    col_labels = ["Node", f"{weight_mode.title()}", "Dist from MSA", "Change", "Tracts"]
    if table_data:
        table = ax4.table(
            cellText=table_data, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.6)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#1B3A5C")
            table[0, j].set_text_props(color="white", fontweight="bold")
        for i, nid in enumerate(node_ids):
            if i < len(table_data):
                color = _NODE_COLORS[int(nid) % len(_NODE_COLORS)]
                table[i + 1, 0].set_text_props(color=color, fontweight="bold")

    ax4.set_title("Node Summary", fontsize=11, color="#333", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        console.print(f"[green]Node evolution chart saved:[/green] {out}")

    return fig



# ═══════════════════════════════════════════════════════════════════════════
# Trajectory + growth-share map — where it's heading, who captured growth
#
# Two panels: a map with the historical center path, a dashed projected vector
# (with fit-quality caveat), and quadrant growth labels; beside it a horizontal
# bar chart of net growth by county. Answers the IC questions a drifting
# centroid only hints at: how fast, which direction, and who absorbed it.
# ═══════════════════════════════════════════════════════════════════════════

def gravity_trajectory_map(
    msa_name: str,
    start_year: int,
    end_year: int,
    target_year: int | None = None,
    weight_mode: str = "population",
    years: list[int] | None = None,
    output_path: str | pathlib.Path | None = None,
    dpi: int = 300,
    use_crosswalk: bool = True,
) -> "matplotlib.figure.Figure":
    """Trajectory projection + growth-share decomposition in one IC figure.

    Args:
        msa_name: MSA name or partial match.
        start_year, end_year: Period bounds for the center path & growth share.
        target_year: Year to project the center to. Defaults to end_year + 10.
        weight_mode: Weight variable.
        years: Explicit years for the historical path; default spreads the span.
        output_path: Save if provided.
        dpi: Resolution.
        use_crosswalk: Normalize pre-2020 GEOIDs.

    Returns:
        Matplotlib Figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except ImportError:
        raise ImportError("matplotlib required. Run: pip install 'geostack[print-maps]'")

    from geostack.gravity import (
        gravity_time_series, compute_shift_metrics, project_center, growth_share,
        fetch_tract_geometries, get_landmarks, resolve_msa,
    )

    if target_year is None:
        target_year = end_year + 10

    msa = resolve_msa(msa_name)
    state_fips = msa["state_fips"]
    counties = msa["counties"]
    proj = msa.get("crs", "EPSG:2276")  # per-MSA feet-based State Plane
    FT_PER_MI = 5280.0

    wlabel = {"population": "people", "jobs": "jobs",
              "workers": "workers", "income_mass": "income mass"}.get(weight_mode, weight_mode)

    console.print(f"[bold]Building trajectory map:[/bold] {msa['name']} ({start_year}→{end_year}, →{target_year})")

    # ── Data ──
    if years is None:
        span = end_year - start_year
        years = sorted({start_year, start_year + span // 3, start_year + 2 * span // 3, end_year})
    series = compute_shift_metrics(
        gravity_time_series(msa_name, years=years, weight_mode=weight_mode, use_crosswalk=use_crosswalk)
    )
    proj_res = project_center(series, target_year, crs=proj)
    gs_county = growth_share(msa_name, start_year, end_year, weight_mode, by="county", use_crosswalk=use_crosswalk)
    gs_quad = growth_share(msa_name, start_year, end_year, weight_mode, by="quadrant", use_crosswalk=use_crosswalk)

    # ── Figure ──
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    g = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.18)
    ax = fig.add_subplot(g[0, 0])
    axb = fig.add_subplot(g[0, 1])

    # ── LEFT PANEL: directional growth compass + projected heading ──
    # A metro basemap can't show a 1-mile center path legibly, so we abstract
    # to a compass: arrows sized by each quadrant's share of net growth, with
    # the projected center heading overlaid.
    qd = {r["unit"]: r for _, r in gs_quad.iterrows()}
    units = {"North": (0, 1), "East": (1, 0), "South": (0, -1), "West": (-1, 0)}
    style = _MODE_STYLES.get(weight_mode, _MODE_STYLES["population"])
    max_share = max((qd[q]["share_of_growth"] for q in units if q in qd), default=1)

    # Faint reference cross
    ax.plot([-1.25, 1.25], [0, 0], color="#eee", linewidth=0.8, zorder=1)
    ax.plot([0, 0], [-1.25, 1.25], color="#eee", linewidth=0.8, zorder=1)

    for q, (dx, dy) in units.items():
        if q not in qd:
            continue
        r = qd[q]
        L = 0.35 + (r["share_of_growth"] / max_share) * 0.75  # arrow length by share
        lw = 4 + (r["share_of_growth"] / max_share) * 10       # width by share
        ax.annotate("", xy=(dx * L, dy * L), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=style["color"],
                                    linewidth=lw, mutation_scale=14, alpha=0.85), zorder=3)
        va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
        ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
        ax.annotate(
            f"{q}\n{r['share_of_growth']:.0f}% of growth\n+{r['net_change']/1000:,.0f}k ({r['pct_change']:+.0f}%)",
            (dx * (L + 0.04), dy * (L + 0.04)),
            fontsize=9, fontfamily=_TUFTE_FONT, color="#333", ha=ha, va=va,
            fontweight="bold", zorder=4)

    # Projected center heading (orange dashed arrow at the trend bearing).
    # Shortened so it tucks under the quadrant arrow tip; label parked in the
    # corner so it never collides with the (often near-parallel) quadrant label.
    th = np.radians(proj_res["trend_bearing_deg"])
    hx, hy = np.sin(th), np.cos(th)
    ax.annotate("", xy=(hx * 0.95, hy * 0.95), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#C75B12", linewidth=2.2,
                                linestyle=(0, (4, 2)), mutation_scale=16), zorder=7)
    ax.scatter([0], [0], s=60, c="#222", zorder=8)
    ax.text(
        -1.62, -1.5,
        f"projected center heading: {_compass(proj_res['trend_bearing_deg'])} "
        f"at {proj_res['pace_mi_per_decade']:.2f} mi/decade",
        fontsize=8.5, fontfamily=_TUFTE_FONT, color="#C75B12", fontweight="bold",
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7f0", edgecolor="#e8c4a8"))

    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.set_axis_off()

    fit_word = "tight linear fit" if proj_res["r2"] >= 0.9 else (
        "moderate fit" if proj_res["r2"] >= 0.7 else "loose fit — caution")
    ax.set_title(
        f"Direction of growth & projected heading to {target_year}  ·  "
        f"R²={proj_res['r2']:.2f} ({fit_word})",
        fontsize=10.5, fontfamily=_TUFTE_FONT, color="#333", loc="center", pad=8)

    # ── BAR PANEL: county growth share ──
    top = gs_county.head(8).iloc[::-1]  # largest at top
    style = _MODE_STYLES.get(weight_mode, _MODE_STYLES["population"])
    colors = [style["color"] if v >= 0 else "#999" for v in top["net_change"]]
    bars = axb.barh(range(len(top)), top["net_change"] / 1000, color=colors, alpha=0.85, zorder=3)
    axb.set_yticks(range(len(top)))
    axb.set_yticklabels(top["unit"], fontsize=9, fontfamily=_TUFTE_FONT)
    for i, (_, r) in enumerate(top.iterrows()):
        axb.annotate(f"{r['share_of_growth']:.0f}%  (+{r['net_change']/1000:,.0f}k, {r['pct_change']:+.0f}%)",
                     (r["net_change"] / 1000, i), fontsize=8, fontfamily=_TUFTE_FONT,
                     color="#333", ha="left", va="center", xytext=(4, 0), textcoords="offset points")
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.spines["left"].set_visible(False)
    axb.spines["bottom"].set_linewidth(0.6); axb.spines["bottom"].set_color("#999")
    axb.tick_params(axis="x", labelsize=8, color="#999", labelcolor="#666")
    axb.tick_params(axis="y", length=0)
    axb.set_xlabel(f"net new {wlabel}, {start_year}–{end_year} (thousands)",
                   fontsize=9, fontfamily=_TUFTE_FONT, color="#555")
    axb.margins(x=0.22)
    axb.set_title(f"Who captured net new {wlabel}", fontsize=10.5, fontfamily=_TUFTE_FONT,
                  color="#333", loc="left", pad=8)
    axb.grid(False)

    # ── Title + headline + source ──
    fig.text(0.02, 0.97, msa["name"], fontsize=15, fontfamily=_TUFTE_FONT,
             fontweight="bold", color="#222", va="top")
    lead_q = gs_quad.iloc[0]
    lead_c = gs_county.iloc[0]
    fig.text(0.02, 0.94,
             f"The metro's {lead_q['unit'].lower()} side absorbed {lead_q['share_of_growth']:.0f}% of net {wlabel} growth "
             f"({start_year}–{end_year}); {lead_c['unit']} County alone took {lead_c['share_of_growth']:.0f}%. "
             f"At the current pace the center reaches {_compass(proj_res['trend_bearing_deg'])} "
             f"by {target_year}.",
             fontsize=9.5, fontfamily=_TUFTE_FONT, color="#555", va="top")

    src = "U.S. Census ACS 5-Year" if weight_mode in ("population", "income_mass") else "U.S. Census LEHD LODES8"
    fig.text(0.98, 0.02,
             f"Source: {src}, {start_year} & {end_year}. Projection is a linear "
             f"extrapolation of the {start_year}–{end_year} trend (illustrative, not a forecast). "
             f"Estimates carry survey margins of error.",
             fontsize=6.5, fontfamily=_TUFTE_FONT, color="#aaa", ha="right", va="bottom")

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    if output_path:
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        console.print(f"[green]Trajectory map saved:[/green] {out}")

    return fig


def _compass(bearing: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((bearing + 11.25) / 22.5) % 16]
