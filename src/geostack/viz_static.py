"""Publication-quality static maps via Matplotlib + Contextily.

Produces print-ready PNG/PDF/SVG maps suitable for IC decks, lender
packages, and broker marketing materials. Existing viz.py static_map()
is a quick-look utility; this module is the full-featured version with
basemap tiles, radius rings, isochrone overlays, annotations, legends,
and institutional styling.

Install:  pip install 'geostack[print-maps]'

Usage:
    from geostack.viz_static import (
        broker_radius_map,
        broker_isochrone_map,
        demographic_choropleth,
        save_map,
        MF_INCOME_BREAKS,
    )

    # Broker-style radius ring map — ready for PDF embedding
    fig, ax = broker_radius_map(
        tracts_gdf, center=(-96.77, 32.78),
        radii_miles=[1, 3, 5], column="median_household_income",
        property_name="Example Property",
    )
    save_map(fig, "example_property_demographics.png", dpi=300)

    # Isochrone version
    fig, ax = broker_isochrone_map(
        tracts_gdf, isochrone_gdf, column="median_household_income",
    )

    # MF class-aware income choropleth (A/B/C/D segmentation)
    # Uses fixed breakpoints tied to 30% rent affordability rule
    fig, ax = broker_isochrone_map(
        tracts_gdf, isochrone_gdf, column="median_household_income",
        property_name="Example Property",
        mf_class_breaks="austin",  # or "dallas", "houston", "birmingham", "default"
    )

    # Custom breaks (upper bound of each class)
    fig, ax = broker_radius_map(
        tracts_gdf, center=(-97.76, 30.19),
        column="median_household_income",
        mf_class_breaks=[36_000, 48_000, 64_000, 80_000],
    )
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from rich.console import Console

console = Console()

# Institutional styling constants
FONT_FAMILY = "Helvetica Neue"
COLORS = {
    "primary": "#1B3A5C",
    "text": "#333333",
    "ring_1mi": "#D32F2F",
    "ring_3mi": "#F57C00",
    "ring_5mi": "#1976D2",
    "iso_5min": "#D32F2F",
    "iso_10min": "#F57C00",
    "iso_15min": "#1976D2",
    "property": "#D32F2F",
}

# Blues sequential colormap for income/demographic choropleths
CMAP_BLUES = "Blues"
CMAP_YLGNBU = "YlGnBu"

# ── Multifamily asset-class income breaks ──────────────────────────────────────
# Fixed breakpoints tied to rent affordability at 30% gross income.
# Use these instead of data-driven quantiles when visualizing MHI for
# A/B/C/D class multifamily targeting.
#
# Example: Austin Class B rents $1,200–$1,600/mo → HHI $48K–$64K required.
#
# Metro-specific breaks account for local rent levels and renter income
# distributions. Breaks are upper bounds for each class.
MF_INCOME_BREAKS: dict[str, dict] = {
    "austin": {
        "breaks": [36_000, 48_000, 64_000, 80_000],
        "labels": ["Class D (<$36K)", "Class C ($36K–$48K)", "Class B ($48K–$64K)",
                   "Class A- ($64K–$80K)", "Class A ($80K+)"],
        "rent_ranges": ["<$900/mo", "$900–$1,200", "$1,200–$1,600", "$1,600–$2,000", "$2,000+"],
        "source": "ACS 2024 Travis County renter MHI ~$55K; 30% affordability rule",
    },
    "dallas": {
        "breaks": [34_000, 45_000, 60_000, 75_000],
        "labels": ["Class D (<$34K)", "Class C ($34K–$45K)", "Class B ($45K–$60K)",
                   "Class A- ($60K–$75K)", "Class A ($75K+)"],
        "rent_ranges": ["<$850/mo", "$850–$1,125", "$1,125–$1,500", "$1,500–$1,875", "$1,875+"],
        "source": "ACS 2023 Dallas County renter MHI ~$52K; 30% affordability rule",
    },
    "houston": {
        "breaks": [32_000, 42_000, 56_000, 72_000],
        "labels": ["Class D (<$32K)", "Class C ($32K–$42K)", "Class B ($42K–$56K)",
                   "Class A- ($56K–$72K)", "Class A ($72K+)"],
        "rent_ranges": ["<$800/mo", "$800–$1,050", "$1,050–$1,400", "$1,400–$1,800", "$1,800+"],
        "source": "ACS 2023 Harris County renter MHI ~$48K; 30% affordability rule",
    },
    "birmingham": {
        "breaks": [28_000, 38_000, 50_000, 65_000],
        "labels": ["Class D (<$28K)", "Class C ($28K–$38K)", "Class B ($38K–$50K)",
                   "Class A- ($50K–$65K)", "Class A ($65K+)"],
        "rent_ranges": ["<$700/mo", "$700–$950", "$950–$1,250", "$1,250–$1,625", "$1,625+"],
        "source": "ACS 2023 Jefferson County renter MHI ~$42K; 30% affordability rule",
    },
    # Fallback for metros without specific calibration
    "default": {
        "breaks": [35_000, 47_000, 62_000, 78_000],
        "labels": ["Class D (<$35K)", "Class C ($35K–$47K)", "Class B ($47K–$62K)",
                   "Class A- ($62K–$78K)", "Class A ($78K+)"],
        "rent_ranges": ["<$875/mo", "$875–$1,175", "$1,175–$1,550", "$1,550–$1,950", "$1,950+"],
        "source": "National renter MHI proxy; 30% affordability rule",
    },
}


def _ensure_deps():
    """Import matplotlib + contextily, raising helpful errors."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
    except ImportError:
        raise ImportError("matplotlib is required. Run: pip install 'geostack[print-maps]'")

    try:
        import contextily as ctx
    except ImportError:
        raise ImportError("contextily is required. Run: pip install 'geostack[print-maps]'")

    return plt, matplotlib, ctx


def _format_legend_labels(ax, column: str, fmt: str = "auto"):
    """Rewrite legend labels to human-readable format.

    Replaces raw floats like '6311.00, 53821.00' with '$6K – $54K' for
    dollar columns, or '6,311 – 53,821' for population columns.
    """
    legend = ax.get_legend()
    if legend is None:
        return

    col_lower = column.lower()
    is_dollar = any(
        kw in col_lower
        for kw in ("income", "rent", "value", "wage", "hhi",
                    "b19013", "b25064", "b25077", "b19301")
    )
    is_population = any(
        kw in col_lower
        for kw in ("population", "units", "occupied", "households",
                    "b01003", "b25001", "b25002", "b25003")
    )

    for text in legend.get_texts():
        label = text.get_text()
        # Parse "low, high" format from mapclassify
        if "," in label:
            parts = label.split(",")
            try:
                lo = float(parts[0].strip())
                hi = float(parts[1].strip())
            except ValueError:
                continue

            if is_dollar:
                lo_s = f"${lo / 1000:.0f}K" if lo >= 1000 else f"${lo:,.0f}"
                hi_s = f"${hi / 1000:.0f}K" if hi >= 1000 else f"${hi:,.0f}"
                text.set_text(f"{lo_s} \u2013 {hi_s}")
            elif is_population:
                lo_s = f"{lo / 1000:.0f}K" if lo >= 10000 else f"{lo:,.0f}"
                hi_s = f"{hi / 1000:.0f}K" if hi >= 10000 else f"{hi:,.0f}"
                text.set_text(f"{lo_s} \u2013 {hi_s}")
            else:
                text.set_text(f"{lo:,.0f} \u2013 {hi:,.0f}")


def _apply_mf_legend_labels(ax, labels: list[str]):
    """Replace legend text entries with predefined MF class labels.

    Args:
        ax: Matplotlib axes with legend.
        labels: List of label strings matching the number of legend entries.
    """
    legend = ax.get_legend()
    if legend is None:
        return

    texts = legend.get_texts()
    for i, text in enumerate(texts):
        if i < len(labels):
            text.set_text(labels[i])


def _add_scale_bar(ax, length_mi: float = 1.0):
    """Add a minimal scale bar in the lower-left corner."""
    plt, _, _ = _ensure_deps()
    from matplotlib.patches import FancyBboxPatch

    length_m = length_mi * 1609.344
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Position: 5% from left, 3% from bottom
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.03
    bar_height = (ylim[1] - ylim[0]) * 0.005

    # White background behind scale bar
    bg_pad = length_m * 0.08
    bg = FancyBboxPatch(
        (x0 - bg_pad, y0 - bg_pad * 3),
        length_m + bg_pad * 2,
        bar_height + bg_pad * 8,
        boxstyle="round,pad=0",
        facecolor="white",
        edgecolor="none",
        alpha=0.85,
        zorder=8,
    )
    ax.add_patch(bg)

    # Scale bar rectangle
    bar = plt.Rectangle(
        (x0, y0), length_m, bar_height,
        facecolor=COLORS["primary"],
        edgecolor=COLORS["primary"],
        linewidth=0.5,
        zorder=9,
    )
    ax.add_patch(bar)

    # Label
    label = f"{length_mi:.0f} mi" if length_mi == int(length_mi) else f"{length_mi} mi"
    ax.text(
        x0 + length_m / 2, y0 + bar_height * 2.5,
        label,
        ha="center", va="bottom",
        fontsize=8, fontweight="bold",
        color=COLORS["primary"],
        fontfamily=FONT_FAMILY,
        zorder=9,
    )


def demographic_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str | None = None,
    cmap: str = CMAP_BLUES,
    figsize: tuple[float, float] = (12, 10),
    basemap: str = "CartoDB.Positron",
    scheme: str = "quantiles",
    n_bins: int = 6,
    legend_title: str | None = None,
    alpha: float = 0.50,
    mf_class_breaks: str | list[float] | None = None,
) -> tuple:
    """High-quality static demographic choropleth with basemap tiles.

    Args:
        gdf: GeoDataFrame with polygon geometry and demographic columns.
        column: Column for choropleth fill.
        title: Map title (rendered top-left, institutional style).
        cmap: Matplotlib colormap name.
        figsize: Figure size in inches.
        basemap: Contextily tile provider (e.g., "CartoDB.Positron",
                 "CartoDB.DarkMatter", "Stamen.TonerLite").
        scheme: mapclassify scheme for binning. Ignored if mf_class_breaks set.
        n_bins: Number of classification bins. Ignored if mf_class_breaks set.
        legend_title: Legend label. Defaults to column name.
        alpha: Fill opacity.
        mf_class_breaks: Multifamily asset-class income breaks. Either:
            - A metro key ("austin", "dallas", "houston", "birmingham", "default")
              to use predefined MF_INCOME_BREAKS thresholds, or
            - A list of break values (e.g., [36000, 48000, 64000, 80000]).
            When set, overrides scheme/n_bins with UserDefined classification
            and applies semantic MF class labels to the legend.

    Returns:
        (fig, ax) tuple. Call save_map(fig, path) for export.
    """
    plt, matplotlib, ctx = _ensure_deps()
    import mapclassify

    gdf = gdf.to_crs("EPSG:4326") if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf.copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    # Project to Web Mercator for contextily basemap alignment
    gdf_merc = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # Resolve MF class breaks if requested
    mf_labels = None
    if mf_class_breaks is not None:
        if isinstance(mf_class_breaks, str):
            metro_key = mf_class_breaks.lower()
            mf_config = MF_INCOME_BREAKS.get(metro_key, MF_INCOME_BREAKS["default"])
            breaks = mf_config["breaks"]
            mf_labels = mf_config["labels"]
        else:
            breaks = list(mf_class_breaks)
            mf_labels = None  # Will use formatted break ranges

        scheme = "user_defined"
        classification_kwds = {"bins": breaks}
        n_bins = len(breaks) + 1
    else:
        classification_kwds = {}

    # Plot choropleth — semi-transparent fill with subtle edges
    plot_kwargs = {
        "column": column,
        "ax": ax,
        "cmap": cmap,
        "scheme": scheme,
        "alpha": alpha,
        "edgecolor": "white",
        "linewidth": 0.2,
        "legend": True,
        "legend_kwds": {
            "loc": "lower right",
            "fontsize": 9,
            "title": legend_title or column.replace("_", " ").title(),
            "title_fontsize": 10,
            "frameon": True,
            "framealpha": 0.95,
            "edgecolor": "#cccccc",
            "fancybox": True,
            "shadow": False,
        },
    }
    if classification_kwds:
        plot_kwargs["classification_kwds"] = classification_kwds
    else:
        plot_kwargs["k"] = n_bins

    gdf_merc.plot(**plot_kwargs)

    # Format legend labels
    if mf_labels:
        _apply_mf_legend_labels(ax, mf_labels)
    else:
        _format_legend_labels(ax, column)

    # Style legend text
    legend = ax.get_legend()
    if legend:
        legend.get_title().set_fontfamily(FONT_FAMILY)
        legend.get_title().set_color(COLORS["primary"])
        for text in legend.get_texts():
            text.set_fontfamily(FONT_FAMILY)
            text.set_color(COLORS["text"])

    # Add basemap tiles
    try:
        provider = getattr(ctx.providers.CartoDB, "Positron", None)
        if basemap == "CartoDB.DarkMatter":
            provider = ctx.providers.CartoDB.DarkMatter
        elif basemap == "CartoDB.PositronNoLabels":
            provider = ctx.providers.CartoDB.PositronNoLabels
        elif basemap == "Stamen.TonerLite":
            provider = ctx.providers.Stamen.TonerLite

        ctx.add_basemap(ax, source=provider or basemap, zoom="auto")
    except Exception as e:
        console.print(f"[yellow]Basemap failed ({e}), continuing without[/yellow]")

    # Title — with white panel behind it for readability
    if title:
        ax.text(
            0.02, 0.97, title,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            color=COLORS["primary"],
            fontfamily=FONT_FAMILY,
            verticalalignment="top",
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
            },
        )

    ax.set_axis_off()
    plt.tight_layout(pad=1.0)

    return fig, ax


def broker_radius_map(
    tracts_gdf: gpd.GeoDataFrame,
    center: tuple[float, float],
    radii_miles: list[float] | None = None,
    column: str = "median_household_income",
    property_name: str | None = None,
    ring_colors: dict[float, str] | None = None,
    figsize: tuple[float, float] = (14, 11),
    cmap: str = CMAP_BLUES,
    basemap: str = "CartoDB.Positron",
    annotate_rings: bool = True,
    show_property_marker: bool = True,
    subtitle: str | None = None,
    mf_class_breaks: str | list[float] | None = None,
    **kwargs,
) -> tuple:
    """Broker-quality radius ring map with demographic choropleth.

    Produces the classic 1/3/5 mile ring map seen in CBRE, JLL, and
    Cushman marketing materials. Print-ready at 300 dpi.

    Args:
        tracts_gdf: GeoDataFrame with tract geometry + demographics.
        center: (lon, lat) of the subject property.
        radii_miles: Ring radii (default [1, 3, 5]).
        column: Demographic column for choropleth.
        property_name: Property label shown on map.
        ring_colors: Dict mapping radius → hex color.
        figsize: Figure size in inches.
        cmap: Matplotlib colormap.
        basemap: Contextily tile provider.
        annotate_rings: If True, label each ring with "Xmi" text.
        show_property_marker: If True, show red dot at center.
        subtitle: Subtitle text below title.
        mf_class_breaks: Metro key or list of income breaks for MF class legend.
            See demographic_choropleth() for details.
        **kwargs: Passed to demographic_choropleth().

    Returns:
        (fig, ax) tuple.
    """
    plt, matplotlib, ctx = _ensure_deps()

    if radii_miles is None:
        radii_miles = [1.0, 3.0, 5.0]

    if ring_colors is None:
        default_colors = [COLORS["ring_1mi"], COLORS["ring_3mi"], COLORS["ring_5mi"]]
        ring_colors = dict(zip(sorted(radii_miles), default_colors[:len(radii_miles)]))

    # Build base choropleth
    fig, ax = demographic_choropleth(
        tracts_gdf, column=column,
        title=property_name or "Demographic Analysis",
        figsize=figsize, cmap=cmap, basemap=basemap,
        mf_class_breaks=mf_class_breaks,
        **kwargs,
    )

    # Project center point to Web Mercator
    from shapely.geometry import Point
    center_wgs = gpd.GeoSeries([Point(center[0], center[1])], crs="EPSG:4326")
    center_merc = center_wgs.to_crs(epsg=3857)
    cx, cy = center_merc.iloc[0].x, center_merc.iloc[0].y

    # Draw radius rings — bolder dashes, white shadow for contrast
    for radius in sorted(radii_miles):
        radius_m = radius * 1609.344
        color = ring_colors.get(radius, COLORS["primary"])

        # White shadow ring (slightly wider, behind the colored one)
        shadow = plt.Circle(
            (cx, cy), radius_m,
            fill=False,
            edgecolor="white",
            linewidth=4.0,
            linestyle="--",
            alpha=0.6,
            zorder=4,
        )
        ax.add_patch(shadow)

        # Colored ring
        circle = plt.Circle(
            (cx, cy), radius_m,
            fill=False,
            edgecolor=color,
            linewidth=2.5,
            linestyle=(0, (8, 4)),  # dash pattern: 8 on, 4 off
            alpha=0.90,
            zorder=5,
        )
        ax.add_patch(circle)

        # Ring label — positioned at top of circle
        if annotate_rings:
            label_x = cx
            label_y = cy + radius_m
            label_text = f"{radius:.0f} mi" if radius == int(radius) else f"{radius} mi"
            ax.annotate(
                label_text,
                xy=(label_x, label_y),
                fontsize=11,
                fontweight="bold",
                color=color,
                fontfamily=FONT_FAMILY,
                ha="center",
                va="bottom",
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.92,
                    "linewidth": 1.2,
                },
            )

    # Property marker — larger dot with halo
    if show_property_marker:
        # White halo
        ax.plot(
            cx, cy,
            marker="o",
            markersize=16,
            markerfacecolor="white",
            markeredgecolor="white",
            markeredgewidth=0,
            alpha=0.7,
            zorder=6,
        )
        # Red dot
        ax.plot(
            cx, cy,
            marker="o",
            markersize=11,
            markerfacecolor=COLORS["property"],
            markeredgecolor="white",
            markeredgewidth=2.5,
            zorder=7,
        )
        if property_name:
            ax.annotate(
                property_name,
                xy=(cx, cy),
                xytext=(18, -18),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=COLORS["primary"],
                fontfamily=FONT_FAMILY,
                zorder=7,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": COLORS["primary"],
                    "alpha": 0.92,
                    "linewidth": 1.0,
                },
            )

    # Subtitle — below the title with white background
    if subtitle:
        ax.text(
            0.02, 0.91, subtitle,
            transform=ax.transAxes,
            fontsize=10,
            color="#555555",
            fontfamily=FONT_FAMILY,
            fontstyle="italic",
            verticalalignment="top",
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )

    # Zoom to largest ring + 15% padding
    max_radius_m = max(radii_miles) * 1609.344
    pad = max_radius_m * 0.15
    ax.set_xlim(cx - max_radius_m - pad, cx + max_radius_m + pad)
    ax.set_ylim(cy - max_radius_m - pad, cy + max_radius_m + pad)

    # Scale bar
    _add_scale_bar(ax, length_mi=1.0)

    return fig, ax


def broker_isochrone_map(
    tracts_gdf: gpd.GeoDataFrame,
    isochrone_gdf: gpd.GeoDataFrame,
    column: str = "median_household_income",
    center: tuple[float, float] | None = None,
    property_name: str | None = None,
    iso_colors: dict[float, str] | None = None,
    figsize: tuple[float, float] = (14, 11),
    cmap: str = CMAP_BLUES,
    basemap: str = "CartoDB.Positron",
    annotate_isos: bool = True,
    subtitle: str | None = None,
    mf_class_breaks: str | list[float] | None = None,
    **kwargs,
) -> tuple:
    """Broker-quality drive-time isochrone map with demographic choropleth.

    Same visual style as broker_radius_map() but with organic drive-time
    polygons instead of geometric circles.

    Args:
        tracts_gdf: GeoDataFrame with tract geometry + demographics.
        isochrone_gdf: GeoDataFrame with 'minutes' column + polygon geometry
                       (from routing_multi.isochrones_nested()).
        column: Demographic column for choropleth.
        center: (lon, lat) for property marker. If None, uses isochrone centroid.
        property_name: Property label.
        iso_colors: Dict mapping minutes → hex color.
        figsize: Figure size.
        cmap: Matplotlib colormap.
        basemap: Contextily tile provider.
        annotate_isos: If True, label each isochrone.
        mf_class_breaks: Metro key or list of income breaks for MF class legend.
            See demographic_choropleth() for details.
        **kwargs: Passed to demographic_choropleth().

    Returns:
        (fig, ax) tuple.
    """
    plt, matplotlib, ctx = _ensure_deps()

    if iso_colors is None:
        iso_colors = {
            5: COLORS["iso_5min"],
            10: COLORS["iso_10min"],
            15: COLORS["iso_15min"],
        }

    # Build base choropleth
    fig, ax = demographic_choropleth(
        tracts_gdf, column=column,
        title=property_name or "Drive-Time Analysis",
        figsize=figsize, cmap=cmap, basemap=basemap,
        mf_class_breaks=mf_class_breaks,
        **kwargs,
    )

    # Project isochrones to Web Mercator
    iso_merc = isochrone_gdf.to_crs(epsg=3857)

    # Draw isochrones (largest first for correct layering)
    iso_sorted = iso_merc.sort_values("minutes", ascending=False)

    for _, row in iso_sorted.iterrows():
        mins = row["minutes"]
        color = iso_colors.get(mins, COLORS["primary"])

        # Semi-transparent fill
        gpd.GeoSeries([row.geometry]).plot(
            ax=ax,
            facecolor=color,
            edgecolor=color,
            alpha=0.12,
            linewidth=2.5,
            zorder=4,
        )

        # Solid border on top
        gpd.GeoSeries([row.geometry]).boundary.plot(
            ax=ax,
            color=color,
            linewidth=2.0,
            alpha=0.8,
            zorder=5,
        )

        # Label at the top of the isochrone
        if annotate_isos:
            bounds = row.geometry.bounds  # minx, miny, maxx, maxy
            label_x = (bounds[0] + bounds[2]) / 2
            label_y = bounds[3]
            ax.annotate(
                f"{int(mins)} min",
                xy=(label_x, label_y),
                fontsize=10,
                fontweight="bold",
                color=color,
                ha="center",
                va="bottom",
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.85,
                    "linewidth": 1,
                },
            )

    # Property marker at center
    if center is not None:
        from shapely.geometry import Point
        center_merc = gpd.GeoSeries(
            [Point(center[0], center[1])], crs="EPSG:4326"
        ).to_crs(epsg=3857)
        cx, cy = center_merc.iloc[0].x, center_merc.iloc[0].y

        ax.plot(
            cx, cy,
            marker="o",
            markersize=10,
            markerfacecolor=COLORS["property"],
            markeredgecolor="white",
            markeredgewidth=2,
            zorder=7,
        )
        if property_name:
            ax.annotate(
                property_name,
                xy=(cx, cy),
                xytext=(15, -15),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=COLORS["primary"],
                zorder=7,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": COLORS["primary"],
                    "alpha": 0.9,
                },
            )

    # Subtitle
    if subtitle:
        ax.text(
            0.02, 0.91, subtitle,
            transform=ax.transAxes,
            fontsize=10,
            color="#555555",
            fontfamily=FONT_FAMILY,
            fontstyle="italic",
            verticalalignment="top",
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )

    # Zoom to isochrone extent + padding
    total_bounds = iso_merc.total_bounds  # minx, miny, maxx, maxy
    dx = total_bounds[2] - total_bounds[0]
    dy = total_bounds[3] - total_bounds[1]
    pad = max(dx, dy) * 0.1
    ax.set_xlim(total_bounds[0] - pad, total_bounds[2] + pad)
    ax.set_ylim(total_bounds[1] - pad, total_bounds[3] + pad)

    # Scale bar
    _add_scale_bar(ax, length_mi=1.0)

    return fig, ax


# ---------------------------------------------------------------------------
# Demographic infographic table (paired with radius / isochrone maps)
# ---------------------------------------------------------------------------

# ACS variables needed for the infographic table
INFOGRAPHIC_VARIABLES = [
    # Population
    "B01003_001E",  # total_population
    "B01002_001E",  # median_age
    # Income
    "B19013_001E",  # median_household_income
    "B19301_001E",  # per_capita_income
    # Housing
    "B25001_001E",  # total_housing_units
    "B25002_002E",  # occupied_units
    "B25003_003E",  # renter_occupied
    "B25003_002E",  # owner_occupied
    "B25064_001E",  # median_rent
    "B25077_001E",  # median_home_value
    # Education (25+ population)
    "B15003_001E",  # edu_total_25plus
    "B15003_022E",  # edu_bachelors
    "B15003_023E",  # edu_masters
    "B15003_024E",  # edu_professional
    "B15003_025E",  # edu_doctorate
    # Income distribution (for $75K+ share)
    "B19001_001E",  # hh_income_total
    "B19001_013E",  # hh_income_75k_100k
    "B19001_014E",  # hh_income_100k_125k
    "B19001_015E",  # hh_income_125k_150k
    "B19001_016E",  # hh_income_150k_200k
    "B19001_017E",  # hh_income_200k_plus
]


def _compute_infographic_rows(ring_data: dict) -> list[dict]:
    """Derive display rows from raw ACS ring data.

    Args:
        ring_data: Dict mapping radius → dict of ACS variable code → value.

    Returns:
        List of row dicts with keys: label, format, and one key per radius.
    """
    rows = []

    def _pct(num, denom):
        if denom and denom > 0:
            return num / denom * 100
        return float("nan")

    for radius, d in ring_data.items():
        pop = d.get("B01003_001E", 0) or 0
        hh_total = d.get("B19001_001E", 0) or 0
        # If total missing, sum the 16 income brackets
        if not hh_total:
            hh_total = sum(
                d.get(f"B19001_{i:03d}E", 0) or 0 for i in range(2, 18)
            )
        edu_total = d.get("B15003_001E", 0) or 0
        occupied = d.get("B25002_002E", 0) or 0
        renter = d.get("B25003_003E", 0) or 0
        owner = d.get("B25003_002E", 0) or 0

        # Bachelors+ = bachelors + masters + professional + doctorate
        bachelors_plus = sum(
            d.get(v, 0) or 0
            for v in ["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]
        )

        # HH income $75K+
        hh_75k_plus = sum(
            d.get(v, 0) or 0
            for v in ["B19001_013E", "B19001_014E", "B19001_015E", "B19001_016E", "B19001_017E"]
        )

        ring_data[radius] = {
            "population": pop,
            "median_age": d.get("B01002_001E", 0) or 0,
            "median_hhi": d.get("B19013_001E", 0) or 0,
            "per_capita_income": d.get("B19301_001E", 0) or 0,
            "total_housing_units": d.get("B25001_001E", 0) or 0,
            "renter_pct": _pct(renter, occupied),
            "owner_pct": _pct(owner, occupied),
            "median_rent": d.get("B25064_001E", 0) or 0,
            "median_home_value": d.get("B25077_001E", 0) or 0,
            "bachelors_plus_pct": _pct(bachelors_plus, edu_total),
            "hh_75k_plus_pct": _pct(hh_75k_plus, hh_total),
        }

    # Build display rows: (label, format_type)
    metric_defs = [
        ("Population", "population", "comma"),
        ("Median Age", "median_age", "decimal1"),
        ("Median Household Income", "median_hhi", "dollar"),
        ("Per Capita Income", "per_capita_income", "dollar"),
        ("HH Income $75K+", "hh_75k_plus_pct", "pct"),
        ("Total Housing Units", "total_housing_units", "comma"),
        ("Renter-Occupied", "renter_pct", "pct"),
        ("Owner-Occupied", "owner_pct", "pct"),
        ("Median Rent", "median_rent", "dollar"),
        ("Median Home Value", "median_home_value", "dollar"),
        ("Bachelor's Degree+", "bachelors_plus_pct", "pct"),
    ]

    result = []
    for label, key, fmt in metric_defs:
        row = {"label": label, "fmt": fmt}
        for radius, metrics in ring_data.items():
            row[radius] = metrics.get(key, 0)
        result.append(row)

    return result


def _format_cell(value, fmt: str) -> str:
    """Format a cell value for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if fmt == "dollar":
        return f"${value:,.0f}"
    elif fmt == "comma":
        return f"{value:,.0f}"
    elif fmt == "pct":
        return f"{value:.1f}%"
    elif fmt == "decimal1":
        return f"{value:.1f}"
    return str(value)


def demographic_infographic(
    ring_data: dict[float, dict[str, float]],
    radii_miles: list[float] | None = None,
    property_name: str | None = None,
    subtitle: str = "ACS 5-Year Estimates (2023)",
    figsize: tuple[float, float] = (14, 9),
    ring_colors: dict[float, str] | None = None,
    mode: str = "radius",
) -> tuple:
    """Render a broker-style demographic infographic table.

    Pairs with broker_radius_map() or broker_isochrone_map() — shows key
    metrics per ring/isochrone in a clean, print-ready table.

    Args:
        ring_data: Dict mapping key (radius in miles or minutes) → dict of
                   ACS variable codes → aggregated values.
        radii_miles: Ordered list of keys to display as columns.
        property_name: Title text.
        subtitle: Source attribution line.
        figsize: Figure size in inches.
        ring_colors: Dict mapping key → header color.
        mode: "radius" for "X-Mile" headers, "isochrone" for "X-Min" headers.

    Returns:
        (fig, ax) tuple. Call save_map(fig, path) for export.
    """
    plt, matplotlib, _ = _ensure_deps()
    from matplotlib.patches import FancyBboxPatch

    if radii_miles is None:
        radii_miles = sorted(ring_data.keys())

    if ring_colors is None:
        if mode == "isochrone":
            default_colors = [COLORS["iso_5min"], COLORS["iso_10min"], COLORS["iso_15min"]]
        else:
            default_colors = [COLORS["ring_1mi"], COLORS["ring_3mi"], COLORS["ring_5mi"]]
        ring_colors = dict(zip(sorted(radii_miles), default_colors[:len(radii_miles)]))

    # Compute derived metrics
    rows = _compute_infographic_rows(ring_data)

    n_rows = len(rows)
    n_cols = len(radii_miles)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_rows + 3.5)
    ax.set_axis_off()

    # Layout constants
    label_x = 0.3          # left edge of metric labels
    col_width = (10 - 4.0) / n_cols  # evenly divide remaining space
    col_starts = [4.0 + i * col_width + col_width / 2 for i in range(n_cols)]
    header_y = n_rows + 2.2
    row_height = 1.0
    first_row_y = n_rows + 0.8

    # --- Title ---
    title_suffix = "Drive-Time Profile" if mode == "isochrone" else "Demographic Profile"
    if property_name:
        ax.text(
            0.2, n_rows + 3.2,
            f"{property_name} — {title_suffix}",
            fontsize=18, fontweight="bold",
            color=COLORS["primary"], fontfamily=FONT_FAMILY,
            va="top",
        )

    # --- Subtitle / source ---
    ax.text(
        0.2, n_rows + 2.55,
        subtitle,
        fontsize=10, color="#777777", fontfamily=FONT_FAMILY,
        fontstyle="italic", va="top",
    )

    # --- Column headers ---
    for i, radius in enumerate(radii_miles):
        cx = col_starts[i]
        color = ring_colors.get(radius, COLORS["primary"])
        if mode == "isochrone":
            label = f"{radius:.0f}-Min" if radius == int(radius) else f"{radius}-Min"
        else:
            label = f"{radius:.0f}-Mile" if radius == int(radius) else f"{radius}-Mile"

        # Colored header pill
        pill_w = col_width * 0.75
        pill = FancyBboxPatch(
            (cx - pill_w / 2, header_y - 0.25), pill_w, 0.55,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="none",
            zorder=3,
        )
        ax.add_patch(pill)

        ax.text(
            cx, header_y,
            label,
            fontsize=13, fontweight="bold",
            color="white", fontfamily=FONT_FAMILY,
            ha="center", va="center", zorder=4,
        )

    # --- Divider line below header ---
    ax.plot(
        [0.1, 9.9], [header_y - 0.55, header_y - 0.55],
        color=COLORS["primary"], linewidth=1.5, alpha=0.3,
    )

    # --- Data rows ---
    for row_idx, row in enumerate(rows):
        y = first_row_y - row_idx * row_height

        # Alternating row backgrounds
        if row_idx % 2 == 0:
            bg = FancyBboxPatch(
                (0.1, y - 0.35), 9.8, 0.7,
                boxstyle="square,pad=0",
                facecolor="#f8f9fa", edgecolor="none",
                zorder=0,
            )
            ax.add_patch(bg)

        # Metric label (left column)
        ax.text(
            label_x, y,
            row["label"],
            fontsize=12, color=COLORS["text"], fontfamily=FONT_FAMILY,
            va="center", ha="left",
        )

        # Values (one per radius column)
        for i, radius in enumerate(radii_miles):
            value = row.get(radius, 0)
            formatted = _format_cell(value, row["fmt"])
            color = ring_colors.get(radius, COLORS["primary"])

            ax.text(
                col_starts[i], y,
                formatted,
                fontsize=13, fontweight="bold",
                color=COLORS["primary"], fontfamily=FONT_FAMILY,
                va="center", ha="center",
            )

    # --- Bottom divider ---
    bottom_y = first_row_y - n_rows * row_height + 0.3
    ax.plot(
        [0.1, 9.9], [bottom_y, bottom_y],
        color=COLORS["primary"], linewidth=1.0, alpha=0.2,
    )

    # --- Source footnote ---
    ax.text(
        0.2, bottom_y - 0.4,
        "Source: U.S. Census Bureau, ACS 5-Year Estimates. "
        "Area-weighted by tract intersection with radius buffer.",
        fontsize=8, color="#999999", fontfamily=FONT_FAMILY,
        va="top",
    )

    plt.tight_layout(pad=0.5)
    return fig, ax


def save_map(
    fig,
    path: str | Path,
    dpi: int = 300,
    bbox_inches: str = "tight",
    transparent: bool = False,
) -> Path:
    """Save a matplotlib figure to file (PNG, PDF, SVG).

    Args:
        fig: Matplotlib figure from any of the map functions.
        path: Output path. Format inferred from extension.
        dpi: Resolution (300 for print, 150 for web).
        bbox_inches: "tight" to crop whitespace.
        transparent: If True, transparent background (useful for overlays).

    Returns:
        Path to saved file.
    """
    path = Path(path)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        facecolor="white" if not transparent else "none",
        edgecolor="none",
        pad_inches=0.3,
    )

    size_kb = path.stat().st_size / 1024
    console.print(f"[green]Saved: {path} ({size_kb:.0f} KB, {dpi} dpi)[/green]")
    return path
