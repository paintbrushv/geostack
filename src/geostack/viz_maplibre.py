"""Interactive demographic maps via py-maplibregl.

Modern WebGL-based alternative to Folium choropleth — uses MapLibre GL JS
for smooth zooming, 3D tilt, and better large-dataset performance.
Existing viz.py (Folium + PMTiles) is unchanged.

Install:  pip install 'geostack[maplibre-interactive]'

Usage:
    from geostack.viz_maplibre import (
        demographic_map,
        radius_ring_map,
        isochrone_map,
    )

    # Choropleth of median income by tract
    m = demographic_map(tracts_gdf, column="median_household_income")

    # Broker-style radius ring map
    m = radius_ring_map(
        tracts_gdf, center=(-96.77, 32.78),
        radii_miles=[1, 3, 5], column="median_household_income",
    )

    # Isochrone overlay map
    m = isochrone_map(
        tracts_gdf, isochrone_gdf,
        column="median_household_income",
    )
"""

from __future__ import annotations

import json

import geopandas as gpd
from rich.console import Console

console = Console()

# Institutional color palette (matches submarket-atlas branding)
COLORS = {
    "primary": "#1B3A5C",
    "accent": "#2E86AB",
    "light": "#A4C3D2",
    "ring_1mi": "#D32F2F",
    "ring_3mi": "#F57C00",
    "ring_5mi": "#1976D2",
    "iso_5min": "#D32F2F",
    "iso_10min": "#F57C00",
    "iso_15min": "#1976D2",
}

# Sequential blue palette for choropleths
CHOROPLETH_STOPS = [
    "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1",
    "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b",
]


def _ensure_maplibregl():
    """Import and return maplibregl module."""
    try:
        import maplibre
        return maplibre
    except ImportError:
        raise ImportError(
            "maplibre is required. Run: pip install 'geostack[maplibre-interactive]'"
        )


def demographic_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str | None = None,
    cmap: list[str] | None = None,
    style: str = "positron",
    zoom: int = 11,
    center: tuple[float, float] | None = None,
    height: int = 600,
    opacity: float = 0.6,
    tooltip_cols: list[str] | None = None,
) -> object:
    """Render a demographic choropleth using MapLibre GL JS.

    Args:
        gdf: GeoDataFrame with polygon geometry and demographic columns.
        column: Column name for choropleth fill color.
        title: Map title.
        cmap: List of hex colors for the fill gradient (low to high).
        style: Basemap style — "positron", "dark-matter", "voyager", or URL.
        zoom: Initial zoom level.
        center: (lon, lat) center. If None, auto-computed from data.
        height: Map height in pixels.
        opacity: Fill opacity (0-1).
        tooltip_cols: Columns to show on hover. Defaults to [column].

    Returns:
        maplibregl.Map object (renders in Jupyter or saves to HTML).
    """
    maplibre = _ensure_maplibregl()

    if cmap is None:
        cmap = CHOROPLETH_STOPS

    gdf = gdf.to_crs("EPSG:4326") if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf.copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    if tooltip_cols is None:
        tooltip_cols = [column]

    # Auto-center
    if center is None:
        centroid = gdf.geometry.union_all().centroid
        center = (centroid.x, centroid.y)

    style_urls = {
        "positron": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "dark-matter": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    }

    # Compute classification breaks for the color ramp
    col_data = gdf[column].dropna()
    if col_data.empty:
        raise ValueError(f"Column {column!r} has no non-null values")

    n_stops = len(cmap)
    quantiles = [col_data.quantile(i / (n_stops - 1)) for i in range(n_stops)]

    # Build MapLibre fill-color expression
    fill_color_expr = ["interpolate", ["linear"], ["get", column]]
    for q, color in zip(quantiles, cmap):
        fill_color_expr.extend([q, color])

    # Prepare GeoJSON source
    geojson_data = json.loads(gdf.to_json())

    m = maplibre.Map(
        maplibre.MapOptions(
            style=style_urls.get(style, style),
            center=list(center),
            zoom=zoom,
        ),
    )

    m.add_source("demographics", {"type": "geojson", "data": geojson_data})

    m.add_layer(
        {
            "id": "demographics-fill",
            "type": "fill",
            "source": "demographics",
            "paint": {
                "fill-color": fill_color_expr,
                "fill-opacity": opacity,
            },
        }
    )

    m.add_layer(
        {
            "id": "demographics-outline",
            "type": "line",
            "source": "demographics",
            "paint": {
                "line-color": "#666",
                "line-width": 0.5,
                "line-opacity": 0.4,
            },
        }
    )

    # Add hover tooltip
    tooltip_template = "<br>".join(
        f"<b>{col}:</b> {{{{ {col} }}}}" for col in tooltip_cols
    )
    m.add_tooltip("demographics-fill", template=tooltip_template)

    if title:
        console.print(f"[dim]maplibre: rendered {title!r} ({len(gdf)} features)[/dim]")
    else:
        console.print(f"[dim]maplibre: rendered choropleth ({len(gdf)} features)[/dim]")

    return m


def radius_ring_map(
    tracts_gdf: gpd.GeoDataFrame,
    center: tuple[float, float],
    radii_miles: list[float] | None = None,
    column: str = "median_household_income",
    property_name: str | None = None,
    ring_colors: dict[float, str] | None = None,
    **kwargs,
) -> object:
    """Broker-style radius ring map with demographic choropleth.

    Overlays 1/3/5 mile dashed circles on a tract-level choropleth.

    Args:
        tracts_gdf: GeoDataFrame with tract geometry + demographics.
        center: (lon, lat) of the subject property.
        radii_miles: List of ring radii (default [1, 3, 5]).
        column: Demographic column for choropleth fill.
        property_name: Label for the center marker popup.
        ring_colors: Dict mapping radius → color. Defaults to branded palette.
        **kwargs: Passed to demographic_map().

    Returns:
        maplibregl.Map object.
    """
    maplibre = _ensure_maplibregl()

    if radii_miles is None:
        radii_miles = [1.0, 3.0, 5.0]

    if ring_colors is None:
        default_colors = [COLORS["ring_1mi"], COLORS["ring_3mi"], COLORS["ring_5mi"]]
        ring_colors = dict(zip(sorted(radii_miles), default_colors[:len(radii_miles)]))

    # Build base demographic choropleth
    m = demographic_map(
        tracts_gdf,
        column=column,
        center=center,
        title=property_name,
        **kwargs,
    )

    # Add radius ring circles as GeoJSON line layers
    from shapely.geometry import Point, mapping

    center_point = gpd.GeoSeries([Point(center[0], center[1])], crs="EPSG:4326")
    utm_crs = center_point.estimate_utm_crs()

    for radius in sorted(radii_miles):
        radius_m = radius * 1609.344
        # Create circle polygon in UTM, project back to WGS84
        circle = center_point.to_crs(utm_crs).iloc[0].buffer(radius_m)
        circle_wgs = gpd.GeoSeries([circle], crs=utm_crs).to_crs("EPSG:4326").iloc[0]

        ring_geojson = {
            "type": "Feature",
            "geometry": mapping(circle_wgs),
            "properties": {"radius_miles": radius, "label": f"{radius}mi"},
        }

        color = ring_colors.get(radius, COLORS["primary"])
        source_id = f"ring-{radius}"

        m.add_source(source_id, {"type": "geojson", "data": ring_geojson})
        m.add_layer(
            {
                "id": f"ring-line-{radius}",
                "type": "line",
                "source": source_id,
                "paint": {
                    "line-color": color,
                    "line-width": 2.5,
                    "line-dasharray": [4, 3],
                    "line-opacity": 0.9,
                },
            }
        )

    # Add center property marker
    marker_geojson = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": list(center)},
        "properties": {"name": property_name or "Subject Property"},
    }
    m.add_source("property-marker", {"type": "geojson", "data": marker_geojson})
    m.add_layer(
        {
            "id": "property-marker-circle",
            "type": "circle",
            "source": "property-marker",
            "paint": {
                "circle-radius": 8,
                "circle-color": "#D32F2F",
                "circle-stroke-width": 2,
                "circle-stroke-color": "#FFFFFF",
            },
        }
    )

    console.print(
        f"[dim]maplibre: radius ring map with {len(radii_miles)} rings[/dim]"
    )
    return m


def isochrone_map(
    tracts_gdf: gpd.GeoDataFrame,
    isochrone_gdf: gpd.GeoDataFrame,
    column: str = "median_household_income",
    center: tuple[float, float] | None = None,
    property_name: str | None = None,
    iso_colors: dict[float, str] | None = None,
    **kwargs,
) -> object:
    """Drive-time isochrone map with demographic choropleth.

    Overlays isochrone polygons (5/10/15 min) on a tract-level choropleth.

    Args:
        tracts_gdf: GeoDataFrame with tract geometry + demographics.
        isochrone_gdf: GeoDataFrame from isochrones_nested() — must have
                       'minutes' column and polygon geometry.
        column: Demographic column for choropleth fill.
        center: (lon, lat) for the map center. If None, auto-computed.
        property_name: Label for popup.
        iso_colors: Dict mapping minutes → color.
        **kwargs: Passed to demographic_map().

    Returns:
        maplibregl.Map object.
    """
    maplibre = _ensure_maplibregl()

    if iso_colors is None:
        iso_colors = {
            5: COLORS["iso_5min"],
            10: COLORS["iso_10min"],
            15: COLORS["iso_15min"],
        }

    if center is None:
        centroid = isochrone_gdf.geometry.union_all().centroid
        center = (centroid.x, centroid.y)

    # Build base choropleth
    m = demographic_map(tracts_gdf, column=column, center=center, **kwargs)

    # Add isochrone polygons (largest first for correct stacking)
    iso_sorted = isochrone_gdf.sort_values("minutes", ascending=False)

    for _, row in iso_sorted.iterrows():
        mins = row["minutes"]
        geojson = json.loads(gpd.GeoSeries([row.geometry], crs="EPSG:4326").to_json())
        color = iso_colors.get(mins, COLORS["primary"])
        source_id = f"iso-{mins}min"

        m.add_source(source_id, {"type": "geojson", "data": geojson})

        # Semi-transparent fill
        m.add_layer(
            {
                "id": f"iso-fill-{mins}",
                "type": "fill",
                "source": source_id,
                "paint": {
                    "fill-color": color,
                    "fill-opacity": 0.1,
                },
            }
        )

        # Solid border
        m.add_layer(
            {
                "id": f"iso-line-{mins}",
                "type": "line",
                "source": source_id,
                "paint": {
                    "line-color": color,
                    "line-width": 2.5,
                    "line-opacity": 0.8,
                },
            }
        )

    # Property marker
    marker_geojson = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": list(center)},
        "properties": {"name": property_name or "Subject Property"},
    }
    m.add_source("property-marker", {"type": "geojson", "data": marker_geojson})
    m.add_layer(
        {
            "id": "property-marker-circle",
            "type": "circle",
            "source": "property-marker",
            "paint": {
                "circle-radius": 8,
                "circle-color": "#D32F2F",
                "circle-stroke-width": 2,
                "circle-stroke-color": "#FFFFFF",
            },
        }
    )

    console.print(
        f"[dim]maplibre: isochrone map with {len(iso_sorted)} rings[/dim]"
    )
    return m
