"""Visualization utilities for geospatial CRE analysis.

Two rendering tiers:

  1. Folium choropleth — quick interactive HTML maps in Jupyter, no extra deps
  2. PMTiles + MapLibre — tile-based rendering for large datasets (optional deps:
     `pip install geostack[tiles]`)

PMTiles workflow (walkerke-inspired):
    from geostack.viz import to_pmtiles, maplibre_map

    to_pmtiles(tract_gdf, "tracts.pmtiles")          # generate tiles once
    maplibre_map("tracts.pmtiles", column="total_jobs") # render in Jupyter

Usage:
    from geostack.viz import choropleth_map, employment_map

    m = choropleth_map(tract_gdf, column="total_jobs", title="Jobs by Tract")
    m.save("jobs.html")
"""

from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Folium choropleth (no extra deps — folium is in geostack[viz])
# ---------------------------------------------------------------------------

def choropleth_map(
    gdf,
    column,
    title=None,
    scheme="quantiles",
    n_bins=5,
    cmap="YlOrRd",
    tooltip_cols=None,
    tiles="CartoDB positron",
    zoom_start=10,
    opacity=0.7,
):
    """Render a choropleth map of a GeoDataFrame column using Folium.

    Args:
        gdf: GeoDataFrame with polygon geometry (any CRS — will be projected to WGS84).
        column: Column name to map.
        title: Map title shown in a top-left legend.
        scheme: mapclassify scheme — 'quantiles', 'naturalbreaks', 'equalinterval',
                'headtailbreaks', 'jenks'. Default 'quantiles'.
        n_bins: Number of classification bins.
        cmap: Matplotlib colormap name.
        tooltip_cols: List of columns to show in hover tooltip. Defaults to [column].
        tiles: Basemap tile provider string accepted by Folium.
        zoom_start: Initial zoom level.
        opacity: Fill opacity (0–1).

    Returns:
        folium.Map object. Call .save("map.html") or display inline in Jupyter.
    """
    try:
        import folium
        import mapclassify
    except ImportError:
        raise ImportError(
            "folium and mapclassify are required. Run: pip install 'geostack[viz]'"
        )

    # Ensure WGS84 for Folium
    gdf = gdf.to_crs("EPSG:4326") if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf.copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    centroid = gdf.geometry.union_all().centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=zoom_start, tiles=tiles)

    if tooltip_cols is None:
        tooltip_cols = [column]

    # Build classification bins
    col_clean = gdf[column].dropna()
    classifier = getattr(mapclassify, scheme.replace("naturalbreaks", "NaturalBreaks").title(), None)
    if classifier is None:
        classifier = mapclassify.Quantiles
    breaks = classifier(col_clean, k=n_bins)

    # mapclassify .bins returns upper edges only; Folium needs full range
    # including the data minimum as the first edge.
    bin_edges = [col_clean.min()] + list(breaks.bins)
    # Deduplicate (can happen with low-variance data)
    bin_edges = sorted(set(bin_edges))

    # Folium Choropleth needs a two-column DataFrame: [join_key, value].
    # The join_key must match GeoJSON feature.id (which is the GDF index).
    # Reset index to make it a column, then reference it by name.
    choro_df = gdf[[column]].copy()
    choro_df = choro_df.reset_index()
    join_col = choro_df.columns[0]  # Whatever the index was named (or "index")

    folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        data=choro_df,
        columns=[join_col, column],
        key_on="feature.id",
        fill_color=cmap,
        fill_opacity=opacity,
        line_opacity=0.3,
        bins=bin_edges,
        legend_name=title or column,
        name=column,
    ).add_to(m)

    # Hover tooltip
    folium.GeoJson(
        gdf,
        style_function=lambda x: {"fillOpacity": 0, "weight": 0},
        tooltip=folium.GeoJsonTooltip(fields=tooltip_cols, localize=True),
    ).add_to(m)

    if title:
        _add_title(m, title)

    return m


def employment_map(tract_gdf, lodes_df, metric="total_jobs", year=None, **kwargs):
    """Convenience wrapper: join LODES tract data to geometry and choropleth.

    Args:
        tract_gdf: GeoDataFrame from census_tracts (must have 'geoid' column).
        lodes_df: DataFrame from lodes.tract_employment() (has 'tract_geoid' column).
        metric: Column from lodes_df to map.
        year: Optional year label for the map title.
        **kwargs: Passed to choropleth_map().

    Returns:
        folium.Map with employment choropleth.
    """
    merged = tract_gdf.merge(
        lodes_df.rename(columns={"tract_geoid": "geoid"}),
        on="geoid",
        how="left",
    )
    merged[metric] = merged[metric].fillna(0)
    title = kwargs.pop("title", f"{metric.replace('_', ' ').title()}" + (f" ({year})" if year else ""))
    return choropleth_map(merged, column=metric, title=title, **kwargs)


def _add_title(m, title):
    """Inject a title HTML element into a Folium map."""
    import folium

    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50px; z-index: 1000;
                background-color: white; padding: 8px 14px;
                border-radius: 4px; box-shadow: 0 1px 5px rgba(0,0,0,0.3);
                font-family: sans-serif; font-size: 14px; font-weight: bold;">
        {title}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))


# ---------------------------------------------------------------------------
# PMTiles generation (requires: pip install geostack[tiles])
# ---------------------------------------------------------------------------

def to_pmtiles(gdf, output_path, min_zoom=4, max_zoom=14, layer_name=None):
    """Generate a PMTiles archive from a GeoDataFrame.

    PMTiles is a single-file tile archive format designed for efficient
    cloud and local serving. Use with maplibre_map() to render large datasets
    (100k+ features) that would be slow in Folium GeoJSON.

    Requires: pip install 'geostack[tiles]'  (installs freestiler)

    Args:
        gdf: GeoDataFrame with polygon or point geometry (any CRS).
        output_path: Path for the output .pmtiles file.
        min_zoom: Minimum zoom level to generate tiles for.
        max_zoom: Maximum zoom level (higher = more detail, larger file).
        layer_name: Name of the vector layer inside the tileset. Defaults to
                    the output filename stem.

    Returns:
        Path to the generated .pmtiles file.

    Example:
        to_pmtiles(tracts_gdf, "tracts.pmtiles", min_zoom=6, max_zoom=12)
        maplibre_map("tracts.pmtiles", column="total_jobs")
    """
    try:
        import freestiler
    except ImportError:
        raise ImportError(
            "freestiler is required for PMTiles generation. "
            "Run: pip install 'geostack[tiles]'"
        )

    import os

    if layer_name is None:
        layer_name = os.path.splitext(os.path.basename(output_path))[0]

    gdf = gdf.to_crs("EPSG:4326") if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf.copy()

    console.print(f"[dim]Generating PMTiles: {output_path} (zoom {min_zoom}–{max_zoom})...[/dim]")
    freestiler.freestile(
        gdf,
        output_path,
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        layer_name=layer_name,
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    console.print(f"[green]PMTiles written: {output_path} ({size_mb:.1f} MB)[/green]")
    return output_path


def maplibre_map(
    pmtiles_path,
    column=None,
    layer_name=None,
    style="positron",
    color_ramp=("white", "#08306b"),
    zoom=10,
    center=None,
    height=500,
):
    """Render a PMTiles archive in Jupyter using MapLibre GL JS.

    Requires: pip install 'geostack[tiles]'  (installs maplibre)

    Args:
        pmtiles_path: Path to a .pmtiles file generated by to_pmtiles().
        column: Property name to use for fill-color expression. If None,
                renders a flat-color fill.
        layer_name: Vector layer name inside the PMTiles file. Defaults to
                    the filename stem.
        style: Base map style — 'positron', 'dark-matter', or a MapLibre style URL.
        color_ramp: Tuple of (low_color, high_color) for the fill gradient.
        zoom: Initial zoom level.
        center: (lon, lat) tuple for initial map center. If None, uses file bounds.
        height: Map height in pixels.

    Returns:
        maplibre.Map widget (displays inline in Jupyter).
    """
    try:
        from maplibre import Map, MapOptions
    except ImportError:
        raise ImportError(
            "maplibre is required for MapLibre rendering. "
            "Run: pip install 'geostack[tiles]'"
        )

    import os

    if layer_name is None:
        layer_name = os.path.splitext(os.path.basename(pmtiles_path))[0]

    style_urls = {
        "positron": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "dark-matter": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    }
    style_url = style_urls.get(style, style)

    map_opts = MapOptions(style=style_url, zoom=zoom)
    if center:
        map_opts.center = (center[0], center[1])

    m = Map(map_opts)

    tile_url = f"pmtiles://{os.path.abspath(pmtiles_path)}"
    m.add_source(layer_name, {"type": "vector", "url": tile_url})

    if column:
        fill_color = [
            "interpolate", ["linear"],
            ["get", column],
            0, color_ramp[0],
            1, color_ramp[1],
        ]
    else:
        fill_color = color_ramp[1]

    m.add_layer(
        {
            "id": f"{layer_name}_fill",
            "type": "fill",
            "source": layer_name,
            "source-layer": layer_name,
            "paint": {"fill-color": fill_color, "fill-opacity": 0.7},
        }
    )
    m.add_layer(
        {
            "id": f"{layer_name}_outline",
            "type": "line",
            "source": layer_name,
            "source-layer": layer_name,
            "paint": {"line-color": "#333", "line-width": 0.5, "line-opacity": 0.5},
        }
    )

    return m


# ---------------------------------------------------------------------------
# Quick static plot (matplotlib — no extra deps for basic exploration)
# ---------------------------------------------------------------------------

def static_map(gdf, column=None, title=None, cmap="YlOrRd", figsize=(10, 8), **kwargs):
    """Quick static choropleth using GeoPandas + matplotlib.

    No extra dependencies — useful for non-Jupyter environments or saving
    high-resolution images.

    Args:
        gdf: GeoDataFrame with polygon geometry.
        column: Column to choropleth. If None, renders a flat fill.
        title: Plot title.
        cmap: Matplotlib colormap.
        figsize: Figure size tuple.
        **kwargs: Additional kwargs passed to gdf.plot().

    Returns:
        (fig, ax) tuple.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required. Run: pip install 'geostack[viz]'")

    gdf = gdf.to_crs("EPSG:4326") if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf

    fig, ax = plt.subplots(figsize=figsize)
    plot_kwargs = {"cmap": cmap, "linewidth": 0.3, "edgecolor": "#aaa", **kwargs}

    if column:
        gdf.plot(column=column, ax=ax, legend=True, **plot_kwargs)
    else:
        gdf.plot(ax=ax, **plot_kwargs)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    ax.set_axis_off()
    plt.tight_layout()
    return fig, ax
