"""Multi-provider isochrone generation via routingpy.

Drop-in alternative to geostack.routing — uses routingpy for access to
Mapbox, Valhalla, ORS, Google, OSRM, and other routing backends through
a single consistent interface. Existing routing.py is unchanged.

Install:  pip install 'geostack[routing-multi]'

Usage:
    from geostack.routing_multi import (
        isochrone_multi,
        isochrones_nested,
        list_providers,
    )

    # Single isochrone via Mapbox
    gdf = isochrone_multi(lon=-96.77, lat=32.78, minutes=15, provider="mapbox")

    # Nested isochrones (5, 10, 15 min) for broker-style maps
    gdf = isochrones_nested(lon=-96.77, lat=32.78, minutes=[5, 10, 15])
"""

from __future__ import annotations

import os

import geopandas as gpd
from dotenv import load_dotenv
from rich.console import Console
from shapely.geometry import Polygon, shape

console = Console()
load_dotenv()


# Provider configuration: env var for API key, routingpy client class name
_PROVIDERS = {
    "mapbox": {
        "env_key": "MAPBOX_ACCESS_TOKEN",
        "client": "MapboxOSRM",
        "profile": "driving",
    },
    "ors": {
        "env_key": "ORS_API_KEY",
        "client": "ORS",
        "profile": "driving-car",
    },
    "valhalla": {
        "env_key": "VALHALLA_API_KEY",
        "client": "Valhalla",
        "profile": "auto",
    },
    "google": {
        "env_key": "GOOGLE_MAPS_API_KEY",
        "client": "Google",
        "profile": "driving",
    },
    "osrm": {
        "env_key": None,  # Public server, no key needed
        "client": "OSRM",
        "profile": "driving",
    },
}


def list_providers() -> dict[str, bool]:
    """List available routing providers and whether their API key is configured.

    Returns:
        Dict mapping provider name to availability (True if key is set or not needed).
    """
    result = {}
    for name, cfg in _PROVIDERS.items():
        if cfg["env_key"] is None:
            result[name] = True
        else:
            result[name] = bool(os.environ.get(cfg["env_key"]))
    return result


def _get_client(provider: str):
    """Instantiate a routingpy client for the given provider."""
    try:
        import routingpy
    except ImportError:
        raise ImportError(
            "routingpy is required. Run: pip install 'geostack[routing-multi]'"
        )

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            f"Available: {', '.join(_PROVIDERS)}"
        )

    cfg = _PROVIDERS[provider]
    api_key = os.environ.get(cfg["env_key"]) if cfg["env_key"] else None

    if cfg["env_key"] and not api_key:
        raise ValueError(
            f"API key not set for {provider}. "
            f"Set {cfg['env_key']} env var."
        )

    client_cls = getattr(routingpy.routers, cfg["client"], None)
    if client_cls is None:
        # Fallback: try direct import
        client_cls = getattr(routingpy, cfg["client"], None)

    if client_cls is None:
        raise ImportError(
            f"routingpy does not have client {cfg['client']}. "
            f"Check routingpy version."
        )

    if api_key:
        return client_cls(api_key=api_key)
    return client_cls()


def _auto_provider() -> str:
    """Pick the best available provider based on configured API keys.

    Priority: mapbox > ors > valhalla > google > osrm (public, last resort).
    """
    priority = ["mapbox", "ors", "valhalla", "google", "osrm"]
    for name in priority:
        cfg = _PROVIDERS[name]
        if cfg["env_key"] is None or os.environ.get(cfg["env_key"]):
            return name
    return "osrm"


def isochrone_multi(
    lon: float,
    lat: float,
    minutes: int | float = 15,
    provider: str = "auto",
    profile: str | None = None,
) -> gpd.GeoDataFrame:
    """Generate a drive-time isochrone polygon via routingpy.

    Args:
        lon, lat: Center point (WGS84).
        minutes: Drive time in minutes.
        provider: "mapbox", "ors", "valhalla", "google", "osrm", or "auto".
        profile: Routing profile override. If None, uses provider default.

    Returns:
        GeoDataFrame with columns: minutes, provider, geometry.
    """
    if provider == "auto":
        provider = _auto_provider()

    cfg = _PROVIDERS[provider]
    profile = profile or cfg["profile"]

    console.print(
        f"[dim]routingpy: {provider} isochrone {minutes}min at ({lat:.4f}, {lon:.4f})[/dim]"
    )

    client = _get_client(provider)

    # routingpy isochrones() takes seconds
    range_seconds = [int(minutes * 60)]

    isochrones = client.isochrones(
        locations=[lon, lat],
        profile=profile,
        intervals=range_seconds,
    )

    if not isochrones:
        raise ValueError(f"No isochrone returned from {provider}")

    # routingpy returns Isochrone objects with .geometry attribute
    iso = isochrones[0]
    geom = _parse_isochrone_geometry(iso.geometry)

    return gpd.GeoDataFrame(
        {"minutes": [minutes], "provider": [provider]},
        geometry=[geom],
        crs="EPSG:4326",
    )


def _parse_isochrone_geometry(geom):
    """Convert routingpy geometry to a Shapely geometry.

    routingpy providers return geometry in different formats:
    - dict (GeoJSON) → use shape()
    - list of [lon, lat] pairs → exterior ring of a Polygon
    - Shapely geometry → pass through
    """
    if isinstance(geom, dict):
        return shape(geom)
    elif isinstance(geom, list):
        # List of coordinate pairs → Polygon exterior ring
        if len(geom) > 0 and isinstance(geom[0], (list, tuple)):
            return Polygon(geom)
        else:
            raise ValueError(f"Cannot parse geometry list: {geom[:3]}...")
    else:
        # Already a Shapely geometry
        return geom


def isochrones_nested(
    lon: float,
    lat: float,
    minutes: list[int | float] | None = None,
    provider: str = "auto",
    profile: str | None = None,
) -> gpd.GeoDataFrame:
    """Generate nested isochrones (e.g., 5/10/15 min) in a single GeoDataFrame.

    Returns polygons sorted largest-first for correct overlay rendering
    (paint the 15-min ring first, then 10, then 5 on top).

    Args:
        lon, lat: Center point (WGS84).
        minutes: List of drive times. Default [5, 10, 15].
        provider: Routing provider or "auto".
        profile: Routing profile override.

    Returns:
        GeoDataFrame with one row per isochrone, sorted descending by minutes.
    """
    if minutes is None:
        minutes = [5, 10, 15]

    if provider == "auto":
        provider = _auto_provider()

    cfg = _PROVIDERS[provider]
    profile = profile or cfg["profile"]

    console.print(
        f"[dim]routingpy: {provider} nested isochrones {minutes} at ({lat:.4f}, {lon:.4f})[/dim]"
    )

    client = _get_client(provider)

    # Some providers support multiple intervals in one call
    range_seconds = sorted([int(m * 60) for m in minutes])

    try:
        isochrones = client.isochrones(
            locations=[lon, lat],
            profile=profile,
            intervals=range_seconds,
        )
    except Exception:
        # Fallback: individual calls
        console.print(f"[dim]Batch isochrone failed, falling back to individual calls[/dim]")
        frames = []
        for m in minutes:
            gdf = isochrone_multi(lon, lat, minutes=m, provider=provider, profile=profile)
            frames.append(gdf)
        return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326").sort_values(
            "minutes", ascending=False
        ).reset_index(drop=True)

    rows = []
    for iso in isochrones:
        geom = _parse_isochrone_geometry(iso.geometry)
        # Determine which minute value this isochrone corresponds to
        iso_minutes = getattr(iso, "interval", None)
        if iso_minutes is not None:
            iso_minutes = iso_minutes / 60
        rows.append({"geometry": geom, "provider": provider, "minutes": iso_minutes})

    import pandas as pd

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # If minutes weren't populated from the API, assign from input
    if gdf["minutes"].isna().all():
        sorted_minutes = sorted(minutes, reverse=False)
        for i, m in enumerate(sorted_minutes):
            if i < len(gdf):
                gdf.loc[i, "minutes"] = m

    # Sort descending for correct overlay rendering
    gdf = gdf.sort_values("minutes", ascending=False).reset_index(drop=True)

    console.print(f"[dim]routingpy: {len(gdf)} isochrones generated[/dim]")
    return gdf
