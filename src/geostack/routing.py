"""Drive-time isochrone and routing utilities.

Isochrones use OpenRouteService (ORS) by default — a single API call
returns the full drive-time polygon. Set ORS_API_KEY env var (free at
https://openrouteservice.org). Falls back to OSRM ray-casting if no
ORS key is set.

Point-to-point routing and matrices still use OSRM.

Usage:
    from geostack.routing import isochrone, drive_time_matrix

    # Get a 15-minute drive-time polygon (ORS, single API call)
    poly = isochrone(lon=-96.77, lat=32.78, minutes=15)

    # Matrix of drive times between points (OSRM)
    matrix = drive_time_matrix(origins, destinations)
"""

import os

import geopandas as gpd
import requests
from dotenv import load_dotenv
from shapely.geometry import MultiPoint, Point, shape

load_dotenv()

OSRM_BASE = os.environ.get("OSRM_BASE_URL", "https://router.project-osrm.org")
ORS_BASE = os.environ.get("ORS_BASE_URL", "https://api.openrouteservice.org")
ORS_API_KEY = os.environ.get("ORS_API_KEY") or os.environ.get("ORS_API") or ""


def drive_time(origin, destination):
    """Get drive time between two points.

    Args:
        origin: (lon, lat) tuple.
        destination: (lon, lat) tuple.

    Returns:
        dict with 'duration_seconds', 'duration_minutes', 'distance_meters'.
    """
    coords = f"{origin[0]},{origin[1]};{destination[0]},{destination[1]}"
    url = f"{OSRM_BASE}/route/v1/driving/{coords}"
    resp = requests.get(url, params={"overview": "false"})
    resp.raise_for_status()
    data = resp.json()

    if data["code"] != "Ok":
        raise ValueError(f"OSRM error: {data.get('message', data['code'])}")

    route = data["routes"][0]
    return {
        "duration_seconds": route["duration"],
        "duration_minutes": round(route["duration"] / 60, 1),
        "distance_meters": route["distance"],
        "distance_miles": round(route["distance"] / 1609.34, 2),
    }


def drive_time_matrix(origins, destinations=None):
    """Compute a drive-time matrix between sets of points.

    Args:
        origins: List of (lon, lat) tuples.
        destinations: List of (lon, lat) tuples. If None, computes NxN matrix.

    Returns:
        List of lists with duration in minutes.
    """
    all_points = list(origins) + (list(destinations) if destinations else [])
    coords = ";".join(f"{p[0]},{p[1]}" for p in all_points)

    n_origins = len(origins)
    sources = ";".join(str(i) for i in range(n_origins))

    params = {"sources": sources, "annotations": "duration"}
    if destinations:
        dest_start = n_origins
        dest_end = n_origins + len(destinations) - 1
        params["destinations"] = ";".join(str(i) for i in range(dest_start, dest_end + 1))

    url = f"{OSRM_BASE}/table/v1/driving/{coords}"
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    if data["code"] != "Ok":
        raise ValueError(f"OSRM error: {data.get('message', data['code'])}")

    # Convert seconds to minutes
    return [[round(d / 60, 1) if d else None for d in row] for row in data["durations"]]


def isochrone(lon, lat, minutes=15, backend="auto"):
    """Generate a drive-time isochrone polygon.

    Preferred backend is OpenRouteService (ORS) — a single API call returns
    the full polygon. Falls back to OSRM ray-casting if ORS_API_KEY is not set.

    Args:
        lon, lat: Center point coordinates (WGS84).
        minutes: Drive time in minutes.
        backend: "auto" (ORS if key available, else OSRM), "ors", or "osrm".

    Returns:
        GeoDataFrame with isochrone polygon(s) and a 'minutes' column.
    """
    if backend == "auto":
        backend = "ors" if ORS_API_KEY else "osrm"

    if backend == "ors":
        return _isochrone_ors(lon, lat, minutes)
    else:
        return _isochrone_osrm(lon, lat, minutes)


def _isochrone_ors(lon, lat, minutes):
    """Isochrone via OpenRouteService — single API call.

    Requires ORS_API_KEY env var. Free tier: 500 isochrones/day.
    Sign up at https://openrouteservice.org/dev/#/signup
    """
    if not ORS_API_KEY:
        raise ValueError(
            "ORS_API_KEY not set. Get a free key at "
            "https://openrouteservice.org/dev/#/signup "
            "and set it: export ORS_API_KEY=your-key"
        )

    url = f"{ORS_BASE}/v2/isochrones/driving-car"
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "locations": [[lon, lat]],
        "range": [minutes * 60],  # ORS uses seconds
        "range_type": "time",
    }

    resp = requests.post(url, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "features" not in data or not data["features"]:
        raise ValueError(f"ORS returned no isochrone features: {data}")

    # ORS returns GeoJSON FeatureCollection
    geom = shape(data["features"][0]["geometry"])

    return gpd.GeoDataFrame(
        {"minutes": [minutes]},
        geometry=[geom],
        crs="EPSG:4326",
    )


def _isochrone_osrm(lon, lat, minutes):
    """Isochrone via OSRM ray-casting (fallback — slow with public server).

    Sends rays in 36 directions, binary-searches for max reachable distance,
    then builds a convex hull. Makes ~180 API calls per isochrone.
    """
    import numpy as np

    max_dist_deg = minutes * 0.01
    bearings = np.linspace(0, 360, 37)[:-1]
    reachable_points = []

    for bearing in bearings:
        rad = np.radians(bearing)
        test_dist = max_dist_deg
        for _ in range(5):
            dest_lon = lon + test_dist * np.sin(rad)
            dest_lat = lat + test_dist * np.cos(rad)
            try:
                result = drive_time((lon, lat), (dest_lon, dest_lat))
                if result["duration_minutes"] <= minutes:
                    reachable_points.append(Point(dest_lon, dest_lat))
                    test_dist *= 1.3
                else:
                    test_dist *= 0.7
            except (requests.HTTPError, ValueError):
                test_dist *= 0.5

    if len(reachable_points) < 3:
        raise ValueError("Could not compute isochrone — too few reachable points")

    hull = MultiPoint(reachable_points).convex_hull
    return gpd.GeoDataFrame(
        {"minutes": [minutes]},
        geometry=[hull],
        crs="EPSG:4326",
    )
