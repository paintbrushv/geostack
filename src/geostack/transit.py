"""Transit and walkability analysis for workforce housing.

Data sources:
  - Walk Score API (requires API key)
  - GTFS feeds (e.g., DART for Dallas-Fort Worth)

Usage:
    from geostack.transit import transit_profile

    profile = transit_profile(lon=-96.6183, lat=32.8028,
                              address="245 E I-30, Garland TX",
                              api_key="YOUR_KEY",
                              gtfs_path="data/dart_gtfs.zip")
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from rich.console import Console
from shapely.geometry import Point

console = Console()

_WALKSCORE_URL = "https://api.walkscore.com/score"
_MILES_PER_DEGREE = 69.0  # approximate at DFW latitude


def walk_score(lon: float, lat: float, address: str,
               api_key: str | None = None) -> dict:
    """Fetch Walk Score, Transit Score, and Bike Score for a location."""
    key = api_key or os.environ.get("WALKSCORE_API_KEY")
    if not key:
        console.print("[yellow]No Walk Score API key. Set WALKSCORE_API_KEY.[/yellow]")
        return {"walkscore": None, "transit_score": None, "bike_score": None,
                "description": None}

    resp = requests.get(_WALKSCORE_URL, params={
        "format": "json",
        "address": address,
        "lat": lat,
        "lon": lon,
        "transit": 1,
        "bike": 1,
        "wsapikey": key,
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != 1:
        console.print(f"[yellow]Walk Score API status {data.get('status')}[/yellow]")
        return {"walkscore": None, "transit_score": None, "bike_score": None,
                "description": None}

    result = {
        "walkscore": data.get("walkscore"),
        "description": data.get("description"),
        "transit_score": data.get("transit", {}).get("score") if data.get("transit") else None,
        "bike_score": data.get("bike", {}).get("score") if data.get("bike") else None,
    }

    console.print(
        f"\n[bold]Walk Score:[/bold]"
        f"\n  Walk: {result['walkscore']} — {result['description']}"
        f"\n  Transit: {result['transit_score']}"
        f"\n  Bike: {result['bike_score']}"
    )
    return result


def _load_gtfs_stops(gtfs_path: str) -> gpd.GeoDataFrame:
    """Load stops from a GTFS zip file into a GeoDataFrame."""
    with zipfile.ZipFile(gtfs_path) as zf:
        stops = pd.read_csv(zf.open("stops.txt"), dtype=str)
        stop_times = pd.read_csv(zf.open("stop_times.txt"), dtype=str,
                                 usecols=["trip_id", "stop_id"])
        trips = pd.read_csv(zf.open("trips.txt"), dtype=str,
                            usecols=["trip_id", "route_id"])
        routes = pd.read_csv(zf.open("routes.txt"), dtype=str,
                             usecols=["route_id", "route_type"])

    trip_routes = trips.merge(routes, on="route_id", how="left")
    stop_routes = stop_times.merge(trip_routes, on="trip_id", how="left")
    stop_routes["route_type"] = pd.to_numeric(stop_routes["route_type"], errors="coerce")
    stop_best = stop_routes.groupby("stop_id")["route_type"].min().reset_index()

    stops = stops.merge(stop_best, on="stop_id", how="left")
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])

    gdf = gpd.GeoDataFrame(
        stops[["stop_id", "stop_name", "route_type"]],
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )
    return gdf


def _count_nearby_stops(lon: float, lat: float, stops_gdf: gpd.GeoDataFrame,
                        radius_miles: float = 0.5) -> dict:
    """Count transit stops within a radius of the property."""
    pt = Point(lon, lat)
    stops_gdf = stops_gdf.copy()
    stops_gdf["dist_deg"] = stops_gdf.geometry.distance(pt)
    stops_gdf["dist_mi"] = stops_gdf["dist_deg"] * _MILES_PER_DEGREE

    nearby = stops_gdf[stops_gdf["dist_mi"] <= radius_miles]
    rail_stops = nearby[nearby["route_type"].isin([0, 1, 2])]
    bus_stops = nearby[nearby["route_type"] == 3]

    nearest_rail_name = None
    nearest_rail_dist = None
    if not rail_stops.empty:
        closest = rail_stops.sort_values("dist_mi").iloc[0]
        nearest_rail_name = closest["stop_name"]
        nearest_rail_dist = closest["dist_mi"]

    return {
        "stops_within_radius": len(nearby),
        "rail_stops_within_radius": len(rail_stops),
        "bus_stops_within_radius": len(bus_stops),
        "nearest_rail_name": nearest_rail_name,
        "nearest_rail_dist_mi": nearest_rail_dist,
        "radius_miles": radius_miles,
    }


def transit_profile(lon: float, lat: float, address: str,
                    api_key: str | None = None,
                    gtfs_path: str | None = None,
                    radius_miles: float = 0.5) -> dict:
    """Combined transit and walkability profile for a property."""
    ws = walk_score(lon, lat, address, api_key)

    gtfs_result = {}
    if gtfs_path:
        stops = _load_gtfs_stops(gtfs_path)
        gtfs_result = _count_nearby_stops(lon, lat, stops, radius_miles)

    return {
        **ws,
        "stops_within_half_mile": gtfs_result.get("stops_within_radius", 0),
        "rail_stops": gtfs_result.get("rail_stops_within_radius", 0),
        "bus_stops": gtfs_result.get("bus_stops_within_radius", 0),
        "nearest_rail": gtfs_result.get("nearest_rail_name"),
        "nearest_rail_dist_mi": gtfs_result.get("nearest_rail_dist_mi"),
    }
