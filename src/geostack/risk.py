"""Flood and climate risk lookup for property underwriting.

Data source: FEMA National Flood Hazard Layer (NFHL) via ArcGIS REST.

Usage:
    from geostack.risk import flood_zone
    result = flood_zone(lon=-95.373, lat=29.761)
"""

from __future__ import annotations

import requests
from rich.console import Console

console = Console()

_NFHL_URL = (
    "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
    "/28/query"
)


def flood_zone(lon: float, lat: float) -> dict:
    """Query FEMA NFHL for the flood zone at a given coordinate.

    Returns:
        dict with zone, sfha (bool), zone_subtype, unmapped (bool).
    """
    resp = requests.get(_NFHL_URL, params={
        "where": "1=1",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF",
        "returnGeometry": "false",
        "f": "json",
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    features = data.get("features", [])
    if not features:
        console.print(f"[yellow]No NFHL coverage at ({lat:.4f}, {lon:.4f})[/yellow]")
        return {"zone": None, "sfha": False, "zone_subtype": None, "unmapped": True}

    attrs = features[0]["attributes"]
    zone = attrs.get("FLD_ZONE")
    sfha = attrs.get("SFHA_TF", "F") == "T"
    subtype = attrs.get("ZONE_SUBTY")

    icon = "[red]SFHA[/red]" if sfha else "[green]OK[/green]"
    console.print(f"Flood zone: {zone} ({subtype}) — {icon}")
    return {"zone": zone, "sfha": sfha, "zone_subtype": subtype, "unmapped": False}
