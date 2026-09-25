"""Zoning risk analysis for CRE due diligence.

Queries municipal ArcGIS zoning services, classifies surrounding zoning
into risk categories (supply threat, quality threat, protective buffer),
and generates severity-coded risk flags.

Usage:
    from geostack.zoning import zoning_risk
    report = zoning_risk(lat=32.78, lon=-96.80, radius_miles=1.0)
"""

from __future__ import annotations

import re

import requests
from rich.console import Console
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

console = Console()

# ---------------------------------------------------------------------------
# City registry -- maps city name to ArcGIS zoning endpoint config
# ---------------------------------------------------------------------------

ZONING_REGISTRY: dict[str, dict] = {
    "dallas": {
        "url": "https://gis.dallascityhall.com/arcgis/rest/services/sdc_public/Zoning/MapServer/15/query",
        "code_field": "ZONE_DIST",
        "desc_field": "LONG_ZONE_DIST",
    },
    "fort_worth": {
        "url": "https://mapit.fortworthtexas.gov/ags/rest/services/Planning_Development/Zoning/MapServer/25/query",
        "code_field": "ZONING",
        "desc_field": "BASE_ZONING",
    },
    "arlington": {
        "url": "https://gis.arlingtontx.gov/agsext/rest/services/Planning/Zoning/MapServer/0/query",
        "code_field": "ZONINGDETAIL",
        "desc_field": "ZONINGDESCRIPTION",
    },
    "plano": {
        "url": "https://gismaps.plano.gov/arcgis/rest/services/Planning/Zoning/MapServer/0/query",
        "code_field": "ZN",
        "desc_field": "IMS_ZONE",
    },
    "frisco": {
        "url": "https://gis.friscotexas.gov/arcgis/rest/services/Planning/Zoning/MapServer/0/query",
        "code_field": "ZONE_",
        "desc_field": None,
    },
    "mckinney": {
        "url": "https://gis.mckinneytexas.org/arcgis/rest/services/Planning/Zoning/MapServer/0/query",
        "code_field": "Zone",
        "desc_field": "ZoningType",
    },
    "denton": {
        "url": "https://gis.cityofdenton.com/arcgis/rest/services/Planning/Zoning/MapServer/0/query",
        "code_field": "ZONING",
        "desc_field": None,
    },
    "lewisville": {
        "url": "https://gis.cityoflewisville.com/arcgis/rest/services/Planning/Zoning/MapServer/0/query",
        "code_field": "ZONING_CLASS",
        "desc_field": "Zoning_Desc",
    },
    "grand_prairie": {
        "url": "https://gis.gptx.org/arcgis/rest/services/Planning/Zoning/MapServer/0/query",
        "code_field": "ZONE_CLASS",
        "desc_field": "ZONE_DESC",
    },
    "irving": {
        "url": "https://maps.cityofirving.org/arcgis/rest/services/Planning/Zoning/MapServer/0/query",
        "code_field": "District",
        "desc_field": "Description",
    },
    "garland": {
        "url": "https://gis.garlandtx.gov/arcgis/rest/services/Planning/Zoning/MapServer/0/query",
        "code_field": "GDC_ZONING",
        "desc_field": "BASE_ZONE",
    },
    "mesquite": {
        "url": "https://gis.cityofmesquite.com/arcgis/rest/services/Planning/Zoning/MapServer/0/query",
        "code_field": "DISTRICT",
        "desc_field": "BASE_ZONE",
    },
    "birmingham": {
        "url": "https://gis.birminghamal.gov/arcgis/rest/services/Planning/Zoning/MapServer/0/query",
        "code_field": "BASEZONE",
        "desc_field": "ZONING",
    },
}


# ---------------------------------------------------------------------------
# Zoning classification — prefix / regex matching
# ---------------------------------------------------------------------------

# Order matters: more specific patterns first, broader patterns last.
# Each tuple: (compiled regex, category)
_ZONE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Multifamily / mixed-use / transit-oriented (supply threat)
    (re.compile(r"^MF", re.IGNORECASE), "mf_permitted"),
    (re.compile(r"^MU", re.IGNORECASE), "mf_permitted"),
    (re.compile(r"^MX", re.IGNORECASE), "mf_permitted"),
    (re.compile(r"^TOD", re.IGNORECASE), "mf_permitted"),
    (re.compile(r"^UR$", re.IGNORECASE), "mf_permitted"),

    # Industrial (quality threat) -- must come before generic M match
    (re.compile(r"^I-?\d*$", re.IGNORECASE), "industrial"),
    (re.compile(r"^LI$", re.IGNORECASE), "industrial"),
    (re.compile(r"^HI$", re.IGNORECASE), "industrial"),
    (re.compile(r"^M-\d", re.IGNORECASE), "industrial"),
    (re.compile(r"^IP$", re.IGNORECASE), "industrial"),
    (re.compile(r"^BP$", re.IGNORECASE), "industrial"),

    # Heavy commercial (quality threat) -- C-3, C-4, HC, GC, CR
    (re.compile(r"^C-[3-9]", re.IGNORECASE), "heavy_commercial"),
    (re.compile(r"^HC$", re.IGNORECASE), "heavy_commercial"),
    (re.compile(r"^GC$", re.IGNORECASE), "heavy_commercial"),
    (re.compile(r"^CR$", re.IGNORECASE), "heavy_commercial"),

    # Light commercial -- C-1, C-2, O, NS, LR, LC, B
    (re.compile(r"^C-[12]", re.IGNORECASE), "light_commercial"),
    (re.compile(r"^O-?\d*$", re.IGNORECASE), "light_commercial"),
    (re.compile(r"^NS$", re.IGNORECASE), "light_commercial"),
    (re.compile(r"^LR$", re.IGNORECASE), "light_commercial"),
    (re.compile(r"^LC$", re.IGNORECASE), "light_commercial"),
    (re.compile(r"^B-?\d*$", re.IGNORECASE), "light_commercial"),

    # Single family / residential / agricultural (protective buffer)
    (re.compile(r"^SF", re.IGNORECASE), "single_family"),
    (re.compile(r"^R-?\d", re.IGNORECASE), "single_family"),
    (re.compile(r"^RS", re.IGNORECASE), "single_family"),
    (re.compile(r"^A$", re.IGNORECASE), "single_family"),
    (re.compile(r"^AG$", re.IGNORECASE), "single_family"),
    (re.compile(r"^RE$", re.IGNORECASE), "single_family"),
    (re.compile(r"^E$", re.IGNORECASE), "single_family"),

    # Other / special districts
    (re.compile(r"^PD", re.IGNORECASE), "other"),
    (re.compile(r"^P$", re.IGNORECASE), "other"),
    (re.compile(r"^PK", re.IGNORECASE), "other"),
    (re.compile(r"^FW$", re.IGNORECASE), "other"),
    (re.compile(r"^SUP", re.IGNORECASE), "other"),
]

# Keywords in PD descriptions that indicate a specific base category
_PD_DESC_KEYWORDS: dict[str, str] = {
    "multifamily": "mf_permitted",
    "multi-family": "mf_permitted",
    "mixed use": "mf_permitted",
    "mixed-use": "mf_permitted",
    "apartment": "mf_permitted",
    "single family": "single_family",
    "single-family": "single_family",
    "residential": "single_family",
    "industrial": "industrial",
    "commercial": "light_commercial",
}


def classify_zone(code: str, description: str | None = None) -> str:
    """Classify a zoning code into one of 6 risk categories.

    Categories: mf_permitted, industrial, heavy_commercial, light_commercial,
    single_family, other.

    Args:
        code: Raw zoning code from the city's GIS service (e.g., "MF-2", "I-1").
        description: Optional description field. Used for PD districts to infer
            the base zoning from the ordinance description.

    Returns:
        One of the 6 category strings.
    """
    code = (code or "").strip()
    if not code:
        return "other"

    # Check regex patterns
    for pattern, category in _ZONE_PATTERNS:
        if pattern.match(code):
            # For PD districts, try to refine using the description
            if category == "other" and code.upper().startswith("PD") and description:
                desc_lower = description.lower()
                for keyword, pd_category in _PD_DESC_KEYWORDS.items():
                    if keyword in desc_lower:
                        return pd_category
            return category

    return "other"


def _query_zoning_service(
    city: str,
    lat: float,
    lon: float,
    radius_miles: float | None = None,
    return_geometry: bool = True,
) -> list[dict]:
    """Query a city's ArcGIS zoning endpoint.

    Args:
        city: City key from ZONING_REGISTRY (e.g., "dallas").
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        radius_miles: If set, query within this radius. If None, point query.
        return_geometry: Whether to include polygon geometry in response.

    Returns:
        List of raw feature dicts (each with "attributes" and optionally "geometry").

    Raises:
        ValueError: If city is not in ZONING_REGISTRY.
    """
    city_key = city.lower().replace(" ", "_")
    if city_key not in ZONING_REGISTRY:
        raise ValueError(f"{city!r} not in ZONING_REGISTRY")

    config = ZONING_REGISTRY[city_key]
    url = config["url"]
    code_field = config["code_field"]
    desc_field = config.get("desc_field")

    # Build outFields
    out_fields = code_field
    if desc_field:
        out_fields += f",{desc_field}"

    params: dict[str, str] = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "where": "1=1",
        "outFields": out_fields,
        "returnGeometry": "true" if return_geometry else "false",
        "outSR": "4326",
        "f": "json",
        "resultRecordCount": "1000",
    }

    if radius_miles is not None:
        params["distance"] = str(radius_miles)
        params["units"] = "esriSRUnit_StatuteMile"

    label = city.replace("_", " ").title()
    console.print(f"[dim]Zoning: querying {label} ArcGIS service...[/dim]")

    all_features: list[dict] = []
    offset = 0

    while True:
        if offset > 0:
            params["resultOffset"] = str(offset)

        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        all_features.extend(features)

        # Check for pagination
        if data.get("exceededTransferLimit") and len(features) > 0:
            offset += len(features)
        else:
            break

    console.print(f"[green]Zoning: {len(all_features)} features from {label}[/green]")
    return all_features


def zoning_lookup(lat: float, lon: float) -> dict | None:
    """Get zoning classification for a single point.

    Tries each city's ArcGIS service sequentially until one returns a result.
    For v1, this is simpler than loading TIGER boundaries for city detection.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        Dict with keys: code, description, category, city, overlay, pd_number.
        Returns None if the point is not covered by any registered city.
    """
    for city_key, config in ZONING_REGISTRY.items():
        try:
            features = _query_zoning_service(city=city_key, lat=lat, lon=lon)
        except Exception:
            # Service timeout or error -- skip to next city
            continue

        if not features:
            continue

        attrs = features[0].get("attributes", {})
        code_field = config["code_field"]
        desc_field = config.get("desc_field")

        code = attrs.get(code_field, "")
        description = attrs.get(desc_field) if desc_field else None

        return {
            "code": code,
            "description": description,
            "category": classify_zone(code, description=description),
            "city": city_key,
            "overlay": None,  # v1: not extracted from most services
            "pd_number": None,  # v1: not extracted from most services
        }

    return None


def _compute_polygon_acres(geometry: dict) -> float:
    """Compute acreage from an ArcGIS polygon geometry dict.

    Converts the ArcGIS rings format to a Shapely polygon, projects
    to EPSG:2276 (TX State Plane North Central, feet), and computes area.

    Args:
        geometry: ArcGIS geometry dict with "rings" key.

    Returns:
        Area in acres. Returns 0.0 if geometry is invalid.
    """
    if not geometry or not geometry.get("rings"):
        return 0.0

    try:
        # ArcGIS returns rings as [[[x, y], ...]]
        # Convert to GeoJSON-compatible format for Shapely
        rings = geometry["rings"]
        if len(rings) == 1:
            geojson = {"type": "Polygon", "coordinates": rings}
        else:
            # First ring is exterior, rest are holes
            geojson = {"type": "Polygon", "coordinates": rings}

        poly = shape(geojson)
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Project to EPSG:2276 (TX State Plane North Central, feet)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2276", always_xy=True)
        projected = transform(transformer.transform, poly)

        # Convert sq feet to acres (1 acre = 43,560 sq ft)
        acres = projected.area / 43_560.0
        return round(acres, 2)
    except Exception:
        return 0.0


def zoning_within_radius(
    lat: float,
    lon: float,
    radius_miles: float = 1.0,
) -> list[dict]:
    """Get all zoning polygons within a radius, classified with acreage.

    Queries all registered cities' ArcGIS services and combines results.
    Each zone polygon is classified and its acreage computed via projection
    to EPSG:2276 (TX State Plane North Central).

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        radius_miles: Search radius in miles (default 1.0).

    Returns:
        List of zone dicts, each with keys: code, description, category,
        acres, city.
    """
    all_zones: list[dict] = []

    for city_key, config in ZONING_REGISTRY.items():
        try:
            features = _query_zoning_service(
                city=city_key, lat=lat, lon=lon,
                radius_miles=radius_miles, return_geometry=True,
            )
        except Exception:
            # Service timeout or error -- skip to next city
            continue

        code_field = config["code_field"]
        desc_field = config.get("desc_field")

        for feat in features:
            attrs = feat.get("attributes", {})
            code = attrs.get(code_field, "")
            description = attrs.get(desc_field) if desc_field else None
            geometry = feat.get("geometry")

            acres = _compute_polygon_acres(geometry) if geometry else 0.0

            all_zones.append({
                "code": code,
                "description": description,
                "category": classify_zone(code, description=description),
                "acres": acres,
                "city": city_key,
            })

    return all_zones


def zoning_risk(
    lat: float,
    lon: float,
    radius_miles: float = 1.0,
) -> dict:
    """Assess zoning-based supply and quality threats around a property.

    Primary function. Queries surrounding zoning within a radius, classifies
    into risk categories, and generates severity-coded risk flags.

    Note: vacant_mf_zoned_acres requires CAD parcel data in PostGIS. Without
    it, this field will be None and supply threat is based on total MF-zoned
    acreage and percentage instead.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        radius_miles: Search radius in miles (default 1.0).

    Returns:
        Dict with keys: query, city, subject_zoning, summary, risk_flags, zones.
    """
    console.print(
        f"\n[bold]Zoning risk scan at ({lat:.4f}, {lon:.4f}), "
        f"{radius_miles}mi radius[/bold]"
    )

    # 1. Get subject site zoning
    subject = zoning_lookup(lat, lon)

    # 2. Get all zoning within radius
    zones = zoning_within_radius(lat, lon, radius_miles=radius_miles)

    # 3. Aggregate by category
    total_acres = sum(z["acres"] for z in zones)
    mf_acres = sum(z["acres"] for z in zones if z["category"] == "mf_permitted")
    industrial_acres = sum(z["acres"] for z in zones if z["category"] == "industrial")
    heavy_commercial_acres = sum(z["acres"] for z in zones if z["category"] == "heavy_commercial")
    single_family_acres = sum(z["acres"] for z in zones if z["category"] == "single_family")
    light_commercial_acres = sum(z["acres"] for z in zones if z["category"] == "light_commercial")

    mf_pct = (mf_acres / total_acres * 100) if total_acres > 0 else 0.0
    industrial_pct = (industrial_acres / total_acres * 100) if total_acres > 0 else 0.0

    # v1: vacant MF detection requires CAD data in PostGIS -- set to None
    vacant_mf_zoned_acres = None

    # 4. Risk flag rules
    risk_flags: list[dict] = []

    # Supply threat
    if vacant_mf_zoned_acres is not None and vacant_mf_zoned_acres > 20:
        supply_threat = "HIGH"
        risk_flags.append({
            "flag": f"{vacant_mf_zoned_acres} acres of vacant MF-zoned land within {radius_miles} mile(s)",
            "severity": "HIGH",
        })
    elif vacant_mf_zoned_acres is not None and vacant_mf_zoned_acres > 5:
        supply_threat = "MEDIUM"
        risk_flags.append({
            "flag": f"{vacant_mf_zoned_acres} acres of vacant MF-zoned land within {radius_miles} mile(s)",
            "severity": "MEDIUM",
        })
    elif mf_pct > 30:
        supply_threat = "MEDIUM"
        risk_flags.append({
            "flag": f"High density of MF-zoned land ({mf_pct:.0f}% > 30%)",
            "severity": "MEDIUM",
        })
    elif mf_acres > 20:
        supply_threat = "MEDIUM"
        risk_flags.append({
            "flag": f"{mf_acres:.1f} acres of MF-zoned land within {radius_miles} mile(s)",
            "severity": "MEDIUM",
        })
    else:
        supply_threat = "LOW"

    # Quality threat
    if industrial_pct > 15:
        quality_threat = "HIGH"
        risk_flags.append({
            "flag": f"Significant industrial zoning nearby ({industrial_pct:.0f}% of area)",
            "severity": "HIGH",
        })
    elif industrial_acres + heavy_commercial_acres > 20:
        quality_threat = "MEDIUM"
        risk_flags.append({
            "flag": (
                f"{industrial_acres:.1f} acres industrial + "
                f"{heavy_commercial_acres:.1f} acres heavy commercial within {radius_miles} mile(s)"
            ),
            "severity": "MEDIUM",
        })
    else:
        quality_threat = "LOW"

    # Additional flags
    if industrial_acres > 0 and quality_threat != "HIGH":
        risk_flags.append({
            "flag": f"{industrial_acres:.1f} acres of industrial zoning within {radius_miles} mile(s)",
            "severity": "LOW",
        })

    summary = {
        "total_parcels_in_radius": len(zones),
        "mf_permitted_acres": round(mf_acres, 1),
        "mf_permitted_pct": round(mf_pct, 1),
        "industrial_acres": round(industrial_acres, 1),
        "industrial_pct": round(industrial_pct, 1),
        "heavy_commercial_acres": round(heavy_commercial_acres, 1),
        "light_commercial_acres": round(light_commercial_acres, 1),
        "single_family_acres": round(single_family_acres, 1),
        "vacant_mf_zoned_acres": vacant_mf_zoned_acres,
        "supply_threat": supply_threat,
        "quality_threat": quality_threat,
    }

    report = {
        "query": {
            "lat": lat,
            "lon": lon,
            "radius_miles": radius_miles,
        },
        "city": subject["city"] if subject else None,
        "subject_zoning": subject,
        "summary": summary,
        "risk_flags": risk_flags,
        "zones": zones,
    }

    # Print summary
    supply_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}[supply_threat]
    quality_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}[quality_threat]
    console.print(f"\n[bold]Zoning Risk Summary[/bold]")
    if subject:
        console.print(f"  Subject: {subject['code']} ({subject['category']})")
    console.print(f"  Total zones in radius: {len(zones)}")
    console.print(f"  MF-permitted: {mf_acres:.1f} ac ({mf_pct:.0f}%)")
    console.print(f"  Industrial: {industrial_acres:.1f} ac ({industrial_pct:.0f}%)")
    console.print(f"  Heavy commercial: {heavy_commercial_acres:.1f} ac")
    console.print(f"  Single family: {single_family_acres:.1f} ac")
    console.print(f"  [bold {supply_color}]Supply threat: {supply_threat}[/bold {supply_color}]")
    console.print(f"  [bold {quality_color}]Quality threat: {quality_threat}[/bold {quality_color}]")
    for flag in risk_flags:
        icon = {"HIGH": "[red]!![/red]", "MEDIUM": "[yellow]![/yellow]", "LOW": "[dim].[/dim]"}
        console.print(f"  {icon.get(flag['severity'], '.')} {flag['flag']}")

    return report
