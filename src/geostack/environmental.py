"""EPA environmental screening for CRE due diligence.

Queries EPA ECHO and CIMC databases for regulated facilities,
Superfund/NPL sites, brownfields, and RCRA corrective action
within a radius of a property. Generates severity-coded risk flags
aligned with ASTM E1527-21 Phase I ESA search radii.

Usage:
    from geostack.environmental import screen_site
    report = screen_site(lat=32.78, lon=-96.80)
"""

from __future__ import annotations

import datetime

import requests
from rich.console import Console

from geostack.risk import flood_zone

console = Console()

ECHO_BASE = "https://echodata.epa.gov/echo"


def echo_facility_search(
    lat: float,
    lon: float,
    radius_miles: float = 1.0,
    media: str | None = None,
    compliance_status: int = 1,
    page_size: int = 100,
) -> list[dict]:
    """Query EPA ECHO for regulated facilities within a radius.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        radius_miles: Search radius in miles (1, 2, 3, 4, 5, 10, or 20).
        media: Program filter — 'A' (Air/CAA), 'W' (Water/CWA),
               'R' (RCRA), 'S' (SDWA), or None for all.
        compliance_status: 1=all, 2=any violation, 3=significant violator.
        page_size: Results per page (max 1000).

    Returns:
        List of facility dicts with keys like FacName, FacLat, FacLong,
        RegistryID, FacSNCFlg, FacComplianceStatus, etc.
    """
    params = {
        "output": "JSON",
        "p_lat": str(lat),
        "p_long": str(lon),
        "p_radius": str(int(radius_miles)),
        "p_cs": str(compliance_status),
        "responseset": str(page_size),
    }
    if media:
        params["p_med"] = media

    console.print(f"[dim]ECHO: searching {radius_miles}mi around ({lat:.4f}, {lon:.4f})...[/dim]")

    resp = requests.get(
        f"{ECHO_BASE}/echo_rest_services.get_facilities",
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    results = data.get("Results", {})
    query_id = results.get("QueryID")
    total_rows = int(results.get("QueryRows", "0"))

    if not query_id or total_rows == 0:
        console.print("[dim]ECHO: 0 facilities found[/dim]")
        return []

    console.print(f"[dim]ECHO: {total_rows} facilities, fetching details...[/dim]")

    # Fetch paginated results
    all_facilities = []
    page = 1
    while len(all_facilities) < total_rows:
        qid_resp = requests.get(
            f"{ECHO_BASE}/echo_rest_services.get_qid",
            params={
                "output": "JSON",
                "qid": query_id,
                "pageno": str(page),
                "pagesize": str(page_size),
            },
            timeout=60,
        )
        qid_resp.raise_for_status()
        qid_data = qid_resp.json()
        facilities = qid_data.get("Results", {}).get("Facilities", [])
        if not facilities:
            break
        all_facilities.extend(facilities)
        page += 1

    console.print(f"[green]ECHO: {len(all_facilities)} facilities loaded[/green]")
    return all_facilities


CIMC_URL = "https://map22.epa.gov/arcgis/rest/services/cimc/Cleanups/MapServer/0/query"

_CIMC_PROGRAM_FILTERS = {
    "superfund": "SF_NPL_CODE_ALL IS NOT NULL",
    "brownfield": "BF_PROPERTY_ID IS NOT NULL",
    "rcra": "RCRA_HANDLER_ID IS NOT NULL",
}


def cimc_cleanup_sites(
    lat: float,
    lon: float,
    radius_miles: float = 1.0,
    program: str | None = None,
) -> list[dict]:
    """Query EPA CIMC for cleanup/Superfund/brownfield sites within a radius.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        radius_miles: Search radius in miles.
        program: Filter to 'superfund', 'brownfield', or 'rcra'.
                 None returns all cleanup programs.

    Returns:
        List of site dicts with keys like PRIMARY_NAME, EPA_ID,
        LATITUDE, LONGITUDE, SF_NPL_CODE, BF_PROPERTY_ID, etc.
    """
    where = _CIMC_PROGRAM_FILTERS.get(program, "1=1") if program else "1=1"

    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "distance": str(radius_miles),
        "units": "esriSRUnit_StatuteMile",
        "where": where,
        "outFields": (
            "PRIMARY_NAME,EPA_ID,REGISTRY_ID,LOCATION_ADDRESS,"
            "CITY_NAME,STATE_CODE,POSTAL_CODE,COUNTY_NAME,"
            "LATITUDE,LONGITUDE,"
            "SF_SITE_ID,SF_NPL_CODE,SF_NPL_CODE_ALL,"
            "BF_PROPERTY_ID,BF_ASSESS_IND,BF_CLEANUP_IND,"
            "RCRA_HANDLER_ID,RCRA_CURRENT_IND,RCRA_REMEDY_SEL_IND,"
            "LUST_ID,UST_STATUS"
        ),
        "returnGeometry": "false",
        "f": "json",
        "resultRecordCount": "1000",
    }

    label = program or "all cleanup"
    console.print(f"[dim]CIMC: searching {label} sites within {radius_miles}mi...[/dim]")

    resp = requests.get(CIMC_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    features = data.get("features", [])
    results = [f["attributes"] for f in features]

    console.print(f"[green]CIMC: {len(results)} {label} sites found[/green]")
    return results


def screen_site(
    lat: float,
    lon: float,
    radius_miles: float = 1.0,
    include_flood: bool = True,
) -> dict:
    """Run preliminary environmental screening for a site.

    Queries EPA ECHO and CIMC databases, optionally FEMA flood zones,
    and generates severity-coded risk flags aligned with ASTM E1527-21.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        radius_miles: Search radius in miles (default 1.0).
        include_flood: Also query FEMA flood zone (default True).

    Returns:
        dict with keys: query, summary, risk_flags, echo_facilities,
        cleanup_sites, flood_zone.
    """
    console.print(f"\n[bold]Environmental screening at ({lat:.4f}, {lon:.4f}), {radius_miles}mi radius[/bold]")

    # Query EPA databases
    facilities = echo_facility_search(lat, lon, radius_miles=radius_miles)
    cleanup = cimc_cleanup_sites(lat, lon, radius_miles=radius_miles)

    # Classify cleanup sites
    npl_sites = [s for s in cleanup if s.get("SF_NPL_CODE") in ("F", "P")]
    brownfield_sites = [s for s in cleanup if s.get("BF_PROPERTY_ID")]
    rcra_ca_sites = [s for s in cleanup if s.get("RCRA_HANDLER_ID")]

    # Count violations
    snc_facilities = [f for f in facilities if f.get("FacSNCFlg") == "Y"]
    violation_facilities = [
        f for f in facilities
        if int(f.get("FacQtrsWithNC") or "0") > 0
    ]

    # Generate risk flags
    risk_flags = []

    if npl_sites:
        risk_flags.append({
            "flag": f"{len(npl_sites)} NPL/Superfund site(s) within {radius_miles} mile(s)",
            "severity": "HIGH",
        })
    if rcra_ca_sites:
        risk_flags.append({
            "flag": f"{len(rcra_ca_sites)} RCRA corrective action site(s) within {radius_miles} mile(s)",
            "severity": "HIGH",
        })
    if snc_facilities:
        risk_flags.append({
            "flag": f"{len(snc_facilities)} facility/ies in significant non-compliance within {radius_miles} mile(s)",
            "severity": "HIGH",
        })
    if brownfield_sites:
        severity = "MEDIUM" if len(brownfield_sites) <= 5 else "HIGH"
        risk_flags.append({
            "flag": f"{len(brownfield_sites)} brownfield site(s) within {radius_miles} mile(s)",
            "severity": severity,
        })
    if len(facilities) > 50:
        risk_flags.append({
            "flag": f"{len(facilities)} regulated facilities within {radius_miles} mile(s) (high industrial density)",
            "severity": "MEDIUM",
        })

    # Flood zone
    flood_result = None
    if include_flood:
        flood_result = flood_zone(lon=lon, lat=lat)
        if flood_result.get("sfha"):
            risk_flags.append({
                "flag": f"Site in FEMA Special Flood Hazard Area (Zone {flood_result['zone']})",
                "severity": "MEDIUM",
            })

    # Determine overall risk level
    severities = [f["severity"] for f in risk_flags]
    if "HIGH" in severities:
        risk_level = "HIGH"
    elif "MEDIUM" in severities:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    report = {
        "query": {
            "lat": lat,
            "lon": lon,
            "radius_miles": radius_miles,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        },
        "summary": {
            "total_regulated_facilities": len(facilities),
            "facilities_with_violations": len(violation_facilities),
            "facilities_snc": len(snc_facilities),
            "superfund_npl_sites": len(npl_sites),
            "brownfield_sites": len(brownfield_sites),
            "rcra_corrective_action_sites": len(rcra_ca_sites),
            "risk_level": risk_level,
        },
        "risk_flags": risk_flags,
        "echo_facilities": facilities,
        "cleanup_sites": cleanup,
        "flood_zone": flood_result,
    }

    # Print summary
    color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}[risk_level]
    console.print(f"\n[bold {color}]Risk Level: {risk_level}[/bold {color}]")
    console.print(f"  Regulated facilities: {len(facilities)}")
    console.print(f"  NPL/Superfund: {len(npl_sites)}")
    console.print(f"  Brownfields: {len(brownfield_sites)}")
    console.print(f"  RCRA corrective action: {len(rcra_ca_sites)}")
    console.print(f"  Significant non-compliance: {len(snc_facilities)}")
    if flood_result:
        fz = flood_result.get("zone") or "unmapped"
        console.print(f"  Flood zone: {fz}")
    for flag in risk_flags:
        icon = {"HIGH": "[red]!![/red]", "MEDIUM": "[yellow]![/yellow]", "LOW": "[dim]·[/dim]"}
        console.print(f"  {icon.get(flag['severity'], '·')} {flag['flag']}")

    return report
