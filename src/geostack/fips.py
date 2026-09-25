"""FIPS code → county/state name lookups.

Uses the Census Bureau's FIPS code reference file (bundled as a lightweight
dict for the most common states). Falls back to a Census API call for
unknown codes.

Usage:
    from geostack.fips import county_name, state_name

    county_name("48", "113")   # → "Dallas County"
    county_name("48", "085")   # → "Collin County"
    state_name("48")           # → "Texas"
"""

import requests
from rich.console import Console

console = Console()

# State FIPS → name
STATE_NAMES = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
    "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}

# Cache for county names (populated on first lookup per state)
_county_cache: dict[str, str] = {}


def state_name(state_fips):
    """Look up state name from 2-digit FIPS code."""
    return STATE_NAMES.get(str(state_fips).zfill(2), f"State {state_fips}")


def county_name(state_fips, county_fips):
    """Look up county name from state + county FIPS codes.

    Returns e.g. "Dallas County" for ("48", "113").
    Caches results per state via Census API on first call.

    Args:
        state_fips: 2-digit state FIPS (e.g., "48").
        county_fips: 3-digit county FIPS (e.g., "113").

    Returns:
        County name string, or "County {fips}" if lookup fails.
    """
    state_fips = str(state_fips).zfill(2)
    county_fips = str(county_fips).zfill(3)
    key = f"{state_fips}{county_fips}"

    if key in _county_cache:
        return _county_cache[key]

    # Try to load all counties for this state from Census API
    try:
        _load_state_counties(state_fips)
    except Exception:
        pass

    return _county_cache.get(key, f"County {key}")


def _load_state_counties(state_fips):
    """Fetch all county names for a state from Census API and cache them."""
    url = f"https://api.census.gov/data/2023/acs/acs5"
    params = {
        "get": "NAME",
        "for": "county:*",
        "in": f"state:{state_fips}",
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # data[0] is headers: ["NAME", "state", "county"]
    for row in data[1:]:
        name = row[0]
        s = row[1]
        c = row[2]
        # Census returns "Dallas County, Texas" — take just the county part
        county_part = name.split(",")[0].strip()
        _county_cache[f"{s}{c}"] = county_part

    console.print(f"[dim]Loaded {len(data) - 1} county names for state {state_fips}[/dim]")


def county_name_bulk(fips_pairs):
    """Look up county names for multiple (state_fips, county_fips) pairs.

    More efficient than calling county_name() in a loop — batches the
    Census API call per state.

    Args:
        fips_pairs: List of (state_fips, county_fips) tuples.

    Returns:
        Dict mapping "state_fips+county_fips" → county name.
    """
    # Identify which states we need to load
    states_needed = set()
    for s, c in fips_pairs:
        key = f"{str(s).zfill(2)}{str(c).zfill(3)}"
        if key not in _county_cache:
            states_needed.add(str(s).zfill(2))

    for s in states_needed:
        try:
            _load_state_counties(s)
        except Exception as e:
            console.print(f"[yellow]County name lookup failed for state {s}: {e}[/yellow]")

    result = {}
    for s, c in fips_pairs:
        key = f"{str(s).zfill(2)}{str(c).zfill(3)}"
        result[key] = _county_cache.get(key, f"County {key}")
    return result
