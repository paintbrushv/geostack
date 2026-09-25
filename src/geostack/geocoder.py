"""Census Bureau Batch Geocoder — address → lat/lon + tract GEOID.

Batch geocode addresses via the Census Bureau's free geocoding service.
Returns coordinates and 11-digit census tract GEOIDs for spatial joins.

Usage:
    from geostack.geocoder import geocode_batch

    import pandas as pd
    df = pd.DataFrame({"street": ["1325 TENNIS DR"], "city": ["Bedford"]})
    result = geocode_batch(df, street_col="street", city_col="city", state="TX")
    # result has original columns + latitude, longitude, tract_geoid, match_type, matched_address
"""

import csv
import hashlib
import io
import time
from pathlib import Path

import pandas as pd
import requests
from rich.console import Console

console = Console()


def _parse_response(text: str) -> pd.DataFrame:
    """Parse pipe-delimited Census Batch Geocoder response into a DataFrame.

    The API returns rows like:
    "id","input_addr","Match","Exact","matched_addr","lon,lat","tiger_id","side","st","cty","tract","block"

    No_Match rows are shorter — only id, input_addr, match_type.
    """
    if not text.strip():
        return pd.DataFrame(columns=[
            "id", "match_type", "matched_address",
            "longitude", "latitude", "tract_geoid",
        ])

    rows = []
    reader = csv.reader(io.StringIO(text), delimiter=",", quotechar='"')
    for parts in reader:
        if not parts:
            continue
        rec_id = parts[0].strip('" ')
        match_type = parts[2].strip('" ') if len(parts) > 2 else "No_Match"

        if match_type == "Match" and len(parts) >= 12:
            matched_address = parts[4].strip('" ')
            lonlat = parts[5].strip('" ')
            lon_str, lat_str = lonlat.split(",")
            state_fips = parts[8].strip('" ')
            county_fips = parts[9].strip('" ')
            tract = parts[10].strip('" ')
            rows.append({
                "id": rec_id,
                "match_type": "Match",
                "matched_address": matched_address,
                "longitude": float(lon_str),
                "latitude": float(lat_str),
                "tract_geoid": f"{state_fips}{county_fips}{tract}",
            })
        else:
            rows.append({
                "id": rec_id,
                "match_type": match_type,
                "matched_address": None,
                "longitude": float("nan"),
                "latitude": float("nan"),
                "tract_geoid": None,
            })

    return pd.DataFrame(rows)


def _prepare_batch_csv(records: list[tuple]) -> str:
    """Format records as CSV for Census Batch Geocoder upload.

    Each record is (unique_id, street, city, state, zip).
    Returns CSV text with no header row.
    """
    lines = []
    for uid, street, city, state, zipcode in records:
        city = city or ""
        zipcode = zipcode or ""
        lines.append(f"{uid},{street},{city},{state},{zipcode}")
    return "\n".join(lines) + "\n"


CENSUS_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch"


def _submit_batch(csv_text: str) -> str:
    """POST a CSV batch to the Census Geocoder and return raw response text.

    Retries once on HTTP 5xx with a 5-second backoff.
    """
    files = {"addressFile": ("addresses.csv", csv_text, "text/csv")}
    data = {"benchmark": "Public_AR_Current", "vintage": "Current_Current"}

    for attempt in range(2):
        resp = requests.post(CENSUS_GEOCODER_URL, files=files, data=data, timeout=120)
        try:
            resp.raise_for_status()
            return resp.text
        except Exception:
            if attempt == 0 and resp.status_code >= 500:
                console.print("[yellow]Census API returned 5xx, retrying in 5s...[/yellow]")
                time.sleep(5)
                continue
            raise


CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "geocode_cache"


def _cache_key(records: list[tuple]) -> str:
    """SHA-256 hash of sorted input addresses for cache key."""
    canonical = sorted(str(r) for r in records)
    return hashlib.sha256("\n".join(canonical).encode()).hexdigest()[:16]


def _save_cache(df: pd.DataFrame, key: str, cache_dir: Path | None = None) -> None:
    """Save geocoded results to parquet cache."""
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_dir / f"{key}.parquet", index=False)


def _load_cache(key: str, cache_dir: Path | None = None) -> pd.DataFrame | None:
    """Load cached geocoded results, or None on cache miss."""
    cache_dir = cache_dir or CACHE_DIR
    path = cache_dir / f"{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def geocode_addresses(
    records: list[tuple],
    batch_size: int = 1000,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Geocode a list of address records via Census Batch Geocoder.

    Args:
        records: List of (unique_id, street, city, state, zip) tuples.
        batch_size: Addresses per API request (max 10,000; default 1,000).
        cache_dir: Override cache directory (default: data/geocode_cache/).

    Returns:
        DataFrame with columns: id, latitude, longitude, tract_geoid,
        match_type, matched_address.
    """
    key = _cache_key(records)
    cached = _load_cache(key, cache_dir=cache_dir)
    if cached is not None:
        console.print(f"[green]Cache hit — {len(cached)} addresses loaded[/green]")
        return cached

    all_results = []
    total_batches = (len(records) + batch_size - 1) // batch_size

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        batch_num = i // batch_size + 1

        console.print(
            f"[dim]Geocoding batch {batch_num}/{total_batches} "
            f"({len(batch)} addresses)...[/dim]"
        )

        csv_text = _prepare_batch_csv(batch)
        response_text = _submit_batch(csv_text)
        batch_df = _parse_response(response_text)
        all_results.append(batch_df)

        matched = (batch_df["match_type"] == "Match").sum()
        total = len(batch_df)
        pct = matched / total * 100 if total else 0
        console.print(
            f"[green]Batch {batch_num}: {matched}/{total} matched ({pct:.1f}%)[/green]"
        )

        if batch_num < total_batches:
            time.sleep(1)

    result = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    _save_cache(result, key, cache_dir=cache_dir)

    matched_total = (result["match_type"] == "Match").sum() if len(result) else 0
    console.print(
        f"\n[bold]Geocoding complete: {matched_total}/{len(result)} matched "
        f"({matched_total / len(result) * 100:.1f}%)[/bold]"
    )

    return result


def geocode_batch(
    df: pd.DataFrame,
    street_col: str,
    city_col: str | None = None,
    state: str = "TX",
    zip_col: str | None = None,
    id_col: str | None = None,
    batch_size: int = 1000,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Geocode a DataFrame of addresses via Census Batch Geocoder.

    Args:
        df: DataFrame containing address data.
        street_col: Column name for street address.
        city_col: Column name for city (optional).
        state: State abbreviation (default "TX").
        zip_col: Column name for ZIP code (optional).
        id_col: Column name for unique ID (uses index if None).
        batch_size: Addresses per API request (default 1,000).
        cache_dir: Override cache directory.

    Returns:
        Original DataFrame with added columns: latitude, longitude,
        tract_geoid, match_type, matched_address.
    """
    records = []
    for idx, row in df.iterrows():
        uid = str(row[id_col]) if id_col else str(idx)
        street = str(row[street_col])
        city = str(row[city_col]) if city_col and city_col in df.columns else ""
        zipcode = str(row[zip_col]) if zip_col and zip_col in df.columns else ""
        # Clean up "nan" strings from pandas
        if city.lower() == "nan":
            city = ""
        if zipcode.lower() == "nan":
            zipcode = ""
        # Clean float ZIPs like "75201.0"
        if "." in zipcode:
            zipcode = zipcode.split(".")[0]
        # Truncate ZIP+4 to 5 digits
        if len(zipcode) > 5:
            zipcode = zipcode[:5]
        records.append((uid, street, city, state, zipcode))

    result = geocode_addresses(records, batch_size=batch_size, cache_dir=cache_dir)

    # Join geocoding results back to original DataFrame
    id_series = df[id_col].astype(str) if id_col else df.index.astype(str)
    geo_cols = result.set_index("id")[["latitude", "longitude", "tract_geoid", "match_type", "matched_address"]]
    out = df.copy()
    out["_join_id"] = id_series.values
    out = out.merge(geo_cols, left_on="_join_id", right_index=True, how="left")
    out.drop(columns=["_join_id"], inplace=True)

    return out
