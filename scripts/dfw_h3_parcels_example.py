"""Example: H3 layers from real DCAD commercial parcel data.

Demonstrates two h3grid capabilities on real Dallas County appraisal-district
(DCAD) commercial parcels:

1. points_to_h3  — bin geocoded parcels into hexes: commercial parcel count
   and total appraised value per cell ("where the commercial RE value is").
2. dasymetric_polygons_to_h3 — allocate tract-level jobs to hexes weighted by
   commercial parcels (jobs sit in commercial buildings, not spread over
   residential/empty land), vs the naive area-weighted allocation.

Run:
    python scripts/dfw_h3_parcels_example.py

Geocoding is cached (data/geocode_cache/), so re-runs are fast.
"""

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import pathlib

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from geostack.geocoder import geocode_batch
from geostack.h3grid import (
    points_to_h3, dasymetric_polygons_to_h3, interpolate_polygons_to_h3,
    h3_choropleth_static,
)

OUT = pathlib.Path("outputs/msa_gravity/dfw_h3")
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE = 6000  # parcels to geocode for the demo


def load_geocoded_parcels() -> gpd.GeoDataFrame:
    """Load DCAD commercial parcels, geocode a sample, return points GeoDataFrame."""
    df = pd.read_parquet("data/dcad/dcad_com_2024.parquet")
    df["tot_val"] = pd.to_numeric(df["tot_val"], errors="coerce")
    df = df[(df["tot_val"] > 0) & df["full_street_name"].notna() & df["street_num"].notna()].copy()
    df["street"] = (df["street_num"].astype(str).str.strip() + " "
                    + df["full_street_name"].astype(str).str.strip())
    df["acct"] = df["account_num"].astype(str)
    samp = df.sample(min(SAMPLE, len(df)), random_state=42).reset_index(drop=True)

    print(f"Geocoding {len(samp):,} DCAD commercial parcels (cached after first run)...")
    geo = geocode_batch(samp, street_col="street", city_col="property_city",
                        state="TX", zip_col="property_zipcode", id_col="acct")
    ok = geo[geo["latitude"].notna()].copy()
    print(f"  matched {len(ok):,}/{len(samp):,} ({len(ok)/len(samp)*100:.0f}%)")

    pts = gpd.GeoDataFrame(
        ok[["acct", "tot_val", "impr_val", "gross_bldg_area"]].copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(ok["longitude"], ok["latitude"])],
        crs="EPSG:4326",
    )
    pts["tot_val"] = pd.to_numeric(pts["tot_val"], errors="coerce")
    return pts[pts["tot_val"] > 0]


def main():
    parcels = load_geocoded_parcels()
    print(f"  {len(parcels):,} geocoded commercial parcels, "
          f"${parcels['tot_val'].sum()/1e9:.1f}B total appraised value")

    # ── 1. points_to_h3: commercial value density ──
    print("\n[1] Binning parcels into H3 (res 8)...")
    hexes = points_to_h3(parcels, resolution=8, value_cols=["tot_val"], agg="sum")
    hexes["value_musd"] = hexes["tot_val"] / 1e6
    hexes.attrs["resolution"] = 8
    hexes.attrs["msa_name"] = "Dallas County"
    h3_choropleth_static(
        hexes[hexes["tot_val"] >= 1e6], "value_musd",
        cmap="inferno", scheme="quantiles", k=7,
        title="Dallas County — Commercial appraised value (H3 res 8, DCAD 2024)",
        subtitle=f"Total $M per hex · {len(parcels):,} geocoded commercial parcels · roads/POIs visible",
        output_path=OUT / "dfw_h3_commercial_value.png",
    )

    # ── 2. dasymetric: jobs allocated by commercial parcels vs area ──
    print("\n[2] Dasymetric jobs allocation (parcel-weighted vs area-weighted)...")
    from geostack.gravity import (
        resolve_msa, fetch_tract_geometries, _fetch_lodes_tract_jobs,
    )
    # Dallas County tracts + jobs
    tracts = fetch_tract_geometries("48", "113", 2023)
    jobs = _fetch_lodes_tract_jobs("48", "113", 2023).rename(columns={"weight": "jobs"})
    tj = tracts.merge(jobs, on="GEOID", how="left")
    tj["jobs"] = tj["jobs"].fillna(0)

    dasy = dasymetric_polygons_to_h3(
        tj, parcels, resolution=8, count_cols=["jobs"], weight_col="tot_val",
    )
    dasy["jobs_density"] = dasy["jobs"] / dasy["hex_area_sqmi"].replace(0, float("nan"))
    dasy.attrs["resolution"] = 8
    dasy.attrs["msa_name"] = "Dallas County"
    h3_choropleth_static(
        dasy[dasy["jobs"] >= 50], "jobs_density",
        cmap="inferno", scheme="naturalbreaks", k=7,
        title="Dallas County — Jobs density, parcel-weighted (H3 res 8)",
        subtitle="Tract jobs allocated by commercial parcel value (dasymetric) — sharper than area-weighting",
        output_path=OUT / "dfw_h3_jobs_dasymetric.png",
    )

    print(f"\nDone. Images in {OUT}/")


if __name__ == "__main__":
    main()
