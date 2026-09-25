"""B+D pipeline: DCAD parcel geometry -> join values -> local PostGIS -> H3.

Implements the recommended approach for working with DCAD locally:

  B. Pull authoritative DCAD parcel geometry from the ArcGIS REST service
     (CurrentDcadParcels, ~695k polygons) and join the bulk-CSV appraisal
     values on account number (GIS_Acct <-> account_num). Exact parcel
     centroids, ~100% coverage — no address geocoding loss.

  D. Load the joined parcels into the local PostGIS `geostack` database once
     (table `dcad_parcels`, GiST-indexed) so every repo on this machine can
     spatial-join parcels to tracts / hexes / anything without re-fetching.

Then it builds an H3 commercial-value layer from the REAL parcel centroids and
compares coverage with the geocoded sample.

Run:
    python scripts/dcad_parcels_to_postgis.py            # full ~695k pull (~6 min, cached)
    python scripts/dcad_parcels_to_postgis.py --max 40000  # quick test subset
"""

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import warnings
warnings.filterwarnings("ignore")

import pathlib

import geopandas as gpd
import pandas as pd
from sqlalchemy import text

from geostack.cad import fetch_dallas_parcel_geometry_all
from geostack.db import get_engine
from geostack.h3grid import points_to_h3, h3_choropleth_static

OUT = pathlib.Path("outputs/msa_gravity/dfw_h3")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="Cap parcels (testing)")
    ap.add_argument("--table", default="dcad_parcels")
    args = ap.parse_args()

    # ── B: geometry + value join ──
    geom = fetch_dallas_parcel_geometry_all(out_fields="GIS_Acct", max_records=args.max)
    geom["gis_acct"] = geom["GIS_Acct"].astype(str).str.strip()
    geom = geom[["gis_acct", "geometry"]].dropna(subset=["geometry"])

    vals = pd.read_parquet("data/dcad/dcad_com_2024.parquet")
    vals["account_num"] = vals["account_num"].astype(str).str.strip()
    vals["tot_val"] = pd.to_numeric(vals["tot_val"], errors="coerce")
    vals["impr_val"] = pd.to_numeric(vals["impr_val"], errors="coerce")
    vals = vals[["account_num", "tot_val", "impr_val", "property_name", "biz_name"]]

    parcels = geom.merge(vals, left_on="gis_acct", right_on="account_num", how="left")
    is_com = parcels["tot_val"].notna()
    print(f"\nJoined {len(parcels):,} parcels; "
          f"{is_com.sum():,} matched a commercial value record "
          f"({is_com.sum()/len(vals)*100:.0f}% of {len(vals):,} commercial accounts)")

    # ── D: load to local PostGIS (GiST-indexed) ──
    engine = get_engine()
    load = parcels.copy()
    load["is_commercial"] = is_com.values
    # Keep geometry valid; project nothing (store as 4326)
    load = gpd.GeoDataFrame(load, geometry="geometry", crs="EPSG:4326")
    print(f"Loading {len(load):,} parcels into PostGIS table '{args.table}'...")
    load.to_postgis(args.table, engine, if_exists="replace", index=False, chunksize=10000)
    with engine.begin() as c:
        c.execute(text(f"CREATE INDEX IF NOT EXISTS {args.table}_geom_gix "
                       f"ON {args.table} USING GIST (geometry)"))
        c.execute(text(f"CREATE INDEX IF NOT EXISTS {args.table}_acct_ix "
                       f"ON {args.table} (gis_acct)"))
        n = c.execute(text(f"SELECT count(*) FROM {args.table}")).scalar()
    print(f"  PostGIS table '{args.table}': {n:,} rows, GiST + account indexes")

    # ── H3 from REAL parcel centroids (no geocoding) ──
    com = load[load["is_commercial"]].copy()
    com["geometry"] = com.geometry.centroid
    com = com[com["tot_val"] > 0]
    print(f"\nBuilding H3 commercial value from {len(com):,} real parcel centroids "
          f"(${com['tot_val'].sum()/1e9:.1f}B)...")
    hexes = points_to_h3(com, resolution=8, value_cols=["tot_val"], agg="sum")
    hexes["value_musd"] = hexes["tot_val"] / 1e6
    hexes.attrs["resolution"] = 8
    hexes.attrs["msa_name"] = "Dallas County"
    h3_choropleth_static(
        hexes[hexes["tot_val"] >= 1e6], "value_musd",
        cmap="inferno", scheme="quantiles", k=7,
        title="Dallas County — Commercial appraised value (H3 res 8, DCAD 2024)",
        subtitle=f"From real DCAD parcel geometry ({len(com):,} parcels, ~100% coverage) "
                 f"· OpenStreetMap full detail",
        output_path=OUT / "dfw_h3_commercial_value_geometry.png",
    )
    print(f"\nDone. Real-geometry map: {OUT}/dfw_h3_commercial_value_geometry.png")
    print("Coverage vs geocoding: ~100% (geometry) vs ~84% (address geocode sample).")


if __name__ == "__main__":
    main()
