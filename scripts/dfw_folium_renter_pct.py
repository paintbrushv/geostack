"""Generate Folium interactive renter % choropleth for all of DFW."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from geostack.census import get_tract_demographics
from geostack.db import get_engine, read_postgis

engine = get_engine()

# DFW MSA core counties (12-county MSA)
DFW_COUNTIES = {
    "085": "Collin",
    "113": "Dallas",
    "121": "Denton",
    "139": "Ellis",
    "221": "Hood",
    "231": "Hunt",
    "257": "Kaufman",
    "251": "Johnson",
    "367": "Parker",
    "397": "Rockwall",
    "439": "Tarrant",
    "497": "Wise",
}
STATE = "48"
YEAR = 2023
OUTPUT = str(Path(__file__).resolve().parents[1] / "showcase" / "dfw_folium_renter_pct.html")

# Fetch ACS demographics for all DFW counties
print("Fetching ACS data for DFW counties...")
frames = []
for fips, name in DFW_COUNTIES.items():
    try:
        gdf = get_tract_demographics(STATE, fips, YEAR, engine=engine)
        print(f"  {name} ({fips}): {len(gdf)} tracts")
        frames.append(gdf)
    except Exception as e:
        print(f"  {name} ({fips}): SKIPPED — {e}")

tract_gdf = pd.concat(frames)
print(f"Total tracts: {len(tract_gdf)}")

# Compute renter %
tract_gdf["renter_pct"] = (
    tract_gdf["renter_occupied"] / tract_gdf["occupied_units"].replace(0, float("nan")) * 100
).round(1)

# Drop tracts with no housing data
tract_gdf = tract_gdf[tract_gdf["renter_pct"].notna()].copy()
print(f"Tracts with renter data: {len(tract_gdf)}, Mean: {tract_gdf['renter_pct'].mean():.1f}%")

# Build Folium choropleth
from geostack.viz import choropleth_map
import folium

m = choropleth_map(
    tract_gdf,
    column="renter_pct",
    title="DFW Metro — Renter-Occupied %  (ACS 2023)",
    scheme="quantiles",
    n_bins=6,
    cmap="RdBu_r",
    tooltip_cols=["renter_pct", "renter_occupied", "occupied_units",
                  "total_population", "median_household_income", "median_rent"],
    zoom_start=10,
    opacity=0.65,
)

m.save(OUTPUT)
print(f"Saved: {OUTPUT}")
