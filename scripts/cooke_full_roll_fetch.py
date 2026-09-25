"""Pull the full Cooke County (CCAD) parcel roll — thin wrapper over cad.py.

The True Prodigy puller now lives in geostack.cad as a shared source
(fetch_cooke_parcels_trueprodigy / fetch_trueprodigy_roll), so other repos
import it instead of duplicating the Playwright/pagination logic. This script
just runs it and caches the normalized roll.

Requires:  pip install 'geostack[cad-scrape]' && playwright install chromium

Run:
    python scripts/cooke_full_roll_fetch.py
"""

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import pathlib

from geostack.cad import fetch_cooke_parcels_trueprodigy

OUT = pathlib.Path("data/cooke/ccad_full_roll.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    df = fetch_cooke_parcels_trueprodigy()          # normalized canonical schema + lat/lon
    df.to_parquet(OUT, index=False)
    geo = df["latitude"].notna().sum()
    print(f"Saved {len(df):,} parcels ({geo:,} with coordinates) → {OUT}")


if __name__ == "__main__":
    main()
