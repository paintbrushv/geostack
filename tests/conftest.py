"""Shared pytest fixtures for the geostack test suite."""

import io
import os

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box


# ---------------------------------------------------------------------------
# Geographic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dfw_point():
    """Central Dallas point (WGS84)."""
    return Point(-96.7970, 32.7767)


@pytest.fixture
def dfw_gdf():
    """Small polygon GeoDataFrame centered on DFW."""
    geom = box(-97.05, 32.60, -96.55, 33.05)
    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")


@pytest.fixture
def austin_gdf():
    """Small polygon GeoDataFrame centered on Austin (TX Central plane)."""
    geom = box(-97.85, 30.15, -97.55, 30.45)
    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")


@pytest.fixture
def seattle_gdf():
    """Small polygon GeoDataFrame in Seattle (non-Texas, UTM 10N)."""
    geom = box(-122.5, 47.5, -122.2, 47.7)
    return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")


@pytest.fixture
def parcels_gdf(dfw_point):
    """A tiny set of synthetic parcel polygons around DFW for spatial tests."""
    import numpy as np
    from shapely.affinity import translate

    rng = np.random.default_rng(42)
    geoms = []
    for _ in range(5):
        dx = rng.uniform(-0.01, 0.01)
        dy = rng.uniform(-0.01, 0.01)
        geoms.append(translate(box(-0.002, -0.002, 0.002, 0.002), dfw_point.x + dx, dfw_point.y + dy))

    return gpd.GeoDataFrame(
        {"parcel_id": range(len(geoms)), "area_sqft": rng.uniform(5000, 50000, len(geoms))},
        geometry=geoms,
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# LODES / Census mock helpers
# ---------------------------------------------------------------------------

def _make_wac_csv_bytes(num_blocks=5, county_prefix="48085"):
    """Generate a minimal LODES WAC CSV (gzip-compressed) for mocking."""
    import gzip
    import random

    rows = [["w_geocode", "C000", "CA01", "CA02", "CA03",
             "CE01", "CE02", "CE03"] + [f"CNS{i:02d}" for i in range(1, 21)] + ["createdate"]]

    for i in range(num_blocks):
        geocode = f"{county_prefix}{str(i + 1).zfill(10)}"
        jobs = random.randint(10, 500)
        row = [geocode, jobs,
               int(jobs * 0.3), int(jobs * 0.5), int(jobs * 0.2),
               int(jobs * 0.25), int(jobs * 0.45), int(jobs * 0.30)]
        row += [random.randint(0, jobs // 5) for _ in range(20)]
        row.append("20230101")
        rows.append(row)

    csv_content = "\n".join(",".join(str(c) for c in row) for row in rows)
    return gzip.compress(csv_content.encode())


@pytest.fixture
def wac_csv_bytes():
    """Compressed WAC CSV bytes for Collin County TX (48085)."""
    return _make_wac_csv_bytes(num_blocks=5, county_prefix="48085")


@pytest.fixture
def wac_parquet_file(tmp_path):
    """A small WAC parquet file in the expected cache path format."""
    import random

    rows = []
    for i in range(5):
        geocode = f"480850001{str(i + 1).zfill(6)}"
        jobs = random.randint(50, 300)
        row = {"w_geocode": geocode, "C000": jobs,
               "CA01": int(jobs * 0.3), "CA02": int(jobs * 0.5), "CA03": int(jobs * 0.2),
               "CE01": int(jobs * 0.25), "CE02": int(jobs * 0.45), "CE03": int(jobs * 0.30)}
        for j in range(1, 21):
            row[f"CNS{j:02d}"] = int(jobs * 0.05)
        rows.append(row)

    df = pd.DataFrame(rows)
    parquet_dir = tmp_path / "tx"
    parquet_dir.mkdir()
    path = parquet_dir / "wac_S000_JT01_2021.parquet"
    df.to_parquet(path, index=False)
    return str(path)
