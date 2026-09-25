"""Parcel analytics — ILR, age, trend analysis, and spatial aggregation.

Core analytical primitives for appraisal parcel data. All functions are pure
(no I/O, no database) and operate on pandas/geopandas DataFrames.

Usage:
    from geostack.analysis import compute_ilr, ilr_trend, aggregate_ilr_to_tracts

    # Single-year ILR
    df = compute_ilr(parcels_df)

    # Multi-year trend
    frames = {2021: df_2021, 2022: df_2022, 2023: df_2023}
    trend = ilr_trend(frames)

    # Spatial aggregation to census tracts
    tracts_with_ilr = aggregate_ilr_to_tracts(parcels_gdf, tracts_gdf)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data cleaning utilities
# ---------------------------------------------------------------------------

def clean_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, replacing negatives and non-parseable values with NaN."""
    result = pd.to_numeric(series, errors="coerce")
    result = result.where(result >= 0, np.nan)
    return result


def clean_year(
    series: pd.Series,
    min_year: int = 1800,
    max_year: int = 2030,
) -> pd.Series:
    """Coerce to integer year, replacing implausible values with NaN."""
    result = pd.to_numeric(series, errors="coerce")
    result = result.where((result >= min_year) & (result <= max_year), np.nan)
    return result


def normalize_address(series: pd.Series) -> pd.Series:
    """Uppercase, strip whitespace, collapse runs, replace blank/NaN with NaN."""
    result = series.astype(str).str.strip().str.upper()
    result = result.str.replace(r"\s+", " ", regex=True)
    result = result.replace(["NAN", "NONE", ""], np.nan)
    return result


# ---------------------------------------------------------------------------
# Single-year derived fields
# ---------------------------------------------------------------------------

def compute_ilr(
    df: pd.DataFrame,
    land_col: str = "land_value",
    imprv_col: str = "improvement_value",
) -> pd.DataFrame:
    """Add an ``ilr`` column (improvement_value / land_value).

    ILR (Improvement-to-Land Ratio) measures how much of a parcel's value
    is in the structure vs. the land. A declining ILR signals that land is
    appreciating faster than the building — a leading indicator of
    redevelopment pressure.

    Values are NaN where land_value is zero, negative, or missing.

    Args:
        df: DataFrame with land and improvement value columns.
        land_col: Column name for land value (default ``land_value``).
        imprv_col: Column name for improvement value (default ``improvement_value``).

    Returns:
        Copy of df with an added ``ilr`` column.
    """
    df = df.copy()
    valid_land = df[land_col].notna() & (df[land_col] > 0)
    df["ilr"] = np.where(
        valid_land,
        df[imprv_col] / df[land_col],
        np.nan,
    )
    return df


def compute_age(
    df: pd.DataFrame,
    built_col: str = "year_built",
    year_col: str = "tax_year",
) -> pd.DataFrame:
    """Add an ``age`` column (tax_year - year_built).

    Args:
        df: DataFrame with year_built and tax_year columns.
        built_col: Column name for construction year (default ``year_built``).
        year_col: Column name for the assessment/tax year (default ``tax_year``).

    Returns:
        Copy of df with an added ``age`` column (NaN where year_built is missing).
    """
    df = df.copy()
    df["age"] = np.where(
        df[built_col].notna(),
        df[year_col] - df[built_col],
        np.nan,
    )
    return df


# ---------------------------------------------------------------------------
# Multi-year trend analysis
# ---------------------------------------------------------------------------

def ilr_trend(
    frames: dict[int, pd.DataFrame],
    parcel_id_col: str = "parcel_id",
    land_col: str = "land_value",
    imprv_col: str = "improvement_value",
    flat_threshold: float = 0.01,
) -> pd.DataFrame:
    """Compute per-parcel ILR trend across multiple years.

    Produces a wide DataFrame with one column per year (``ilr_YYYY``) plus
    ``ilr_trend_slope`` (OLS slope across available years) and
    ``ilr_direction`` (``"up"`` / ``"down"`` / ``"flat"``).

    This format is designed for choropleth mapping (direction column) while
    preserving the full time series (ilr_YYYY columns) for chart overlays and
    future statistical tests.

    Args:
        frames: Mapping of tax year → normalized parcel DataFrame. Each
            DataFrame must have ``parcel_id_col``, ``land_col``, and
            ``imprv_col`` columns. ILR is computed internally so the caller
            does not need to pre-compute it.
        parcel_id_col: Column identifying each parcel (default ``parcel_id``).
        land_col: Land value column name.
        imprv_col: Improvement value column name.
        flat_threshold: Absolute slope below which a trend is called ``"flat"``
            (default 0.01). Tune this for your market's ILR scale.

    Returns:
        DataFrame indexed by ``parcel_id_col`` with columns:
        - ``ilr_YYYY`` for each year in *frames*
        - ``ilr_trend_slope``: float, OLS slope (NaN if < 2 non-null years)
        - ``ilr_direction``: ``"up"`` / ``"down"`` / ``"flat"`` / None

    Example::

        frames = {2021: df21, 2022: df22, 2023: df23}
        trend = ilr_trend(frames)
        # parcel_id | ilr_2021 | ilr_2022 | ilr_2023 | ilr_trend_slope | ilr_direction
        # 48085001  | 2.1      | 1.8      | 1.4      | -0.35           | down
    """
    if not frames:
        raise ValueError("frames must contain at least one year")

    years = sorted(frames.keys())

    # Compute ILR per year and collect one column per year
    year_series = {}
    for yr in years:
        df = frames[yr]
        if "ilr" not in df.columns:
            df = compute_ilr(df, land_col=land_col, imprv_col=imprv_col)
        year_series[yr] = df.set_index(parcel_id_col)["ilr"].rename(f"ilr_{yr}")

    wide = pd.concat(year_series.values(), axis=1)

    ilr_cols = [f"ilr_{yr}" for yr in years]

    # Per-parcel OLS slope using numpy.polyfit, vectorised via apply
    def _slope(row):
        values = row.values.astype(float)
        finite_mask = np.isfinite(values)
        finite_years = np.array(years)[finite_mask]
        finite_vals = values[finite_mask]
        if len(finite_years) < 2:
            return np.nan
        # Normalise years to small integers to avoid floating-point scale issues
        x = finite_years - finite_years[0]
        return np.polyfit(x, finite_vals, 1)[0]

    wide["ilr_trend_slope"] = wide[ilr_cols].apply(_slope, axis=1)

    def _direction(slope):
        if pd.isna(slope):
            return None
        if slope > flat_threshold:
            return "up"
        if slope < -flat_threshold:
            return "down"
        return "flat"

    wide["ilr_direction"] = wide["ilr_trend_slope"].map(_direction)

    wide.index.name = parcel_id_col
    return wide.reset_index()


# ---------------------------------------------------------------------------
# Spatial aggregation
# ---------------------------------------------------------------------------

def aggregate_ilr_to_tracts(
    parcels_gdf,
    tracts_gdf,
    ilr_col: str = "ilr",
    tract_id_col: str = "geoid",
) -> "gpd.GeoDataFrame":
    """Spatially join parcel ILR values to census tracts and aggregate.

    Performs a point-in-polygon (or polygon-centroid-in-polygon) join of
    parcels to tracts, then computes tract-level ILR statistics suitable for
    choropleth mapping.

    Args:
        parcels_gdf: GeoDataFrame with geometry (Point or Polygon) and an
            ``ilr`` column. If geometry is Polygon, centroids are used for
            the join.
        tracts_gdf: GeoDataFrame with polygon geometry and a tract ID column
            (default ``geoid``). Typically from ``get_tract_demographics()``.
        ilr_col: Name of the ILR column in parcels_gdf (default ``ilr``).
        tract_id_col: Name of the tract ID column in tracts_gdf
            (default ``geoid``).

    Returns:
        tracts_gdf with four added columns:
        - ``median_ilr``: median ILR of parcels within the tract
        - ``mean_ilr``: mean ILR of parcels within the tract
        - ``parcel_count``: number of parcels with a valid ILR in the tract
        - ``pct_underimproved``: fraction of parcels with ILR < 1.0
        Tracts with no parcels get NaN for all four metrics.
    """
    import geopandas as gpd

    # Auto-reproject parcels to match tracts CRS
    if parcels_gdf.crs != tracts_gdf.crs:
        from rich.console import Console
        Console().print(
            f"[yellow]aggregate_ilr_to_tracts: reprojecting parcels from "
            f"{parcels_gdf.crs} to {tracts_gdf.crs}[/yellow]"
        )
        parcels_gdf = parcels_gdf.to_crs(tracts_gdf.crs)

    # Use centroids for polygon parcel geometry
    join_gdf = parcels_gdf.copy()
    if join_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        join_gdf["geometry"] = join_gdf.geometry.centroid

    # Keep only parcels with a valid ILR for the aggregation
    join_gdf = join_gdf[join_gdf[ilr_col].notna()]

    # Ensure tracts have the expected id column accessible as a regular column
    tracts = tracts_gdf.copy()
    if tract_id_col not in tracts.columns and tracts.index.name == tract_id_col:
        tracts = tracts.reset_index()

    joined = gpd.sjoin(
        join_gdf[[ilr_col, "geometry"]],
        tracts[[tract_id_col, "geometry"]],
        how="left",
        predicate="within",
    )

    unmatched = joined[tract_id_col].isna().sum()
    if unmatched > 0:
        import warnings
        warnings.warn(
            f"aggregate_ilr_to_tracts: {unmatched} parcel(s) did not fall within any tract.",
            stacklevel=2,
        )

    def _pct_under(s):
        return (s < 1.0).sum() / len(s) if len(s) > 0 else np.nan

    stats = (
        joined.groupby(tract_id_col)[ilr_col]
        .agg(
            median_ilr="median",
            mean_ilr="mean",
            parcel_count="count",
            pct_underimproved=_pct_under,
        )
        .reset_index()
    )

    result = tracts.merge(stats, on=tract_id_col, how="left")
    return result
