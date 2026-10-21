"""Load CoStar market/submarket time-series exports ("DataExport" workbooks).

CoStar's Data Export is wide: one row per (geography, slice, concept) with a
column per quarter ("2000 Q1" ...), including forecast quarters through the
horizon. This module melts it to a tidy long frame and flags history vs. the
current (quarter-to-date) quarter vs. forecast, using the export's "As Of".

Drop raw exports in data/costar/raw/ (gitignored; CoStar data is licensed).
Anything published from it must cite "Source: CoStar".

Usage:
    from geostack.costar_market import load_costar

    df = load_costar()                       # every export in data/costar/raw
    dfw = df.query("cbsa_code == '19100' and geo_type == 'Metro' and slice == 'All'")
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "costar" / "raw"
_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "costar" / "processed"

_ID_COLS = {
    "Property Class Name": "property_class",
    "Slice": "slice",
    "As Of": "as_of",
    "Geography Name": "geo_name",
    "Property Type": "property_type",
    "Geography Code": "geo_code",
    "Forecast Scenario": "scenario",
    "CBSA Code": "cbsa_code",
    "Geography Type": "geo_type",
    "Concept Name": "concept",
}
_QUARTER = re.compile(r"^(\d{4}) Q([1-4])$")


def _quarter_ts(label: str) -> pd.Period:
    y, q = _QUARTER.match(label).groups()
    return pd.Period(year=int(y), quarter=int(q), freq="Q")


def tidy_export(wide: pd.DataFrame) -> pd.DataFrame:
    """Melt one CoStar DataExport sheet to long form.

    Returns columns: geo_type, geo_name, geo_code, cbsa_code, market, submarket,
    property_type, slice, scenario, concept, period (pd.Period Q), value,
    as_of (pd.Period Q), status ('history' | 'current' | 'forecast').
    """
    qcols = [c for c in wide.columns if _QUARTER.match(str(c))]
    ids = [c for c in _ID_COLS if c in wide.columns]
    long = (wide[ids + qcols].rename(columns=_ID_COLS)
            .melt(id_vars=[_ID_COLS[c] for c in ids], var_name="quarter", value_name="value")
            .dropna(subset=["value"]))
    long["period"] = long["quarter"].map(_quarter_ts)
    long["as_of"] = long["as_of"].map(_quarter_ts)
    long["status"] = "history"
    long.loc[long.period == long.as_of, "status"] = "current"
    long.loc[long.period > long.as_of, "status"] = "forecast"
    long["cbsa_code"] = long["cbsa_code"].astype("Int64").astype(str)
    parts = long["geo_name"].str.split(" - ")
    long["market"] = parts.str[0].str.strip()
    # metro-only exports have no third part, so cast before .str
    long["submarket"] = (parts.str[2].astype("string").str.strip()
                         .where(long["geo_type"] == "Submarket"))
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.drop(columns="quarter").reset_index(drop=True)


def load_costar(paths=None, refresh: bool = False) -> pd.DataFrame:
    """Load and stack CoStar exports (default: every .xlsx/.csv in data/costar/raw).

    Parsed files are cached as parquet next to the raw folder. When the same
    geography/slice/concept/period appears in several exports, the one with the
    latest As Of wins.
    """
    if paths is None:
        paths = sorted(p for p in RAW_DIR.glob("**/*") if p.suffix.lower() in (".xlsx", ".csv"))
    frames = []
    for p in map(Path, paths):
        cache = _CACHE_DIR / f"{p.stem}.parquet"
        if cache.exists() and not refresh and cache.stat().st_mtime >= p.stat().st_mtime:
            frames.append(pd.read_parquet(cache))
            continue
        wide = (pd.read_excel(p, sheet_name=0) if p.suffix.lower() == ".xlsx"
                else pd.read_csv(p))
        long = tidy_export(wide).assign(source_file=p.name)
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        long.astype({"period": str, "as_of": str}).to_parquet(cache, index=False)
        frames.append(long)
    if not frames:
        raise FileNotFoundError(f"No CoStar exports found in {RAW_DIR}")
    df = pd.concat(frames, ignore_index=True)
    for col in ("period", "as_of"):  # cached parquet stores periods as "2026Q3" strings
        if not isinstance(df[col].dtype, pd.PeriodDtype):
            df[col] = pd.PeriodIndex(df[col].astype(str), freq="Q")
    keys = ["geo_code", "slice", "scenario", "concept", "period"]
    return (df.sort_values("as_of").drop_duplicates(keys, keep="last")
            .reset_index(drop=True))


def series(df: pd.DataFrame, concept: str, *, geo_type: str = "Metro", slice_: str = "All",
           by: str = "cbsa_code") -> pd.DataFrame:
    """Wide quarterly frame (index = period) of one concept, one column per geography."""
    sub = df[(df.concept == concept) & (df.geo_type == geo_type) & (df.slice == slice_)]
    return sub.pivot_table(index="period", columns=by, values="value", aggfunc="first")


def as_of(df: pd.DataFrame) -> pd.Period:
    """Latest As Of quarter across the loaded exports."""
    return df["as_of"].max()
