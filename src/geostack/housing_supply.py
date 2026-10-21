"""Metro housing supply vs. demand panel from public, keyless sources.

Builds an annual metro (CBSA) panel of new housing supply (building permits)
alongside the demand side (population, households, payroll jobs) so you can
compute permits per new household, permits per 1,000 residents, permits per
job added, etc. Also pulls national/regional starts and completions from FRED.

Sources (none require an API key):
  - Census Building Permits Survey (BPS), county annual + year-to-date files,
    aggregated to CBSAs with the OMB 2023 delineation (consistent boundaries
    across all years).
  - Census Population Estimates (PEP), county totals 2000-2025.
  - ACS 1-year households (B25003) for CBSAs. Uses the Census API when
    CENSUS_API_KEY is set (2005-2025), otherwise the keyless table-based
    summary files (2019, 2021+). There is no standard 2020 1-year release.
  - FRED graph CSV endpoint (keyless) for starts/completions and metro
    nonfarm payrolls.

Usage:
    from geostack.housing_supply import metro_panel, TEXAS_METROS

    panel = metro_panel(list(TEXAS_METROS), start=2000, end=2025)
"""

from __future__ import annotations

import csv
import io
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from rich.console import Console

console = Console()

_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "housing_supply"
_UA = {"User-Agent": "geostack-research (housing supply panel)"}

_BPS_COUNTY_URL = "https://www2.census.gov/econ/bps/County/co{stamp}.txt"
_DELINEATION_URL = (
    "https://www2.census.gov/programs-surveys/metro-micro/geographies/"
    "reference-files/{v}/delineation-files/list1_{v}.xlsx"
)
_PEP_URLS = {
    "2000s": "https://www2.census.gov/programs-surveys/popest/datasets/"
             "2000-2010/intercensal/county/co-est00int-tot.csv",
    "2010s": "https://www2.census.gov/programs-surveys/popest/datasets/"
             "2010-2020/counties/totals/co-est2020-alldata.csv",
    "2020s": "https://www2.census.gov/programs-surveys/popest/datasets/"
             "2020-2025/counties/totals/co-est2025-alldata.csv",
}
_PEP_HU_URL = ("https://www2.census.gov/programs-surveys/popest/datasets/"
               "2010-2020/housing/HU-EST2020_ALL.csv")
_ACS_SF_URL = {
    2019: "https://www2.census.gov/programs-surveys/acs/summary_file/2019/"
          "prototype/1YRData/acsdt1y2019-b25003.dat",
    "default": "https://www2.census.gov/programs-surveys/acs/summary_file/{y}/"
               "table-based-SF/data/1YRData/acsdt1y{y}-b25003.dat",
}
_ACS_API_URL = "https://api.census.gov/data/{y}/acs/acs1"
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# Label for the national aggregate row in metro panels
US = "US"

TEXAS_METROS: dict[str, str] = {
    "19100": "Dallas-Fort Worth",
    "26420": "Houston",
    "12420": "Austin",
    "41700": "San Antonio",
}

# Fast-growing Sun Belt / Mountain West comparison set
PEER_METROS: dict[str, str] = {
    "38060": "Phoenix",
    "12060": "Atlanta",
    "34980": "Nashville",
    "16740": "Charlotte",
    "19740": "Denver",
    "36740": "Orlando",
    "45300": "Tampa",
    "39580": "Raleigh",
}

# FRED total nonfarm payroll series (SA, thousands) by CBSA
FRED_PAYROLL_IDS: dict[str, str] = {
    "19100": "DALL148NA",
    "26420": "HOUS448NA",
    "12420": "AUST448NA",
    "41700": "SANA748NA",
    "38060": "PHOE004NA",
    "12060": "ATLA013NA",
    "34980": "NASH947NA",
    "16740": "CHAR737NA",
    "19740": "DENV708NA",
    "36740": "ORLA712NA",
    "45300": "TAMP312NA",
    "39580": "RALE537NA",
    US: "PAYEMS",
}

# FRED new residential construction series (SAAR, thousands of units)
FRED_CONSTRUCTION_IDS: dict[str, str] = {
    "starts_total_us": "HOUST",
    "starts_1u_us": "HOUST1F",
    "starts_5p_us": "HOUST5F",
    "completions_total_us": "COMPUTSA",
    "completions_1u_us": "COMPU1USA",
    "completions_5p_us": "COMPU5MUSA",
    "under_construction_total_us": "UNDCONTSA",
    "starts_total_south": "HOUSTS",
    "completions_total_south": "COMPUSTSA",
    "under_construction_total_south": "UNDCONSTSA",
    "starts_total_west": "HOUSTW",
    "completions_total_west": "COMPUWTSA",
    "starts_total_midwest": "HOUSTMW",
    "completions_total_midwest": "COMPUMWTSA",
    "starts_total_northeast": "HOUSTNE",
    "completions_total_northeast": "COMPUNETSA",
    "permits_total_south": "PERMITS",
}


# ---------------------------------------------------------------------------
# Download / cache helpers
# ---------------------------------------------------------------------------

def _cache(name: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / name


def _get(url: str, timeout: int = 120, headers: dict | None = _UA) -> requests.Response:
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def _fresh(path: Path, max_age_days: float | None) -> bool:
    if not path.exists():
        return False
    if max_age_days is None:
        return True
    return (time.time() - path.stat().st_mtime) < max_age_days * 86400


# ---------------------------------------------------------------------------
# CBSA delineation
# ---------------------------------------------------------------------------

def load_cbsa_delineation(vintage: int = 2023) -> pd.DataFrame:
    """County -> CBSA crosswalk from the OMB delineation file.

    Returns DataFrame with: cbsa_code, cbsa_title, metro_micro, county_fips (5-digit),
    county_name, state_name.
    """
    cache = _cache(f"cbsa_delineation_{vintage}.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    raw = pd.read_excel(io.BytesIO(_get(_DELINEATION_URL.format(v=vintage)).content),
                        header=2, dtype=str)
    raw = raw.dropna(subset=["FIPS County Code"])
    df = pd.DataFrame({
        "cbsa_code": raw["CBSA Code"],
        "cbsa_title": raw["CBSA Title"],
        "metro_micro": raw["Metropolitan/Micropolitan Statistical Area"],
        "county_fips": raw["FIPS State Code"].str.zfill(2) + raw["FIPS County Code"].str.zfill(3),
        "county_name": raw["County/County Equivalent"],
        "state_name": raw["State Name"],
    })
    df.to_parquet(cache, index=False)
    return df


def cbsa_counties(cbsa_code: str, vintage: int = 2023) -> list[str]:
    """5-digit county FIPS codes that make up a CBSA."""
    d = load_cbsa_delineation(vintage)
    return d.loc[d["cbsa_code"] == cbsa_code, "county_fips"].tolist()


# ---------------------------------------------------------------------------
# Building Permits Survey (county files)
# ---------------------------------------------------------------------------

_BPS_UNIT_COLS = {"units_1u": 7, "units_2u": 10, "units_3_4u": 13, "units_5p": 16}


def _bps_int(val: str) -> int:
    cleaned = val.strip().replace(",", "")
    return int(cleaned) if cleaned.lstrip("-").isdigit() else 0


def parse_bps_county(text: str) -> pd.DataFrame:
    """Parse a BPS county file (annual 'a' or year-to-date 'y').

    Uses the imputed-estimate columns (not the 'reported only' columns), so totals
    include Census imputation for non-responding permit offices.

    Returns DataFrame with: county_fips, units_1u, units_2u, units_3_4u, units_5p,
    units_total, units_2p.
    """
    rows = []
    for parts in csv.reader(io.StringIO(text)):
        if len(parts) < 18:
            continue
        st, co = parts[1].strip(), parts[2].strip()
        if not (st.isdigit() and co.isdigit()):
            continue
        row = {"county_fips": st.zfill(2) + co.zfill(3)}
        for col, idx in _BPS_UNIT_COLS.items():
            row[col] = _bps_int(parts[idx])
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["units_2p"] = df["units_2u"] + df["units_3_4u"] + df["units_5p"]
    df["units_total"] = df["units_1u"] + df["units_2p"]
    return df


def bps_county(stamp: str, max_age_days: float | None = None) -> pd.DataFrame:
    """Fetch one BPS county file by its stamp: '2024a' (annual) or '2608y' (YTD Aug 2026).

    Annual files are immutable once published; YTD files get revised, so pass
    max_age_days to re-pull them.
    """
    cache = _cache(f"bps_county_{stamp}.parquet")
    if _fresh(cache, max_age_days):
        return pd.read_parquet(cache)
    console.print(f"[dim]Fetching BPS county {stamp}...[/dim]")
    df = parse_bps_county(_get(_BPS_COUNTY_URL.format(stamp=stamp)).text)
    df.to_parquet(cache, index=False)
    return df


def bps_county_annual(years) -> pd.DataFrame:
    """County permits for each year in `years` (adds a `year` column)."""
    frames = []
    for y in years:
        df = bps_county(f"{y}a")
        frames.append(df.assign(year=int(y)))
    return pd.concat(frames, ignore_index=True)


def latest_bps_ytd_stamp(year: int) -> str | None:
    """Latest available YTD stamp (e.g. '2608y') for a calendar year, or None."""
    listing = _get("https://www2.census.gov/econ/bps/County/").text
    yy = f"{year % 100:02d}"
    months = sorted({int(m) for m in re.findall(rf"co{yy}(\d\d)y\.txt", listing)})
    return f"{yy}{months[-1]:02d}y" if months else None


def _to_cbsa(county_df: pd.DataFrame, cbsa_codes, value_cols, by=("year",),
             vintage: int = 2023) -> pd.DataFrame:
    """Aggregate a county-level frame to CBSAs (plus a national `US` row)."""
    d = load_cbsa_delineation(vintage)[["cbsa_code", "county_fips"]]
    d = d[d["cbsa_code"].isin(list(cbsa_codes))]
    by = list(by)
    metro = (county_df.merge(d, on="county_fips")
             .groupby(["cbsa_code", *by], as_index=False)[list(value_cols)].sum())
    nation = county_df.groupby(by, as_index=False)[list(value_cols)].sum().assign(cbsa_code=US)
    return pd.concat([metro, nation], ignore_index=True)


def cbsa_permits_annual(cbsa_codes, years) -> pd.DataFrame:
    """Annual permitted units by structure type for CBSAs (+ US), 2023 boundaries."""
    cols = ["units_1u", "units_2u", "units_3_4u", "units_5p", "units_2p", "units_total"]
    return _to_cbsa(bps_county_annual(years), cbsa_codes, cols)


def cbsa_permits_ytd(cbsa_codes, year: int, month: int | None = None) -> pd.DataFrame:
    """Year-to-date permits for `year` and the same months of the prior year.

    Returns columns: cbsa_code, period ('YYYY YTD'), year, through_month, units_*.
    """
    if month is None:
        stamp = latest_bps_ytd_stamp(year)
        if stamp is None:
            raise ValueError(f"No BPS YTD file published yet for {year}")
        month = int(stamp[2:4])
    cols = ["units_1u", "units_2u", "units_3_4u", "units_5p", "units_2p", "units_total"]
    frames = []
    for y in (year - 1, year):
        stamp = f"{y % 100:02d}{month:02d}y"
        df = bps_county(stamp, max_age_days=7 if y == year else None)
        frames.append(df.assign(year=y))
    out = _to_cbsa(pd.concat(frames, ignore_index=True), cbsa_codes, cols)
    return out.assign(through_month=month)


# ---------------------------------------------------------------------------
# Population estimates (PEP)
# ---------------------------------------------------------------------------

def _pep_frame(key: str) -> pd.DataFrame:
    cache = _cache(f"pep_county_{key}.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    console.print(f"[dim]Fetching PEP county {key}...[/dim]")
    raw = pd.read_csv(io.BytesIO(_get(_PEP_URLS[key]).content), encoding="latin-1",
                      dtype={"STATE": str, "COUNTY": str})
    raw = raw[raw["SUMLEV"].astype(int) == 50].copy()
    raw["county_fips"] = raw["STATE"].str.zfill(2) + raw["COUNTY"].str.zfill(3)
    pop_cols = [c for c in raw.columns if c.startswith("POPESTIMATE")]
    long = raw.melt(id_vars="county_fips", value_vars=pop_cols, value_name="population")
    long["year"] = long["variable"].str[-4:].astype(int)
    long = long.drop(columns="variable")
    long.to_parquet(cache, index=False)
    return long


# Which PEP vintage supplies each year's level, and each year-over-year change.
# Changes are computed within one vintage so methodology breaks don't show up as
# fake growth (e.g. the 2020 Census re-benchmark).
def _pep_vintage_for_level(year: int) -> str:
    return "2000s" if year < 2010 else "2010s" if year < 2020 else "2020s"


def _pep_vintage_for_change(year: int) -> str:
    return "2000s" if year <= 2010 else "2010s" if year <= 2020 else "2020s"


def county_population(years) -> pd.DataFrame:
    """County July-1 population and within-vintage annual change.

    Returns: county_fips, year, population, pop_change.
    """
    frames = {k: _pep_frame(k) for k in _PEP_URLS}
    out = []
    for y in years:
        lv = frames[_pep_vintage_for_level(y)]
        level = lv[lv["year"] == y][["county_fips", "population"]]
        cv = frames[_pep_vintage_for_change(y)]
        cur = cv[cv["year"] == y].set_index("county_fips")["population"]
        prev = cv[cv["year"] == y - 1].set_index("county_fips")["population"]
        chg = (cur - prev).rename("pop_change").reset_index()
        out.append(level.merge(chg, on="county_fips", how="left").assign(year=y))
    return pd.concat(out, ignore_index=True)


def cbsa_population(cbsa_codes, years) -> pd.DataFrame:
    """CBSA population and annual change (+ US), 2023 boundaries."""
    return _to_cbsa(county_population(years), cbsa_codes, ["population", "pop_change"])


def cbsa_housing_units_2020() -> pd.DataFrame:
    """July 2020 housing-unit estimates (PEP vintage 2020) for every CBSA (+ US).

    The stock base for expressing permits as a supply growth rate (% of stock).
    Returns: cbsa_code, housing_units_2020.
    """
    cache = _cache("pep_hu_county_2020.parquet")
    if cache.exists():
        county = pd.read_parquet(cache)
    else:
        raw = pd.read_csv(io.BytesIO(_get(_PEP_HU_URL).content), encoding="latin-1",
                          dtype={"STATE": str, "COUNTY": str})
        raw = raw[raw["SUMLEV"].astype(int) == 50]
        county = pd.DataFrame({
            "county_fips": raw["STATE"].str.zfill(2) + raw["COUNTY"].str.zfill(3),
            "housing_units_2020": raw["HUESTIMATE2020"].astype(int),
        })
        county.to_parquet(cache, index=False)
    d = load_cbsa_delineation()
    return _to_cbsa(county.assign(_k=0), d["cbsa_code"].unique(), ["housing_units_2020"],
                    by=("_k",)).drop(columns="_k")


# ---------------------------------------------------------------------------
# ACS households (B25003: occupied housing units = households)
# ---------------------------------------------------------------------------

def _acs_households_sf(year: int) -> pd.DataFrame | None:
    cache = _cache(f"acs1_b25003_{year}.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    url = _ACS_SF_URL.get(year, _ACS_SF_URL["default"].format(y=year))
    resp = requests.get(url, headers=_UA, timeout=180)
    if 400 <= resp.status_code < 500:  # not published (yet) for this year
        return None
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text), sep="|", dtype={"GEO_ID": str})
    # Whole-CBSA rows look like 310M700US19100 (summary level 310 + vintage component)
    cbsa = raw["GEO_ID"].str.extract(r"^310\w{4}US(\d{5})$")[0]
    raw = raw.assign(cbsa_code=cbsa.where(raw["GEO_ID"] != "0100000US", US)).dropna(
        subset=["cbsa_code"])
    df = pd.DataFrame({
        "cbsa_code": raw["cbsa_code"],
        "households": raw["B25003_E001"].astype(int),
        "renter_households": raw["B25003_E003"].astype(int),
        "year": year,
    })
    df.to_parquet(cache, index=False)
    return df


def _acs_households_api(year: int, key: str) -> pd.DataFrame | None:
    cache = _cache(f"acs1_b25003_{year}.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for geo in ("metropolitan statistical area/micropolitan statistical area:*", "us:1"):
        resp = requests.get(_ACS_API_URL.format(y=year), headers=_UA, timeout=120, params={
            "get": "B25003_001E,B25003_003E", "for": geo, "key": key})
        if resp.status_code in (204, 404):
            return None
        resp.raise_for_status()
        rows = resp.json()
        f = pd.DataFrame(rows[1:], columns=rows[0])
        code_col = f.columns[-1]
        frames.append(pd.DataFrame({
            "cbsa_code": f[code_col] if code_col != "us" else US,
            "households": f["B25003_001E"].astype(int),
            "renter_households": f["B25003_003E"].astype(int),
            "year": year,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache, index=False)
    return df


def _cbsa_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep whole-CBSA rows (310M700US19100 style GEO_IDs) and the national row."""
    cbsa = raw["GEO_ID"].str.extract(r"^310\w{4}US(\d{5})$")[0]
    return (raw.assign(cbsa_code=cbsa.where(raw["GEO_ID"] != "0100000US", US))
            .dropna(subset=["cbsa_code"]).drop(columns="GEO_ID"))


def acs1_table(table: str, year: int) -> pd.DataFrame | None:
    """One ACS 1-year detailed table for every CBSA (+ US).

    Columns are named like the summary files: B25004_E002 (estimate),
    B25004_M002 (90% margin of error). Uses the Census API when CENSUS_API_KEY
    is set, else the keyless table-based summary files (2019, 2021+).
    Returns None when the year isn't published.
    """
    table = table.upper()
    cache = _cache(f"acs1_{table.lower()}_{year}_raw.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    key = os.environ.get("CENSUS_API_KEY")
    if key:
        frames = []
        for geo in ("metropolitan statistical area/micropolitan statistical area:*", "us:1"):
            resp = requests.get(_ACS_API_URL.format(y=year), headers=_UA, timeout=120, params={
                "get": f"group({table})", "for": geo, "key": key})
            if resp.status_code in (204, 404):
                return None
            resp.raise_for_status()
            rows = resp.json()
            f = pd.DataFrame(rows[1:], columns=rows[0])
            code_col = f.columns[-1]
            est = {c: f"{table}_E{c[len(table) + 1:-1]}" for c in f.columns
                   if re.fullmatch(rf"{table}_\d{{3}}[EM]", c)}
            est = {c: v if c.endswith("E") else v.replace("_E", "_M") for c, v in est.items()}
            out = f[list(est)].rename(columns=est).apply(pd.to_numeric, errors="coerce")
            out["cbsa_code"] = f[code_col].values if code_col != "us" else US
            frames.append(out)
        df = pd.concat(frames, ignore_index=True)
    else:
        url = (_ACS_SF_URL.get(year, _ACS_SF_URL["default"].format(y=year))
               .replace("b25003", table.lower()))
        resp = requests.get(url, headers=_UA, timeout=180)
        if 400 <= resp.status_code < 500:
            return None
        resp.raise_for_status()
        raw = pd.read_csv(io.StringIO(resp.text), sep="|", dtype={"GEO_ID": str})
        df = _cbsa_rows(raw)
        num = [c for c in df.columns if c != "cbsa_code"]
        df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df = df.assign(year=year)
    df.to_parquet(cache, index=False)
    return df


def cbsa_tenure_vacancy(cbsa_codes, years) -> pd.DataFrame:
    """ACS 1-year housing stock, tenure and vacancy for CBSAs (+ US).

    Columns: cbsa_code, year, housing_units, households, owner_hh, renter_hh,
    renter_hh_5plus (renters in buildings with 5+ units, the apartment-renter
    match for CoStar multifamily), vacant, for_rent, rented_not_occupied,
    for_sale, sold_not_occupied, rental_stock, rental_vacancy, owner_stock,
    homeowner_vacancy, plus 90% MOEs for for_rent, renter_hh and renter_hh_5plus
    (suffix _moe).
    """
    codes = [*cbsa_codes, US]
    frames = []
    for y in years:
        if y == 2020:
            continue
        t = {name: acs1_table(name, y) for name in ("B25001", "B25003", "B25004", "B25032")}
        if any(v is None for v in t.values()):
            continue
        m = t["B25001"][["cbsa_code", "B25001_E001"]]
        for name in ("B25003", "B25004", "B25032"):
            m = m.merge(t[name].drop(columns="year"), on="cbsa_code")
        m = m[m.cbsa_code.isin(codes)]
        frames.append(pd.DataFrame({
            "cbsa_code": m.cbsa_code, "year": y,
            "housing_units": m.B25001_E001,
            "households": m.B25003_E001,
            "owner_hh": m.B25003_E002,
            "renter_hh": m.B25003_E003,
            "renter_hh_moe": m.B25003_M003,
            # B25032 renter block: E013 total, E018-E021 = 5-9, 10-19, 20-49, 50+ units
            "renter_hh_5plus": m[[f"B25032_E{i:03d}" for i in range(18, 22)]].sum(axis=1),
            # Census approximation for the MOE of a sum: root of summed squared MOEs
            "renter_hh_5plus_moe": (m[[f"B25032_M{i:03d}" for i in range(18, 22)]] ** 2)
            .sum(axis=1) ** 0.5,
            "vacant": m.B25004_E001,
            "for_rent": m.B25004_E002,
            "for_rent_moe": m.B25004_M002,
            "rented_not_occupied": m.B25004_E003,
            "for_sale": m.B25004_E004,
            "sold_not_occupied": m.B25004_E005,
        }))
    df = pd.concat(frames, ignore_index=True)
    df["rental_stock"] = df.renter_hh + df.for_rent + df.rented_not_occupied
    df["rental_vacancy"] = df.for_rent / df.rental_stock
    df["owner_stock"] = df.owner_hh + df.for_sale + df.sold_not_occupied
    df["homeowner_vacancy"] = df.for_sale / df.owner_stock
    return df.sort_values(["cbsa_code", "year"]).reset_index(drop=True)


def cbsa_households(cbsa_codes, years) -> pd.DataFrame:
    """ACS 1-year households for CBSAs (+ US), as published (ACS-vintage boundaries).

    Years without a release (2020, or years not yet published / not available
    keylessly) are skipped silently.
    """
    key = os.environ.get("CENSUS_API_KEY")
    frames = []
    for y in years:
        if y == 2020:
            continue
        df = _acs_households_api(y, key) if key else _acs_households_sf(y)
        if df is None and key:
            df = _acs_households_sf(y)
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["cbsa_code", "households", "renter_households", "year"])
    df = pd.concat(frames, ignore_index=True)
    return df[df["cbsa_code"].isin([*cbsa_codes, US])].reset_index(drop=True)


# ---------------------------------------------------------------------------
# FRED (keyless graph CSV)
# ---------------------------------------------------------------------------

def fred_series(series_id: str, max_age_days: float = 1.0) -> pd.Series:
    """Fetch a FRED series via the public graph CSV endpoint (no API key).

    If FRED is unreachable or returns something other than the CSV (e.g. its
    maintenance page), falls back to the cached copy with a warning.
    """
    cache = _cache(f"fred_{series_id}.parquet")
    if _fresh(cache, max_age_days):
        df = pd.read_parquet(cache)
    else:
        try:
            # FRED's edge stalls on custom UAs; the requests default works
            text = _get(_FRED_CSV_URL.format(sid=series_id), timeout=60, headers=None).text
            if not text.startswith("observation_date"):
                raise ValueError("FRED returned a non-CSV page (maintenance or error)")
            df = pd.read_csv(io.StringIO(text))
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df.to_parquet(cache, index=False)
        except (requests.RequestException, ValueError) as exc:
            if not cache.exists():
                raise
            console.print(f"[yellow]FRED {series_id} unavailable ({exc}); using cached copy "
                          f"from {time.ctime(cache.stat().st_mtime)}[/yellow]")
            df = pd.read_parquet(cache)
    return df.set_index(pd.to_datetime(df["date"]))["value"].rename(series_id)


def fred_frame(ids: dict[str, str], max_age_days: float = 1.0) -> pd.DataFrame:
    """Wide frame of several FRED series, columns named by the dict keys."""
    return pd.concat({k: fred_series(v, max_age_days) for k, v in ids.items()}, axis=1)


def construction_monthly() -> pd.DataFrame:
    """Monthly starts / completions / under construction (SAAR thousands)."""
    return fred_frame(FRED_CONSTRUCTION_IDS)


def cbsa_payrolls_annual(cbsa_codes) -> pd.DataFrame:
    """Annual-average total nonfarm payrolls (thousands) for CBSAs with a known FRED id."""
    frames = []
    for code in [*cbsa_codes, US]:
        sid = FRED_PAYROLL_IDS.get(code)
        if sid is None:
            continue
        s = fred_series(sid).dropna()
        counts = s.groupby(s.index.year).count()
        annual = s.groupby(s.index.year).mean()[counts == 12]  # full years only
        frames.append(pd.DataFrame({"cbsa_code": code, "year": annual.index,
                                    "jobs_k": annual.values}))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def metro_panel(cbsa_codes, start: int = 2000, end: int = 2025) -> pd.DataFrame:
    """Annual supply/demand panel for CBSAs plus a national `US` row.

    Columns: cbsa_code, year, units_1u ... units_total (permits), population,
    pop_change, households (ACS years only), renter_households, jobs_k, job_change_k.
    """
    years = list(range(start, end + 1))
    permits = cbsa_permits_annual(cbsa_codes, years)
    pop = cbsa_population(cbsa_codes, years)
    hh = cbsa_households(cbsa_codes, years)
    jobs = cbsa_payrolls_annual(cbsa_codes)
    jobs = jobs.sort_values(["cbsa_code", "year"])
    jobs["job_change_k"] = jobs.groupby("cbsa_code")["jobs_k"].diff()
    panel = (permits.merge(pop, on=["cbsa_code", "year"], how="left")
             .merge(hh, on=["cbsa_code", "year"], how="left")
             .merge(jobs, on=["cbsa_code", "year"], how="left"))
    return panel.sort_values(["cbsa_code", "year"]).reset_index(drop=True)
