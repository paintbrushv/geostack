# ruff: noqa: E501  (fixtures reproduce raw Census file rows)
"""Offline tests for geostack.housing_supply parsing and aggregation."""

import pandas as pd

from geostack import housing_supply as hs

BPS_SAMPLE = """\
Survey,FIPS,FIPS,Region,Division,County,,1-unit,,,2-units,,,3-4 units,,,5+ units,,,1-unit rep,,,2-units rep,,,3-4 units rep,,, 5+units rep
Date,State,County,Code,Code,Name,Bldgs,Units,Value,Bldgs,Units,Value,Bldgs,Units,Value,Bldgs,Units,Value,Bldgs,Units,Value,Bldgs,Units,Value,Bldgs,Units,Value,Bldgs,Units,Value

2025,48,113,3,7,Dallas County                 ,900,900,1,5,10,1,2,8,1,40,2000,1,800,800,1,5,10,1,2,8,1,40,1900,1
2025,48,085,3,7,Collin County                 ,5000,5000,1,0,0,0,0,0,0,10,600,1,4900,4900,1,0,0,0,0,0,0,10,600,1
2025,01,001,3,6,Autauga County                ,200,200,1,0,0,0,0,0,0,0,0,0,200,200,1,0,0,0,0,0,0,0,0,0
"""


def test_parse_bps_county_uses_imputed_columns_and_totals():
    df = hs.parse_bps_county(BPS_SAMPLE).set_index("county_fips")
    dallas = df.loc["48113"]
    assert dallas.units_1u == 900  # imputed estimate, not the 800 reported
    assert dallas.units_2u == 10 and dallas.units_3_4u == 8 and dallas.units_5p == 2000
    assert dallas.units_2p == 2018
    assert dallas.units_total == 2918
    assert len(df) == 3


def test_parse_bps_county_handles_old_padded_format():
    old = "9599  ,     01  , 001  , 3  , 6  , Autauga County  , 219  , 219  , 1  , 0  , 0  , 0  , " \
          "0  , 0  , 0  , 5  , 40  , 1  , 204  , 204  , 1  , 0  , 0  , 0  , 0  , 0  , 0  , 4  , 32  , 1"
    df = hs.parse_bps_county(old)
    assert df.iloc[0].county_fips == "01001"
    assert df.iloc[0].units_total == 259


def test_to_cbsa_aggregates_metros_and_nation(monkeypatch):
    delin = pd.DataFrame({"cbsa_code": ["19100", "19100"], "county_fips": ["48113", "48085"]})
    monkeypatch.setattr(hs, "load_cbsa_delineation", lambda vintage=2023: delin)
    county = hs.parse_bps_county(BPS_SAMPLE).assign(year=2025)
    out = hs._to_cbsa(county, ["19100"], ["units_total"]).set_index("cbsa_code")
    assert out.loc["19100", "units_total"] == 2918 + 5600
    assert out.loc[hs.US, "units_total"] == 2918 + 5600 + 200


def test_pep_vintage_selection_keeps_changes_within_one_vintage():
    assert hs._pep_vintage_for_level(2009) == "2000s"
    assert hs._pep_vintage_for_level(2010) == "2010s"
    assert hs._pep_vintage_for_level(2020) == "2020s"
    # 2009->2010 and 2019->2020 changes come from the vintage holding both years
    assert hs._pep_vintage_for_change(2010) == "2000s"
    assert hs._pep_vintage_for_change(2020) == "2010s"
    assert hs._pep_vintage_for_change(2021) == "2020s"


def test_acs_summary_file_parses_cbsa_geo_ids(monkeypatch, tmp_path):
    text = ("GEO_ID|B25003_E001|B25003_M001|B25003_E002|B25003_M002|B25003_E003|B25003_M003\n"
            "0100000US|1000|1|600|1|400|1\n"
            "310M700US19100|300|1|180|1|120|1\n"
            "312M700US191004819000|50|1|30|1|20|1\n"
            "0400000US48|90|1|60|1|30|1\n")

    class Resp:
        status_code = 200

        def __init__(self, t):
            self.text = t

        def raise_for_status(self):
            pass

    monkeypatch.setattr(hs, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(hs.requests, "get", lambda *a, **k: Resp(text))
    df = hs._acs_households_sf(2024).set_index("cbsa_code")
    assert set(df.index) == {"19100", hs.US}
    assert df.loc["19100", "households"] == 300
    assert df.loc["19100", "renter_households"] == 120


def test_costar_tidy_export_flags_history_and_forecast():
    from geostack.costar_market import tidy_export

    wide = pd.DataFrame({
        "Property Class Name": ["Multi-Family"] * 2,
        "Slice": ["All"] * 2,
        "As Of": ["2026 Q4"] * 2,
        "Geography Name": ["Dallas-Fort Worth - TX USA", "Austin - TX USA - East Austin "],
        "Property Type": ["Multi-Family"] * 2,
        "Geography Code": ["G2-19100", "APT-1-1"],
        "Forecast Scenario": ["Base Case"] * 2,
        "CBSA Code": [19100, 12420],
        "Geography Type": ["Metro", "Submarket"],
        "Concept Name": ["Vacancy Rate", "Inventory Units"],
        "2026 Q3": [0.11, 1000],
        "2026 Q4": [None, None],
        "2027 Q1": [0.10, 1010],
    })
    long = tidy_export(wide)
    assert set(long.status) == {"history", "forecast"}  # empty current quarter is dropped
    assert long[long.period == pd.Period("2027Q1")].status.eq("forecast").all()
    sub = long[long.geo_type == "Submarket"].iloc[0]
    assert sub.market == "Austin" and sub.submarket == "East Austin"
    assert sub.cbsa_code == "12420"
    assert pd.isna(long[long.geo_type == "Metro"].iloc[0].submarket)
