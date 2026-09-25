"""Tests for geostack.cad — CAD fetch functions for all supported counties.

All HTTP calls are mocked — no real network requests.
"""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest

from geostack.cad import (
    COLLIN_DATASET_IDS,
    COLLIN_GIS_URL,
    COOKE_GIS_URL,
    COOKE_SOCRATA_DATASET,
    COUNTY_REGISTRY,
    DALLAS_GIS_URL,
    DALLAS_SOCRATA_DATASET,
    GRAYSON_GIS_URL,
    GRAYSON_SOCRATA_DATASET,
    TCAD_GIS_URL,
    TCAD_SOCRATA_DATASET,
    _arcgis_geojson_fetch,
    fetch_collin_parcel_geometry,
    fetch_collin_parcels,
    fetch_collin_parcels_socrata,
    fetch_cooke_parcel_geometry,
    fetch_cooke_parcels,
    fetch_dallas_parcel_geometry,
    fetch_dallas_parcels,
    fetch_grayson_parcel_geometry,
    fetch_grayson_parcels,
    fetch_parcels,
    fetch_tcad_parcel_geometry,
    fetch_tcad_parcels,
    normalize_parcels,
)


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _mock_resp(json_data):
    """Build a minimal mock requests.Response."""
    m = MagicMock()
    m.json.return_value = json_data
    m.raise_for_status.return_value = None
    return m


def _socrata_rows():
    return [{"prop_id": "11111", "tax_year": "2024", "market_value": "450000"}]


def _geojson_features():
    """One-polygon GeoJSON FeatureCollection body (as returned by ArcGIS REST)."""
    return {
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-96.80, 33.00], [-96.70, 33.00],
                        [-96.70, 33.10], [-96.80, 33.10],
                        [-96.80, 33.00],
                    ]],
                },
                "properties": {"PROP_ID": "11111", "ACCOUNT_NUM": "22222"},
            }
        ]
    }


def _txgio_square_county_boundary():
    from shapely.geometry import box

    return box(0, 0, 10, 10)


def _txgio_parcel_feature(object_id, coordinates):
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
        "properties": {"OBJECTID": object_id, "geo_id": f"GEO-{object_id}"},
    }


def _txgio_statewide_features():
    return {
        "type": "FeatureCollection",
        "features": [
            _txgio_parcel_feature(
                101,
                [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]],
            ),
            # Intersects the county boundary, but its centroid is outside.
            _txgio_parcel_feature(
                202,
                [[9, 9], [13, 9], [13, 13], [9, 13], [9, 9]],
            ),
        ],
    }


def _empty_geojson_features():
    return {"type": "FeatureCollection", "features": []}


def _txgio_keyset_responder(first_page):
    responses = [_mock_resp(first_page), _mock_resp(_empty_geojson_features())]

    def _get(*args, **kwargs):
        if responses:
            return responses.pop(0)
        return _mock_resp(_empty_geojson_features())

    return _get


# ---------------------------------------------------------------------------
# Socrata attribute fetch — parametrized across all 5 counties
# ---------------------------------------------------------------------------

# Note: fetch_collin_parcels_socrata is deprecated and tested separately below.
# Its URL is dynamic (year-specific) and its $where uses propyear instead of tax_year.
# fetch_cooke_parcels and fetch_grayson_parcels no longer use Socrata (dead endpoints)
# and are tested separately below.
SOCRATA_CASES = [
    (fetch_dallas_parcels,  DALLAS_SOCRATA_DATASET,  "account_num"),
    (fetch_tcad_parcels,    TCAD_SOCRATA_DATASET,    "prop_id"),
]


@pytest.mark.parametrize("fetch_fn,expected_url,expected_order", SOCRATA_CASES)
class TestSocrataFetch:
    def test_returns_dataframe(self, fetch_fn, expected_url, expected_order):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())):
            result = fetch_fn()
        assert isinstance(result, pd.DataFrame)

    def test_calls_correct_url(self, fetch_fn, expected_url, expected_order):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())) as mock_get:
            fetch_fn()
        mock_get.assert_called_once()
        assert mock_get.call_args[0][0] == expected_url

    def test_where_uses_single_quoted_year(self, fetch_fn, expected_url, expected_order):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())) as mock_get:
            fetch_fn(year=2024)
        params = mock_get.call_args[1]["params"]
        assert "$where" in params
        assert params["$where"] == "tax_year='2024'"

    def test_default_year_is_2024(self, fetch_fn, expected_url, expected_order):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())) as mock_get:
            fetch_fn()
        params = mock_get.call_args[1]["params"]
        assert "2024" in params["$where"]

    def test_custom_year_passed_through(self, fetch_fn, expected_url, expected_order):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())) as mock_get:
            fetch_fn(year=2023)
        params = mock_get.call_args[1]["params"]
        assert "2023" in params["$where"]

    def test_limit_in_params(self, fetch_fn, expected_url, expected_order):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())) as mock_get:
            fetch_fn(limit=100)
        params = mock_get.call_args[1]["params"]
        assert params["$limit"] == 100

    def test_order_param_present(self, fetch_fn, expected_url, expected_order):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())) as mock_get:
            fetch_fn()
        params = mock_get.call_args[1]["params"]
        assert "$order" in params
        assert params["$order"] == expected_order

    def test_raise_for_status_called(self, fetch_fn, expected_url, expected_order):
        mock_resp = _mock_resp(_socrata_rows())
        with patch("geostack.cad.requests.get", return_value=mock_resp):
            fetch_fn()
        mock_resp.raise_for_status.assert_called_once()

    def test_empty_response_returns_empty_dataframe(self, fetch_fn, expected_url, expected_order):
        with patch("geostack.cad.requests.get", return_value=_mock_resp([])):
            result = fetch_fn()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# ArcGIS geometry fetch — parametrized across all 5 counties
# ---------------------------------------------------------------------------

GIS_CASES = [
    (fetch_collin_parcel_geometry,  COLLIN_GIS_URL,  "PROP_ID"),
    (fetch_dallas_parcel_geometry,  DALLAS_GIS_URL,  "ACCOUNT_NUM"),
    (fetch_cooke_parcel_geometry,   COOKE_GIS_URL,   "PROP_ID"),
    (fetch_tcad_parcel_geometry,    TCAD_GIS_URL,    "PROP_ID"),
    (fetch_grayson_parcel_geometry, GRAYSON_GIS_URL, "PROP_ID"),
]


@pytest.mark.parametrize("fetch_fn,expected_url,id_field", GIS_CASES)
class TestGeometryFetch:
    def test_returns_geodataframe(self, fetch_fn, expected_url, id_field):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())):
            result = fetch_fn()
        assert isinstance(result, gpd.GeoDataFrame)

    def test_crs_is_4326(self, fetch_fn, expected_url, id_field):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())):
            result = fetch_fn()
        assert result.crs.to_epsg() == 4326

    def test_has_geometry_column(self, fetch_fn, expected_url, id_field):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())):
            result = fetch_fn()
        assert result.geometry is not None
        assert len(result) == 1

    def test_calls_correct_url(self, fetch_fn, expected_url, id_field):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())) as mock_get:
            fetch_fn()
        mock_get.assert_called_once()
        assert mock_get.call_args[0][0] == expected_url

    def test_no_parcel_ids_uses_fallback_where(self, fetch_fn, expected_url, id_field):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())) as mock_get:
            fetch_fn(where_clause=None)
        params = mock_get.call_args[1]["params"]
        assert params["where"] == "1=1"

    def test_parcel_ids_build_in_clause(self, fetch_fn, expected_url, id_field):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())) as mock_get:
            fetch_fn(parcel_ids=["AAA", "BBB"])
        params = mock_get.call_args[1]["params"]
        assert id_field in params["where"]
        assert "'AAA'" in params["where"]
        assert "'BBB'" in params["where"]
        assert "IN" in params["where"]

    def test_explicit_where_clause_passed_through(self, fetch_fn, expected_url, id_field):
        custom = "LAND_USE = 'RES'"
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())) as mock_get:
            fetch_fn(where_clause=custom)
        params = mock_get.call_args[1]["params"]
        assert params["where"] == custom

    def test_outfields_is_star(self, fetch_fn, expected_url, id_field):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())) as mock_get:
            fetch_fn()
        params = mock_get.call_args[1]["params"]
        assert params["outFields"] == "*"

    def test_format_is_geojson(self, fetch_fn, expected_url, id_field):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())) as mock_get:
            fetch_fn()
        params = mock_get.call_args[1]["params"]
        assert params["f"] == "geojson"

    def test_max_records_default(self, fetch_fn, expected_url, id_field):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())) as mock_get:
            fetch_fn()
        params = mock_get.call_args[1]["params"]
        assert params["resultRecordCount"] == 1000

    def test_custom_max_records(self, fetch_fn, expected_url, id_field):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())) as mock_get:
            fetch_fn(max_records=500)
        params = mock_get.call_args[1]["params"]
        assert params["resultRecordCount"] == 500

    def test_raise_for_status_called(self, fetch_fn, expected_url, id_field):
        mock_resp = _mock_resp(_geojson_features())
        with patch("geostack.cad.requests.get", return_value=mock_resp):
            fetch_fn()
        mock_resp.raise_for_status.assert_called_once()



class TestTxgioWiseDentonGeometry:
    def _install_square_county_boundary(self, monkeypatch, cad):
        monkeypatch.setattr(
            cad,
            "_txgio_county_boundary",
            lambda *args, **kwargs: _txgio_square_county_boundary(),
            raising=False,
        )

    def _assert_uses_services1_bbox_keyset(self, cad, fetch_name, monkeypatch):
        self._install_square_county_boundary(monkeypatch, cad)
        with patch(
            "geostack.cad.requests.get",
            side_effect=_txgio_keyset_responder(_txgio_statewide_features()),
        ) as mock_get:
            getattr(cad, fetch_name)()

        first_call = mock_get.call_args_list[0]
        url = first_call[0][0]
        params = first_call[1]["params"]

        assert "services1.arcgis.com" in url
        assert "FeatureServer/0/query" in url
        assert "MapServer" not in url
        assert params["f"] == "geojson"
        assert params["geometryType"] == "esriGeometryEnvelope"
        assert params["spatialRel"] == "esriSpatialRelIntersects"
        assert "geometry" in params
        assert params["geometry"]
        assert "OBJECTID" in params["where"]
        assert ">" in params["where"]
        assert "fips" not in params["where"].lower()
        assert "OBJECTID" in params["orderByFields"]
        assert "resultOffset" not in params

    def _assert_clips_by_county_centroid(self, cad, fetch_name, monkeypatch):
        self._install_square_county_boundary(monkeypatch, cad)
        with patch(
            "geostack.cad.requests.get",
            side_effect=_txgio_keyset_responder(_txgio_statewide_features()),
        ):
            result = getattr(cad, fetch_name)()

        assert list(result["OBJECTID"]) == [101]
        assert result.geometry.iloc[0].centroid.within(_txgio_square_county_boundary())

    def test_wise_geometry_uses_services1_bbox_keyset_not_fips_where_or_offset(self, monkeypatch):
        from geostack import cad

        self._assert_uses_services1_bbox_keyset(
            cad, "fetch_wise_parcel_geometry", monkeypatch
        )

    def test_denton_geometry_uses_services1_bbox_keyset_not_fips_where_or_offset(self, monkeypatch):
        from geostack import cad

        self._assert_uses_services1_bbox_keyset(
            cad, "fetch_denton_parcel_geometry", monkeypatch
        )

    def test_wise_geometry_clips_statewide_response_by_centroid_within_county_boundary(self, monkeypatch):
        from geostack import cad

        self._assert_clips_by_county_centroid(
            cad, "fetch_wise_parcel_geometry", monkeypatch
        )

    def test_denton_geometry_clips_statewide_response_by_centroid_within_county_boundary(self, monkeypatch):
        from geostack import cad

        self._assert_clips_by_county_centroid(
            cad, "fetch_denton_parcel_geometry", monkeypatch
        )

# ---------------------------------------------------------------------------
# _arcgis_geojson_fetch (the shared helper)
# ---------------------------------------------------------------------------

class TestArcgisGeojsonFetch:
    def test_returns_geodataframe(self):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())):
            result = _arcgis_geojson_fetch("https://example.com/query", "1=1", 100, "Test")
        assert isinstance(result, gpd.GeoDataFrame)

    def test_correct_url_used(self):
        url = "https://example.com/query"
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())) as mock_get:
            _arcgis_geojson_fetch(url, "1=1", 100, "Test")
        assert mock_get.call_args[0][0] == url

    def test_where_and_count_in_params(self):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_geojson_features())) as mock_get:
            _arcgis_geojson_fetch("https://example.com/query", "LAND_USE='COM'", 250, "Test")
        params = mock_get.call_args[1]["params"]
        assert params["where"] == "LAND_USE='COM'"
        assert params["resultRecordCount"] == 250

    def test_all_geometry_functions_delegate_to_helper(self):
        """Dallas, Cooke, Travis, Grayson geometry functions all route through _arcgis_geojson_fetch."""
        for fetch_fn in (
            fetch_dallas_parcel_geometry,
            fetch_cooke_parcel_geometry,
            fetch_tcad_parcel_geometry,
            fetch_grayson_parcel_geometry,
        ):
            with patch("geostack.cad._arcgis_geojson_fetch", return_value=gpd.GeoDataFrame()) as mock_helper:
                fetch_fn()
            mock_helper.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_cooke_parcels — True Prodigy path (Socrata mgnk-vbdk dead since 2025)
# ---------------------------------------------------------------------------

class TestFetchCookeParcels:
    def _tp_raw_rows(self):
        return pd.DataFrame([{
            "pAccountID": "9871740",
            "pYear": "2025",
            "landValue": 470255,
            "improvementValue": 1142882,
            "appraisedValue": 1613137,
            "marketValue": 1613137,
            "name": "LYNGHOLM JAMES M",
            "fullSitus": "100 NOWETA PL, LAKE KIOWA, TX, 76240",
            "city": "LAKE KIOWA",
            "zip": "76240",
            "propType": "R",
            "useCd": "A1",
            "legalAcreage": 0.5,
            "latitude": 33.5583587794,
            "longitude": -97.0158099731,
            "geoID": "7800-001-0073R",
        }])

    def test_delegates_to_trueprodigy(self):
        """fetch_cooke_parcels should delegate to fetch_cooke_parcels_trueprodigy."""
        with patch("geostack.cad.fetch_cooke_parcels_trueprodigy",
                   return_value=self._tp_raw_rows()) as mock_tp:
            result = fetch_cooke_parcels()
        mock_tp.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_returns_dataframe(self):
        with patch("geostack.cad.fetch_cooke_parcels_trueprodigy",
                   return_value=self._tp_raw_rows()):
            result = fetch_cooke_parcels()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_normalize_maps_tp_columns(self):
        """After normalize_parcels, raw TP fields map to canonical column names."""
        with patch("geostack.cad.fetch_cooke_parcels_trueprodigy",
                   return_value=self._tp_raw_rows()):
            raw = fetch_cooke_parcels()
        norm = normalize_parcels(raw, "cooke")
        assert "parcel_id" in norm.columns
        assert "owner_name" in norm.columns
        assert "total_value" in norm.columns
        assert norm.iloc[0]["owner_name"] == "LYNGHOLM JAMES M"
        assert norm.iloc[0]["total_value"] == 1613137.0

    def test_normalize_coerces_na_strings_to_numeric(self):
        """True Prodigy emits literal 'N/A' strings for missing market_value/lat/lon.
        normalize_parcels must coerce them to NaN (regression: pyarrow parquet write
        failed on object-dtype market_value during the 2026-06-09 live run)."""
        rows = self._tp_raw_rows()
        na_row = rows.iloc[0].copy()
        na_row["pAccountID"] = "9999999"
        na_row["marketValue"] = "N/A"
        na_row["latitude"] = "N/A"
        na_row["longitude"] = "N/A"
        raw = pd.concat([rows, na_row.to_frame().T], ignore_index=True)
        norm = normalize_parcels(raw, "cooke")
        for col in ("market_value", "latitude", "longitude"):
            assert pd.api.types.is_numeric_dtype(norm[col]), f"{col} must be numeric"
        assert pd.isna(norm.iloc[1]["market_value"])
        assert pd.isna(norm.iloc[1]["latitude"])
        # parquet round-trip must not raise
        import io
        norm.to_parquet(io.BytesIO())


# ---------------------------------------------------------------------------
# fetch_grayson_parcels — PACS bulk path (Socrata e4rz-6rb7 dead since 2025)
# ---------------------------------------------------------------------------

class TestFetchGraysonParcels:
    def _grayson_raw_records(self):
        return [{
            "prop_id": "000000100008",
            "prop_val_yr": "2025",
            "prop_type_cd": "R",
            "imprv_state_cd": "",
            "land_state_cd": "A1",
            "geo_id": "002 106900401",
            "py_owner_name": "JONES MICHAEL STEVEN",
            "situs_address": "631 DELAWARE BEND RD",
            "situs_city": "WHITESBORO",
            "situs_zip": "76273",
            "hood_cd": "001",
            "land_value": 288267,
            "improvement_value": 118740,
            "appraised_val": 407007,
            "assessed_val": 407007,
            "legal_acreage": 1.234,
        }]

    def _grayson_raw_rows(self):
        df = pd.DataFrame(self._grayson_raw_records())
        df["yr_built"] = 1978
        df["main_area_sqft"] = 1850.0
        return df

    def _run_fetch(self, tmp_path, year=None, parse_records=None):
        """Run fetch_grayson_parcels with download + parse mocked, cache in tmp_path."""
        if parse_records is None:
            parse_records = self._grayson_raw_records()
        fake_zip = tmp_path / "GCAD_Export.zip"
        fake_zip.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty ZIP
        with patch("geostack.cad._grayson_cache_dir", return_value=tmp_path), \
             patch("geostack.cad._parse_grayson_appraisal_info",
                   return_value=parse_records), \
             patch("geostack.cad._parse_grayson_improvement_detail",
                   return_value=pd.DataFrame(columns=["prop_id", "yr_built", "main_area_sqft"])):
            return fetch_grayson_parcels(year=year)

    def test_default_year_none_detects_roll_year(self, tmp_path):
        """Bare call (year=None) parses without a year filter and detects 2025."""
        result = self._run_fetch(tmp_path, year=None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["prop_val_yr"] == "2025"
        # Cache file named after the DETECTED year
        assert (tmp_path / "grayson_2025.parquet").exists()

    def test_year_none_rebuilds_stale_cache_missing_land_state_code(self, tmp_path):
        """A cached roll missing newly parsed PACS columns is stale and must be rebuilt."""
        stale = self._grayson_raw_rows().drop(columns=["land_state_cd"])
        stale.loc[0, "py_owner_name"] = "STALE CACHE OWNER"
        stale.to_parquet(tmp_path / "grayson_2025.parquet", index=False)
        fake_zip = tmp_path / "GCAD_Export.zip"
        fake_zip.write_bytes(b"PK\x05\x06" + b"\x00" * 18)

        with patch("geostack.cad._grayson_cache_dir", return_value=tmp_path), \
             patch("geostack.cad._parse_grayson_appraisal_info",
                   return_value=self._grayson_raw_records()) as parse_appraisal, \
             patch("geostack.cad._parse_grayson_improvement_detail",
                   return_value=pd.DataFrame(columns=["prop_id", "yr_built", "main_area_sqft"])):
            result = fetch_grayson_parcels(year=None)

        parse_appraisal.assert_called_once()
        assert "land_state_cd" in result.columns
        assert result.iloc[0]["land_state_cd"] == "A1"
        assert result.iloc[0]["py_owner_name"] == "JONES MICHAEL STEVEN"


    def test_explicit_stale_year_raises(self, tmp_path):
        """Requesting a year absent from the export raises instead of returning empty."""
        with pytest.raises(ValueError, match="2024"):
            self._run_fetch(tmp_path, year=2024, parse_records=[])

    def test_year_none_empty_export_raises(self, tmp_path):
        """Zero certified records is never a valid roll — raise even with year=None."""
        with pytest.raises(ValueError, match="no certified"):
            self._run_fetch(tmp_path, year=None, parse_records=[])

    def test_dispatcher_bare_call_returns_roll(self, tmp_path):
        """fetch_parcels('grayson') with NO year (npl's call pattern) must work."""
        fake_zip = tmp_path / "GCAD_Export.zip"
        fake_zip.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
        with patch("geostack.cad._grayson_cache_dir", return_value=tmp_path), \
             patch("geostack.cad._parse_grayson_appraisal_info",
                   return_value=self._grayson_raw_records()), \
             patch("geostack.cad._parse_grayson_improvement_detail",
                   return_value=pd.DataFrame(columns=["prop_id", "yr_built", "main_area_sqft"])):
            raw = fetch_parcels("grayson")
        norm = normalize_parcels(raw, "grayson")
        assert len(norm) == 1
        assert norm.iloc[0]["tax_year"] == 2025

    def test_cache_dir_is_repo_anchored(self):
        """Cache dir must live inside the geostack repo, not be cwd-relative."""
        from pathlib import Path
        import geostack.cad as cad_mod
        from geostack.cad import _grayson_cache_dir
        cache_dir = _grayson_cache_dir()
        assert cache_dir.is_absolute()
        # <repo>/src/geostack/cad.py → <repo>/data/grayson
        repo_root = Path(cad_mod.__file__).resolve().parent.parent.parent
        assert cache_dir == repo_root / "data" / "grayson"

    def test_normalize_maps_pacs_columns(self):
        """After normalize_parcels, PACS field names map to canonical column names."""
        raw = self._grayson_raw_rows()
        norm = normalize_parcels(raw, "grayson")
        assert norm.iloc[0]["parcel_id"] == "000000100008"
        assert norm.iloc[0]["owner_name"] == "JONES MICHAEL STEVEN"
        assert norm.iloc[0]["situs_address"] == "631 DELAWARE BEND RD"
        assert norm.iloc[0]["total_value"] == 407007.0
        assert norm.iloc[0]["tax_year"] == 2025

    def test_normalize_prefers_improvement_state_code_then_land_state_fallback(self):
        """state_use_code uses improvement state when present, else land state."""
        primary = self._grayson_raw_records()[0].copy()
        primary["prop_id"] = "000000100008"
        primary["imprv_state_cd"] = "A1"
        primary["land_state_cd"] = "B2"

        fallback = primary.copy()
        fallback["prop_id"] = "000000100009"
        fallback["imprv_state_cd"] = ""
        fallback["land_state_cd"] = "F1"

        raw = pd.DataFrame([primary, fallback])
        raw["yr_built"] = 1978
        raw["main_area_sqft"] = 1850.0

        norm = normalize_parcels(raw, "grayson")
        state_use_by_parcel = norm.set_index("parcel_id")["state_use_code"].to_dict()
        assert state_use_by_parcel["000000100008"] == "A1"
        assert state_use_by_parcel["000000100009"] == "F1"

    def test_pacs_property_type_mapped(self):
        """PACS 'R' should map to 'Real' by the pacs=True code path."""
        raw = self._grayson_raw_rows()
        norm = normalize_parcels(raw, "grayson")
        assert norm.iloc[0]["property_type"] == "Real"

    def test_dead_socrata_url_still_exists_as_constant(self):
        """GRAYSON_SOCRATA_DATASET kept for historical reference."""
        assert GRAYSON_SOCRATA_DATASET.endswith(".json")
        assert "e4rz-6rb7" in GRAYSON_SOCRATA_DATASET


# ---------------------------------------------------------------------------
# PACS legal_acreage — 4 IMPLIED decimal places
# ---------------------------------------------------------------------------

def _make_pacs_appraisal_info_line(
    acreage_field: str = "0000000000001865",
    land_state_cd: str = "",
) -> str:
    """Build a synthetic PACS APPRAISAL_INFO.TXT fixed-width line.

    Field positions per 'Appraisal Export Layout - 8.0.31.xlsx' (shipped inside
    the Grayson GCAD_Export.zip). legal_acreage occupies 1-based 1660-1675;
    land_state_cd occupies 1-based 2742-2751.
    """
    line = list(" " * 2760)

    def put(start: int, text: str) -> None:
        line[start:start + len(text)] = list(text)

    put(0, "000000123456")        # prop_id
    put(12, "R")                  # prop_type_cd — real property
    put(17, "02025")              # prop_val_yr (zero-padded, as in live exports)
    put(22, "000000000000")       # sup_num == 0 (certified)
    put(546, "TEST GEO ID")       # geo_id
    put(608, "DOE JOHN")          # py_owner_name
    put(1659, acreage_field)      # legal_acreage (16 chars, 4 implied decimals)
    put(2741, land_state_cd)      # land_state_cd
    put(1915, "000000000400000")  # appraised_val
    put(1945, "000000000400000")  # assessed_val
    return "".join(line)


def _write_pacs_zip(tmp_path, line: str) -> str:
    """Write *line* into a minimal PACS-style ZIP and return its path."""
    import zipfile

    zip_path = tmp_path / "PACS_Export.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("2026-01-01_000001_APPRAISAL_INFO.TXT", line + "\r\n")
    return str(zip_path)


class TestPacsLegalAcreageImpliedDecimals:
    """PACS stores legal_acreage as a zero-padded integer with 4 IMPLIED
    decimal places — the layout spec says 'Legal Acreage (4 Decimals)' and the
    raw field has no decimal point: '0000000000001865' means 0.1865 acres.

    Regression for the unit bug where the raw integer passed straight through
    to land_acres, rendering a 0.19-acre Denton lot as 1,865 'acres' (npl
    initially misdiagnosed the inflated values as square feet; live medians
    Denton 1,800 / Grayson 3,134 are acres x 10,000, i.e. 0.18 / 0.31 ac).
    """

    def test_pacs_acres_implied_decimals(self):
        from geostack.cad import _pacs_acres
        assert _pacs_acres("0000000000001865") == pytest.approx(0.1865)
        assert _pacs_acres("0000000031552000") == pytest.approx(3155.2)
        assert _pacs_acres("0000000000100000") == pytest.approx(10.0)

    def test_pacs_acres_explicit_decimal_point_honored(self):
        """If a vintage ever emits a literal decimal point, take it as-is."""
        from geostack.cad import _pacs_acres
        assert _pacs_acres("         12.5000") == pytest.approx(12.5)

    def test_pacs_acres_blank_and_garbage(self):
        from geostack.cad import _pacs_acres
        assert _pacs_acres("                ") is None
        assert _pacs_acres("ABC") is None

    def test_grayson_parser_decodes_implied_decimals(self, tmp_path):
        from geostack.cad import _parse_grayson_appraisal_info
        zip_path = _write_pacs_zip(tmp_path, _make_pacs_appraisal_info_line())
        records = _parse_grayson_appraisal_info(zip_path, None, None, use_python_zip=True)
        assert len(records) == 1
        assert records[0]["legal_acreage"] == pytest.approx(0.1865)

    def test_grayson_parser_decodes_land_state_code(self, tmp_path):
        from geostack.cad import _parse_grayson_appraisal_info
        zip_path = _write_pacs_zip(
            tmp_path,
            _make_pacs_appraisal_info_line(land_state_cd="F1"),
        )
        records = _parse_grayson_appraisal_info(zip_path, None, None, use_python_zip=True)
        assert len(records) == 1
        assert records[0]["land_state_cd"] == "F1"

    def test_denton_parser_decodes_implied_decimals(self, tmp_path):
        from geostack.cad import _parse_denton_appraisal_info
        zip_path = _write_pacs_zip(tmp_path, _make_pacs_appraisal_info_line())
        records = _parse_denton_appraisal_info(zip_path, 2025, None, use_python_zip=True)
        assert len(records) == 1
        assert records[0]["legal_acreage"] == pytest.approx(0.1865)

    def test_wise_parser_decodes_implied_decimals(self, tmp_path):
        from geostack.cad import _parse_wise_appraisal_info
        zip_path = _write_pacs_zip(tmp_path, _make_pacs_appraisal_info_line())
        records = _parse_wise_appraisal_info(zip_path, 2025, None)
        assert len(records) == 1
        assert records[0]["legal_acreage"] == pytest.approx(0.1865)


# ---------------------------------------------------------------------------
# Denton PACS parser — mail address emit
#
# Verified 2026-06-28 against live Denton 2024 and 2025 exports:
#   - 2025 format: street in py_addr_line1 (pos 693-753), line2 blank
#   - 2024 format: street in py_addr_line2 (pos 753-813), line1 blank
#   - city=873-923, state=923-973, zip=978-983
#   - 99.7%+ non-blank rate on 5k sample (2024); 100% on 2k sample (2025)
#   - Absentee owner confirmed: CROSS MARKETA, situs=456 APPLEGATE DR LEWISVILLE
#     75067, mail=1825 ARBOR CREEK DR APT 119 CARROLLTON 75010 (zips differ)
# ---------------------------------------------------------------------------

def _make_denton_mail_line(mail_line1: str = "", mail_line2: str = "",
                            mail_city: str = "", mail_state: str = "",
                            mail_zip: str = "") -> str:
    """Build a synthetic Denton APPRAISAL_INFO.TXT line with mailing address fields."""
    line = list(" " * 2760)

    def put(start: int, text: str) -> None:
        line[start:start + len(text)] = list(text)

    put(0, "000000000012")        # prop_id
    put(12, "R")                  # prop_type_cd — real property
    put(17, "02025")              # prop_val_yr (zero-padded)
    put(22, "000000000000")       # sup_num == 0 (certified)
    put(546, "SL0100A-00000A-")   # geo_id
    put(608, "CROSS MARKETA")     # py_owner_name
    # Mailing address block (PACS py_addr_* — same positions as Grayson/Wise)
    put(693, mail_line1)          # py_addr_line1: care-of / PO Box (often blank)
    put(753, mail_line2)          # py_addr_line2: street delivery line (primary)
    put(873, mail_city)           # py_addr_city
    put(923, mail_state)          # py_addr_state
    put(978, mail_zip)            # py_addr_zip (5-digit)
    put(1659, "0000000000001865") # legal_acreage
    put(1915, "000000000400000")  # appraised_val
    put(1945, "000000000400000")  # assessed_val
    return "".join(line)


class TestDentonParserMailAddr:
    """Unit tests for the Denton APPRAISAL_INFO parser mailing-address emit.

    The slice positions are byte-identical to Grayson/Wise (PACS 8.0.30/8.0.31);
    verified against live Denton 2024 and 2025 exports. Two vintage cases:
      - 2025: street in py_addr_line1 (pos 693); line2 blank
      - 2024: street in py_addr_line2 (pos 753); line1 blank
    The parser uses ``_m_line2 or _m_line1`` so both work without branching.
    """

    def _parse(self, tmp_path, line: str) -> list[dict]:
        from geostack.cad import _parse_denton_appraisal_info
        zip_path = _write_pacs_zip(tmp_path, line)
        return _parse_denton_appraisal_info(zip_path, 2025, None, use_python_zip=True)

    def test_mail_addr_in_line1_2025_vintage(self, tmp_path):
        """2025 export pattern: street in line1, line2 blank — mirror of live record 000000000012."""
        line = _make_denton_mail_line(
            mail_line1="1825 ARBOR CREEK DR APT 119",
            mail_city="CARROLLTON",
            mail_state="TX",
            mail_zip="75010",
        )
        records = self._parse(tmp_path, line)
        assert len(records) == 1
        r = records[0]
        assert r["mail_addr1"] == "1825 ARBOR CREEK DR APT 119"
        assert r["mail_city"] == "CARROLLTON"
        assert r["mail_state"] == "TX"
        assert r["mail_zip"] == "75010"

    def test_mail_addr_in_line2_2024_vintage(self, tmp_path):
        """2024 export pattern: street in line2, line1 blank — parser prefers line2."""
        line = _make_denton_mail_line(
            mail_line1="",
            mail_line2="1444 APPLEGATE DR",
            mail_city="LEWISVILLE",
            mail_state="TX",
            mail_zip="75067",
        )
        records = self._parse(tmp_path, line)
        assert len(records) == 1
        r = records[0]
        # prefer line2 (street) over line1 (blank)
        assert r["mail_addr1"] == "1444 APPLEGATE DR"
        assert r["mail_city"] == "LEWISVILLE"
        assert r["mail_state"] == "TX"
        assert r["mail_zip"] == "75067"

    def test_mail_addr_falls_back_to_line1_when_line2_blank(self, tmp_path):
        """PO Box case: line1 has PO Box, line2 blank — falls back to line1."""
        line = _make_denton_mail_line(
            mail_line1="PO BOX 1234",
            mail_line2="",
            mail_city="DENTON",
            mail_state="TX",
            mail_zip="76201",
        )
        records = self._parse(tmp_path, line)
        assert len(records) == 1
        assert records[0]["mail_addr1"] == "PO BOX 1234"
        assert records[0]["mail_city"] == "DENTON"

    def test_mail_addr_empty_when_both_lines_blank(self, tmp_path):
        """When both address lines are blank, mail_addr1 is empty string."""
        line = _make_denton_mail_line()
        records = self._parse(tmp_path, line)
        assert len(records) == 1
        assert records[0]["mail_addr1"] == ""

    def test_mail_keys_present_in_every_record(self, tmp_path):
        """Every parsed record must carry the four mail_ keys (even if blank)."""
        line = _make_denton_mail_line(
            mail_line1="456 OAK ST", mail_city="FLOWER MOUND",
            mail_state="TX", mail_zip="75028",
        )
        records = self._parse(tmp_path, line)
        assert len(records) == 1
        for key in ("mail_addr1", "mail_city", "mail_state", "mail_zip"):
            assert key in records[0], f"Expected key {key!r} in parsed record"

    def test_normalize_parcels_carries_mail_columns_for_denton(self, tmp_path):
        """normalize_parcels(_DENTON_COLUMN_MAP) must surface mail_addr1 → mail_addr1,
        mail_city, mail_state, mail_zip from the raw PACS record."""
        import pandas as pd
        from geostack.cad import _parse_denton_appraisal_info, normalize_parcels
        line = _make_denton_mail_line(
            mail_line1="1825 ARBOR CREEK DR APT 119",
            mail_city="CARROLLTON",
            mail_state="TX",
            mail_zip="75010",
        )
        zip_path = _write_pacs_zip(tmp_path, line)
        records = _parse_denton_appraisal_info(zip_path, 2025, None, use_python_zip=True)
        raw_df = pd.DataFrame(records)
        # Simulate the yr_built / main_area_sqft columns that fetch_denton_parcels adds
        raw_df["yr_built"] = None
        raw_df["main_area_sqft"] = None
        norm = normalize_parcels(raw_df, "denton")
        assert "mail_addr1" in norm.columns, "normalize_parcels must carry mail_addr1 for denton"
        assert norm["mail_addr1"].iloc[0] == "1825 ARBOR CREEK DR APT 119"
        assert norm["mail_city"].iloc[0] == "CARROLLTON"
        assert norm["mail_state"].iloc[0] == "TX"
        assert norm["mail_zip"].iloc[0] == "75010"


# ---------------------------------------------------------------------------
# URL constants sanity checks
# ---------------------------------------------------------------------------

class TestUrlConstants:
    def test_socrata_urls_are_json_endpoints(self):
        for url in (DALLAS_SOCRATA_DATASET, COOKE_SOCRATA_DATASET,
                    TCAD_SOCRATA_DATASET, GRAYSON_SOCRATA_DATASET):
            assert url.endswith(".json"), f"Socrata endpoint should end in .json: {url}"

    def test_collin_dataset_ids_are_nonempty_strings(self):
        for yr, dataset_id in COLLIN_DATASET_IDS.items():
            assert isinstance(dataset_id, str) and len(dataset_id) > 0, (
                f"COLLIN_DATASET_IDS[{yr}] must be a non-empty string"
            )

    def test_gis_urls_end_with_query(self):
        for url in (COLLIN_GIS_URL, DALLAS_GIS_URL,
                    COOKE_GIS_URL, TCAD_GIS_URL, GRAYSON_GIS_URL):
            assert url.endswith("/query"), f"ArcGIS REST endpoint should end with /query: {url}"

    def test_all_gis_urls_are_https(self):
        for url in (COLLIN_GIS_URL, DALLAS_GIS_URL,
                    COOKE_GIS_URL, TCAD_GIS_URL, GRAYSON_GIS_URL):
            assert url.startswith("https://"), f"Expected HTTPS: {url}"

    def test_all_socrata_urls_are_https(self):
        for url in (DALLAS_SOCRATA_DATASET, COOKE_SOCRATA_DATASET,
                    TCAD_SOCRATA_DATASET, GRAYSON_SOCRATA_DATASET):
            assert url.startswith("https://"), f"Expected HTTPS: {url}"


# ---------------------------------------------------------------------------
# fetch_collin_parcels_socrata (deprecated single-page interface)
# ---------------------------------------------------------------------------

class TestCollinSocrataDeprecated:
    def test_returns_dataframe(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())):
                result = fetch_collin_parcels_socrata(year=2024)
        assert isinstance(result, pd.DataFrame)

    def test_uses_year_specific_url(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())) as mock_get:
                fetch_collin_parcels_socrata(year=2024)
        url = mock_get.call_args[0][0]
        assert COLLIN_DATASET_IDS[2024] in url

    def test_emits_deprecation_warning(self):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())):
            with pytest.warns(DeprecationWarning, match="deprecated"):
                fetch_collin_parcels_socrata(year=2024)

    def test_where_uses_propyear(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())) as mock_get:
                fetch_collin_parcels_socrata(year=2023)
        params = mock_get.call_args[1]["params"]
        assert "propyear" in params["$where"]
        assert "2023" in params["$where"]


# ---------------------------------------------------------------------------
# fetch_collin_parcels (new full-pagination interface)
# ---------------------------------------------------------------------------

class TestFetchCollinParcels:
    def _two_page_mock(self):
        """Returns two responses: first with 50K rows stub (two records), second empty."""
        page1 = _mock_resp([{"propid": "A", "propyear": "2024"}] * 2)
        page2 = _mock_resp([])
        return [page1, page2]

    def test_returns_dataframe(self):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())):
            result = fetch_collin_parcels(year=2024)
        assert isinstance(result, pd.DataFrame)

    def test_raises_for_unknown_year(self):
        with pytest.raises(ValueError, match="2000"):
            fetch_collin_parcels(year=2000)

    def test_uses_year_specific_dataset_id(self):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())) as mock_get:
            fetch_collin_parcels(year=2024)
        url = mock_get.call_args[0][0]
        assert COLLIN_DATASET_IDS[2024] in url

    def test_limit_passed_through(self):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())) as mock_get:
            fetch_collin_parcels(year=2024, limit=100)
        params = mock_get.call_args[1]["params"]
        assert params["$limit"] == 100

    def test_core_only_adds_select_param(self):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())) as mock_get:
            fetch_collin_parcels(year=2024, core_only=True)
        params = mock_get.call_args[1]["params"]
        assert "$select" in params

    def test_full_mode_no_select_param(self):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())) as mock_get:
            fetch_collin_parcels(year=2024, core_only=False)
        params = mock_get.call_args[1]["params"]
        assert "$select" not in params

    def test_pagination_stops_on_empty_page(self):
        responses = [_mock_resp([{"propid": "X"}]), _mock_resp([])]
        with patch("geostack.cad.requests.get", side_effect=responses):
            result = fetch_collin_parcels(year=2024)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# COUNTY_REGISTRY
# ---------------------------------------------------------------------------

class TestCountyRegistry:
    def test_all_five_counties_present(self):
        for county in ("collin", "dallas", "cooke", "travis", "grayson"):
            assert county in COUNTY_REGISTRY

    def test_each_entry_has_required_keys(self):
        required = {"state_fips", "county_fips", "state", "gis_url", "column_map"}
        for county, meta in COUNTY_REGISTRY.items():
            for key in required:
                assert key in meta, f"{county} missing key {key!r}"

    def test_collin_has_dataset_ids(self):
        assert COUNTY_REGISTRY["collin"]["dataset_ids"] is not None
        assert 2024 in COUNTY_REGISTRY["collin"]["dataset_ids"]

    def test_state_fips_are_strings(self):
        for county, meta in COUNTY_REGISTRY.items():
            assert isinstance(meta["state_fips"], str)
            assert meta["state_fips"].isdigit()


# ---------------------------------------------------------------------------
# normalize_parcels
# ---------------------------------------------------------------------------

class TestNormalizeParcels:
    def _collin_raw(self):
        """Minimal raw Collin CAD Socrata row after fetch_collin_parcels."""
        return pd.DataFrame([{
            "propid": "123456",
            "propyear": "2024",
            "currvalland": "150000",
            "currvalimprv": "350000",
            "currvalmarket": "500000",
            "imprvyearbuilt": "1995",
            "ownername": "ACME LLC",
            "situsconcat": "  123 main st  ",
            "situscity": "plano",
            "situszip": "75025",
            "proptype": "R",
        }])

    def test_returns_dataframe(self):
        result = normalize_parcels(self._collin_raw(), "collin")
        assert isinstance(result, pd.DataFrame)

    def test_parcel_id_renamed(self):
        result = normalize_parcels(self._collin_raw(), "collin")
        assert "parcel_id" in result.columns
        assert result["parcel_id"].iloc[0] == "123456"

    def test_land_value_is_numeric(self):
        result = normalize_parcels(self._collin_raw(), "collin")
        assert result["land_value"].iloc[0] == 150000.0

    def test_ilr_computed(self):
        result = normalize_parcels(self._collin_raw(), "collin")
        assert "ilr" in result.columns
        assert result["ilr"].iloc[0] == pytest.approx(350000 / 150000)

    def test_age_computed(self):
        result = normalize_parcels(self._collin_raw(), "collin")
        assert "age" in result.columns
        assert result["age"].iloc[0] == pytest.approx(2024 - 1995)

    def test_address_normalized(self):
        result = normalize_parcels(self._collin_raw(), "collin")
        assert result["situs_address"].iloc[0] == "123 MAIN ST"

    def test_county_and_state_added(self):
        result = normalize_parcels(self._collin_raw(), "collin")
        assert result["county"].iloc[0] == "collin"
        assert result["state"].iloc[0] == "tx"

    def test_unknown_county_raises(self):
        with pytest.raises(ValueError, match="Unknown county"):
            normalize_parcels(pd.DataFrame(), "atlantis")

    def test_unmappable_columns_raises(self):
        """DataFrame with zero mappable columns raises ValueError."""
        df = pd.DataFrame([{"col_x": "1", "col_y": "2"}])
        with pytest.raises(ValueError, match="Could not map"):
            normalize_parcels(df, "collin")


# ---------------------------------------------------------------------------
# fetch_parcels dispatcher
# ---------------------------------------------------------------------------

class TestFetchParcels:
    def test_unknown_county_raises(self):
        with pytest.raises(ValueError, match="Unknown county"):
            fetch_parcels("atlantis", year=2024)

    def test_collin_routes_to_collin_function(self):
        with patch("geostack.cad.fetch_collin_parcels", return_value=pd.DataFrame()) as mock_fn:
            fetch_parcels("collin", year=2024)
        mock_fn.assert_called_once()

    def test_dallas_calls_socrata(self):
        with patch("geostack.cad.requests.get", return_value=_mock_resp(_socrata_rows())):
            result = fetch_parcels("dallas", year=2024)
        assert isinstance(result, pd.DataFrame)

    def test_cooke_routes_to_trueprodigy(self):
        with patch("geostack.cad.fetch_cooke_parcels_trueprodigy",
                   return_value=pd.DataFrame()) as mock_fn:
            fetch_parcels("cooke", year=2025)
        mock_fn.assert_called_once()
        assert mock_fn.call_args[1]["year"] == 2025

    def test_cooke_explicit_2024_passed_through(self):
        """No sentinel: an explicit year=2024 must reach trueprodigy as 2024, not None."""
        with patch("geostack.cad.fetch_cooke_parcels_trueprodigy",
                   return_value=pd.DataFrame()) as mock_fn:
            fetch_parcels("cooke", year=2024)
        assert mock_fn.call_args[1]["year"] == 2024

    def test_cooke_bare_call_passes_none(self):
        """Bare call means 'current year' — trueprodigy receives year=None."""
        with patch("geostack.cad.fetch_cooke_parcels_trueprodigy",
                   return_value=pd.DataFrame()) as mock_fn:
            fetch_parcels("cooke")
        assert mock_fn.call_args[1]["year"] is None

    def test_grayson_routes_to_grayson_pacs(self):
        with patch("geostack.cad.fetch_grayson_parcels",
                   return_value=pd.DataFrame()) as mock_fn:
            fetch_parcels("grayson", year=2025)
        mock_fn.assert_called_once()

    def test_case_insensitive(self):
        with patch("geostack.cad.fetch_collin_parcels", return_value=pd.DataFrame()) as mock_fn:
            fetch_parcels("COLLIN", year=2024)
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# True Prodigy shared CAD source
# ---------------------------------------------------------------------------

class TestTrueProdigy:
    def test_sources_registered(self):
        from geostack.cad import TRUEPRODIGY_SOURCES
        assert "cooke" in TRUEPRODIGY_SOURCES
        src = TRUEPRODIGY_SOURCES["cooke"]
        assert src["portal"].startswith("https://")
        assert "76240" in src["zips"]

    def test_normalize_maps_fields_and_coords(self):
        import pandas as pd
        from geostack.cad import normalize_trueprodigy
        raw = pd.DataFrame([{
            "pAccountID": 9871740, "pYear": "2026",
            "landValue": "470255", "improvementValue": "1142882",
            "appraisedValue": "1613137", "marketValue": "1613137",
            "name": "DOE JANE", "fullSitus": "100 NOWETA PL, LAKE KIOWA, TX",
            "city": "LAKE KIOWA", "zip": "76240", "propType": "R",
            "useCd": "A1", "legalAcreage": "0.5",
            "latitude": "33.5583587794", "longitude": "-97.0158099731",
            "geoID": "7800-001-0073R", "ignored": "x",
        }])
        out = normalize_trueprodigy(raw)
        assert out.iloc[0]["parcel_id"] == 9871740
        assert out.iloc[0]["total_value"] == 1613137.0   # coerced to numeric
        assert abs(out.iloc[0]["latitude"] - 33.5583587794) < 1e-6
        assert out.iloc[0]["situs_address"].startswith("100 NOWETA")
        assert "ignored" not in out.columns  # only mapped fields kept

    def test_normalize_handles_missing_columns(self):
        import pandas as pd
        from geostack.cad import normalize_trueprodigy
        # Only a subset present — should not raise
        out = normalize_trueprodigy(pd.DataFrame([{"pAccountID": 1, "appraisedValue": "100"}]))
        assert out.iloc[0]["total_value"] == 100.0
        assert "latitude" not in out.columns
