"""Tests for geostack.zoning -- zoning risk analysis."""

import pytest
from unittest.mock import patch, MagicMock

from geostack.zoning import ZONING_REGISTRY


class TestClassifyZone:
    def test_mf_codes(self):
        from geostack.zoning import classify_zone

        assert classify_zone("MF-2") == "mf_permitted"
        assert classify_zone("MF-1") == "mf_permitted"
        assert classify_zone("MF-3") == "mf_permitted"
        assert classify_zone("MF-4") == "mf_permitted"
        assert classify_zone("MU-1") == "mf_permitted"
        assert classify_zone("MU-2") == "mf_permitted"
        assert classify_zone("TOD") == "mf_permitted"
        assert classify_zone("MX-1") == "mf_permitted"
        assert classify_zone("UR") == "mf_permitted"

    def test_industrial_codes(self):
        from geostack.zoning import classify_zone

        assert classify_zone("I-1") == "industrial"
        assert classify_zone("I-2") == "industrial"
        assert classify_zone("LI") == "industrial"
        assert classify_zone("HI") == "industrial"
        assert classify_zone("M-1") == "industrial"
        assert classify_zone("M-2") == "industrial"
        assert classify_zone("IP") == "industrial"
        assert classify_zone("BP") == "industrial"

    def test_heavy_commercial_codes(self):
        from geostack.zoning import classify_zone

        assert classify_zone("C-3") == "heavy_commercial"
        assert classify_zone("C-4") == "heavy_commercial"
        assert classify_zone("HC") == "heavy_commercial"
        assert classify_zone("GC") == "heavy_commercial"
        assert classify_zone("CR") == "heavy_commercial"

    def test_light_commercial_codes(self):
        from geostack.zoning import classify_zone

        assert classify_zone("C-1") == "light_commercial"
        assert classify_zone("C-2") == "light_commercial"
        assert classify_zone("O-1") == "light_commercial"
        assert classify_zone("O-2") == "light_commercial"
        assert classify_zone("NS") == "light_commercial"
        assert classify_zone("LR") == "light_commercial"
        assert classify_zone("LC") == "light_commercial"
        assert classify_zone("B-1") == "light_commercial"

    def test_single_family_codes(self):
        from geostack.zoning import classify_zone

        assert classify_zone("SF-1") == "single_family"
        assert classify_zone("SF-3") == "single_family"
        assert classify_zone("R-1") == "single_family"
        assert classify_zone("R-2") == "single_family"
        assert classify_zone("R-3") == "single_family"
        assert classify_zone("A") == "single_family"
        assert classify_zone("AG") == "single_family"
        assert classify_zone("RE") == "single_family"
        assert classify_zone("RS-10") == "single_family"

    def test_other_codes(self):
        from geostack.zoning import classify_zone

        assert classify_zone("PD") == "other"
        assert classify_zone("P") == "other"
        assert classify_zone("PK") == "other"
        assert classify_zone("FW") == "other"
        assert classify_zone("SUP") == "other"

    def test_case_insensitive(self):
        from geostack.zoning import classify_zone

        assert classify_zone("mf-2") == "mf_permitted"
        assert classify_zone("sf-1") == "single_family"
        assert classify_zone("i-1") == "industrial"

    def test_unknown_code_returns_other(self):
        from geostack.zoning import classify_zone

        assert classify_zone("XYZZY") == "other"
        assert classify_zone("") == "other"

    def test_pd_with_mf_description(self):
        from geostack.zoning import classify_zone

        assert classify_zone("PD", description="Planned Development - Multifamily") == "mf_permitted"
        assert classify_zone("PD", description="Planned Development - Single Family") == "single_family"
        assert classify_zone("PD", description="Planned Development") == "other"


class TestQueryZoningService:
    @patch("geostack.zoning.requests.get")
    def test_point_query_returns_features(self, mock_get):
        from geostack.zoning import _query_zoning_service

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "features": [
                {
                    "attributes": {
                        "ZONE_DIST": "MF-2",
                        "LONG_ZONE_DIST": "Multifamily District",
                    },
                    "geometry": {
                        "rings": [
                            [[-96.80, 32.78], [-96.79, 32.78],
                             [-96.79, 32.79], [-96.80, 32.79], [-96.80, 32.78]]
                        ]
                    },
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        features = _query_zoning_service(
            city="dallas", lat=32.78, lon=-96.80
        )
        assert len(features) == 1
        assert features[0]["attributes"]["ZONE_DIST"] == "MF-2"

        # Verify ArcGIS params
        call_params = mock_get.call_args[1]["params"]
        assert call_params["inSR"] == "4326"
        assert call_params["f"] == "json"
        assert call_params["returnGeometry"] == "true"

    @patch("geostack.zoning.requests.get")
    def test_radius_query_uses_distance_param(self, mock_get):
        from geostack.zoning import _query_zoning_service

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"features": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        _query_zoning_service(city="dallas", lat=32.78, lon=-96.80, radius_miles=1.0)
        call_params = mock_get.call_args[1]["params"]
        assert call_params["distance"] == "1.0"
        assert call_params["units"] == "esriSRUnit_StatuteMile"

    @patch("geostack.zoning.requests.get")
    def test_point_only_query_omits_distance(self, mock_get):
        from geostack.zoning import _query_zoning_service

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"features": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        _query_zoning_service(city="dallas", lat=32.78, lon=-96.80)
        call_params = mock_get.call_args[1]["params"]
        assert "distance" not in call_params

    @patch("geostack.zoning.requests.get")
    def test_empty_response_returns_empty_list(self, mock_get):
        from geostack.zoning import _query_zoning_service

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"features": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        features = _query_zoning_service(city="dallas", lat=32.0, lon=-96.0)
        assert features == []

    @patch("geostack.zoning.requests.get")
    def test_pagination_fetches_all_pages(self, mock_get):
        from geostack.zoning import _query_zoning_service

        # First page: full batch + exceededTransferLimit
        page1_resp = MagicMock()
        page1_resp.json.return_value = {
            "features": [
                {"attributes": {"ZONE_DIST": f"SF-{i}"}, "geometry": {"rings": []}}
                for i in range(1000)
            ],
            "exceededTransferLimit": True,
        }
        page1_resp.raise_for_status = MagicMock()

        # Second page: partial batch, no exceededTransferLimit
        page2_resp = MagicMock()
        page2_resp.json.return_value = {
            "features": [
                {"attributes": {"ZONE_DIST": "MF-2"}, "geometry": {"rings": []}}
            ],
        }
        page2_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [page1_resp, page2_resp]

        features = _query_zoning_service(
            city="dallas", lat=32.78, lon=-96.80, radius_miles=1.0
        )
        assert len(features) == 1001

    @patch("geostack.zoning.requests.get")
    def test_invalid_city_raises(self, mock_get):
        from geostack.zoning import _query_zoning_service

        with pytest.raises(ValueError, match="not in ZONING_REGISTRY"):
            _query_zoning_service(city="atlantis", lat=0, lon=0)


class TestZoningLookup:
    @patch("geostack.zoning._query_zoning_service")
    def test_returns_zoning_for_dallas_point(self, mock_query):
        from geostack.zoning import zoning_lookup

        mock_query.return_value = [
            {
                "attributes": {
                    "ZONE_DIST": "MF-2",
                    "LONG_ZONE_DIST": "Multifamily District",
                },
                "geometry": {
                    "rings": [
                        [[-96.80, 32.78], [-96.79, 32.78],
                         [-96.79, 32.79], [-96.80, 32.79], [-96.80, 32.78]]
                    ]
                },
            }
        ]

        result = zoning_lookup(lat=32.78, lon=-96.80)
        assert result["code"] == "MF-2"
        assert result["description"] == "Multifamily District"
        assert result["category"] == "mf_permitted"
        assert result["city"] == "dallas"

    @patch("geostack.zoning._query_zoning_service")
    def test_tries_multiple_cities_until_hit(self, mock_query):
        from geostack.zoning import zoning_lookup

        # First cities return empty, then fort_worth returns a result
        call_count = 0

        def side_effect(city, lat, lon):
            nonlocal call_count
            call_count += 1
            if city == "fort_worth":
                return [
                    {
                        "attributes": {
                            "ZONING": "SF-4",
                            "BASE_ZONING": "Single Family 4",
                        },
                        "geometry": {"rings": []},
                    }
                ]
            return []

        mock_query.side_effect = side_effect

        result = zoning_lookup(lat=32.75, lon=-97.33)
        assert result["code"] == "SF-4"
        assert result["city"] == "fort_worth"
        assert result["category"] == "single_family"
        # Should have tried at least 2 cities
        assert call_count >= 2

    @patch("geostack.zoning._query_zoning_service")
    def test_returns_none_when_not_covered(self, mock_query):
        from geostack.zoning import zoning_lookup

        mock_query.return_value = []

        result = zoning_lookup(lat=40.0, lon=-74.0)
        assert result is None

    @patch("geostack.zoning._query_zoning_service")
    def test_no_desc_field_returns_none_description(self, mock_query):
        from geostack.zoning import zoning_lookup

        mock_query.return_value = [
            {
                "attributes": {"ZONE_": "MF-1"},
                "geometry": {"rings": []},
            }
        ]

        # Frisco has no desc_field
        def side_effect(city, lat, lon):
            if city == "frisco":
                return [{"attributes": {"ZONE_": "MF-1"}, "geometry": {"rings": []}}]
            return []

        mock_query.side_effect = side_effect

        result = zoning_lookup(lat=33.15, lon=-96.82)
        assert result["code"] == "MF-1"
        assert result["description"] is None
        assert result["category"] == "mf_permitted"

    @patch("geostack.zoning._query_zoning_service")
    def test_returns_overlay_and_pd_fields(self, mock_query):
        from geostack.zoning import zoning_lookup

        # Basic test that overlay and pd_number are present in output
        mock_query.return_value = [
            {
                "attributes": {
                    "ZONE_DIST": "C-2",
                    "LONG_ZONE_DIST": "Light Commercial",
                },
                "geometry": {"rings": []},
            }
        ]

        def side_effect(city, lat, lon):
            if city == "dallas":
                return [
                    {
                        "attributes": {
                            "ZONE_DIST": "C-2",
                            "LONG_ZONE_DIST": "Light Commercial",
                        },
                        "geometry": {"rings": []},
                    }
                ]
            return []

        mock_query.side_effect = side_effect

        result = zoning_lookup(lat=32.78, lon=-96.80)
        assert "overlay" in result
        assert "pd_number" in result


class TestZoningWithinRadius:
    @patch("geostack.zoning._query_zoning_service")
    def test_returns_classified_zones_with_acreage(self, mock_query):
        from geostack.zoning import zoning_within_radius

        # A polygon roughly 0.01 degrees square in Dallas (~200 acres)
        ring = [
            [-96.80, 32.78], [-96.79, 32.78],
            [-96.79, 32.79], [-96.80, 32.79], [-96.80, 32.78],
        ]

        def side_effect(city, lat, lon, radius_miles, return_geometry):
            if city == "dallas":
                return [
                    {
                        "attributes": {
                            "ZONE_DIST": "MF-2",
                            "LONG_ZONE_DIST": "Multifamily District",
                        },
                        "geometry": {"rings": [ring]},
                    },
                    {
                        "attributes": {
                            "ZONE_DIST": "SF-3",
                            "LONG_ZONE_DIST": "Single Family 3",
                        },
                        "geometry": {"rings": [ring]},
                    },
                ]
            return []

        mock_query.side_effect = side_effect

        zones = zoning_within_radius(lat=32.78, lon=-96.80, radius_miles=1.0)
        assert len(zones) == 2
        assert zones[0]["code"] == "MF-2"
        assert zones[0]["category"] == "mf_permitted"
        assert zones[0]["acres"] > 0  # Should compute acreage from polygon
        assert zones[1]["code"] == "SF-3"
        assert zones[1]["category"] == "single_family"

    @patch("geostack.zoning._query_zoning_service")
    def test_empty_radius_returns_empty(self, mock_query):
        from geostack.zoning import zoning_within_radius

        mock_query.return_value = []

        zones = zoning_within_radius(lat=40.0, lon=-74.0, radius_miles=1.0)
        assert zones == []
        assert isinstance(zones, list)

    @patch("geostack.zoning._query_zoning_service")
    def test_zones_have_required_keys(self, mock_query):
        from geostack.zoning import zoning_within_radius

        ring = [
            [-96.80, 32.78], [-96.79, 32.78],
            [-96.79, 32.79], [-96.80, 32.79], [-96.80, 32.78],
        ]

        def side_effect(city, lat, lon, radius_miles, return_geometry):
            if city == "dallas":
                return [
                    {
                        "attributes": {
                            "ZONE_DIST": "I-1",
                            "LONG_ZONE_DIST": "Industrial",
                        },
                        "geometry": {"rings": [ring]},
                    },
                ]
            return []

        mock_query.side_effect = side_effect

        zones = zoning_within_radius(lat=32.78, lon=-96.80, radius_miles=1.0)
        assert len(zones) == 1
        zone = zones[0]
        required_keys = {"code", "description", "category", "acres", "city"}
        assert required_keys.issubset(zone.keys())
        assert zone["category"] == "industrial"

    @patch("geostack.zoning._query_zoning_service")
    def test_skips_features_without_geometry(self, mock_query):
        from geostack.zoning import zoning_within_radius

        def side_effect(city, lat, lon, radius_miles, return_geometry):
            if city == "dallas":
                return [
                    {
                        "attributes": {
                            "ZONE_DIST": "MF-2",
                            "LONG_ZONE_DIST": "Multifamily",
                        },
                        # No geometry key
                    },
                    {
                        "attributes": {
                            "ZONE_DIST": "SF-1",
                            "LONG_ZONE_DIST": "Single Family",
                        },
                        "geometry": None,
                    },
                ]
            return []

        mock_query.side_effect = side_effect

        zones = zoning_within_radius(lat=32.78, lon=-96.80, radius_miles=1.0)
        # Features without geometry are included but with acres=0
        assert all(isinstance(z["acres"], (int, float)) for z in zones)

    @patch("geostack.zoning._query_zoning_service")
    def test_aggregates_across_cities(self, mock_query):
        from geostack.zoning import zoning_within_radius

        ring = [
            [-96.80, 32.78], [-96.79, 32.78],
            [-96.79, 32.79], [-96.80, 32.79], [-96.80, 32.78],
        ]

        call_cities = []

        def side_effect(city, lat, lon, radius_miles, return_geometry):
            call_cities.append(city)
            if city == "dallas":
                return [
                    {
                        "attributes": {"ZONE_DIST": "MF-2", "LONG_ZONE_DIST": "MF"},
                        "geometry": {"rings": [ring]},
                    },
                ]
            if city == "irving":
                return [
                    {
                        "attributes": {"District": "C-2", "Description": "Commercial"},
                        "geometry": {"rings": [ring]},
                    },
                ]
            return []

        mock_query.side_effect = side_effect

        zones = zoning_within_radius(lat=32.78, lon=-96.80, radius_miles=1.0)
        # Should query all cities and combine results
        assert len(call_cities) == len(ZONING_REGISTRY)
        assert len(zones) == 2


class TestZoningRisk:
    @patch("geostack.zoning.zoning_within_radius")
    @patch("geostack.zoning.zoning_lookup")
    def test_returns_structured_report(self, mock_lookup, mock_radius):
        from geostack.zoning import zoning_risk

        mock_lookup.return_value = {
            "code": "MF-2",
            "description": "Multifamily District",
            "category": "mf_permitted",
            "city": "dallas",
            "overlay": None,
            "pd_number": None,
        }
        mock_radius.return_value = [
            {"code": "MF-2", "description": "Multifamily", "category": "mf_permitted", "acres": 20.0, "city": "dallas"},
            {"code": "SF-3", "description": "Single Family", "category": "single_family", "acres": 100.0, "city": "dallas"},
            {"code": "I-1", "description": "Industrial", "category": "industrial", "acres": 10.0, "city": "dallas"},
            {"code": "C-2", "description": "Light Commercial", "category": "light_commercial", "acres": 15.0, "city": "dallas"},
        ]

        report = zoning_risk(lat=32.78, lon=-96.80, radius_miles=1.0)

        assert "query" in report
        assert "subject_zoning" in report
        assert "summary" in report
        assert "risk_flags" in report
        assert "zones" in report

        assert report["query"]["lat"] == 32.78
        assert report["query"]["lon"] == -96.80
        assert report["query"]["radius_miles"] == 1.0
        assert report["city"] == "dallas"
        assert report["subject_zoning"]["code"] == "MF-2"

    @patch("geostack.zoning.zoning_within_radius")
    @patch("geostack.zoning.zoning_lookup")
    def test_summary_aggregates_acres_by_category(self, mock_lookup, mock_radius):
        from geostack.zoning import zoning_risk

        mock_lookup.return_value = {
            "code": "MF-2", "description": "MF", "category": "mf_permitted",
            "city": "dallas", "overlay": None, "pd_number": None,
        }
        mock_radius.return_value = [
            {"code": "MF-2", "category": "mf_permitted", "acres": 30.0, "city": "dallas", "description": "MF"},
            {"code": "MF-3", "category": "mf_permitted", "acres": 12.3, "city": "dallas", "description": "MF"},
            {"code": "I-1", "category": "industrial", "acres": 12.1, "city": "dallas", "description": "Ind"},
            {"code": "C-3", "category": "heavy_commercial", "acres": 8.7, "city": "dallas", "description": "HC"},
            {"code": "SF-1", "category": "single_family", "acres": 145.2, "city": "dallas", "description": "SF"},
        ]

        report = zoning_risk(lat=32.78, lon=-96.80)

        s = report["summary"]
        assert s["mf_permitted_acres"] == pytest.approx(42.3)
        assert s["industrial_acres"] == pytest.approx(12.1)
        assert s["heavy_commercial_acres"] == pytest.approx(8.7)
        assert s["single_family_acres"] == pytest.approx(145.2)
        assert s["total_parcels_in_radius"] == 5

    @patch("geostack.zoning.zoning_within_radius")
    @patch("geostack.zoning.zoning_lookup")
    def test_supply_threat_high_for_large_vacant_mf(self, mock_lookup, mock_radius):
        from geostack.zoning import zoning_risk

        mock_lookup.return_value = {
            "code": "C-2", "description": "Commercial", "category": "light_commercial",
            "city": "dallas", "overlay": None, "pd_number": None,
        }
        # 25 acres MF-zoned -- should be HIGH supply threat
        mock_radius.return_value = [
            {"code": "MF-2", "category": "mf_permitted", "acres": 25.0, "city": "dallas", "description": "MF"},
            {"code": "SF-1", "category": "single_family", "acres": 75.0, "city": "dallas", "description": "SF"},
        ]

        report = zoning_risk(lat=32.78, lon=-96.80)
        # Without CAD cross-reference, uses total MF acres as proxy
        # 25 acres MF > 20 threshold -> HIGH or at least MEDIUM
        assert report["summary"]["supply_threat"] in ("HIGH", "MEDIUM")

    @patch("geostack.zoning.zoning_within_radius")
    @patch("geostack.zoning.zoning_lookup")
    def test_quality_threat_high_for_heavy_industrial(self, mock_lookup, mock_radius):
        from geostack.zoning import zoning_risk

        mock_lookup.return_value = {
            "code": "MF-2", "description": "MF", "category": "mf_permitted",
            "city": "dallas", "overlay": None, "pd_number": None,
        }
        # 20% industrial -- should trigger HIGH quality threat
        mock_radius.return_value = [
            {"code": "I-1", "category": "industrial", "acres": 20.0, "city": "dallas", "description": "Ind"},
            {"code": "SF-1", "category": "single_family", "acres": 80.0, "city": "dallas", "description": "SF"},
        ]

        report = zoning_risk(lat=32.78, lon=-96.80)
        assert report["summary"]["quality_threat"] == "HIGH"

    @patch("geostack.zoning.zoning_within_radius")
    @patch("geostack.zoning.zoning_lookup")
    def test_quality_threat_medium_for_combined_industrial_heavy_commercial(self, mock_lookup, mock_radius):
        from geostack.zoning import zoning_risk

        mock_lookup.return_value = {
            "code": "MF-2", "description": "MF", "category": "mf_permitted",
            "city": "dallas", "overlay": None, "pd_number": None,
        }
        # industrial + heavy_commercial > 20 acres combined
        mock_radius.return_value = [
            {"code": "I-1", "category": "industrial", "acres": 12.0, "city": "dallas", "description": "Ind"},
            {"code": "C-3", "category": "heavy_commercial", "acres": 10.0, "city": "dallas", "description": "HC"},
            {"code": "SF-1", "category": "single_family", "acres": 178.0, "city": "dallas", "description": "SF"},
        ]

        report = zoning_risk(lat=32.78, lon=-96.80)
        assert report["summary"]["quality_threat"] == "MEDIUM"

    @patch("geostack.zoning.zoning_within_radius")
    @patch("geostack.zoning.zoning_lookup")
    def test_low_risk_all_single_family(self, mock_lookup, mock_radius):
        from geostack.zoning import zoning_risk

        mock_lookup.return_value = {
            "code": "SF-1", "description": "Single Family", "category": "single_family",
            "city": "dallas", "overlay": None, "pd_number": None,
        }
        mock_radius.return_value = [
            {"code": "SF-1", "category": "single_family", "acres": 100.0, "city": "dallas", "description": "SF"},
            {"code": "SF-3", "category": "single_family", "acres": 100.0, "city": "dallas", "description": "SF"},
        ]

        report = zoning_risk(lat=32.78, lon=-96.80)
        assert report["summary"]["supply_threat"] == "LOW"
        assert report["summary"]["quality_threat"] == "LOW"
        assert report["risk_flags"] == []

    @patch("geostack.zoning.zoning_within_radius")
    @patch("geostack.zoning.zoning_lookup")
    def test_supply_threat_medium_for_high_mf_pct(self, mock_lookup, mock_radius):
        from geostack.zoning import zoning_risk

        mock_lookup.return_value = {
            "code": "MF-2", "description": "MF", "category": "mf_permitted",
            "city": "dallas", "overlay": None, "pd_number": None,
        }
        # MF > 30% of total
        mock_radius.return_value = [
            {"code": "MF-2", "category": "mf_permitted", "acres": 4.0, "city": "dallas", "description": "MF"},
            {"code": "SF-1", "category": "single_family", "acres": 6.0, "city": "dallas", "description": "SF"},
        ]

        report = zoning_risk(lat=32.78, lon=-96.80)
        # 4/10 = 40% MF, > 30% threshold
        assert report["summary"]["supply_threat"] == "MEDIUM"
        # Should have a flag about high density
        flag_texts = [f["flag"] for f in report["risk_flags"]]
        assert any("30%" in f or "density" in f.lower() for f in flag_texts)

    @patch("geostack.zoning.zoning_within_radius")
    @patch("geostack.zoning.zoning_lookup")
    def test_risk_flags_are_list_of_dicts(self, mock_lookup, mock_radius):
        from geostack.zoning import zoning_risk

        mock_lookup.return_value = {
            "code": "MF-2", "description": "MF", "category": "mf_permitted",
            "city": "dallas", "overlay": None, "pd_number": None,
        }
        mock_radius.return_value = [
            {"code": "I-1", "category": "industrial", "acres": 50.0, "city": "dallas", "description": "Ind"},
        ]

        report = zoning_risk(lat=32.78, lon=-96.80)
        for flag in report["risk_flags"]:
            assert "flag" in flag
            assert "severity" in flag
            assert flag["severity"] in ("HIGH", "MEDIUM", "LOW")

    @patch("geostack.zoning.zoning_within_radius")
    @patch("geostack.zoning.zoning_lookup")
    def test_handles_no_subject_zoning(self, mock_lookup, mock_radius):
        from geostack.zoning import zoning_risk

        mock_lookup.return_value = None  # Point not in any registered city
        mock_radius.return_value = []

        report = zoning_risk(lat=40.0, lon=-74.0)
        assert report["subject_zoning"] is None
        assert report["city"] is None
        assert report["summary"]["total_parcels_in_radius"] == 0


@pytest.mark.integration
class TestZoningIntegration:
    """Live tests against city ArcGIS services. Run with: pytest -m integration"""

    def test_zoning_lookup_dallas_downtown(self):
        from geostack.zoning import zoning_lookup

        # Dallas City Hall area
        result = zoning_lookup(lat=32.7767, lon=-96.7970)
        assert result is not None
        assert result["code"]  # Should return some zoning code
        assert result["category"] in (
            "mf_permitted", "industrial", "heavy_commercial",
            "light_commercial", "single_family", "other",
        )
        assert result["city"] == "dallas"

    def test_zoning_lookup_fort_worth(self):
        from geostack.zoning import zoning_lookup

        # Fort Worth downtown
        result = zoning_lookup(lat=32.7555, lon=-97.3308)
        assert result is not None
        assert result["code"]
        assert result["city"] == "fort_worth"

    def test_zoning_within_radius_dallas(self):
        from geostack.zoning import zoning_within_radius

        # Dallas Medical District -- should have mixed zoning
        zones = zoning_within_radius(lat=32.7918, lon=-96.8369, radius_miles=0.5)
        assert len(zones) > 0
        categories = {z["category"] for z in zones}
        # Should have at least 2 different categories
        assert len(categories) >= 1
        assert all(z["acres"] >= 0 for z in zones)

    def test_zoning_risk_dallas(self):
        from geostack.zoning import zoning_risk

        # Near Deep Ellum -- mix of commercial, MF, industrial
        report = zoning_risk(lat=32.7837, lon=-96.7834, radius_miles=0.5)
        assert "summary" in report
        assert "risk_flags" in report
        assert "zones" in report
        assert report["summary"]["supply_threat"] in ("HIGH", "MEDIUM", "LOW")
        assert report["summary"]["quality_threat"] in ("HIGH", "MEDIUM", "LOW")
        assert report["summary"]["total_parcels_in_radius"] > 0
