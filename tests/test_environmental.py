"""Tests for geostack.environmental — EPA environmental screening."""

import pytest
from unittest.mock import patch, MagicMock


class TestEchoFacilitySearch:
    @patch("geostack.environmental.requests.get")
    def test_returns_list_of_facilities(self, mock_get):
        from geostack.environmental import echo_facility_search

        # Step 1: get_facilities returns a QueryID
        query_resp = MagicMock()
        query_resp.json.return_value = {
            "Results": {
                "QueryID": "abc123",
                "QueryRows": "2",
                "Message": "Success",
            }
        }
        query_resp.raise_for_status = MagicMock()

        # Step 2: get_qid returns facility data
        qid_resp = MagicMock()
        qid_resp.json.return_value = {
            "Results": {
                "Facilities": [
                    {
                        "FacName": "ACME CHEMICAL",
                        "FacLat": "32.78",
                        "FacLong": "-96.80",
                        "RegistryID": "110000123456",
                        "FacSNCFlg": "Y",
                        "FacQtrsWithNC": "4",
                        "RCRAComplianceStatus": "Violation Identified",
                    },
                    {
                        "FacName": "CLEAN WATER CO",
                        "FacLat": "32.79",
                        "FacLong": "-96.81",
                        "RegistryID": "110000789012",
                        "FacSNCFlg": "N",
                        "FacQtrsWithNC": "0",
                        "CWAComplianceStatus": "No Violation Identified",
                    },
                ]
            }
        }
        qid_resp.raise_for_status = MagicMock()
        mock_get.side_effect = [query_resp, qid_resp]

        results = echo_facility_search(lat=32.78, lon=-96.80, radius_miles=1)
        assert len(results) == 2
        assert results[0]["FacName"] == "ACME CHEMICAL"
        assert results[0]["FacSNCFlg"] == "Y"

        # Verify correct API calls
        calls = mock_get.call_args_list
        assert "get_facilities" in calls[0][0][0]
        assert "get_qid" in calls[1][0][0]

    @patch("geostack.environmental.requests.get")
    def test_returns_empty_for_no_results(self, mock_get):
        from geostack.environmental import echo_facility_search

        resp = MagicMock()
        resp.json.return_value = {
            "Results": {"QueryID": "abc123", "QueryRows": "0", "Message": "Success"}
        }
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        results = echo_facility_search(lat=32.0, lon=-96.0, radius_miles=1)
        assert results == []

    @patch("geostack.environmental.requests.get")
    def test_passes_media_filter(self, mock_get):
        from geostack.environmental import echo_facility_search

        resp = MagicMock()
        resp.json.return_value = {
            "Results": {"QueryID": "abc", "QueryRows": "0", "Message": "Success"}
        }
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        echo_facility_search(lat=32.78, lon=-96.80, media="R")
        call_params = mock_get.call_args_list[0][1]["params"]
        assert call_params["p_med"] == "R"


class TestCimcCleanupSites:
    @patch("geostack.environmental.requests.get")
    def test_returns_superfund_sites(self, mock_get):
        from geostack.environmental import cimc_cleanup_sites

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "features": [
                {
                    "attributes": {
                        "PRIMARY_NAME": "OLD CHEMICAL PLANT",
                        "EPA_ID": "TXD000123456",
                        "LATITUDE": 32.79,
                        "LONGITUDE": -96.81,
                        "SF_NPL_CODE": "F",
                        "CITY_NAME": "DALLAS",
                        "STATE_CODE": "TX",
                    }
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        results = cimc_cleanup_sites(lat=32.78, lon=-96.80, radius_miles=1)
        assert len(results) == 1
        assert results[0]["PRIMARY_NAME"] == "OLD CHEMICAL PLANT"
        assert results[0]["SF_NPL_CODE"] == "F"

    @patch("geostack.environmental.requests.get")
    def test_filters_by_program(self, mock_get):
        from geostack.environmental import cimc_cleanup_sites

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"features": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        cimc_cleanup_sites(lat=32.78, lon=-96.80, program="superfund")
        call_params = mock_get.call_args[1]["params"]
        assert "SF_NPL_CODE" in call_params["where"]

    @patch("geostack.environmental.requests.get")
    def test_passes_inSR_4326(self, mock_get):
        from geostack.environmental import cimc_cleanup_sites

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"features": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        cimc_cleanup_sites(lat=32.78, lon=-96.80)
        call_params = mock_get.call_args[1]["params"]
        assert call_params["inSR"] == "4326"

    @patch("geostack.environmental.requests.get")
    def test_empty_features_returns_empty_list(self, mock_get):
        from geostack.environmental import cimc_cleanup_sites

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"features": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        results = cimc_cleanup_sites(lat=32.78, lon=-96.80)
        assert results == []


class TestScreenSite:
    @patch("geostack.environmental.cimc_cleanup_sites")
    @patch("geostack.environmental.echo_facility_search")
    def test_returns_structured_report(self, mock_echo, mock_cimc):
        from geostack.environmental import screen_site

        mock_echo.return_value = [
            {"FacName": "ACME", "FacSNCFlg": "Y", "FacLat": "32.78", "FacLong": "-96.80"},
        ]
        mock_cimc.return_value = [
            {"PRIMARY_NAME": "NPL SITE", "SF_NPL_CODE": "F", "LATITUDE": 32.79, "LONGITUDE": -96.81},
        ]

        report = screen_site(lat=32.78, lon=-96.80, include_flood=False)

        assert "summary" in report
        assert "risk_flags" in report
        assert "echo_facilities" in report
        assert "cleanup_sites" in report
        assert "query" in report
        assert report["query"]["lat"] == 32.78
        assert report["summary"]["total_regulated_facilities"] == 1

    @patch("geostack.environmental.cimc_cleanup_sites")
    @patch("geostack.environmental.echo_facility_search")
    def test_flags_npl_site_as_high(self, mock_echo, mock_cimc):
        from geostack.environmental import screen_site

        mock_echo.return_value = []
        mock_cimc.return_value = [
            {"PRIMARY_NAME": "SUPERFUND", "SF_NPL_CODE": "F", "LATITUDE": 32.79, "LONGITUDE": -96.81},
        ]

        report = screen_site(lat=32.78, lon=-96.80, include_flood=False)
        high_flags = [f for f in report["risk_flags"] if f["severity"] == "HIGH"]
        assert len(high_flags) >= 1
        assert report["summary"]["risk_level"] == "HIGH"

    @patch("geostack.environmental.cimc_cleanup_sites")
    @patch("geostack.environmental.echo_facility_search")
    def test_clean_site_is_low_risk(self, mock_echo, mock_cimc):
        from geostack.environmental import screen_site

        mock_echo.return_value = []
        mock_cimc.return_value = []

        report = screen_site(lat=32.78, lon=-96.80, include_flood=False)
        assert report["summary"]["risk_level"] == "LOW"
        assert report["summary"]["total_regulated_facilities"] == 0

    @patch("geostack.environmental.cimc_cleanup_sites")
    @patch("geostack.environmental.echo_facility_search")
    def test_flags_snc_facility_as_high(self, mock_echo, mock_cimc):
        from geostack.environmental import screen_site

        mock_echo.return_value = [
            {"FacName": "BAD ACTOR", "FacSNCFlg": "Y", "FacLat": "32.781", "FacLong": "-96.801"},
        ]
        mock_cimc.return_value = []

        report = screen_site(lat=32.78, lon=-96.80, include_flood=False)
        high_flags = [f for f in report["risk_flags"] if f["severity"] == "HIGH"]
        assert len(high_flags) >= 1

    @patch("geostack.environmental.flood_zone")
    @patch("geostack.environmental.cimc_cleanup_sites")
    @patch("geostack.environmental.echo_facility_search")
    def test_includes_flood_zone(self, mock_echo, mock_cimc, mock_flood):
        from geostack.environmental import screen_site

        mock_echo.return_value = []
        mock_cimc.return_value = []
        mock_flood.return_value = {"zone": "AE", "sfha": True, "zone_subtype": "FLOODWAY", "unmapped": False}

        report = screen_site(lat=32.78, lon=-96.80, include_flood=True)
        assert report["flood_zone"]["zone"] == "AE"
        medium_flags = [f for f in report["risk_flags"] if f["severity"] == "MEDIUM"]
        assert any("flood" in f["flag"].lower() for f in medium_flags)


@pytest.mark.integration
class TestEnvironmentalIntegration:
    """Live tests against EPA APIs. Run with: pytest -m integration"""

    def test_echo_dallas_downtown(self):
        from geostack.environmental import echo_facility_search

        results = echo_facility_search(lat=32.78, lon=-96.80, radius_miles=1)
        assert len(results) > 0
        assert all("FacName" in f for f in results)
        assert all("RegistryID" in f for f in results)

    def test_cimc_dallas_downtown(self):
        from geostack.environmental import cimc_cleanup_sites

        results = cimc_cleanup_sites(lat=32.78, lon=-96.80, radius_miles=5)
        assert len(results) > 0
        assert all("PRIMARY_NAME" in s for s in results)

    def test_screen_site_dallas(self):
        from geostack.environmental import screen_site

        report = screen_site(lat=32.78, lon=-96.80, radius_miles=1)
        assert report["summary"]["total_regulated_facilities"] > 0
        assert report["summary"]["risk_level"] in ("HIGH", "MEDIUM", "LOW")
        assert "risk_flags" in report
