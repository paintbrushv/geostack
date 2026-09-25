"""Tests for geostack.risk — FEMA flood zone lookup."""

import pytest
from unittest.mock import patch, MagicMock


class TestFloodZone:
    @patch("geostack.risk.requests.get")
    def test_returns_flood_zone_dict(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "features": [{"attributes": {"FLD_ZONE": "AE", "ZONE_SUBTY": "FLOODWAY", "SFHA_TF": "T"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from geostack.risk import flood_zone
        result = flood_zone(lon=-95.373, lat=29.761)
        assert isinstance(result, dict)
        assert result["zone"] == "AE"
        assert result["sfha"] is True
        assert result["zone_subtype"] == "FLOODWAY"

    @patch("geostack.risk.requests.get")
    def test_zone_x_is_not_sfha(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "features": [{"attributes": {"FLD_ZONE": "X", "ZONE_SUBTY": "AREA OF MINIMAL FLOOD HAZARD", "SFHA_TF": "F"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from geostack.risk import flood_zone
        result = flood_zone(lon=-96.6183, lat=32.8028)
        assert result["zone"] == "X"
        assert result["sfha"] is False

    @patch("geostack.risk.requests.get")
    def test_no_features_returns_unmapped(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"features": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from geostack.risk import flood_zone
        result = flood_zone(lon=-101.0, lat=33.0)
        assert result["zone"] is None
        assert result["sfha"] is False
        assert result["unmapped"] is True

    @patch("geostack.risk.requests.get")
    def test_api_url_uses_correct_layer(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"features": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from geostack.risk import flood_zone
        flood_zone(lon=-96.0, lat=32.0)
        call_url = mock_get.call_args[0][0]
        assert "/28/query" in call_url
