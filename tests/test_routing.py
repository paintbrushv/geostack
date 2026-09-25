"""Tests for geostack.routing — drive_time, drive_time_matrix, isochrone."""

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest

from geostack.routing import drive_time, drive_time_matrix, isochrone


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _route_resp(duration=900.0, distance=12000.0, code="Ok"):
    return MagicMock(
        json=MagicMock(return_value={
            "code": code,
            "routes": [{"duration": duration, "distance": distance}],
        }),
        raise_for_status=MagicMock(),
    )


def _table_resp(durations, code="Ok"):
    return MagicMock(
        json=MagicMock(return_value={"code": code, "durations": durations}),
        raise_for_status=MagicMock(),
    )


# ---------------------------------------------------------------------------
# drive_time
# ---------------------------------------------------------------------------

class TestDriveTime:
    def test_returns_dict_with_expected_keys(self):
        with patch("geostack.routing.requests.get", return_value=_route_resp()):
            result = drive_time((-96.8, 33.0), (-96.7, 33.1))
        assert set(result.keys()) == {
            "duration_seconds", "duration_minutes", "distance_meters", "distance_miles"
        }

    def test_duration_seconds_correct(self):
        with patch("geostack.routing.requests.get", return_value=_route_resp(duration=600.0)):
            result = drive_time((-96.8, 33.0), (-96.7, 33.1))
        assert result["duration_seconds"] == 600.0

    def test_duration_minutes_rounded(self):
        with patch("geostack.routing.requests.get", return_value=_route_resp(duration=601.0)):
            result = drive_time((-96.8, 33.0), (-96.7, 33.1))
        assert result["duration_minutes"] == round(601.0 / 60, 1)

    def test_distance_meters_correct(self):
        with patch("geostack.routing.requests.get", return_value=_route_resp(distance=8000.0)):
            result = drive_time((-96.8, 33.0), (-96.7, 33.1))
        assert result["distance_meters"] == 8000.0

    def test_distance_miles_rounded(self):
        with patch("geostack.routing.requests.get", return_value=_route_resp(distance=1609.34)):
            result = drive_time((-96.8, 33.0), (-96.7, 33.1))
        assert result["distance_miles"] == 1.0

    def test_error_code_raises(self):
        with patch("geostack.routing.requests.get", return_value=_route_resp(code="NoRoute")):
            with pytest.raises(ValueError, match="OSRM error"):
                drive_time((-96.8, 33.0), (-96.7, 33.1))

    def test_calls_osrm_route_endpoint(self):
        with patch("geostack.routing.requests.get", return_value=_route_resp()) as mock_get:
            drive_time((-96.8, 33.0), (-96.7, 33.1))
        url = mock_get.call_args[0][0]
        assert "/route/v1/driving/" in url

    def test_coordinates_in_url(self):
        with patch("geostack.routing.requests.get", return_value=_route_resp()) as mock_get:
            drive_time((-96.8, 33.0), (-96.7, 33.1))
        url = mock_get.call_args[0][0]
        assert "-96.8,33.0" in url
        assert "-96.7,33.1" in url

    def test_raise_for_status_called(self):
        mock_resp = _route_resp()
        with patch("geostack.routing.requests.get", return_value=mock_resp):
            drive_time((-96.8, 33.0), (-96.7, 33.1))
        mock_resp.raise_for_status.assert_called_once()


# ---------------------------------------------------------------------------
# drive_time_matrix
# ---------------------------------------------------------------------------

class TestDriveTimeMatrix:
    def _durations(self, n):
        """NxN matrix of synthetic durations in seconds."""
        return [[float(i * 60 + j * 30) for j in range(n)] for i in range(n)]

    def test_returns_list_of_lists(self):
        origins = [(-96.8, 33.0), (-96.7, 33.1)]
        with patch("geostack.routing.requests.get", return_value=_table_resp(self._durations(2))):
            result = drive_time_matrix(origins)
        assert isinstance(result, list)
        assert all(isinstance(row, list) for row in result)

    def test_values_converted_to_minutes(self):
        durations = [[600.0, 900.0], [900.0, 0.0]]
        origins = [(-96.8, 33.0), (-96.7, 33.1)]
        with patch("geostack.routing.requests.get", return_value=_table_resp(durations)):
            result = drive_time_matrix(origins)
        assert result[0][0] == 10.0   # 600s = 10 min
        assert result[0][1] == 15.0   # 900s = 15 min

    def test_none_for_null_durations(self):
        durations = [[600.0, None], [None, 0.0]]
        origins = [(-96.8, 33.0), (-96.7, 33.1)]
        with patch("geostack.routing.requests.get", return_value=_table_resp(durations)):
            result = drive_time_matrix(origins)
        assert result[0][1] is None
        assert result[1][0] is None

    def test_error_code_raises(self):
        origins = [(-96.8, 33.0)]
        with patch("geostack.routing.requests.get", return_value=_table_resp([[0.0]], code="Error")):
            with pytest.raises(ValueError, match="OSRM error"):
                drive_time_matrix(origins)

    def test_calls_table_endpoint(self):
        origins = [(-96.8, 33.0), (-96.7, 33.1)]
        with patch("geostack.routing.requests.get", return_value=_table_resp(self._durations(2))) as mock_get:
            drive_time_matrix(origins)
        url = mock_get.call_args[0][0]
        assert "/table/v1/driving/" in url

    def test_with_separate_destinations(self):
        origins = [(-96.8, 33.0)]
        destinations = [(-96.7, 33.1), (-96.6, 33.2)]
        durations = [[300.0, 600.0]]
        with patch("geostack.routing.requests.get", return_value=_table_resp(durations)) as mock_get:
            result = drive_time_matrix(origins, destinations)
        # destinations param should be in the query params
        params = mock_get.call_args[1]["params"]
        assert "destinations" in params
        assert result[0][0] == 5.0
        assert result[0][1] == 10.0


# ---------------------------------------------------------------------------
# isochrone
# ---------------------------------------------------------------------------

class TestIsochrone:
    def _make_route_resp(self, duration=600.0):
        """Mock a successful OSRM route response."""
        return MagicMock(
            json=MagicMock(return_value={
                "code": "Ok",
                "routes": [{"duration": duration, "distance": 5000.0}],
            }),
            raise_for_status=MagicMock(),
        )

    def test_returns_geodataframe(self):
        # Patch drive_time to always succeed within budget
        with patch("geostack.routing.drive_time", return_value={
            "duration_minutes": 10.0, "duration_seconds": 600.0,
            "distance_meters": 5000.0, "distance_miles": 3.1,
        }):
            result = isochrone(-96.77, 32.78, minutes=15)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_has_minutes_column(self):
        with patch("geostack.routing.drive_time", return_value={
            "duration_minutes": 10.0, "duration_seconds": 600.0,
            "distance_meters": 5000.0, "distance_miles": 3.1,
        }):
            result = isochrone(-96.77, 32.78, minutes=15)
        assert "minutes" in result.columns
        assert result["minutes"].iloc[0] == 15

    def test_crs_is_4326(self):
        with patch("geostack.routing.drive_time", return_value={
            "duration_minutes": 10.0, "duration_seconds": 600.0,
            "distance_meters": 5000.0, "distance_miles": 3.1,
        }):
            result = isochrone(-96.77, 32.78, minutes=15)
        assert result.crs.to_epsg() == 4326

    def test_too_few_points_raises(self):
        """If all routes raise ValueError we get too few points for a convex hull."""
        with patch("geostack.routing.drive_time", side_effect=ValueError("no route")):
            with pytest.raises(ValueError, match="too few reachable"):
                isochrone(-96.77, 32.78, minutes=15)

    @pytest.mark.integration
    def test_real_osrm_public_server(self):
        """Smoke-test against the public OSRM demo server. Requires internet."""
        result = isochrone(-96.77, 32.78, minutes=5)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) >= 1
