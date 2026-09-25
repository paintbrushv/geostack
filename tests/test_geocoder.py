"""Tests for geostack.geocoder — Census Batch Geocoder."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestParseResponse:
    def test_parses_matched_row(self):
        from geostack.geocoder import _parse_response

        raw = (
            '"1","1325 TENNIS DR, Bedford, TX, 76021","Match","Exact",'
            '"1325 TENNIS DR, BEDFORD, TX, 76021","-97.1234,32.8456",'
            '"12345","L","48","439","010200","1001"\n'
        )
        df = _parse_response(raw)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["id"] == "1"
        assert row["match_type"] == "Match"
        assert abs(row["latitude"] - 32.8456) < 0.0001
        assert abs(row["longitude"] - (-97.1234)) < 0.0001
        assert row["tract_geoid"] == "48439010200"
        assert "BEDFORD" in row["matched_address"]

    def test_parses_no_match_row(self):
        from geostack.geocoder import _parse_response

        raw = '"2","999 FAKE ST, Nowhere, TX, 00000","No_Match"\n'
        df = _parse_response(raw)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["id"] == "2"
        assert row["match_type"] == "No_Match"
        assert pd.isna(row["latitude"])
        assert pd.isna(row["longitude"])
        assert pd.isna(row["tract_geoid"])

    def test_parses_multiple_rows(self):
        from geostack.geocoder import _parse_response

        raw = (
            '"1","1325 TENNIS DR, Bedford, TX, 76021","Match","Exact",'
            '"1325 TENNIS DR, BEDFORD, TX, 76021","-97.1234,32.8456",'
            '"12345","L","48","439","010200","1001"\n'
            '"2","999 FAKE ST, Nowhere, TX, 00000","No_Match"\n'
        )
        df = _parse_response(raw)
        assert len(df) == 2
        assert df.iloc[0]["match_type"] == "Match"
        assert df.iloc[1]["match_type"] == "No_Match"

    def test_empty_response(self):
        from geostack.geocoder import _parse_response

        df = _parse_response("")
        assert len(df) == 0


class TestPrepareBatchCsv:
    def test_formats_records_as_csv(self):
        from geostack.geocoder import _prepare_batch_csv

        records = [
            ("1", "1325 TENNIS DR", "Bedford", "TX", "76021"),
            ("2", "815 W ABRAM ST", "Arlington", "TX", "76013"),
        ]
        csv_text = _prepare_batch_csv(records)
        lines = csv_text.strip().split("\n")
        assert len(lines) == 2
        assert "1,1325 TENNIS DR,Bedford,TX,76021" == lines[0]
        assert "2,815 W ABRAM ST,Arlington,TX,76013" == lines[1]

    def test_handles_empty_city_and_zip(self):
        from geostack.geocoder import _prepare_batch_csv

        records = [("1", "1325 TENNIS DR", "", "TX", "")]
        csv_text = _prepare_batch_csv(records)
        assert "1,1325 TENNIS DR,,TX," == csv_text.strip()

    def test_handles_none_city_and_zip(self):
        from geostack.geocoder import _prepare_batch_csv

        records = [("1", "1325 TENNIS DR", None, "TX", None)]
        csv_text = _prepare_batch_csv(records)
        assert "1,1325 TENNIS DR,,TX," == csv_text.strip()


class TestSubmitBatch:
    @patch("geostack.geocoder.requests.post")
    def test_posts_to_census_api(self, mock_post):
        from geostack.geocoder import _submit_batch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            '"1","1325 TENNIS DR, Bedford, TX, 76021","Match","Exact",'
            '"1325 TENNIS DR, BEDFORD, TX, 76021","-97.1234,32.8456",'
            '"12345","L","48","439","010200","1001"\n'
        )
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        csv_text = "1,1325 TENNIS DR,Bedford,TX,76021\n"
        result_text = _submit_batch(csv_text)
        assert result_text == mock_resp.text

        call_kwargs = mock_post.call_args
        assert "geocoding.geo.census.gov" in call_kwargs[0][0]
        assert call_kwargs[1]["data"]["benchmark"] == "Public_AR_Current"
        assert call_kwargs[1]["data"]["vintage"] == "Current_Current"

    @patch("geostack.geocoder.requests.post")
    @patch("geostack.geocoder.time.sleep")
    def test_retries_on_500(self, mock_sleep, mock_post):
        from geostack.geocoder import _submit_batch

        fail_resp = MagicMock()
        fail_resp.status_code = 500
        fail_resp.raise_for_status.side_effect = Exception("Server Error")

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.text = '"1","addr","No_Match"\n'
        ok_resp.raise_for_status = MagicMock()

        mock_post.side_effect = [fail_resp, ok_resp]

        result = _submit_batch("1,123 MAIN ST,Dallas,TX,75201\n")
        assert result == ok_resp.text
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(5)


class TestCache:
    def test_cache_key_is_deterministic(self):
        from geostack.geocoder import _cache_key

        records_a = [("1", "123 MAIN", "Dallas", "TX", "75201")]
        records_b = [("1", "123 MAIN", "Dallas", "TX", "75201")]
        assert _cache_key(records_a) == _cache_key(records_b)

    def test_cache_key_differs_for_different_input(self):
        from geostack.geocoder import _cache_key

        records_a = [("1", "123 MAIN", "Dallas", "TX", "75201")]
        records_b = [("1", "456 OAK", "Dallas", "TX", "75201")]
        assert _cache_key(records_a) != _cache_key(records_b)

    def test_save_and_load_cache(self, tmp_path):
        from geostack.geocoder import _save_cache, _load_cache

        df = pd.DataFrame({
            "id": ["1"], "latitude": [32.84], "longitude": [-97.12],
            "tract_geoid": ["48439010200"], "match_type": ["Match"],
            "matched_address": ["1325 TENNIS DR, BEDFORD, TX, 76021"],
        })
        key = "abc123"
        _save_cache(df, key, cache_dir=tmp_path)
        loaded = _load_cache(key, cache_dir=tmp_path)
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded.iloc[0]["tract_geoid"] == "48439010200"

    def test_load_cache_returns_none_on_miss(self, tmp_path):
        from geostack.geocoder import _load_cache

        result = _load_cache("nonexistent", cache_dir=tmp_path)
        assert result is None


class TestGeocodeAddresses:
    @patch("geostack.geocoder._submit_batch")
    def test_geocodes_list_of_records(self, mock_submit, tmp_path):
        from geostack.geocoder import geocode_addresses

        mock_submit.return_value = (
            '"1","1325 TENNIS DR, Bedford, TX, 76021","Match","Exact",'
            '"1325 TENNIS DR, BEDFORD, TX, 76021","-97.1234,32.8456",'
            '"12345","L","48","439","010200","1001"\n'
            '"2","999 FAKE ST, Nowhere, TX, 00000","No_Match"\n'
        )

        records = [
            ("1", "1325 TENNIS DR", "Bedford", "TX", "76021"),
            ("2", "999 FAKE ST", "Nowhere", "TX", "00000"),
        ]
        df = geocode_addresses(records, batch_size=1000, cache_dir=tmp_path)
        assert len(df) == 2
        assert df.loc[df["id"] == "1", "tract_geoid"].iloc[0] == "48439010200"
        assert pd.isna(df.loc[df["id"] == "2", "tract_geoid"].iloc[0])

    @patch("geostack.geocoder._submit_batch")
    @patch("geostack.geocoder.time.sleep")
    def test_chunks_into_batches(self, mock_sleep, mock_submit, tmp_path):
        from geostack.geocoder import geocode_addresses

        mock_submit.return_value = '"1","addr","No_Match"\n'

        records = [
            (str(i), f"{i} MAIN ST", "Dallas", "TX", "75201")
            for i in range(5)
        ]
        geocode_addresses(records, batch_size=2, cache_dir=tmp_path)
        # 5 records / batch_size 2 = 3 batches
        assert mock_submit.call_count == 3

    @patch("geostack.geocoder._submit_batch")
    def test_uses_cache_on_second_call(self, mock_submit, tmp_path):
        from geostack.geocoder import geocode_addresses

        mock_submit.return_value = '"1","addr","No_Match"\n'

        records = [("1", "123 MAIN ST", "Dallas", "TX", "75201")]
        geocode_addresses(records, batch_size=1000, cache_dir=tmp_path)
        geocode_addresses(records, batch_size=1000, cache_dir=tmp_path)
        # Second call should hit cache, not API
        assert mock_submit.call_count == 1


class TestGeocodeBatch:
    @patch("geostack.geocoder._submit_batch")
    def test_geocodes_dataframe(self, mock_submit, tmp_path):
        from geostack.geocoder import geocode_batch

        mock_submit.return_value = (
            '"0","1325 TENNIS DR, Bedford, TX, 76021","Match","Exact",'
            '"1325 TENNIS DR, BEDFORD, TX, 76021","-97.1234,32.8456",'
            '"12345","L","48","439","010200","1001"\n'
        )

        df = pd.DataFrame({
            "street": ["1325 TENNIS DR"],
            "city": ["Bedford"],
            "zip": ["76021"],
        })
        result = geocode_batch(
            df, street_col="street", city_col="city", zip_col="zip",
            state="TX", cache_dir=tmp_path,
        )
        assert "latitude" in result.columns
        assert "tract_geoid" in result.columns
        assert len(result) == 1
        assert result.iloc[0]["tract_geoid"] == "48439010200"
        # Original columns preserved
        assert result.iloc[0]["street"] == "1325 TENNIS DR"

    @patch("geostack.geocoder._submit_batch")
    def test_uses_id_col(self, mock_submit, tmp_path):
        from geostack.geocoder import geocode_batch

        mock_submit.return_value = '"PARCEL001","addr","No_Match"\n'

        df = pd.DataFrame({
            "parcel_id": ["PARCEL001"],
            "street": ["123 MAIN ST"],
        })
        result = geocode_batch(
            df, street_col="street", id_col="parcel_id",
            state="TX", cache_dir=tmp_path,
        )
        assert len(result) == 1

    @patch("geostack.geocoder._submit_batch")
    def test_handles_no_city_no_zip(self, mock_submit, tmp_path):
        from geostack.geocoder import geocode_batch

        mock_submit.return_value = '"0","addr","No_Match"\n'

        df = pd.DataFrame({"street": ["123 MAIN ST"]})
        result = geocode_batch(
            df, street_col="street", state="TX", cache_dir=tmp_path,
        )
        assert len(result) == 1
        assert "latitude" in result.columns


@pytest.mark.integration
class TestCensusApiIntegration:
    """Live tests against Census Batch Geocoder. Run with: pytest -m integration"""

    def test_known_dfw_addresses(self, tmp_path):
        from geostack.geocoder import geocode_addresses

        records = [
            ("1", "1325 TENNIS DR", "Bedford", "TX", "76021"),
            ("2", "815 W ABRAM ST", "Arlington", "TX", "76013"),
            ("3", "999 DEFINITELY FAKE ST", "Nowhere", "TX", "00000"),
        ]
        df = geocode_addresses(records, cache_dir=tmp_path)
        assert len(df) == 3

        matched = df[df["match_type"] == "Match"]
        assert len(matched) >= 2  # at least the 2 real addresses

        row1 = df[df["id"] == "1"].iloc[0]
        assert row1["tract_geoid"].startswith("48")  # Texas
        assert abs(row1["latitude"] - 32.84) < 0.1  # roughly Bedford TX

        fake = df[df["id"] == "3"].iloc[0]
        assert fake["match_type"] == "No_Match"
