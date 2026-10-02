"""Tests for external connectors (Open-Meteo, GDELT, AISstream.io) and
the multi-source investigation orchestrator. Mocked HTTP/WebSocket
throughout -- this sandbox cannot reach the real external APIs
(confirmed 403 from each), so these tests verify the connectors handle
real, documented response shapes correctly and degrade gracefully on
failure, without needing live network access.
"""
import asyncio
import json
from unittest.mock import MagicMock, patch

import httpx

from logisense import models as m  # noqa: F401
from logisense.config import Settings
from logisense.models import ProviderStatus
from logisense.services.external.evidence_item import EvidenceItem
from logisense.services.external.gdelt_connector import GDELTConnector
from logisense.services.external.maritime_connector import AISStreamConnector
from logisense.services.external.weather_connector import OpenMeteoConnector


def _settings(aisstream_api_key: str | None = None) -> Settings:
    return Settings(database_url="sqlite:///:memory:", gemini_api_key="x", aisstream_api_key=aisstream_api_key)


def _mock_response(json_data, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("error", request=MagicMock(), response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# EvidenceItem
# ---------------------------------------------------------------------------

def test_evidence_item_citation_text_formats_per_source_type():
    db_item = EvidenceItem(source_type="internal_db", source_name="purchase_orders", source_id="PO-04370", content="x")
    assert db_item.citation_text == "[DB: purchase_orders.PO-04370]"

    doc_item = EvidenceItem(source_type="internal_document", source_name="supplier_contracts", source_id="SUP-014_contract.md", content="x")
    assert doc_item.citation_text == "[DOC: supplier_contracts/SUP-014_contract.md]"

    weather_item = EvidenceItem(source_type="weather_api", source_name="Open-Meteo", source_id="Berlin", content="x", location="Berlin, Germany", timestamp="2026-01-01T12:00")
    assert weather_item.citation_text == "[EXTERNAL: Open-Meteo — Berlin, Germany — 2026-01-01T12:00]"

    gdelt_item = EvidenceItem(source_type="gdelt", source_name="GDELT", source_id="url", content="x", title="Factory Strike", timestamp="20260101")
    assert gdelt_item.citation_text == "[EXTERNAL: GDELT — Factory Strike — 20260101]"


def test_evidence_item_ranking_priority_favors_internal_db_over_external():
    db_item = EvidenceItem(source_type="internal_db", source_name="x", source_id="x", content="x")
    doc_item = EvidenceItem(source_type="internal_document", source_name="x", source_id="x", content="x")
    external_item = EvidenceItem(source_type="weather_api", source_name="x", source_id="x", content="x")
    assert db_item.ranking_priority < doc_item.ranking_priority < external_item.ranking_priority


# ---------------------------------------------------------------------------
# Open-Meteo connector
# ---------------------------------------------------------------------------

def test_open_meteo_returns_evidence_on_success():
    geocode_response = _mock_response({"results": [{"name": "Berlin", "country": "Germany", "latitude": 52.5, "longitude": 13.4}]})
    forecast_response = _mock_response({"current": {"time": "2026-01-01T12:00", "temperature_2m": 5.2, "precipitation": 0.0, "wind_speed_10m": 12.3, "weather_code": 2}})

    with patch("httpx.get", side_effect=[geocode_response, forecast_response]):
        connector = OpenMeteoConnector()
        result = connector.get_weather_for_location("Berlin")

    assert result.status == ProviderStatus.AVAILABLE
    assert len(result.items) == 1
    assert result.items[0].source_type == "weather_api"
    assert "partly cloudy" in result.items[0].content
    assert "Berlin" in result.items[0].location


def test_open_meteo_handles_unresolvable_location_without_crashing():
    geocode_response = _mock_response({"results": []})
    with patch("httpx.get", return_value=geocode_response):
        connector = OpenMeteoConnector()
        result = connector.get_weather_for_location("Nonexistent Place Xyzzy")

    assert result.status == ProviderStatus.NO_EVIDENCE_FOUND
    assert result.items == []


def test_open_meteo_handles_timeout_without_crashing():
    with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
        connector = OpenMeteoConnector()
        result = connector.get_weather_for_location("Berlin")

    assert result.status == ProviderStatus.ERROR
    assert result.items == []
    assert "timed out" in result.error_message.lower()


def test_open_meteo_handles_http_error_without_crashing():
    error_response = _mock_response({}, status_code=500)
    with patch("httpx.get", return_value=error_response):
        connector = OpenMeteoConnector()
        result = connector.get_weather_for_location("Berlin")

    assert result.status == ProviderStatus.ERROR
    assert result.items == []


def test_open_meteo_handles_malformed_response_without_crashing():
    bad_response = MagicMock()
    bad_response.raise_for_status.return_value = None
    bad_response.json.side_effect = ValueError("not JSON")
    with patch("httpx.get", return_value=bad_response):
        connector = OpenMeteoConnector()
        result = connector.get_weather_for_location("Berlin")

    assert result.status == ProviderStatus.ERROR


# ---------------------------------------------------------------------------
# GDELT connector
# ---------------------------------------------------------------------------

def test_gdelt_filters_out_articles_not_actually_mentioning_the_entity():
    response = _mock_response({"articles": [
        {"title": "Meridian Supply Co. announces factory expansion", "url": "https://a.example/1", "seendate": "20260101", "domain": "a.example"},
        {"title": "Unrelated shipping news about a different company", "url": "https://a.example/2", "seendate": "20260102", "domain": "a.example"},
    ]})
    with patch("httpx.get", return_value=response):
        connector = GDELTConnector()
        result = connector.search_events("Meridian Supply Co.")

    assert result.status == ProviderStatus.AVAILABLE
    assert len(result.items) == 1
    assert "Meridian" in result.items[0].content


def test_gdelt_handles_empty_results_without_crashing():
    response = _mock_response({"articles": []})
    with patch("httpx.get", return_value=response):
        connector = GDELTConnector()
        result = connector.search_events("Some Supplier")

    assert result.status == ProviderStatus.NO_EVIDENCE_FOUND
    assert result.items == []


def test_gdelt_handles_rate_limit_without_crashing():
    error_response = _mock_response({}, status_code=429)
    with patch("httpx.get", return_value=error_response):
        connector = GDELTConnector()
        result = connector.search_events("Some Supplier")

    assert result.status == ProviderStatus.ERROR
    assert result.items == []


# ---------------------------------------------------------------------------
# AISstream.io connector -- graceful missing-key behavior + real WebSocket
# protocol logic (subscribe message, bounded listen window, error handling)
# ---------------------------------------------------------------------------

class _FakeAISWebSocket:
    """Stands in for a real websockets connection: records what was
    sent, and yields queued messages from recv() before hanging
    (simulating a live stream with no more traffic) so the connector's
    own listen_duration_seconds timeout is what actually ends the test,
    not the fake running out of data."""

    def __init__(self, messages_to_yield):
        self.sent_messages: list[str] = []
        self._queue = list(messages_to_yield)

    async def send(self, data: str) -> None:
        self.sent_messages.append(data)

    async def recv(self) -> str:
        if self._queue:
            return self._queue.pop(0)
        await asyncio.Event().wait()  # never resolves -- the caller's wait_for() timeout cuts this off
        raise AssertionError("unreachable")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def test_aisstream_marks_unavailable_when_key_missing_and_does_not_crash():
    connector = AISStreamConnector(_settings(aisstream_api_key=None))
    result = connector.get_vessel_activity_near("Port of Rotterdam")

    assert result.status == ProviderStatus.UNAVAILABLE
    assert result.items == []
    assert "not configured" in result.error_message.lower()


def test_aisstream_sends_correct_subscription_message_and_collects_position_reports():
    geocode_response = _mock_response({"results": [{"name": "Rotterdam", "country": "Netherlands", "latitude": 51.9, "longitude": 4.5}]})
    position_message = json.dumps({
        "MessageType": "PositionReport",
        "MetaData": {"MMSI": 123456789, "ShipName": "Ever Given", "latitude": 51.9, "longitude": 4.5, "time_utc": "2026-01-01 08:00:00 UTC"},
        "Message": {"PositionReport": {"Sog": 12.5, "Cog": 90.0}},
    })
    fake_ws = _FakeAISWebSocket([position_message])

    with patch("httpx.get", return_value=geocode_response), \
         patch("websockets.connect", return_value=fake_ws):
        connector = AISStreamConnector(_settings(aisstream_api_key="real-key"), listen_duration_seconds=0.05)
        result = connector.get_vessel_activity_near("Rotterdam")

    assert result.status == ProviderStatus.AVAILABLE
    assert len(result.items) == 1
    assert result.items[0].source_type == "maritime_api"
    assert "Ever Given" in result.items[0].content

    # Verify the actual subscription message sent matches AISstream's
    # documented format -- APIKey + geographic BoundingBoxes, key in the
    # message body (not an HTTP header, since there is no HTTP request here).
    assert len(fake_ws.sent_messages) == 1
    sent = json.loads(fake_ws.sent_messages[0])
    assert sent["APIKey"] == "real-key"
    assert "BoundingBoxes" in sent
    assert isinstance(sent["BoundingBoxes"], list)


def test_aisstream_returns_no_evidence_found_when_no_messages_arrive():
    geocode_response = _mock_response({"results": [{"name": "Rotterdam", "country": "Netherlands", "latitude": 51.9, "longitude": 4.5}]})
    fake_ws = _FakeAISWebSocket([])  # nothing ever arrives during the listen window

    with patch("httpx.get", return_value=geocode_response), \
         patch("websockets.connect", return_value=fake_ws):
        connector = AISStreamConnector(_settings(aisstream_api_key="real-key"), listen_duration_seconds=0.05)
        result = connector.get_vessel_activity_near("Rotterdam")

    assert result.status == ProviderStatus.NO_EVIDENCE_FOUND
    assert result.items == []


def test_aisstream_error_message_in_stream_is_handled_gracefully_not_crashed():
    """AISstream reports errors (invalid key, throttling, etc) as a
    message with an "Error" field on the same socket, not an HTTP
    status -- this must degrade gracefully like any other failure."""
    geocode_response = _mock_response({"results": [{"name": "Rotterdam", "country": "Netherlands", "latitude": 51.9, "longitude": 4.5}]})
    error_message = json.dumps({"Error": "invalid API key"})
    fake_ws = _FakeAISWebSocket([error_message])

    with patch("httpx.get", return_value=geocode_response), \
         patch("websockets.connect", return_value=fake_ws):
        connector = AISStreamConnector(_settings(aisstream_api_key="wrong-key"), listen_duration_seconds=0.05)
        result = connector.get_vessel_activity_near("Rotterdam")

    assert result.status == ProviderStatus.ERROR
    assert result.items == []
    assert "invalid api key" in result.error_message.lower()


def test_aisstream_handles_unresolvable_location_without_crashing():
    geocode_response = _mock_response({"results": []})
    with patch("httpx.get", return_value=geocode_response):
        connector = AISStreamConnector(_settings(aisstream_api_key="real-key"))
        result = connector.get_vessel_activity_near("Nonexistent Place Xyzzy")

    assert result.status == ProviderStatus.NO_EVIDENCE_FOUND
    assert result.items == []


def test_aisstream_to_evidence_item_handles_ship_static_data():
    msg = {
        "MessageType": "ShipStaticData",
        "MetaData": {"MMSI": 987654321, "ShipName": "Maersk Example", "time_utc": "2026-01-01 09:00:00 UTC"},
        "Message": {"ShipStaticData": {"Destination": "ROTTERDAM", "Eta": {"Month": 1, "Day": 15}}},
    }
    item = AISStreamConnector._to_evidence_item(msg, "Rotterdam, Netherlands")

    assert item is not None
    assert item.source_type == "maritime_api"
    assert "Maersk Example" in item.content
    assert "ROTTERDAM" in item.content


def test_aisstream_to_evidence_item_returns_none_for_unhandled_message_types():
    msg = {"MessageType": "SomeOtherAISMessageType", "MetaData": {}}
    assert AISStreamConnector._to_evidence_item(msg, "Rotterdam") is None

