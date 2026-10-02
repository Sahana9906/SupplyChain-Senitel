"""AISstream.io connector -- real-time AIS (vessel tracking) data.
Free, but requires a free API key from aisstream.io (registration only,
no payment). This connector must NEVER crash the application when the
key is missing -- it marks itself unavailable and the caller continues
with whatever other sources ARE available.

Real, verified API contract (confirmed via aisstream.io's own
documentation, not guessed):

  - WebSocket-only, no REST endpoint: wss://stream.aisstream.io/v0/stream
  - Within 3 seconds of connecting, send ONE JSON subscription message:
        {"APIKey": "...", "BoundingBoxes": [[[lat1,lon1],[lat2,lon2]]],
         "FilterMessageTypes": [...]}   (geographic, not port-name-based --
         there is no "give me port X's calls" query in this API)
  - The server then pushes a continuous stream of AIS messages until the
    socket is closed -- this is a live feed, not a request/response
    query, so retrieval here means "listen for a bounded window and
    collect whatever comes in", not "fetch a fixed answer".
  - Errors (invalid key, throttling, etc) arrive as a message with an
    "Error" field on the same socket, not an HTTP status code.

Location is resolved to coordinates via Open-Meteo's free geocoding API
(already used by weather_connector.py) since AISstream itself has no
place-name lookup.

NOTE: this sandbox cannot reach aisstream.io or geocoding-api.open-meteo.com
(same network restriction documented elsewhere in this project). The
WebSocket protocol logic below is tested against a mocked
websockets.connect() matching the documented message shapes exactly;
it has not been exercised against the live service.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import websockets

from logisense.config import Settings
from logisense.models import ProviderStatus
from logisense.services.external.evidence_item import EvidenceItem

_AISSTREAM_WS_URL = "wss://stream.aisstream.io/v0/stream"
_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


class AISStreamConnectorResult:
    def __init__(self, status: ProviderStatus, items: list[EvidenceItem], error_message: str | None = None) -> None:
        self.status = status
        self.items = items
        self.error_message = error_message


class AISStreamConnector:
    def __init__(self, settings: Settings, listen_duration_seconds: float = 8.0, connect_timeout_seconds: float = 10.0) -> None:
        self.api_key = settings.aisstream_api_key
        self.listen_duration_seconds = listen_duration_seconds
        self.connect_timeout_seconds = connect_timeout_seconds

    def get_vessel_activity_near(self, location_name: str) -> AISStreamConnectorResult:
        """location_name should be a real place name (city, port, or
        "city, country") -- derived from the supplier/shipment's own
        location in the internal database, never asked of the user when
        it's already known internally."""
        if not self.api_key:
            return AISStreamConnectorResult(
                ProviderStatus.UNAVAILABLE, [],
                "AISSTREAM_API_KEY is not configured -- maritime evidence was not retrieved for this analysis.",
            )

        try:
            lat, lon, resolved_name = self._geocode(location_name)
        except httpx.HTTPError as exc:
            return AISStreamConnectorResult(ProviderStatus.ERROR, [], f"Could not resolve location for maritime lookup: {exc}")

        if lat is None:
            return AISStreamConnectorResult(ProviderStatus.NO_EVIDENCE_FOUND, [], f"Could not resolve location '{location_name}' to coordinates.")

        bounding_box = self._bounding_box(lat, lon)

        try:
            raw_messages = asyncio.run(self._listen(bounding_box))
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any websocket/protocol failure must degrade gracefully, never crash the caller
            return AISStreamConnectorResult(ProviderStatus.ERROR, [], f"AISstream connection failed: {exc}")

        if not raw_messages:
            return AISStreamConnectorResult(
                ProviderStatus.NO_EVIDENCE_FOUND, [],
                f"No AIS activity observed near {resolved_name} during the {self.listen_duration_seconds}s listening window.",
            )

        items = [item for item in (self._to_evidence_item(msg, resolved_name) for msg in raw_messages) if item is not None]
        if not items:
            return AISStreamConnectorResult(ProviderStatus.NO_EVIDENCE_FOUND, [], "AIS messages were received but none contained usable vessel data.")

        return AISStreamConnectorResult(ProviderStatus.AVAILABLE, items)

    def _geocode(self, location_name: str) -> tuple[float | None, float | None, str | None]:
        resp = httpx.get(_GEOCODING_URL, params={"name": location_name, "count": 1}, timeout=self.connect_timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results")
        if not results:
            return None, None, None
        top = results[0]
        resolved_name = ", ".join(filter(None, [top.get("name"), top.get("country")]))
        return top.get("latitude"), top.get("longitude"), resolved_name

    @staticmethod
    def _bounding_box(lat: float, lon: float, margin_degrees: float = 1.0) -> list[list[float]]:
        return [[lat - margin_degrees, lon - margin_degrees], [lat + margin_degrees, lon + margin_degrees]]

    async def _listen(self, bounding_box: list[list[float]]) -> list[dict]:
        """Connects, subscribes, and collects messages for
        listen_duration_seconds -- this is a live stream, so "retrieval"
        here means listening for a bounded window, not fetching a fixed
        answer. An "Error" field on any received message (AISstream's
        documented error-reporting mechanism -- invalid key, throttling,
        etc) aborts the listen with a RuntimeError, caught by the caller."""
        messages: list[dict] = []
        async with websockets.connect(_AISSTREAM_WS_URL, open_timeout=self.connect_timeout_seconds) as ws:
            subscribe_message = {
                "APIKey": self.api_key, "BoundingBoxes": [bounding_box],
                "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
            }
            await ws.send(json.dumps(subscribe_message))

            loop = asyncio.get_event_loop()
            deadline = loop.time() + self.listen_duration_seconds
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                except websockets.exceptions.ConnectionClosed:
                    break

                try:
                    parsed = json.loads(raw)
                except ValueError:
                    continue  # one malformed frame shouldn't abort the whole listen

                if parsed.get("Error"):
                    raise RuntimeError(f"AISstream error: {parsed['Error']}")
                messages.append(parsed)

        return messages

    @staticmethod
    def _to_evidence_item(msg: dict, location_name: str) -> EvidenceItem | None:
        message_type = msg.get("MessageType")
        meta = msg.get("MetaData", {})
        ship_name = (meta.get("ShipName") or "unknown vessel").strip()
        mmsi = meta.get("MMSI")
        timestamp = meta.get("time_utc")

        if message_type == "PositionReport":
            report = msg.get("Message", {}).get("PositionReport", {})
            content = (
                f"Vessel {ship_name} (MMSI {mmsi}) reported position near {location_name}: "
                f"lat {meta.get('latitude')}, lon {meta.get('longitude')}, "
                f"speed {report.get('Sog')} knots, course {report.get('Cog')}."
            )
        elif message_type == "ShipStaticData":
            static = msg.get("Message", {}).get("ShipStaticData", {})
            content = (
                f"Vessel {ship_name} (MMSI {mmsi}) near {location_name}: "
                f"destination {static.get('Destination')}, ETA {static.get('Eta')}."
            )
        else:
            return None

        return EvidenceItem(
            source_type="maritime_api", source_name="AISstream.io", source_id=str(mmsi or ship_name),
            content=content, entity=ship_name, location=location_name, timestamp=timestamp,
            confidence=0.8, metadata={"message_type": message_type, "mmsi": mmsi},
        )
