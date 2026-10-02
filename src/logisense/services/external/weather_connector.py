"""Open-Meteo weather connector. Free, non-commercial API -- no API key
required. Used to check whether weather conditions at a supplier's
location or a shipment's route are a plausible contributing factor to a
delay.

Real Open-Meteo endpoints (documented, public):
    Geocoding: https://geocoding-api.open-meteo.com/v1/search
    Forecast/current: https://api.open-meteo.com/v1/forecast

NOTE: this sandbox's network egress cannot reach api.open-meteo.com or
geocoding-api.open-meteo.com (confirmed via direct curl -- both return
403 through this environment's proxy, the same restriction already
documented for huggingface.co and generativelanguage.googleapis.com
elsewhere in this project). This connector is implemented against the
real, documented API contract and unit-tested against mocked HTTP
responses shaped like Open-Meteo's actual JSON -- it has not been
exercised against the live API from within this sandbox. Run it for
real in an environment with normal internet access to confirm.
"""
from __future__ import annotations

import httpx

from logisense.models import ProviderStatus
from logisense.services.external.evidence_item import EvidenceItem

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WEATHER_CODE_DESCRIPTIONS = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow fall", 73: "moderate snow fall", 75: "heavy snow fall",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


class OpenMeteoConnectorResult:
    def __init__(self, status: ProviderStatus, items: list[EvidenceItem], error_message: str | None = None) -> None:
        self.status = status
        self.items = items
        self.error_message = error_message


class OpenMeteoConnector:
    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def get_weather_for_location(self, location_name: str) -> OpenMeteoConnectorResult:
        """location_name should be a real place name (city, or
        "city, country") -- derived from the supplier's own address in
        the internal database, never asked of the user when it's
        already known internally."""
        try:
            lat, lon, resolved_name = self._geocode(location_name)
        except _ConnectorHTTPError as exc:
            return OpenMeteoConnectorResult(ProviderStatus.ERROR, [], f"Open-Meteo geocoding failed: {exc}")
        except _ConnectorTimeoutError as exc:
            return OpenMeteoConnectorResult(ProviderStatus.ERROR, [], f"Open-Meteo geocoding timed out: {exc}")

        if lat is None:
            return OpenMeteoConnectorResult(ProviderStatus.NO_EVIDENCE_FOUND, [], f"Could not resolve location '{location_name}'.")

        try:
            weather = self._fetch_current_weather(lat, lon)
        except _ConnectorHTTPError as exc:
            return OpenMeteoConnectorResult(ProviderStatus.ERROR, [], f"Open-Meteo forecast request failed: {exc}")
        except _ConnectorTimeoutError as exc:
            return OpenMeteoConnectorResult(ProviderStatus.ERROR, [], f"Open-Meteo forecast request timed out: {exc}")

        if weather is None:
            return OpenMeteoConnectorResult(ProviderStatus.NO_EVIDENCE_FOUND, [], "Open-Meteo returned no current weather data.")

        code = weather.get("weather_code")
        description = _WEATHER_CODE_DESCRIPTIONS.get(code, f"weather code {code}")
        content = (
            f"Current conditions at {resolved_name}: {description}, "
            f"temperature {weather.get('temperature_2m')}°C, "
            f"precipitation {weather.get('precipitation')}mm, "
            f"wind speed {weather.get('wind_speed_10m')} km/h."
        )
        item = EvidenceItem(
            source_type="weather_api", source_name="Open-Meteo", source_id=resolved_name,
            content=content, location=resolved_name, timestamp=weather.get("time"),
            confidence=1.0, metadata={"raw": weather},
        )
        return OpenMeteoConnectorResult(ProviderStatus.AVAILABLE, [item])

    def _geocode(self, location_name: str) -> tuple[float | None, float | None, str | None]:
        response = self._get(_GEOCODING_URL, params={"name": location_name, "count": 1})
        results = response.get("results")
        if not results:
            return None, None, None
        top = results[0]
        resolved_name = ", ".join(filter(None, [top.get("name"), top.get("country")]))
        return top.get("latitude"), top.get("longitude"), resolved_name

    def _fetch_current_weather(self, lat: float, lon: float) -> dict | None:
        response = self._get(
            _FORECAST_URL,
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,precipitation,wind_speed_10m,weather_code",
            },
        )
        return response.get("current")

    def _get(self, url: str, params: dict) -> dict:
        try:
            resp = httpx.get(url, params=params, timeout=self.timeout_seconds)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException as exc:
            raise _ConnectorTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise _ConnectorHTTPError(str(exc)) from exc
        except ValueError as exc:  # malformed/non-JSON response
            raise _ConnectorHTTPError(f"malformed response: {exc}") from exc


class _ConnectorHTTPError(Exception):
    pass


class _ConnectorTimeoutError(Exception):
    pass
