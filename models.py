from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class WiFiEvent:
    timestamp: datetime
    frame_type: str
    bssid: str | None = None
    source: str | None = None
    destination: str | None = None
    ssid: str | None = None
    channel: int | None = None
    security: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WiFiEvent":
        raw_timestamp = payload["timestamp"]
        timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return cls(
            timestamp=timestamp,
            frame_type=str(payload["frame_type"]).lower(),
            bssid=_normalise_mac(payload.get("bssid")),
            source=_normalise_mac(payload.get("source")),
            destination=_normalise_mac(payload.get("destination")),
            ssid=_clean_text(payload.get("ssid")),
            channel=_optional_int(payload.get("channel")),
            security=_clean_text(payload.get("security")),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["timestamp"] = self.timestamp.isoformat()
        return value


@dataclass(frozen=True)
class Alert:
    alert_type: str
    severity: str
    timestamp: datetime
    message: str
    bssid: str | None = None
    ssid: str | None = None
    evidence_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["timestamp"] = self.timestamp.isoformat()
        return value


def _normalise_mac(value: Any) -> str | None:
    if not value:
        return None
    return str(value).upper()


def _clean_text(value: Any) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None
