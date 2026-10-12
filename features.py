"""Windowed feature engineering for passive Wi-Fi management metadata."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import log2
from typing import Iterable

try:
    from .models import WiFiEvent
except ImportError:  # Direct execution from src/main.py
    from models import WiFiEvent


FRAME_TYPES = ("beacon", "deauth", "disassoc", "probe")
FEATURE_COLUMNS = (
    "event_count",
    "frame_rate",
    "beacon_count",
    "deauth_count",
    "disassoc_count",
    "probe_count",
    "other_frame_count",
    "beacon_ratio",
    "deauth_ratio",
    "disassoc_ratio",
    "probe_ratio",
    "unique_source_count",
    "unique_destination_count",
    "broadcast_destination_ratio",
    "ssid_count",
    "channel_count",
    "security_profile_count",
    "open_or_wep_ratio",
    "frame_type_entropy",
)


@dataclass(frozen=True)
class FeatureWindow:
    """Features and identifying metadata for one BSSID/time bucket."""

    bssid: str
    window_start: str
    values: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {"bssid": self.bssid, "window_start": self.window_start, **self.values}


def extract_feature_windows(
    events: Iterable[WiFiEvent], window_seconds: int = 30
) -> list[FeatureWindow]:
    """Aggregate normalized events into fixed, UTC-aligned BSSID windows.

    Windows are aligned to Unix time boundaries, so separate files and sensors
    produce comparable buckets. Events without a BSSID are retained under an
    explicit ``UNKNOWN`` key rather than silently discarded.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be greater than zero")

    buckets: dict[tuple[str, int], list[WiFiEvent]] = defaultdict(list)
    for event in events:
        epoch = int(event.timestamp.timestamp())
        bucket = epoch - epoch % window_seconds
        buckets[(event.bssid or "UNKNOWN", bucket)].append(event)

    windows = []
    for (bssid, bucket), rows in sorted(buckets.items(), key=lambda item: item[0][1]):
        counts = Counter(event.frame_type for event in rows)
        total = len(rows)
        ratios = {frame: counts[frame] / total for frame in FRAME_TYPES}
        entropy = -sum((count / total) * log2(count / total) for count in counts.values() if count > 0)
        channels = {event.channel for event in rows if event.channel is not None}
        ssids = {event.ssid for event in rows if event.ssid}
        security = {event.security.upper() for event in rows if event.security}
        sources = {event.source for event in rows if event.source}
        destinations = {event.destination for event in rows if event.destination}
        broadcast_count = sum(
            1 for event in rows
            if event.destination and event.destination.upper() in {
                "FF:FF:FF:FF:FF:FF", "BROADCAST"
            }
        )
        weak_count = sum(
            1 for event in rows
            if event.security and event.security.upper() in {"OPEN", "WEP"}
        )
        values = {
            "event_count": float(total),
            "frame_rate": total / window_seconds,
            "beacon_count": float(counts["beacon"]),
            "deauth_count": float(counts["deauth"]),
            "disassoc_count": float(counts["disassoc"]),
            "probe_count": float(counts["probe"]),
            "other_frame_count": float(total - sum(counts[name] for name in FRAME_TYPES)),
            "beacon_ratio": ratios["beacon"],
            "deauth_ratio": ratios["deauth"],
            "disassoc_ratio": ratios["disassoc"],
            "probe_ratio": ratios["probe"],
            "unique_source_count": float(len(sources)),
            "unique_destination_count": float(len(destinations)),
            "broadcast_destination_ratio": broadcast_count / total,
            "ssid_count": float(len(ssids)),
            "channel_count": float(len(channels)),
            "security_profile_count": float(len(security)),
            "open_or_wep_ratio": weak_count / total,
            "frame_type_entropy": entropy,
        }
        start = datetime.fromtimestamp(bucket, tz=timezone.utc).isoformat()
        windows.append(FeatureWindow(bssid=bssid, window_start=start, values=values))
    return windows
