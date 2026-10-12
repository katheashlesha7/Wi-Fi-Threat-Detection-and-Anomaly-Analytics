from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import timedelta

try:  # Package execution: py -m src.main
    from .config import DetectorConfig
    from .models import Alert, WiFiEvent
except ImportError:  # Direct execution/import from the src directory
    from config import DetectorConfig
    from models import Alert, WiFiEvent


class ThreatDetector:
    """Stateful, passive detector for normalized Wi-Fi management events."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.events: deque[WiFiEvent] = deque()
        self._rate_events: dict[tuple[str, str], deque] = defaultdict(deque)
        self.alerts: list[Alert] = []
        self._alert_fingerprints: set[tuple[str, str | None, str | None]] = set()

    def process(self, event: WiFiEvent) -> list[Alert]:
        self.events.append(event)
        if event.bssid and event.frame_type in {"deauth", "disassoc", "beacon", "probe"}:
            self._rate_events[(event.frame_type, event.bssid)].append(event.timestamp)
        self._expire_old_events(event)
        generated = []
        generated.extend(self._rate_alerts(event))
        generated.extend(self._identity_alerts(event))
        new_alerts = []
        for alert in generated:
            fingerprint = (alert.alert_type, alert.bssid, alert.ssid)
            if fingerprint not in self._alert_fingerprints:
                self._alert_fingerprints.add(fingerprint)
                self.alerts.append(alert)
                new_alerts.append(alert)
        return new_alerts

    def _expire_old_events(self, newest: WiFiEvent) -> None:
        cutoff = newest.timestamp - timedelta(seconds=self.config.window_seconds)
        while self.events and self.events[0].timestamp < cutoff:
            expired = self.events.popleft()
            key = (expired.frame_type, expired.bssid) if expired.bssid else None
            if key and key in self._rate_events:
                timestamps = self._rate_events[key]
                if timestamps and timestamps[0] == expired.timestamp:
                    timestamps.popleft()
                if not timestamps:
                    del self._rate_events[key]

    def _rate_alerts(self, event: WiFiEvent) -> list[Alert]:
        specs = {
            "deauth": ("DEAUTH_FLOOD", self.config.deauth_threshold, "high"),
            "disassoc": ("DISASSOC_FLOOD", self.config.disassoc_threshold, "high"),
            "beacon": ("BEACON_FLOOD", self.config.beacon_threshold, "medium"),
            "probe": ("PROBE_FLOOD", self.config.probe_threshold, "medium"),
        }
        if event.frame_type not in specs or not event.bssid:
            return []
        alert_type, threshold, severity = specs[event.frame_type]
        count = len(self._rate_events[(event.frame_type, event.bssid)])
        if count < threshold:
            return []
        return [Alert(
            alert_type=alert_type,
            severity=severity,
            timestamp=event.timestamp,
            message=f"{count} {event.frame_type} frames from {event.bssid} in {self.config.window_seconds}s.",
            bssid=event.bssid,
            ssid=event.ssid,
            evidence_count=count,
        )]

    def _identity_alerts(self, event: WiFiEvent) -> list[Alert]:
        if event.frame_type != "beacon" or not event.ssid:
            return []
        trusted = self.config.trusted_aps.get(event.ssid)
        alerts: list[Alert] = []
        security = (event.security or "UNKNOWN").upper()
        # Unknown means the input adapter could not determine the capability;
        # it is not evidence that the AP is actually open.
        if security in {"OPEN", "WEP"}:
            alerts.append(Alert("WEAK_SECURITY", "medium", event.timestamp,
                f"{event.ssid} advertises {security} security.", event.bssid, event.ssid))
        if not trusted:
            return alerts
        known = next((item for item in trusted if item.get("bssid", "").upper() == event.bssid), None)
        if known is None:
            alerts.append(Alert("ROGUE_BSSID", "high", event.timestamp,
                f"Untrusted BSSID {event.bssid} advertises protected SSID {event.ssid}.", event.bssid, event.ssid))
            return alerts
        valid_channels = set(known.get("channels", []))
        valid_security = {value.upper() for value in known.get("security", [])}
        channel_mismatch = bool(valid_channels and event.channel is not None and event.channel not in valid_channels)
        security_mismatch = bool(valid_security and security != "UNKNOWN" and security not in valid_security)
        if channel_mismatch or security_mismatch:
            alerts.append(Alert("EVIL_TWIN_SUSPECT", "high", event.timestamp,
                f"{event.ssid} profile differs from its trusted baseline (channel/security).", event.bssid, event.ssid))
        return alerts

    def summary(self) -> dict:
        return {
            "events_processed": len(self.events),
            "alerts": [alert.to_dict() for alert in self.alerts],
            "alert_counts": dict(Counter(alert.alert_type for alert in self.alerts)),
        }
