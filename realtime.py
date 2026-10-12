"""Stateful live monitor: immediate rules plus periodic window-level ML."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import joblib

try:
    from .config import DetectorConfig
    from .detector import ThreatDetector
    from .features import FEATURE_COLUMNS, extract_feature_windows
    from .ml_pipeline import MODEL_FORMAT_VERSION, predict_feature_window
    from .models import WiFiEvent
except ImportError:  # Direct execution from src/main.py
    from config import DetectorConfig
    from detector import ThreatDetector
    from features import FEATURE_COLUMNS, extract_feature_windows
    from ml_pipeline import MODEL_FORMAT_VERSION, predict_feature_window
    from models import WiFiEvent


class RealtimeMonitor:
    """Process each event immediately and score active windows on a cadence."""

    def __init__(
        self,
        model_path: str | Path,
        report_path: str | Path,
        config_path: str | Path | None = None,
        event_log_path: str | Path | None = None,
        window_seconds: int | None = None,
        score_every_seconds: float = 5.0,
        min_events: int = 8,
        max_recent_windows: int = 1000,
    ) -> None:
        if score_every_seconds <= 0 or min_events <= 0:
            raise ValueError("score_every_seconds and min_events must be positive")
        self.bundle = joblib.load(model_path)
        self.metadata = self.bundle.get("metadata", {})
        if self.metadata.get("format_version") != MODEL_FORMAT_VERSION:
            raise ValueError("Unsupported model artifact format")
        if self.metadata.get("feature_columns") != list(FEATURE_COLUMNS):
            raise ValueError("Model feature schema does not match this code version")
        model_window = int(self.metadata.get("window_seconds", 30))
        if window_seconds is None:
            window_seconds = model_window
        elif window_seconds != model_window:
            raise ValueError(f"This model expects --window-seconds {model_window}")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        self.config = DetectorConfig.from_file(config_path)
        self.detector = ThreatDetector(self.config)
        self.window_seconds = window_seconds
        self.score_every_seconds = score_every_seconds
        self.min_events = min_events
        self.max_recent_windows = max_recent_windows
        self.report_path = Path(report_path)
        self.event_log_path = Path(event_log_path) if event_log_path else None
        self._event_log = None
        if self.event_log_path:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._event_log = self.event_log_path.open("a", encoding="utf-8")

        self._windows: dict[tuple[str, int], list[WiFiEvent]] = {}
        self._last_scored: dict[tuple[str, int], float] = {}
        self._last_status: dict[tuple[str, int], tuple[str, bool]] = {}
        self._predictions: OrderedDict[tuple[str, int], dict] = OrderedDict()
        self.total_events = 0
        self.total_windows_scored = 0
        self.inference_count = 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.closed = False
        self._last_report_save_monotonic = 0.0
        self._save_report()

    def _key(self, event: WiFiEvent) -> tuple[str, int]:
        epoch = int(event.timestamp.timestamp())
        bucket = epoch - epoch % self.window_seconds
        return event.bssid or "UNKNOWN", bucket

    def process_event(self, event: WiFiEvent) -> None:
        """Handle one newly captured event and persist any immediate alert."""
        self.tick()
        self.total_events += 1
        if self._event_log:
            self._event_log.write(json.dumps(event.to_dict(), separators=(",", ":")) + "\n")
            self._event_log.flush()

        new_alerts = self.detector.process(event)
        for alert in new_alerts:
            print(
                f"\n[{alert.timestamp.isoformat()}] RULE {alert.severity.upper()} "
                f"{alert.alert_type} {alert.bssid or ''} — {alert.message}",
                flush=True,
            )

        key = self._key(event)
        self._windows.setdefault(key, []).append(event)
        now = time.monotonic()
        last_scored = self._last_scored.get(key)
        should_score = (
            len(self._windows[key]) >= self.min_events
            and (last_scored is None or now - last_scored >= self.score_every_seconds)
        )
        if should_score:
            self._score_window(key, final=False)
        self._save_report_if_due(force=bool(new_alerts or should_score))

    def tick(self) -> None:
        """Finalize UTC windows after their boundary, including during idle RF."""
        now = datetime.now(timezone.utc).timestamp()
        due = [
            key for key in self._windows
            if key[1] + self.window_seconds <= now
        ]
        for key in due:
            self._score_window(key, final=True)
            self._windows.pop(key, None)
            self._last_scored.pop(key, None)
            self._last_status.pop(key, None)
        self._save_report_if_due(force=bool(due))

    def _score_window(self, key: tuple[str, int], final: bool) -> None:
        events = self._windows.get(key, [])
        if not events:
            return
        if len(events) < self.min_events:
            if final:
                print(
                    f"\n[ML] {key[0]} window skipped: {len(events)} events "
                    f"(minimum {self.min_events}).",
                    flush=True,
                )
            return
        windows = extract_feature_windows(events, self.window_seconds)
        feature_window = next((item for item in windows if item.bssid == key[0]), None)
        if feature_window is None:
            return
        prediction = predict_feature_window(feature_window, self.bundle)
        prediction.update({
            "event_count": len(events),
            "is_final_window_score": final,
            "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        first_score = key not in self._predictions
        if first_score:
            self.total_windows_scored += 1
        self.inference_count += 1
        self._predictions[key] = prediction
        self._predictions.move_to_end(key)
        while len(self._predictions) > self.max_recent_windows:
            self._predictions.popitem(last=False)
        self._last_scored[key] = time.monotonic()

        status = (prediction["predicted_class"], prediction["is_anomalous"])
        if status != self._last_status.get(key) and (status[0] != "normal" or status[1]):
            print(
                f"\n[{prediction['scored_at_utc']}] ML {key[0]} "
                f"class={status[0]} attack_probability={prediction['attack_probability']:.2f} "
                f"anomaly={status[1]} score={prediction['anomaly_score']:.4f}",
                flush=True,
            )
        self._last_status[key] = status

    def report(self) -> dict:
        predictions = list(self._predictions.values())
        class_counts = Counter(item["predicted_class"] for item in predictions)
        alerts = self.detector.alerts[-1000:]
        return {
            "status": "stopped" if self.closed else "monitoring",
            "started_at_utc": self.started_at,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": self.metadata,
            "window_seconds": self.window_seconds,
            "score_every_seconds": self.score_every_seconds,
            "min_events": self.min_events,
            "total_events_processed": self.total_events,
            "events_processed": len(self.detector.events),
            "active_bssid_windows": len(self._windows),
            "summary": {
                "total_windows_scored": self.total_windows_scored,
                "inference_count": self.inference_count,
                "windows_in_report": len(predictions),
                "predicted_class_counts": dict(class_counts),
                "anomalous_windows": sum(bool(item["is_anomalous"]) for item in predictions),
            },
            "alert_counts": dict(Counter(alert.alert_type for alert in self.detector.alerts)),
            "alerts": [alert.to_dict() for alert in alerts],
            "predictions": predictions,
            "event_log": str(self.event_log_path) if self.event_log_path else None,
            "notice": "Live detections are triage signals. Confirm with authorized controller and sensor evidence.",
        }

    def _save_report(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(self.report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.report(), indent=2), encoding="utf-8")
        os.replace(temporary, self.report_path)
        self._last_report_save_monotonic = time.monotonic()

    def _save_report_if_due(self, force: bool = False) -> None:
        """Persist live counters at least once per second while traffic is quiet."""
        if force or time.monotonic() - self._last_report_save_monotonic >= 1.0:
            self._save_report()

    def close(self) -> None:
        """Score remaining complete-enough windows and write a final snapshot."""
        if self.closed:
            return
        for key in list(self._windows):
            self._score_window(key, final=True)
        self.closed = True
        self._save_report()
        if self._event_log:
            self._event_log.close()
