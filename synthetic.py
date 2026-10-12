"""Reproducible, labelled scenarios for an offline detector demonstration.

These scenarios are intentionally synthetic. They make the full pipeline
reproducible without collecting traffic, and must not be presented as field
performance or a substitute for evaluation on authorized real data.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

try:
    from .features import FEATURE_COLUMNS, extract_feature_windows
    from .models import WiFiEvent
except ImportError:  # Direct execution from src/main.py
    from features import FEATURE_COLUMNS, extract_feature_windows
    from models import WiFiEvent


CLASS_LABELS = (
    "normal",
    "deauth_flood",
    "disassoc_flood",
    "beacon_flood",
    "probe_flood",
    "mixed_management_burst",
)


def _mac(number: int) -> str:
    return "02:42:%02X:%02X:%02X:%02X" % (
        (number >> 24) & 255, (number >> 16) & 255,
        (number >> 8) & 255, number & 255,
    )


def _scenario_events(
    label: str,
    rng: random.Random,
    capture_index: int,
    channel: int,
) -> list[WiFiEvent]:
    bssid = _mac(10_000 + capture_index)
    ssid = f"LabNet-{capture_index % 13:02d}"
    security = "WPA3" if capture_index % 3 == 0 else "WPA2"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=capture_index, minutes=(capture_index * 7) % 60
    )

    if label == "normal":
        counts = {
            "beacon": rng.randint(12, 42),
            "probe": rng.randint(2, 18),
            "deauth": rng.randint(0, 4),
            "disassoc": rng.randint(0, 3),
        }
    elif label == "deauth_flood":
        counts = {"beacon": rng.randint(5, 30), "probe": rng.randint(1, 18),
                  "deauth": rng.randint(18, 110), "disassoc": rng.randint(0, 5)}
    elif label == "disassoc_flood":
        counts = {"beacon": rng.randint(5, 30), "probe": rng.randint(1, 18),
                  "deauth": rng.randint(0, 5), "disassoc": rng.randint(18, 110)}
    elif label == "beacon_flood":
        counts = {"beacon": rng.randint(70, 260), "probe": rng.randint(0, 12),
                  "deauth": rng.randint(0, 5), "disassoc": rng.randint(0, 4)}
    elif label == "probe_flood":
        counts = {"beacon": rng.randint(8, 34), "probe": rng.randint(70, 260),
                  "deauth": rng.randint(0, 5), "disassoc": rng.randint(0, 4)}
    elif label == "mixed_management_burst":
        counts = {"beacon": rng.randint(6, 35), "probe": rng.randint(8, 55),
                  "deauth": rng.randint(12, 55), "disassoc": rng.randint(12, 55)}
    else:
        raise ValueError(f"Unsupported synthetic class: {label}")

    frames = [frame for frame, count in counts.items() for _ in range(count)]
    rng.shuffle(frames)
    events = []
    for index, frame_type in enumerate(frames):
        # Spread events across one half-minute window while keeping burst rate
        # meaningful. Duplicate millisecond values are valid for this dataset.
        timestamp = start + timedelta(milliseconds=rng.randint(0, 29_500))
        source = bssid if frame_type == "beacon" else _mac(
            50_000 + capture_index * 1_000 + (index % rng.randint(1, 14))
        )
        destination = (
            "FF:FF:FF:FF:FF:FF" if frame_type in {"beacon", "probe"}
            else bssid
        )
        events.append(WiFiEvent(
            timestamp=timestamp,
            frame_type=frame_type,
            bssid=bssid,
            source=source,
            destination=destination,
            ssid=ssid if frame_type == "beacon" else None,
            channel=channel if frame_type == "beacon" else None,
            security=security if frame_type == "beacon" else None,
        ))
    return events


def generate_labeled_rows(scenarios_per_class: int = 200, seed: int = 42) -> list[dict]:
    """Return feature rows with labels and capture groups for ML evaluation."""
    if scenarios_per_class < 16:
        raise ValueError("scenarios_per_class must be at least 16 for group holdout evaluation")
    rng = random.Random(seed)
    rows = []
    for capture_index in range(scenarios_per_class):
        # Each held-out capture group contains every class, reducing the chance
        # that evaluation merely measures a particular synthetic environment.
        channel = (capture_index * 3) % 13 + 1
        for label in CLASS_LABELS:
            events = _scenario_events(label, rng, capture_index, channel)
            windows = extract_feature_windows(events, window_seconds=30)
            if len(windows) != 1:
                raise RuntimeError("Synthetic scenario unexpectedly crossed a feature window")
            row = dict(windows[0].values)
            row["window_seconds"] = 30
            row["label"] = label
            row["capture_group"] = f"capture-{capture_index:05d}"
            rows.append(row)
    return rows


def feature_columns() -> tuple[str, ...]:
    """Expose the canonical model input schema for serialization and tooling."""
    return FEATURE_COLUMNS
