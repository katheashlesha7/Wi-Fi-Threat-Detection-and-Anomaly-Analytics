from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectorConfig:
    window_seconds: int = 30
    deauth_threshold: int = 8
    disassoc_threshold: int = 8
    beacon_threshold: int = 40
    probe_threshold: int = 80
    trusted_aps: dict[str, list[dict]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")
        for name in ("deauth_threshold", "disassoc_threshold", "beacon_threshold", "probe_threshold"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")

    @classmethod
    def from_file(cls, path: str | Path | None) -> "DetectorConfig":
        if path is None:
            return cls()
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown detector config keys: {', '.join(unknown)}")
        return cls(**raw)
