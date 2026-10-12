from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Event
import time

try:  # Package execution: py -m src.main
    from .models import WiFiEvent
except ImportError:  # Direct execution/import from the src directory
    from models import WiFiEvent


def read_jsonl(path: str | Path) -> Iterator[WiFiEvent]:
    """Yield valid JSONL events; malformed lines are skipped with a clear error."""
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield WiFiEvent.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid event on line {number}: {exc}") from exc


def _load_scapy():
    try:
        from scapy.all import (
            Dot11, Dot11Beacon, Dot11Deauth, Dot11Disas, Dot11Elt,
            Dot11EltMicrosoftWPA, Dot11EltRSN, Dot11ProbeReq,
        )
    except ImportError as exc:
        raise RuntimeError("Live capture requires scapy. Run: pip install -r requirements.txt") from exc
    return (
        Dot11, Dot11Beacon, Dot11Deauth, Dot11Disas, Dot11Elt,
        Dot11EltMicrosoftWPA, Dot11EltRSN, Dot11ProbeReq,
    )


def get_capture_interfaces() -> list[str]:
    """Return interface identifiers recognized by Scapy on this machine."""
    try:
        from scapy.all import get_if_list
    except ImportError as exc:
        raise RuntimeError("Interface listing requires scapy. Run: pip install -r requirements.txt") from exc
    return list(get_if_list())


def _event_from_packet(packet, layers) -> WiFiEvent | None:
    (Dot11, Dot11Beacon, Dot11Deauth, Dot11Disas, Dot11Elt,
     Dot11EltMicrosoftWPA, Dot11EltRSN, Dot11ProbeReq) = layers
    if not packet.haslayer(Dot11):
        return None
    dot11 = packet[Dot11]
    frame_type = None
    if packet.haslayer(Dot11Beacon):
        frame_type = "beacon"
    elif packet.haslayer(Dot11Deauth):
        frame_type = "deauth"
    elif packet.haslayer(Dot11Disas):
        frame_type = "disassoc"
    elif packet.haslayer(Dot11ProbeReq):
        frame_type = "probe"
    if not frame_type:
        return None
    ssid = None
    channel = None
    security = "UNKNOWN" if frame_type == "beacon" else None
    if frame_type == "beacon" and packet.haslayer(Dot11Elt):
        has_wpa = False
        has_rsn = False
        wpa3_akm = False
        element = packet.getlayer(Dot11Elt)
        while isinstance(element, Dot11Elt):
            try:
                element_id = int(element.ID)
                if element_id == 0 and ssid is None:
                    ssid = bytes(element.info).decode("utf-8", errors="replace")
                elif element_id == 3 and channel is None:
                    channel = int(element.channel)
                elif element_id == 61 and channel is None:
                    channel = int(element.primary_channel)
                if isinstance(element, Dot11EltRSN):
                    has_rsn = True
                    wpa3_akm = wpa3_akm or any(
                        int(suite.oui) == 4012 and int(suite.suite) in {8, 9, 18}
                        for suite in element.akm_suites
                    )
                elif isinstance(element, Dot11EltMicrosoftWPA):
                    has_wpa = True
            except (AttributeError, TypeError, ValueError):
                pass
            next_element = element.payload
            if not isinstance(next_element, Dot11Elt):
                break
            element = next_element
        if has_rsn:
            security = "WPA3" if wpa3_akm else "WPA2"
        elif has_wpa:
            security = "WPA"
        else:
            try:
                privacy = bool(int(packet[Dot11Beacon].cap) & 0x10)
            except (TypeError, ValueError):
                privacy = False
            security = "WEP" if privacy else "OPEN"
    return WiFiEvent.from_dict({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_type": frame_type,
        "bssid": dot11.addr3,
        "source": dot11.addr2,
        "destination": dot11.addr1,
        "ssid": ssid if frame_type == "beacon" else None,
        "channel": channel,
        "security": security if frame_type == "beacon" else None,
    })


def capture_stream(
    interface: str,
    stop_event: Event | None = None,
    heartbeat_seconds: float = 0.5,
) -> Iterator[WiFiEvent | None]:
    """Yield metadata as packets arrive; yield ``None`` periodically when idle."""
    try:
        from scapy.all import AsyncSniffer
        from scapy.config import conf
    except ImportError as exc:
        raise RuntimeError("Live capture requires scapy. Run: pip install -r requirements.txt") from exc

    layers = _load_scapy()
    pending: Queue[WiFiEvent] = Queue()

    def handle(packet) -> None:
        event = _event_from_packet(packet, layers)
        if event is not None:
            pending.put(event)

    sniffer_options = {}
    if sys.platform in {"win32", "darwin"} or conf.use_pcap:
        sniffer_options["monitor"] = True
    sniffer = AsyncSniffer(iface=interface, prn=handle, store=False, **sniffer_options)
    try:
        try:
            sniffer.start()
        except Exception as exc:
            raise RuntimeError(
                f"Could not start passive capture on interface {interface!r}. "
                "Check the interface name, adapter monitor-mode support, capture driver, and permissions."
            ) from exc
        while stop_event is None or not stop_event.is_set():
            try:
                yield pending.get(timeout=heartbeat_seconds)
            except Empty:
                yield None
    finally:
        if sniffer.running:
            sniffer.stop(join=True)


def capture_events(interface: str, duration: int) -> Iterator[WiFiEvent]:
    """Passively capture selected management metadata for a fixed duration."""
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    stream = capture_stream(interface)
    deadline = time.monotonic() + duration
    try:
        while time.monotonic() < deadline:
            event = next(stream)
            if event is not None:
                yield event
    finally:
        stream.close()
