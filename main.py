from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Preferred package execution: py -m src.main
    from .config import DetectorConfig
    from .detector import ThreatDetector
    from .input_adapters import capture_events, capture_stream, get_capture_interfaces, read_jsonl
except ImportError:  # Supports: py src/main.py
    from config import DetectorConfig
    from detector import ThreatDetector
    from input_adapters import capture_events, capture_stream, get_capture_interfaces, read_jsonl


def write_report(path: str | Path, report: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Defensive Wi-Fi threat detector")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="Analyze JSONL events")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--config")
    analyze.add_argument("--output", required=True)
    generate = subparsers.add_parser("generate-synthetic", help="Create a labelled offline ML dataset")
    generate.add_argument("--output", required=True)
    generate.add_argument("--scenarios-per-class", type=int, default=200)
    generate.add_argument("--seed", type=int, default=42)
    train = subparsers.add_parser("train", help="Evaluate and train hybrid ML models")
    train.add_argument("--input", required=True, help="Feature CSV with label and capture_group columns")
    train.add_argument("--model", required=True, help="Output joblib model bundle")
    train.add_argument("--metrics", required=True, help="Output holdout evaluation JSON")
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=42)
    score = subparsers.add_parser("score", help="Score JSONL events with a trained ML model")
    score.add_argument("--input", required=True)
    score.add_argument("--model", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--window-seconds", type=int, help="Must match the model's training feature window")
    score.add_argument("--config", help="Optional trusted AP baseline and rule thresholds")
    extract = subparsers.add_parser("extract-features", help="Create a windowed feature table from event JSONL")
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--window-seconds", type=int, default=30)
    capture = subparsers.add_parser("capture", help="Passively capture management metadata")
    capture.add_argument("--interface", required=True)
    capture.add_argument("--duration", type=int, default=60)
    capture.add_argument("--config")
    capture.add_argument("--output", required=True)
    monitor = subparsers.add_parser("monitor", help="Continuously monitor and score live metadata")
    monitor.add_argument("--interface", required=True, help="Existing authorized monitor-mode interface")
    monitor.add_argument("--model", required=True, help="Trained model bundle from the train command")
    monitor.add_argument("--config", help="Optional thresholds and trusted AP baseline")
    monitor.add_argument("--report", default="reports/live_report.json")
    monitor.add_argument("--event-log", help="Optional local JSONL log of captured metadata")
    monitor.add_argument("--window-seconds", type=int, help="Must match the model's training feature window; defaults from model")
    monitor.add_argument("--score-every-seconds", type=float, default=5.0)
    monitor.add_argument("--min-events", type=int, default=8)
    report = subparsers.add_parser("report", help="Print a JSON report summary")
    report.add_argument("--input", required=True)
    subparsers.add_parser("interfaces", help="List capture interface names known to Scapy")
    args = parser.parse_args()

    if args.command == "generate-synthetic":
        try:
            import pandas as pd
            from .synthetic import generate_labeled_rows
        except ImportError:
            import pandas as pd
            from synthetic import generate_labeled_rows
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataset = pd.DataFrame(generate_labeled_rows(args.scenarios_per_class, args.seed))
        dataset.to_csv(destination, index=False)
        print(f"Generated {len(dataset)} labelled windows across {dataset['capture_group'].nunique()} capture groups.")
        print(f"Dataset written to: {destination}")
        return 0
    if args.command == "train":
        try:
            from .ml_pipeline import train_models
        except ImportError:
            from ml_pipeline import train_models
        result = train_models(args.input, args.model, args.metrics, args.test_size, args.seed)
        print(f"Holdout accuracy: {result['accuracy']:.3f}")
        print(f"Holdout macro F1: {result['macro_f1']:.3f}")
        print(f"Holdout balanced accuracy: {result['balanced_accuracy']:.3f}")
        print(f"Holdout rows/groups: {result['holdout_rows']}/{result['holdout_groups']}")
        print("X: engineered numeric features; y: activity label")
        print(f"Model written to: {result['model_path']}")
        print(f"Evaluation written to: {result['metrics_path']}")
        return 0
    if args.command == "score":
        try:
            from .ml_pipeline import score_jsonl
        except ImportError:
            from ml_pipeline import score_jsonl
        result = score_jsonl(args.input, args.model, args.output, args.window_seconds, args.config)
        print(f"Scored {result['summary']['total_windows_scored']} windows; flagged {result['summary']['anomalous_windows']} as anomalous.")
        print(f"Rule engine created {len(result['alerts'])} alerts.")
        print(f"ML report written to: {args.output}")
        return 0
    if args.command == "extract-features":
        try:
            import pandas as pd
            from .features import FEATURE_COLUMNS, extract_feature_windows
        except ImportError:
            import pandas as pd
            from features import FEATURE_COLUMNS, extract_feature_windows
        windows = extract_feature_windows(read_jsonl(args.input), args.window_seconds)
        rows = [{
            "window_start": item.window_start,
            "bssid": item.bssid,
            "window_seconds": args.window_seconds,
            **item.values,
        } for item in windows]
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["window_start", "bssid", "window_seconds", *FEATURE_COLUMNS]).to_csv(destination, index=False)
        print(f"Extracted {len(rows)} feature windows to: {destination}")
        print("Add reviewed label and capture_group columns before training on an authorized labelled dataset.")
        return 0
    if args.command == "report":
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        events_in_window = data.get("events_processed", data.get("total_events_processed", 0))
        print(f"Events processed / active window: {events_in_window}")
        print("Alert counts:", json.dumps(data.get("alert_counts", {}), indent=2))
        return 0
    if args.command == "interfaces":
        try:
            for interface in get_capture_interfaces():
                print(interface)
        except RuntimeError as exc:
            print(exc)
            return 2
        return 0
    if args.command == "monitor":
        try:
            from .realtime import RealtimeMonitor
        except ImportError:
            from realtime import RealtimeMonitor
        live = RealtimeMonitor(
            model_path=args.model,
            report_path=args.report,
            config_path=args.config,
            event_log_path=args.event_log,
            window_seconds=args.window_seconds,
            score_every_seconds=args.score_every_seconds,
            min_events=args.min_events,
        )
        stream = capture_stream(args.interface)
        print(f"Monitoring {args.interface!r} passively. Press Ctrl+C to stop.", flush=True)
        print(f"Live report: {args.report}", flush=True)
        try:
            for event in stream:
                if event is None:
                    live.tick()
                else:
                    live.process_event(event)
        except KeyboardInterrupt:
            print("\nStopping monitor and writing the final report...", flush=True)
        except RuntimeError as exc:
            print(f"Capture error: {exc}")
            return 2
        finally:
            stream.close()
            live.close()
        print(f"Final report written to: {args.report}")
        return 0
    config = DetectorConfig.from_file(args.config)
    detector = ThreatDetector(config)
    events = read_jsonl(args.input) if args.command == "analyze" else capture_events(args.interface, args.duration)
    processed = 0
    for event in events:
        detector.process(event)
        processed += 1
    result = detector.summary()
    result["total_events_processed"] = processed
    write_report(args.output, result)
    print(f"Processed {processed} events; created {len(result['alerts'])} unique alerts.")
    print(f"Report written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
