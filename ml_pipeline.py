"""Train and apply the hybrid supervised/unsupervised Wi-Fi ML models."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

try:
    from .config import DetectorConfig
    from .detector import ThreatDetector
    from .features import FEATURE_COLUMNS, extract_feature_windows
    from .input_adapters import read_jsonl
    from .synthetic import CLASS_LABELS
except ImportError:  # Direct execution from src/main.py
    from config import DetectorConfig
    from detector import ThreatDetector
    from features import FEATURE_COLUMNS, extract_feature_windows
    from input_adapters import read_jsonl
    from synthetic import CLASS_LABELS


MODEL_FORMAT_VERSION = 1


def _dataset_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_models(
    dataset_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    test_size: float = 0.25,
    seed: int = 42,
) -> dict:
    """Evaluate on held-out capture groups, then fit production artifacts."""
    source = Path(dataset_path)
    model_destination = Path(model_path)
    metrics_destination = Path(metrics_path)
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")
    data = pd.read_csv(source)
    required = set(FEATURE_COLUMNS) | {"label", "capture_group"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("Dataset has no feature rows")
    data = data.dropna(subset=[*FEATURE_COLUMNS, "label", "capture_group"]).copy()
    if data.empty:
        raise ValueError("Dataset has no complete feature rows")
    unexpected_labels = sorted(set(data["label"].astype(str)) - set(CLASS_LABELS))
    if unexpected_labels:
        raise ValueError(f"Unexpected labels: {', '.join(unexpected_labels)}")
    if data["capture_group"].nunique() < 16:
        raise ValueError("At least 16 distinct capture groups are required for group holdout evaluation")
    if "window_seconds" in data.columns:
        window_sizes = pd.to_numeric(data["window_seconds"], errors="raise").unique()
        if (
            len(window_sizes) != 1
            or int(window_sizes[0]) <= 0
            or float(window_sizes[0]) != int(window_sizes[0])
        ):
            raise ValueError("All feature rows must use the same positive window_seconds value")
        window_seconds = int(window_sizes[0])
    else:
        window_seconds = 30

    x = data.loc[:, FEATURE_COLUMNS].astype(float)
    y = data["label"].astype(str)
    groups = data["capture_group"].astype(str)
    if "normal" not in set(y):
        raise ValueError("Dataset must include normal windows for one-class anomaly detection")
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Feature matrix contains NaN or infinite values")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_indices, test_indices = next(splitter.split(x, y, groups))
    x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])
    if train_groups & test_groups:
        raise RuntimeError("Capture group leakage detected between train and holdout sets")

    classifier = RandomForestClassifier(
        n_estimators=400,
        max_features=0.8,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    classifier.fit(x_train, y_train)
    holdout_prediction = classifier.predict(x_test)

    # The one-class model sees only normal training windows. Its threshold is
    # learned from training data and frozen before holdout evaluation.
    normal_train = x_train.loc[y_train == "normal"]
    if len(normal_train) < 8:
        raise ValueError("The training split needs at least 8 normal windows")
    anomaly_model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    anomaly_model.fit(normal_train)
    train_normal_scores = -anomaly_model.score_samples(normal_train)
    anomaly_threshold = float(np.quantile(train_normal_scores, 0.99))
    holdout_anomaly_scores = -anomaly_model.score_samples(x_test)
    holdout_anomaly_flags = holdout_anomaly_scores > anomaly_threshold
    holdout_is_attack = (y_test.to_numpy() != "normal").astype(int)
    anomaly_metrics = {
        "precision": float(precision_score(holdout_is_attack, holdout_anomaly_flags, zero_division=0)),
        "recall": float(recall_score(holdout_is_attack, holdout_anomaly_flags, zero_division=0)),
        "f1": float(f1_score(holdout_is_attack, holdout_anomaly_flags, zero_division=0)),
        "threshold": anomaly_threshold,
        "threshold_quantile": 0.99,
    }

    evaluation = {
        "split": "GroupShuffleSplit by capture_group",
        "train_rows": int(len(train_indices)),
        "holdout_rows": int(len(test_indices)),
        "train_groups": int(len(train_groups)),
        "holdout_groups": int(len(test_groups)),
        "accuracy": float(accuracy_score(y_test, holdout_prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, holdout_prediction)),
        "macro_f1": float(f1_score(y_test, holdout_prediction, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_test, holdout_prediction, labels=CLASS_LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": {
            "labels": list(CLASS_LABELS),
            "values": confusion_matrix(y_test, holdout_prediction, labels=CLASS_LABELS).tolist(),
        },
        "one_class_anomaly_detection": anomaly_metrics,
        "feature_importance": sorted(
            [
                {"feature": name, "importance": float(value)}
                for name, value in zip(FEATURE_COLUMNS, classifier.feature_importances_)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        ),
    }

    # Fit final estimators on the full labelled dataset only after preserving
    # the independent group holdout metrics above.
    final_classifier = RandomForestClassifier(
        n_estimators=400,
        max_features=0.8,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    ).fit(x, y)
    final_normal = x.loc[y == "normal"]
    final_anomaly_model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    ).fit(final_normal)
    final_normal_scores = -final_anomaly_model.score_samples(final_normal)
    final_threshold = float(np.quantile(final_normal_scores, 0.99))
    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "format_version": MODEL_FORMAT_VERSION,
        "trained_at_utc": trained_at,
        "training_rows": int(len(data)),
        "capture_groups": int(groups.nunique()),
        "labels": list(final_classifier.classes_),
        "feature_columns": list(FEATURE_COLUMNS),
        "window_seconds": window_seconds,
        "dataset_sha256": _dataset_digest(source),
        "evaluation_scope": "synthetic or user-supplied labelled dataset; see README",
        "holdout_metrics": {
            "accuracy": evaluation["accuracy"],
            "balanced_accuracy": evaluation["balanced_accuracy"],
            "macro_f1": evaluation["macro_f1"],
            "train_groups": evaluation["train_groups"],
            "holdout_groups": evaluation["holdout_groups"],
        },
    }
    bundle = {
        "classifier": final_classifier,
        "anomaly_detector": final_anomaly_model,
        "anomaly_threshold": final_threshold,
        "metadata": metadata,
    }
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    metrics_destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_destination, compress=3)
    metrics_destination.write_text(
        json.dumps({"model": metadata, "holdout_evaluation": evaluation}, indent=2),
        encoding="utf-8",
    )
    return {"model_path": str(model_destination), "metrics_path": str(metrics_destination), **evaluation}


def predict_feature_window(window, bundle: dict) -> dict:
    """Return an ML triage record using the same path offline and live."""
    frame = pd.DataFrame([window.values], columns=FEATURE_COLUMNS)
    probabilities = bundle["classifier"].predict_proba(frame)[0]
    labels = list(bundle["classifier"].classes_)
    predicted = str(bundle["classifier"].predict(frame)[0])
    score = float(-bundle["anomaly_detector"].score_samples(frame)[0])
    anomaly = score > float(bundle["anomaly_threshold"])
    attack_probability = float(sum(
        probability for label, probability in zip(labels, probabilities) if label != "normal"
    ))
    return {
        "window_start": window.window_start,
        "bssid": window.bssid,
        "predicted_class": predicted,
        "attack_probability": attack_probability,
        "anomaly_score": score,
        "anomaly_threshold": float(bundle["anomaly_threshold"]),
        "is_anomalous": bool(anomaly),
        "class_probabilities": {
            label: float(probability) for label, probability in zip(labels, probabilities)
        },
        "features": window.values,
    }


def score_jsonl(
    input_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    window_seconds: int | None = None,
    config_path: str | Path | None = None,
) -> dict:
    """Score passive event metadata and write an investigation-oriented JSON."""
    bundle = joblib.load(model_path)
    metadata = bundle.get("metadata", {})
    if metadata.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported model artifact format")
    if metadata.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("Model feature schema does not match this code version")
    model_window = int(metadata.get("window_seconds", 30))
    if window_seconds is None:
        window_seconds = model_window
    elif window_seconds != model_window:
        raise ValueError(f"This model expects --window-seconds {model_window}")
    events = list(read_jsonl(input_path))
    windows = extract_feature_windows(events, window_seconds=window_seconds)
    predictions = []
    for window in windows:
        predictions.append(predict_feature_window(window, bundle))
    summary = {
        "total_windows_scored": len(predictions),
        "predicted_class_counts": {},
        "anomalous_windows": sum(row["is_anomalous"] for row in predictions),
    }
    for row in predictions:
        label = row["predicted_class"]
        summary["predicted_class_counts"][label] = summary["predicted_class_counts"].get(label, 0) + 1
    detector = ThreatDetector(DetectorConfig.from_file(config_path))
    for event in events:
        detector.process(event)
    rule_alerts = [alert.to_dict() for alert in detector.alerts]
    report = {
        "model": metadata,
        "window_seconds": window_seconds,
        "summary": summary,
        "predictions": predictions,
        "alerts": rule_alerts,
        "alert_counts": detector.summary()["alert_counts"],
        "events_processed": detector.summary()["events_processed"],
        "total_events_processed": len(events),
        "notice": "ML scores are triage signals. Validate them against authorized controller and sensor evidence.",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
