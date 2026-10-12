"""Live and offline report dashboard for Wi-Fi threat detection.

Run from the project root with: python -m streamlit run streamlit_dashboard.py
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LIVE_REPORT = PROJECT_ROOT / "reports" / "live_report.json"

st.set_page_config(page_title="Wi-Fi Threat Detection", page_icon="📡", layout="wide")
st.title("📡 Wi-Fi Threat Detection Dashboard")
st.caption("Live passive-monitor reports or uploaded offline reports.")


def show_report(report: dict, live_report_path: Path | None = None) -> None:
    """Render one detector report and identify whether its live source is fresh."""
    if not isinstance(report, dict) or not isinstance(report.get("alerts"), list):
        st.error("Expected a detector JSON report containing an 'alerts' list.")
        return

    if live_report_path is not None:
        report_status = str(report.get("status", "unknown")).lower()
        try:
            age_seconds = max(0.0, time.time() - live_report_path.stat().st_mtime)
        except OSError:
            age_seconds = float("inf")
        if report_status == "monitoring" and age_seconds <= 15:
            st.success(f"LIVE MONITOR ACTIVE · updated {age_seconds:.0f}s ago")
        elif report_status == "monitoring":
            st.warning(f"Report is stale ({age_seconds:.0f}s old). Check the monitor process and capture adapter.")
        else:
            st.info(f"Monitor status: {report_status}. Report updated {age_seconds:.0f}s ago.")
        st.caption(f"Live report file: {live_report_path}")
    else:
        st.info("Uploaded report view · this is a saved snapshot, not a live feed.")

    alerts = report["alerts"]
    severity_counts = Counter(str(alert.get("severity", "unknown")).lower() for alert in alerts)
    type_counts = Counter(str(alert.get("alert_type", "UNKNOWN")) for alert in alerts)

    left, middle, right, far_right = st.columns(4)
    left.metric("Events processed", report.get("total_events_processed", report.get("events_processed", 0)))
    middle.metric("Alerts", len(alerts))
    right.metric("High severity", severity_counts["high"])
    far_right.metric("Alert types", len(type_counts))

    holdout_metrics = report.get("model", {}).get("holdout_metrics", {})
    if holdout_metrics.get("accuracy") is None:
        metrics_path = PROJECT_ROOT / "reports" / "model_metrics.json"
        try:
            saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            saved_model = saved_metrics.get("model", {})
            report_digest = report.get("model", {}).get("dataset_sha256")
            saved_digest = saved_model.get("dataset_sha256")
            if report_digest and report_digest == saved_digest:
                holdout_metrics = saved_metrics.get("holdout_evaluation", {})
        except (OSError, json.JSONDecodeError):
            pass
    if holdout_metrics.get("accuracy") is not None:
        st.subheader("Offline holdout benchmark")
        accuracy_columns = st.columns(3)
        accuracy_columns[0].metric("Holdout accuracy", f"{float(holdout_metrics['accuracy']):.1%}")
        accuracy_columns[1].metric("Macro F1", f"{float(holdout_metrics.get('macro_f1', 0)):.3f}")
        accuracy_columns[2].metric("Balanced accuracy", f"{float(holdout_metrics.get('balanced_accuracy', 0)):.3f}")
        st.caption("This is an offline group-held-out benchmark, not an accuracy measurement of current live traffic. Report the dataset type and provenance with this score.")

    predictions = report.get("predictions", [])
    if predictions:
        st.subheader("ML window triage")
        model_summary = report.get("summary", {})
        ml_left, ml_middle, ml_right = st.columns(3)
        ml_left.metric("Windows scored", model_summary.get("total_windows_scored", len(predictions)))
        ml_middle.metric("Anomalous windows", model_summary.get("anomalous_windows", 0))
        ml_right.metric("Rule alerts", len(alerts))
        predicted_counts = model_summary.get("predicted_class_counts", {})
        if predicted_counts:
            st.bar_chart(dict(predicted_counts), horizontal=True)
        prediction_rows = [{
            "Window start": row.get("window_start", "—"),
            "BSSID": row.get("bssid", "—"),
            "Predicted class": row.get("predicted_class", "—"),
            "Attack probability": round(float(row.get("attack_probability", 0)), 3),
            "Anomaly score": round(float(row.get("anomaly_score", 0)), 4),
            "Anomalous": bool(row.get("is_anomalous", False)),
        } for row in predictions]
        st.dataframe(prediction_rows, use_container_width=True, hide_index=True)
        st.caption("ML scores prioritize review; confirm findings with authorized network evidence.")
        st.divider()

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Alerts by type")
        st.bar_chart(dict(type_counts), horizontal=True)
    with chart_right:
        st.subheader("Severity distribution")
        st.bar_chart(dict(severity_counts), horizontal=True)

    st.subheader("Alert investigation")
    available_severities = sorted(severity_counts)
    chosen_severities = st.multiselect("Show severity", available_severities, default=available_severities)
    search = st.text_input("Search SSID, BSSID, alert type, or message")

    rows = []
    for alert in alerts:
        if str(alert.get("severity", "unknown")).lower() not in chosen_severities:
            continue
        searchable = " ".join(str(value) for value in alert.values()).lower()
        if search.lower() not in searchable:
            continue
        rows.append({
            "Timestamp": alert.get("timestamp", "—"),
            "Alert type": alert.get("alert_type", "—"),
            "Severity": str(alert.get("severity", "—")).upper(),
            "SSID": alert.get("ssid", "—"),
            "BSSID": alert.get("bssid", "—"),
            "Evidence count": alert.get("evidence_count", 1),
            "Message": alert.get("message", "—"),
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.download_button(
        "Download displayed alerts as JSON",
        data=json.dumps(rows, indent=2),
        file_name="filtered_wifi_alerts.json",
        mime="application/json",
    )
    st.info("Alerts are investigation indicators. Verify with authorized network-controller and sensor evidence before taking action.")


with st.sidebar:
    st.header("Report source")
    report_source = st.radio("Choose report", ["Live report", "Upload saved report"], index=0)

if report_source == "Live report":
    with st.sidebar:
        live_path_text = st.text_input("Live report file", value=str(DEFAULT_LIVE_REPORT))
        refresh_seconds = st.selectbox("Refresh interval", [2, 5, 10], index=1, format_func=lambda value: f"Every {value} seconds")

    live_report_path = Path(live_path_text).expanduser()

    @st.fragment(run_every=f"{refresh_seconds}s")
    def live_report_view() -> None:
        try:
            report = json.loads(live_report_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            st.warning("No live report exists yet. Start the monitor command; it creates this file when monitoring begins.")
            st.code("python -m src.main monitor --interface <INTERFACE> --model models/wifi_hybrid.joblib --config config.example.json --report reports/live_report.json")
            return
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Could not read the live report: {exc}")
            return
        show_report(report, live_report_path)

    live_report_view()
else:
    with st.sidebar:
        uploaded = st.file_uploader("Upload detector JSON report", type=["json"])
        st.caption("Uploaded reports are saved snapshots. Choose Live report for automatic refresh.")
    if uploaded is None:
        st.info("Choose a saved detector JSON report in the left panel.")
    else:
        try:
            report = json.loads(uploaded.getvalue().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            st.error(f"The uploaded file is not valid UTF-8 JSON: {exc}")
        else:
            show_report(report)
