# Wi-Fi Threat Detection and Anomaly Analytics
Created by: Ashlesha Kathe

A defensive, local-first wireless monitoring portfolio project. It combines an interpretable rule engine with a reproducible data science pipeline: fixed-window feature engineering, a supervised multi-class Random Forest, and a one-class Isolation Forest trained on normal windows. The inference command returns both ML triage scores and rule alerts.

The project is designed for offline demos and authorized passive monitoring. It never transmits frames, changes wireless settings, cracks credentials, or inspects packet payloads.

## What this project demonstrates

- Parsing normalized 802.11 management metadata from JSONL or an already-enabled monitor interface.
- Feature engineering over BSSID/time windows, including event rates, frame mix, address diversity, broadcast ratio, SSID/security/channel variation, and frame-type entropy.
- Multi-class attack-family classification and one-class anomaly scoring.
- Group-held-out evaluation to keep capture groups out of both training and holdout data.
- Model/data lineage, confusion matrix, class-level precision/recall/F1, balanced accuracy, macro F1, feature importance, and anomaly-detector metrics.
- A trusted-AP baseline rule engine, CLI, browser dashboard, and local Streamlit investigation view.

## Quick start

Requires Python 3.10 or newer.

```powershell
cd wifi-threat-detection
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```

Generate a reproducible synthetic feature dataset, train/evaluate the models, and score the included offline events:

```powershell
py -3 -m src.main generate-synthetic --output data/synthetic_features.csv --scenarios-per-class 200 --seed 42
py -3 -m src.main train --input data/synthetic_features.csv --model models/wifi_hybrid.joblib --metrics reports/model_metrics.json --seed 42
py -3 -m src.main score --input data/sample_events.jsonl --model models/wifi_hybrid.joblib --config config.example.json --output reports/hybrid_report.json
```

`generate-synthetic` creates six labelled classes and groups examples by simulated capture environment. The generated data is useful for demonstrating the complete workflow; it is not evidence of performance on real networks. For a resume or interview, report the checked-in evaluation as a synthetic benchmark and describe how you would validate it on authorized, independently collected data.

## Run the live monitor on Windows

The live command needs a wireless adapter and driver that expose 802.11 monitor-mode frames to Scapy. Many built-in laptop adapters do not expose this on Windows. Install a supported capture driver such as Npcap if required by your adapter, use the exact interface name Scapy recognizes, and run with the permissions required by that driver. If your hardware/driver cannot expose monitor frames, the live command cannot be made to work by software configuration alone; use a compatible adapter/driver or a supported Linux capture host.

After installing dependencies, list the interface identifiers Scapy sees:

```powershell
py -3 -m src.main interfaces
```

This lists interfaces; it does not enable monitor mode. Scapy's monitor-mode behavior depends on the operating system, adapter, driver, and capture backend, and Windows has significant limitations. Follow the adapter/driver's authorized setup procedure before starting the monitor. The monitor captures on the channel currently visible to that adapter; it does not channel-hop.

First train the model using the commands above. Then, in PowerShell, start continuous capture and scoring:

```powershell
py -3 -m src.main monitor --interface "<SCAPY_MONITOR_INTERFACE>" --model models\wifi_hybrid.joblib --config config.example.json --report reports\live_report.json --score-every-seconds 5 --min-events 8
```

Replace `<SCAPY_MONITOR_INTERFACE>` with the interface identifier available on your machine. The monitor prints rule alerts immediately and scores active BSSID windows every five seconds after eight events. It writes an incrementally updated JSON report. Press **Ctrl+C** to stop and save a final snapshot.

To keep a local JSONL copy of captured metadata for later review, add `--event-log reports\live_events.jsonl`. It records timestamps, SSIDs, BSSIDs, and frame metadata; keep it protected and delete it according to your retention policy. The event log is optional.

Open a second PowerShell window in the project folder to view the dashboard:

```powershell
py -3 -m streamlit run streamlit_dashboard.py
```

The dashboard opens in live-report mode by default and reads `reports/live_report.json`, refreshing every five seconds. Start the monitor first; the dashboard will show a waiting message until the live report exists, then display the monitor status, event count, ML window scores, and rule alerts. You can choose **Upload saved report** in the sidebar when you want to inspect an offline JSON report. To stop the sensor, return to its terminal and press Ctrl+C.

If you do not have a monitor-capable adapter, you can still demonstrate the full data science workflow with the offline commands above and inspect `reports/hybrid_report.json`; that does not constitute a live RF test.

## Commands

### Rule-only analysis

```powershell
py -3 -m src.main analyze --input data/sample_events.jsonl --config config.example.json --output reports/rules_report.json
py -3 -m src.main report --input reports/rules_report.json
```

### Build a labelled training table from reviewed data

The input contract is one JSON object per line; see `data/sample_events.jsonl`. Extract features from authorized metadata, then add reviewed `label` and `capture_group` columns before training. A capture group represents one independent collection session/site so that related rows stay together during evaluation. Keep the same `window_seconds` value for every training row; the saved model enforces that window size during scoring.

```powershell
py -3 -m src.main extract-features --input data/authorized_events.jsonl --output data/review_features.csv --window-seconds 30
```

The model supports the classes `normal`, `deauth_flood`, `disassoc_flood`, `beacon_flood`, `probe_flood`, and `mixed_management_burst`. Label provenance and collection boundaries should be documented with the dataset. Do not train on labels inferred from the same model output being evaluated.

### Train and evaluate

```powershell
py -3 -m src.main train --input data/synthetic_features.csv --model models/wifi_hybrid.joblib --metrics reports/model_metrics.json --test-size 0.25 --seed 42
```

The trainer makes a group-disjoint holdout split, reports classifier metrics and a confusion matrix, evaluates the Isolation Forest as binary anomaly detection, and only then fits final artifacts on all supplied rows. It writes the model and metrics to local files. Re-run with a fixed seed for a repeatable synthetic benchmark.

### Hybrid inference

```powershell
py -3 -m src.main score --input data/sample_events.jsonl --model models/wifi_hybrid.joblib --config config.example.json --output reports/hybrid_report.json
```

The report includes per-window class probabilities, attack probability, Isolation Forest score/threshold, and alerts from the rule engine. A model class prediction and an anomaly flag are separate signals. Use both to prioritize investigation; neither is proof of malicious activity.

### Passive capture

```powershell
py -3 -m src.main capture --interface wlan0mon --duration 60 --config config.example.json --output reports/live_report.json
```

Only capture networks and equipment you own or are authorized to monitor. The operating system and adapter must already expose a monitor-mode interface. Windows capture commonly requires Npcap and a permitted interface. The capture parser retains selected management metadata only.

For continuous inference, use `monitor` instead of `capture`. `capture` makes a finite-duration report after its capture finishes; `monitor` processes each incoming event, evaluates rules immediately, periodically scores each active fixed window, and updates a local report until interrupted.

## Detection and ML design

### Interpretable rules

The streaming rule engine detects deauthentication, disassociation, beacon, and probe bursts, weak/open advertisements, new BSSIDs advertising protected SSIDs from the configured baseline, and changes to a known AP's channel/security profile. Thresholds are in `config.example.json`. MAC and SSID evidence can be spoofed, so baseline alerts are indicators for review.

### Feature and model pipeline

Events are grouped by BSSID and fixed UTC-aligned windows. The feature table captures volume and rate, management-frame composition, source/destination diversity, broadcast behavior, observed SSID/channel/security profile counts, weak-security ratio, and Shannon entropy of frame-type composition.

The supervised model is a class-balanced Random Forest. Its group-held-out report includes accuracy, balanced accuracy, macro F1, class-level metrics, confusion matrix, and feature importance. The dashboard labels these as an offline holdout benchmark; they are not accuracy measurements of current live traffic. A separate Isolation Forest is fit only on `normal` training windows; its alert threshold is the 99th percentile of normal training anomaly scores. The threshold is not estimated from holdout data.

The default data is generated by `src/synthetic.py`; it is intentionally simplified and can create easier class boundaries than real RF environments. Synthetic scores must not be represented as real-world validation, production detection rates, or protection guarantees. A strong extension is a documented benchmark against a consented dataset with time/site/device-separated evaluation and analyst-reviewed labels.

## Dashboard

Launch the local Streamlit dashboard. It reads the live monitor report automatically and refreshes every five seconds. The sidebar can also load a saved report from `analyze` or `score`:

```powershell
py -3 -m streamlit run streamlit_dashboard.py
```

The live view uses `reports/live_report.json` by default. It never substitutes the built-in demo data; if the monitor has not created a report yet, it displays a clear waiting message. Streamlit refreshes only the report view while the page is open. When the report includes model metadata, the dashboard can also show the model's saved offline holdout accuracy, macro F1, and balanced accuracy, clearly separated from live detections.

Alternatively, open `dashboard/index.html` and load a generated rule report. Both dashboards run locally. The browser-only dashboard focuses on rule alerts.

## Project structure

```text
wifi-threat-detection/
├── src/
│   ├── config.py             # thresholds and trusted AP configuration
│   ├── detector.py           # interpretable stateful rules
│   ├── features.py           # fixed-window feature engineering
│   ├── input_adapters.py     # JSONL and passive Scapy input
│   ├── main.py               # CLI orchestration
│   ├── ml_pipeline.py        # group evaluation, training, scoring
│   ├── models.py             # normalized events and alert records
│   ├── realtime.py           # streaming rules, window scoring, live snapshots
│   └── synthetic.py          # deterministic labelled demo scenarios
├── data/                     # safe event example; generated data is local
├── reports/                  # runtime reports and evaluation artifacts
├── dashboard/index.html      # browser-only rule-alert dashboard
├── streamlit_dashboard.py    # interactive local report dashboard
├── config.example.json
└── requirements.txt
```

## Limitations and responsible use

- Passive visibility depends on sensor placement, channel coverage, adapter/driver support, and time synchronization.
- Synthetic data cannot establish operational performance. Validate on authorized, reviewed, independently split data before making claims.
- MAC addresses, SSIDs, channels, and timings can be sensitive or spoofed. Store reports securely and follow a defined retention policy.
- This is a portfolio/reference system, not an enterprise wireless IDS/IPS. Confirm alerts with network-controller and second-sensor evidence before incident response.


