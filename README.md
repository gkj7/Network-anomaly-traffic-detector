# Network Anomaly Traffic Detector

A real-time network anomaly detection system built with Python and Machine Learning.

## What it does
Monitors network traffic, extracts flow-based features, and automatically flags suspicious activity using an Isolation Forest ML model — no labeled attack data needed.

## Attack types detected
- Port Scan
- SYN Flood
- UDP Flood
- ICMP Ping Flood
- Data Exfiltration

## Tech used
- Python
- Scikit-learn (Isolation Forest)
- Rich (terminal dashboard)

## How to run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```
