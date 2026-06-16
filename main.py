"""
main.py
Entry point — spawns simulator threads and runs the live Rich dashboard.
"""

import time
import random
import threading
from rich.live import Live
from rich.console import Console

from detector import AnomalyDetector, generate_normal_flow, generate_attack_flow
from dashboard import Dashboard

console = Console()

# ─── Config ──────────────────────────────────────────────────────────────────

NORMAL_WORKERS   = 4      # concurrent threads generating normal traffic
ATTACK_INTERVAL  = (8, 20)  # seconds between injected attack flows
REFRESH_RATE     = 2      # dashboard redraws per second

# ─── Worker threads ───────────────────────────────────────────────────────────

def normal_traffic_worker(detector: AnomalyDetector, dash: Dashboard, stop_event: threading.Event):
    """Continuously generates normal flows and scores them."""
    while not stop_event.is_set():
        try:
            flow              = generate_normal_flow()
            is_anom, score, sev = detector.score_flow(flow)
            if is_anom is not None:
                dash.record_flow(flow, is_anom, score, sev)
        except Exception as e:
            pass  # keep thread alive


def attack_injector(detector: AnomalyDetector, dash: Dashboard, stop_event: threading.Event):
    """Periodically injects an attack flow to trigger anomaly alerts."""
    # Wait until model is warmed up before starting attacks
    while not detector.trained and not stop_event.is_set():
        time.sleep(1)

    time.sleep(3)   # brief quiet period after training

    while not stop_event.is_set():
        wait = random.uniform(*ATTACK_INTERVAL)
        # Sleep in small chunks so we can exit cleanly
        for _ in range(int(wait * 10)):
            if stop_event.is_set():
                return
            time.sleep(0.1)

        try:
            flow              = generate_attack_flow()
            is_anom, score, sev = detector.score_flow(flow)
            if is_anom is not None:
                dash.record_flow(flow, is_anom, score, sev)
        except Exception as e:
            pass


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    console.clear()
    console.print("\n[bold #00d4ff]  AI-Based Network Anomaly Detection System[/]")
    console.print("[dim]  Initializing Isolation Forest engine…[/]\n")
    time.sleep(0.8)

    detector   = AnomalyDetector()
    dash       = Dashboard(detector)
    stop_event = threading.Event()

    # Spawn normal-traffic workers
    threads = []
    for _ in range(NORMAL_WORKERS):
        t = threading.Thread(target=normal_traffic_worker, args=(detector, dash, stop_event), daemon=True)
        t.start()
        threads.append(t)

    # Spawn attack injector
    atk = threading.Thread(target=attack_injector, args=(detector, dash, stop_event), daemon=True)
    atk.start()
    threads.append(atk)

    console.print("[green]  ✓ Workers started.[/]  [dim]Warming up model — processing first 40 flows…[/]\n")
    time.sleep(1.0)

    try:
        with Live(
            dash.render(),
            console=console,
            refresh_per_second=REFRESH_RATE,
            screen=True,
        ) as live:
            while True:
                time.sleep(1 / REFRESH_RATE)
                live.update(dash.render())

    except KeyboardInterrupt:
        stop_event.set()
        console.clear()
        console.print("\n[bold #00d4ff]  Session Summary[/]")
        console.print(f"  Total flows processed : [bold]{detector.total_flows}[/]")
        console.print(f"  Normal flows          : [green]{detector.total_normal}[/]")
        console.print(f"  Anomalies detected    : [red]{detector.total_anomalies}[/]")
        console.print(f"  Anomaly rate          : [yellow]{detector.anomaly_rate}%[/]")
        console.print(f"  Total alerts fired    : [bold red]{len(dash.alerts)}[/]\n")
        console.print("[dim]  Goodbye.[/]\n")


if __name__ == "__main__":
    main()
