"""
detector.py
Core engine: traffic simulation, flow feature extraction, Isolation Forest model.
"""

import time
import random
import threading
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.ensemble import IsolationForest

# ─── Data Structures ────────────────────────────────────────────────────────

PROTOCOLS = ["TCP", "UDP", "ICMP"]

NORMAL_PROFILES = [
    {"name": "HTTP",    "proto": "TCP",  "dport": 80,   "size_range": (200, 1500), "rate": 0.05},
    {"name": "HTTPS",   "proto": "TCP",  "dport": 443,  "size_range": (300, 1400), "rate": 0.05},
    {"name": "DNS",     "proto": "UDP",  "dport": 53,   "size_range": (60,  200),  "rate": 0.08},
    {"name": "SSH",     "proto": "TCP",  "dport": 22,   "size_range": (100, 600),  "rate": 0.10},
    {"name": "NTP",     "proto": "UDP",  "dport": 123,  "size_range": (48,  80),   "rate": 0.12},
    {"name": "SMTP",    "proto": "TCP",  "dport": 25,   "size_range": (200, 800),  "rate": 0.15},
    {"name": "FTP",     "proto": "TCP",  "dport": 21,   "size_range": (100, 500),  "rate": 0.20},
]

ATTACK_PROFILES = [
    {
        "name": "Port Scan",
        "proto": "TCP",
        "dport_range": (1, 65535),
        "size_range": (40, 80),
        "pkt_count": (50, 200),
        "duration": (0.5, 2.0),
    },
    {
        "name": "UDP Flood",
        "proto": "UDP",
        "dport_range": (1, 65535),
        "size_range": (512, 1500),
        "pkt_count": (200, 800),
        "duration": (0.5, 3.0),
    },
    {
        "name": "SYN Flood",
        "proto": "TCP",
        "dport_range": (80, 443),
        "size_range": (40, 60),
        "pkt_count": (300, 1000),
        "duration": (1.0, 5.0),
    },
    {
        "name": "ICMP Ping Flood",
        "proto": "ICMP",
        "dport_range": (0, 0),
        "size_range": (64, 128),
        "pkt_count": (100, 500),
        "duration": (1.0, 4.0),
    },
    {
        "name": "Data Exfiltration",
        "proto": "TCP",
        "dport_range": (443, 443),
        "size_range": (1400, 1500),
        "pkt_count": (150, 400),
        "duration": (2.0, 8.0),
    },
]


@dataclass
class Flow:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: str
    start_time: float
    packets: list = field(default_factory=list)   # list of (timestamp, size)
    label: str = "NORMAL"
    attack_name: str = ""

    def add_packet(self, size: int):
        self.packets.append((time.time(), size))

    def to_features(self):
        """Extract 8 numeric features from the flow."""
        if not self.packets:
            return None
        timestamps, sizes = zip(*self.packets)
        duration   = max(timestamps) - min(timestamps) + 1e-6
        pkt_count  = len(self.packets)
        byte_total = sum(sizes)
        avg_size   = byte_total / pkt_count
        pps        = pkt_count / duration          # packets per second
        bps        = byte_total / duration         # bytes per second
        size_std   = float(np.std(sizes)) if pkt_count > 1 else 0.0
        proto_num  = {"TCP": 0, "UDP": 1, "ICMP": 2}.get(self.proto, 3)

        return [pkt_count, byte_total, avg_size, duration, pps, bps, size_std, proto_num]


# ─── Traffic Simulator ───────────────────────────────────────────────────────

def _random_ip(private=True):
    if private:
        return f"192.168.{random.randint(1,10)}.{random.randint(2,254)}"
    return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def generate_normal_flow() -> Flow:
    profile = random.choice(NORMAL_PROFILES)
    src = _random_ip(private=True)
    dst = _random_ip(private=random.random() < 0.6)
    flow = Flow(
        src_ip=src, dst_ip=dst,
        src_port=random.randint(1024, 65535),
        dst_port=profile["dport"],
        proto=profile["proto"],
        start_time=time.time(),
        label="NORMAL",
    )
    # Simulate realistic packet bursts
    n_packets = int(random.expovariate(1 / 12)) + 1
    delay = profile["rate"]
    for _ in range(n_packets):
        size = random.randint(*profile["size_range"])
        flow.add_packet(size)
        time.sleep(delay * random.uniform(0.5, 1.5))
    return flow


def generate_attack_flow() -> Flow:
    profile = random.choice(ATTACK_PROFILES)
    src = _random_ip(private=random.random() < 0.3)   # often external
    dst = _random_ip(private=True)
    dport = random.randint(*profile["dport_range"]) if profile["dport_range"][0] != profile["dport_range"][1] else profile["dport_range"][0]
    flow = Flow(
        src_ip=src, dst_ip=dst,
        src_port=random.randint(1024, 65535),
        dst_port=dport,
        proto=profile["proto"],
        start_time=time.time(),
        label="ANOMALY",
        attack_name=profile["name"],
    )
    n_packets = random.randint(*profile["pkt_count"])
    duration  = random.uniform(*profile["duration"])
    delay     = duration / n_packets
    for _ in range(n_packets):
        size = random.randint(*profile["size_range"])
        flow.add_packet(size)
        time.sleep(delay)
    return flow


# ─── Anomaly Detector ────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Wraps sklearn IsolationForest.
    Phase 1 – warm-up: collect WARMUP_FLOWS flows to train the model.
    Phase 2 – detection: score every incoming flow in real time.
    """

    WARMUP_FLOWS = 40          # flows before model is trained
    CONTAMINATION = 0.08       # expected anomaly fraction

    def __init__(self):
        self.model      = None
        self.trained    = False
        self.warmup_buf = []   # raw feature rows collected during warm-up
        self.lock       = threading.Lock()

        # Stats
        self.total_flows    = 0
        self.total_anomalies = 0
        self.total_normal   = 0

    def _train(self, X: np.ndarray):
        self.model = IsolationForest(
            n_estimators=150,
            contamination=self.CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X)
        self.trained = True

    def score_flow(self, flow: Flow):
        """
        Returns (is_anomaly: bool, score: float, severity: str)
        score is in [-1, 0]: more negative → more anomalous
        """
        features = flow.to_features()
        if features is None:
            return False, 0.0, "NONE"

        with self.lock:
            self.total_flows += 1

            if not self.trained:
                self.warmup_buf.append(features)
                if len(self.warmup_buf) >= self.WARMUP_FLOWS:
                    X = np.array(self.warmup_buf)
                    self._train(X)
                # During warm-up, return no verdict
                return None, 0.0, "WARMING"

            X      = np.array([features])
            pred   = self.model.predict(X)[0]           # 1 = normal, -1 = anomaly
            score  = self.model.score_samples(X)[0]     # lower = more anomalous

            is_anomaly = pred == -1
            if is_anomaly:
                self.total_anomalies += 1
                severity = "HIGH" if score < -0.3 else "LOW"
            else:
                self.total_normal += 1
                severity = "OK"

            return is_anomaly, round(float(score), 4), severity

    @property
    def warmup_progress(self):
        return min(len(self.warmup_buf), self.WARMUP_FLOWS)

    @property
    def anomaly_rate(self):
        if self.total_flows == 0:
            return 0.0
        return round(self.total_anomalies / self.total_flows * 100, 1)
