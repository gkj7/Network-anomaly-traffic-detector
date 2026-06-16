"""
dashboard.py
Live terminal UI using Rich — scrolling alert log, stats panel, top talkers.
"""

import time
from collections import deque, Counter
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn

console = Console()

MAX_ALERTS  = 14     # rows visible in alert log
MAX_HISTORY = 200    # internal ring-buffer size


class Dashboard:
    def __init__(self, detector):
        self.detector  = detector
        self.alerts    = deque(maxlen=MAX_HISTORY)
        self.all_flows = deque(maxlen=MAX_HISTORY)
        self.start_time = time.time()
        self.ip_counter  = Counter()

    # ── Feed methods (called from main thread) ──────────────────────────────

    def record_flow(self, flow, is_anomaly, score, severity):
        entry = {
            "ts":       time.strftime("%H:%M:%S"),
            "src":      flow.src_ip,
            "dst":      flow.dst_ip,
            "proto":    flow.proto,
            "dport":    flow.dst_port,
            "pkts":     len(flow.packets),
            "bytes":    sum(s for _, s in flow.packets),
            "score":    score,
            "anomaly":  is_anomaly,
            "severity": severity,
            "attack":   flow.attack_name,
        }
        self.all_flows.append(entry)
        self.ip_counter[flow.src_ip] += 1
        if is_anomaly:
            self.alerts.append(entry)

    # ── Render helpers ──────────────────────────────────────────────────────

    def _header(self) -> Panel:
        uptime = int(time.time() - self.start_time)
        h, m, s = uptime // 3600, (uptime % 3600) // 60, uptime % 60
        status = "[bold green]● ACTIVE[/]" if self.detector.trained else "[bold yellow]◌ WARMING UP[/]"
        title = Text("  AI-Based Network Anomaly Detection System  ", style="bold white on #1a1a2e")
        sub = Text(f"Isolation Forest  |  Status: {status}  |  Uptime: {h:02d}:{m:02d}:{s:02d}", style="dim")
        return Panel(
            Align.center(Text.assemble(title, "\n", sub)),
            style="bold #00d4ff",
            box=box.DOUBLE,
        )

    def _stats_panel(self) -> Panel:
        d = self.detector
        table = Table.grid(padding=(0, 4))
        table.add_column(style="dim cyan", justify="right")
        table.add_column(style="bold white", justify="left")
        table.add_column(style="dim cyan", justify="right")
        table.add_column(style="bold white", justify="left")

        warmup_bar = f"[{'█' * d.warmup_progress}{'░' * (d.WARMUP_FLOWS - d.warmup_progress)}]" if not d.trained else "[bold green]  Trained ✓"
        model_state = f"[yellow]{warmup_bar}[/]  ({d.warmup_progress}/{d.WARMUP_FLOWS} flows)" if not d.trained else "[bold green]Trained ✓  (IsolationForest n=150)[/]"

        anomaly_color = "red" if d.anomaly_rate > 15 else "yellow" if d.anomaly_rate > 5 else "green"

        table.add_row(
            "Total Flows",       f"[bold]{d.total_flows}[/]",
            "Anomaly Rate",      f"[bold {anomaly_color}]{d.anomaly_rate}%[/]",
        )
        table.add_row(
            "Normal",            f"[green]{d.total_normal}[/]",
            "Anomalies",         f"[red]{d.total_anomalies}[/]",
        )
        table.add_row(
            "Model",             model_state,
            "Alerts",            f"[bold red]{len(self.alerts)}[/]",
        )

        return Panel(table, title="[bold cyan]── System Stats ──[/]", border_style="#00d4ff", box=box.ROUNDED)

    def _alert_log(self) -> Panel:
        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold #00d4ff",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Time",     width=9,  style="dim")
        table.add_column("Severity", width=8)
        table.add_column("Attack Type",  width=18, style="bold")
        table.add_column("Source IP",    width=17)
        table.add_column("Destination",  width=17)
        table.add_column("Proto", width=6, style="cyan")
        table.add_column("Pkts",  width=6, justify="right")
        table.add_column("Bytes", width=9, justify="right")
        table.add_column("Score", width=8, justify="right")

        recent_alerts = list(self.alerts)[-MAX_ALERTS:]
        for e in reversed(recent_alerts):
            sev_style = "[bold red]▲ HIGH [/]" if e["severity"] == "HIGH" else "[bold yellow]▼ LOW  [/]"
            score_col = f"[red]{e['score']:.3f}[/]" if e["severity"] == "HIGH" else f"[yellow]{e['score']:.3f}[/]"
            table.add_row(
                e["ts"],
                sev_style,
                f"[bold red]{e['attack'] or 'Unknown'}[/]",
                f"[white]{e['src']}[/]",
                f"[dim]{e['dst']}[/]",
                e["proto"],
                str(e["pkts"]),
                f"{e['bytes']:,}",
                score_col,
            )

        # Pad empty rows
        for _ in range(MAX_ALERTS - len(recent_alerts)):
            table.add_row("", "", "─" * 14, "", "", "", "", "", "")

        count = len(self.alerts)
        title = f"[bold red]── 🚨 Alert Log  ({count} total) ──[/]" if count else "[bold cyan]── Alert Log (no anomalies yet) ──[/]"
        return Panel(table, title=title, border_style="red" if count else "#00d4ff", box=box.ROUNDED)

    def _recent_flows(self) -> Panel:
        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Time",  width=9,  style="dim")
        table.add_column("Src",   width=17)
        table.add_column("Dst",   width=17)
        table.add_column("Proto", width=6,  style="cyan")
        table.add_column("Port",  width=6,  justify="right")
        table.add_column("Pkts",  width=5,  justify="right")
        table.add_column("Status", width=10)

        recent = list(self.all_flows)[-8:]
        for e in reversed(recent):
            if e["anomaly"]:
                status = "[bold red]ANOMALY[/]"
                src_s  = f"[red]{e['src']}[/]"
            else:
                status = "[green]  OK   [/]"
                src_s  = f"[dim white]{e['src']}[/]"
            table.add_row(
                e["ts"], src_s, f"[dim]{e['dst']}[/]",
                e["proto"], str(e["dport"]),
                str(e["pkts"]), status,
            )

        return Panel(table, title="[bold cyan]── Recent Flows ──[/]", border_style="#00d4ff", box=box.ROUNDED)

    def _top_talkers(self) -> Panel:
        table = Table(box=box.SIMPLE_HEAD, show_header=True,
                      header_style="bold cyan", expand=True, padding=(0, 1))
        table.add_column("Source IP",  width=18)
        table.add_column("Flows",      width=8, justify="right")
        table.add_column("Share",      width=24)

        total = sum(self.ip_counter.values()) or 1
        for ip, cnt in self.ip_counter.most_common(6):
            pct   = cnt / total
            bar_w = int(pct * 20)
            bar   = f"[cyan]{'█' * bar_w}[/][dim]{'░' * (20 - bar_w)}[/] {pct*100:.1f}%"
            table.add_row(ip, str(cnt), bar)

        return Panel(table, title="[bold cyan]── Top Talkers ──[/]", border_style="#00d4ff", box=box.ROUNDED)

    def _footer(self) -> Panel:
        tips = "[dim]  Press [bold]Ctrl+C[/] to stop   |   Anomaly Score: closer to -1.0 = more suspicious   |   Model: Isolation Forest (sklearn)[/]"
        return Panel(Align.center(tips), style="dim", box=box.SIMPLE)

    # ── Main render ─────────────────────────────────────────────────────────

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(),       name="header",  size=5),
            Layout(self._stats_panel(),  name="stats",   size=7),
            Layout(self._alert_log(),    name="alerts",  size=MAX_ALERTS + 5),
            Layout(name="bottom",        size=14),
            Layout(self._footer(),       name="footer",  size=3),
        )
        layout["bottom"].split_row(
            Layout(self._recent_flows(),  name="flows",   ratio=3),
            Layout(self._top_talkers(),   name="talkers", ratio=2),
        )
        return layout
