"""
metrics.py — Mocked Prometheus metrics for the SRE telemetry layer.

In production these would be real prometheus_client Counters/Histograms
registered against a CollectorRegistry. Here we wrap them in a thin
abstraction so the rest of the codebase never imports prometheus_client
directly — swap the backend without touching callers.
"""
from __future__ import annotations

import threading
import time
from typing import Optional


class _Counter:
    """Thread-safe monotonically increasing counter."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class _Histogram:
    """Thread-safe histogram with configurable buckets."""

    def __init__(
        self,
        name: str,
        description: str,
        buckets: tuple[float, ...] = (10, 50, 100, 200, 350, 500, 750, 1000, 2000, float("inf")),
    ) -> None:
        self.name = name
        self.description = description
        self._buckets = buckets
        self._counts: dict[float, int] = {b: 0 for b in buckets}
        self._sum: float = 0.0
        self._total_count: int = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._total_count += 1
            for bucket in self._buckets:
                if value <= bucket:
                    self._counts[bucket] += 1

    @property
    def sum(self) -> float:
        with self._lock:
            return self._sum

    @property
    def count(self) -> int:
        with self._lock:
            return self._total_count


class _Gauge:
    """Thread-safe gauge (can go up and down)."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class PipelineMetrics:
    """
    Central metrics registry for the Autonomous Alpha Pipeline.
    All metrics are exposed as properties — the SRE monitor reads them
    and formats them as Prometheus text exposition for Grafana scraping.
    """

    def __init__(self) -> None:
        # Ingestion
        self.packets_received = _Counter(
            "pipeline_packets_received_total",
            "Total WebSocket depth update messages received.",
        )
        self.packets_dropped = _Counter(
            "pipeline_packets_dropped_total",
            "Total messages dropped (parse errors, sequence gaps, queue full).",
        )
        self.alerts_emitted = _Counter(
            "pipeline_alerts_emitted_total",
            "Total MarketAlert objects enqueued for graph processing.",
        )

        # Graph
        self.graph_invocations = _Counter(
            "pipeline_graph_invocations_total",
            "Total LangGraph pipeline invocations.",
        )
        self.graph_latency_ms = _Histogram(
            "pipeline_graph_latency_ms",
            "End-to-end graph latency in milliseconds.",
        )
        self.p99_latency_gauge = _Gauge(
            "pipeline_graph_p99_latency_ms",
            "Current 99th percentile end-to-end graph latency.",
        )
        self.llm_fallbacks = _Counter(
            "pipeline_llm_fallbacks_total",
            "Total LLM calls that triggered the CANCEL_ALL fallback.",
        )

        # System health
        self.queue_depth = _Gauge(
            "pipeline_alert_queue_depth",
            "Current number of alerts waiting in the async queue.",
        )
        self.ws_reconnects = _Counter(
            "pipeline_ws_reconnects_total",
            "Total WebSocket reconnection attempts.",
        )
        self.circuit_breaker_opens = _Counter(
            "pipeline_circuit_breaker_opens_total",
            "Total times the LLM circuit breaker transitioned to OPEN.",
        )

        # Business value
        self.analyst_hours_saved = _Counter(
            "pipeline_analyst_hours_saved_total",
            "Estimated analyst-hours saved by automated alert processing.",
        )
        self._started_at: float = time.time()

    def uptime_seconds(self) -> float:
        return time.time() - self._started_at

    def drop_rate(self) -> float:
        received = self.packets_received.value
        if received == 0:
            return 0.0
        return self.packets_dropped.value / received

    def prometheus_text(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines: list[str] = []

        def _counter(metric: _Counter) -> None:
            lines.append(f"# HELP {metric.name} {metric.description}")
            lines.append(f"# TYPE {metric.name} counter")
            lines.append(f"{metric.name} {metric.value}")

        def _gauge(metric: _Gauge) -> None:
            lines.append(f"# HELP {metric.name} {metric.description}")
            lines.append(f"# TYPE {metric.name} gauge")
            lines.append(f"{metric.name} {metric.value}")

        def _hist(metric: _Histogram) -> None:
            lines.append(f"# HELP {metric.name} {metric.description}")
            lines.append(f"# TYPE {metric.name} histogram")
            for bound, count in sorted(metric._counts.items()):
                b = "+Inf" if bound == float("inf") else str(bound)
                lines.append(f'{metric.name}_bucket{{le="{b}"}} {count}')
            lines.append(f"{metric.name}_sum {metric.sum}")
            lines.append(f"{metric.name}_count {metric.count}")

        # Ingestion
        _counter(self.packets_received)
        _counter(self.packets_dropped)
        _counter(self.alerts_emitted)

        # Graph
        _counter(self.graph_invocations)
        _hist(self.graph_latency_ms)
        _gauge(self.p99_latency_gauge)
        _counter(self.llm_fallbacks)

        # Health
        _gauge(self.queue_depth)
        _counter(self.ws_reconnects)
        _counter(self.circuit_breaker_opens)

        # Value
        _counter(self.analyst_hours_saved)

        return "\n".join(lines) + "\n"