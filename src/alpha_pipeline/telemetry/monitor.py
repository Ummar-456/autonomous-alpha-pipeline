"""
monitor.py — Telemetry & Evaluation Platform (SRE Layer).

Background asyncio task that:
  1. Polls orchestrator + graph runner for health metrics
  2. Exposes Prometheus text format on HTTP :8000/metrics
  3. Logs structured health summaries every heartbeat_interval_s
  4. Tracks business value (analyst-hours saved)
  5. Alerts on high drop-rate or queue backup
"""
from __future__ import annotations

import asyncio
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import TYPE_CHECKING, Optional

from .metrics import PipelineMetrics  # needed for _make_metrics_handler signature

from .business_value import BusinessValueTracker
from .metrics import PipelineMetrics
from ..config import TelemetryConfig

if TYPE_CHECKING:
    from ..ingestion.orchestrator import LOBOrchestrator
    from ..agents.graph import GraphRunner

logger = logging.getLogger(__name__)

def _make_metrics_handler(metrics: PipelineMetrics) -> type:
    """
    Factory that returns a _MetricsHandler class closing over a specific
    PipelineMetrics instance.

    FIX: The original code used a module-level _METRICS_REGISTRY global that
    was overwritten on every SREMonitor.__init__ call. If a second SREMonitor
    was instantiated (e.g. during a restart, test, or accidental double-init),
    the global pointed at the new empty instance while the original one (with
    real counts) kept running the heartbeat loop. Prometheus would then scrape
    all zeros. Using a closure per HTTP server instance eliminates the race
    entirely — each server is permanently bound to exactly one metrics object.
    """

    class _MetricsHandler(BaseHTTPRequestHandler):
        _metrics = metrics  # captured at server-creation time, never reassigned

        def do_GET(self) -> None:
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return

            try:
                body = self._metrics.prometheus_text().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                logger.error(f"Error serving metrics: {e}")
                self.send_response(500)
                self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            pass  # suppress access logs

    return _MetricsHandler


class SREMonitor:
    """
    Background SRE monitor. Runs as an asyncio.Task alongside orchestrator
    and graph runner. The Prometheus HTTP server runs in a daemon thread.
    """

    def __init__(
        self,
        config: Optional[TelemetryConfig] = None,
        alert_queue: Optional[asyncio.Queue] = None,
        orchestrator: Optional["LOBOrchestrator"] = None,
        # FIX: Accept graph_runner at construction time instead of patching
        # it in manually after the fact in main.py. This eliminates the
        # startup race where _collect_metrics could fire before the manual
        # cross-link was set.
        graph_runner: Optional["GraphRunner"] = None,
    ) -> None:
        self._config = config or TelemetryConfig()
        self._queue = alert_queue
        self._orchestrator = orchestrator
        self._graph_runner = graph_runner

        self.metrics = PipelineMetrics()
        self.business_value = BusinessValueTracker()
        self._is_running: bool = False
        self._http_thread: Optional[Thread] = None
        self._http_server: Optional[HTTPServer] = None
        # No global registry assignment — handler is bound via closure in
        # _start_http_server, so multiple SREMonitor instances can't stomp
        # each other's scrape target.

    async def run(self) -> None:
        """Main loop for the monitor task."""
        self._is_running = True
        self._start_http_server()

        logger.info(
            "SREMonitor started | prometheus=http://localhost:%d/metrics "
            "heartbeat_interval=%.0fs",
            self._config.prometheus_port,
            self._config.heartbeat_interval_s,
        )

        try:
            while self._is_running:
                await asyncio.sleep(self._config.heartbeat_interval_s)
                self._collect_metrics()
                self._log_health_summary()

        except asyncio.CancelledError:
            logger.info("SREMonitor shutting down.")
            raise
        finally:
            self._stop_http_server()
            self._is_running = False

    async def stop(self) -> None:
        """Stops the async loop."""
        self._is_running = False

    def record_graph_invocation(
        self,
        latency_ms: float,
        action: str,
        fallback_triggered: bool,
    ) -> None:
        """
        Callback invoked by GraphRunner after each completed pipeline run.

        This is the ONLY place graph_invocations, graph_latency_ms,
        llm_fallbacks, and analyst_hours_saved are incremented. The poll
        loop in _collect_metrics intentionally does NOT touch these counters
        to avoid the double-increment race.
        """
        self.metrics.graph_invocations.inc()
        self.metrics.graph_latency_ms.observe(latency_ms)

        if fallback_triggered:
            self.metrics.llm_fallbacks.inc()

        self.business_value.record_decision(action, fallback_triggered, latency_ms)

        self.metrics.analyst_hours_saved.inc(
            self._config.analyst_hours_per_alert
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _collect_metrics(self) -> None:
        """
        Sync orchestrator counters and queue depth into PipelineMetrics.

        FIX: graph_invocations is intentionally NOT synced here. It is
        managed exclusively via record_graph_invocation() (push path).
        Syncing it here too created a double-increment race: after the
        first heartbeat the Prometheus counter would be ahead of the
        runner's internal counter, making new_invokes permanently <= 0
        and occasionally double-counting on startup.
        """
        if self._orchestrator:
            # Sync packets received (delta from orchestrator's own counter)
            new_recv = self._orchestrator.packets_received - self.metrics.packets_received._value
            if new_recv > 0:
                self.metrics.packets_received.inc(new_recv)

            # Sync packets dropped
            new_drop = self._orchestrator.packets_dropped - self.metrics.packets_dropped._value
            if new_drop > 0:
                self.metrics.packets_dropped.inc(new_drop)

            # Sync alerts emitted
            new_alerts = self._orchestrator.alerts_emitted - self.metrics.alerts_emitted._value
            if new_alerts > 0:
                self.metrics.alerts_emitted.inc(new_alerts)

        if self._queue:
            self.metrics.queue_depth.set(float(self._queue.qsize()))

        if self._graph_runner:
            # p99 gauge is read-only from the runner; safe to poll
            p99 = self._graph_runner.p99_latency_ms()
            self.metrics.p99_latency_gauge.set(p99)
            # NOTE: graph_invocations delta sync removed — see docstring above

    def _log_health_summary(self) -> None:
        """Structured logging for terminal observability."""
        drop_rate = self.metrics.drop_rate()

        if drop_rate > 0.05:
            logger.warning("🚨 HIGH DROP RATE | drop_rate=%.2f%%", drop_rate * 100)

        current_queue_depth = self.metrics.queue_depth.value
        if current_queue_depth > 50:
            logger.warning("🚨 QUEUE BACKPRESSURE | depth=%d", int(current_queue_depth))

        p99 = self._graph_runner.p99_latency_ms() if self._graph_runner else 0.0
        hours_saved = self.business_value.total_hours_saved
        fte = self.business_value.annualized_fte_equivalent()

        logger.info(
            "SRE Heartbeat | "
            "uptime=%.0fs | "
            "recv=%d | "
            "drop=%d (%.1f%%) | "
            "alerts=%d | "
            "runs=%d | "
            "p99=%.1fms | "
            "q=%d | "
            "hours=%.2f | "
            "fte=%.4f",
            self.metrics.uptime_seconds(),
            int(self.metrics.packets_received.value),
            int(self.metrics.packets_dropped.value),
            drop_rate * 100,
            int(self.metrics.alerts_emitted.value),
            int(self.metrics.graph_invocations.value),
            p99,
            int(current_queue_depth),
            hours_saved,
            fte,
        )

    def _start_http_server(self) -> None:
        """Starts the Prometheus scrape endpoint in a background thread."""
        try:
            handler = _make_metrics_handler(self.metrics)
            server = HTTPServer(("0.0.0.0", self._config.prometheus_port), handler)
            self._http_server = server
            self._http_thread = Thread(target=server.serve_forever, daemon=True)
            self._http_thread.start()
            logger.info("Prometheus HTTP server live on :%d/metrics", self._config.prometheus_port)
        except OSError as exc:
            logger.error("Failed to start Prometheus HTTP server: %s", exc)

    def _stop_http_server(self) -> None:
        """Graceful shutdown of the HTTP thread."""
        if self._http_server:
            self._http_server.shutdown()
            self._http_server.server_close()
            logger.info("Prometheus HTTP server stopped.")