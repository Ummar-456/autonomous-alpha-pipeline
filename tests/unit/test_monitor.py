"""Unit tests for SREMonitor and its HTTP metrics endpoint — 9 tests."""
from __future__ import annotations

import asyncio
import socket
import urllib.request

import pytest

from alpha_pipeline.config import TelemetryConfig
from alpha_pipeline.telemetry.monitor import SREMonitor


def _free_port() -> int:
    """Find an available TCP port for the test HTTP server."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestSREMonitorMetricCollection:
    def test_record_graph_invocation_increments_metrics(self):
        config = TelemetryConfig(prometheus_port=_free_port())
        monitor = SREMonitor(config=config)
        monitor.record_graph_invocation(latency_ms=120.0, action="CANCEL_ALL", fallback_triggered=False)
        assert monitor.metrics.graph_invocations.value == 1
        assert monitor.metrics.graph_latency_ms.count == 1

    def test_fallback_increments_fallback_counter(self):
        config = TelemetryConfig(prometheus_port=_free_port())
        monitor = SREMonitor(config=config)
        monitor.record_graph_invocation(latency_ms=80.0, action="CANCEL_ALL", fallback_triggered=True)
        assert monitor.metrics.llm_fallbacks.value == 1

    def test_non_fallback_does_not_increment_fallback_counter(self):
        config = TelemetryConfig(prometheus_port=_free_port())
        monitor = SREMonitor(config=config)
        monitor.record_graph_invocation(latency_ms=80.0, action="HOLD", fallback_triggered=False)
        assert monitor.metrics.llm_fallbacks.value == 0

    def test_business_value_recorded_per_invocation(self):
        config = TelemetryConfig(prometheus_port=_free_port(), analyst_hours_per_alert=0.5)
        monitor = SREMonitor(config=config)
        monitor.record_graph_invocation(latency_ms=100.0, action="CANCEL_ALL", fallback_triggered=False)
        assert monitor.metrics.analyst_hours_saved.value == 0.5

    def test_multiple_invocations_accumulate(self):
        config = TelemetryConfig(prometheus_port=_free_port(), analyst_hours_per_alert=0.25)
        monitor = SREMonitor(config=config)
        for _ in range(4):
            monitor.record_graph_invocation(latency_ms=50.0, action="HOLD", fallback_triggered=False)
        assert monitor.metrics.analyst_hours_saved.value == 1.0
        assert monitor.metrics.graph_invocations.value == 4


class TestSREMonitorOrchestatorSync:
    def test_collect_metrics_syncs_orchestrator_counters(self):
        from unittest.mock import MagicMock
        config = TelemetryConfig(prometheus_port=_free_port())
        mock_orch = MagicMock()
        mock_orch.packets_received = 1000
        mock_orch.packets_dropped = 15
        mock_orch.alerts_emitted = 42
        monitor = SREMonitor(config=config, orchestrator=mock_orch)
        monitor._collect_metrics()
        assert monitor.metrics.packets_received.value == 1000.0
        assert monitor.metrics.packets_dropped.value == 15.0
        assert monitor.metrics.alerts_emitted.value == 42.0

    def test_collect_metrics_syncs_queue_depth(self):
        config = TelemetryConfig(prometheus_port=_free_port())
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        for _ in range(7):
            queue.put_nowait("item")
        monitor = SREMonitor(config=config, alert_queue=queue)
        monitor._collect_metrics()
        assert monitor.metrics.queue_depth.value == 7.0


class TestSREMonitorHTTP:
    def test_prometheus_http_server_serves_metrics(self):
        port = _free_port()
        config = TelemetryConfig(prometheus_port=port)
        monitor = SREMonitor(config=config)
        monitor._start_http_server()
        import time; time.sleep(0.05)  # let the daemon thread bind

        try:
            monitor.record_graph_invocation(latency_ms=150.0, action="CANCEL_ALL", fallback_triggered=False)
            resp = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=2)
            body = resp.read().decode()
            assert "pipeline_graph_invocations_total" in body
            assert "pipeline_analyst_hours_saved_total" in body
            assert resp.getcode() == 200
        finally:
            monitor._stop_http_server()

    def test_prometheus_http_server_404_for_non_metrics_path(self):
        port = _free_port()
        config = TelemetryConfig(prometheus_port=port)
        monitor = SREMonitor(config=config)
        monitor._start_http_server()
        import time; time.sleep(0.05)

        try:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"http://localhost:{port}/unknown", timeout=2)
            assert exc_info.value.code == 404
        finally:
            monitor._stop_http_server()
