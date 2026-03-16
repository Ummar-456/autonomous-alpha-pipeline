"""Unit tests for telemetry — PipelineMetrics and BusinessValueTracker — 10 tests."""
from __future__ import annotations

import time

import pytest

from alpha_pipeline.telemetry.business_value import BusinessValueTracker
from alpha_pipeline.telemetry.metrics import PipelineMetrics, _Counter, _Gauge, _Histogram


class TestPrimitiveMetrics:
    def test_counter_increments(self):
        c = _Counter("test", "desc")
        c.inc(3.0)
        c.inc(2.0)
        assert c.value == 5.0

    def test_counter_default_increment_is_one(self):
        c = _Counter("test", "desc")
        c.inc()
        assert c.value == 1.0

    def test_gauge_set_and_inc(self):
        g = _Gauge("test", "desc")
        g.set(10.0)
        g.inc(2.0)
        assert g.value == 12.0

    def test_gauge_dec(self):
        g = _Gauge("test", "desc")
        g.set(10.0)
        g.dec(3.0)
        assert g.value == 7.0

    def test_histogram_count_and_sum(self):
        h = _Histogram("test", "desc")
        h.observe(100.0)
        h.observe(200.0)
        assert h.count == 2
        assert h.sum == 300.0

    def test_histogram_quantile_reasonable(self):
        h = _Histogram("test", "desc", buckets=(10, 50, 100, 200, float("inf")))
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            h.observe(float(v))
        p50 = h.quantile(0.5)
        assert p50 >= 50.0  # at least median


class TestPipelineMetrics:
    def test_drop_rate_zero_when_no_packets(self):
        m = PipelineMetrics()
        assert m.drop_rate() == 0.0

    def test_drop_rate_calculated_correctly(self):
        m = PipelineMetrics()
        m.packets_received.inc(100)
        m.packets_dropped.inc(10)
        assert abs(m.drop_rate() - 0.10) < 0.001

    def test_prometheus_text_contains_metric_names(self):
        m = PipelineMetrics()
        m.packets_received.inc(5)
        text = m.prometheus_text()
        assert "pipeline_packets_received_total" in text
        assert "pipeline_graph_latency_ms" in text
        assert "pipeline_analyst_hours_saved_total" in text

    def test_uptime_increases(self):
        m = PipelineMetrics()
        t0 = m.uptime_seconds()
        time.sleep(0.05)
        assert m.uptime_seconds() > t0


class TestBusinessValueTracker:
    def test_hours_accumulated_per_action(self):
        tracker = BusinessValueTracker()
        tracker.record_decision("CANCEL_ALL", False, 120.0)
        assert tracker.total_hours_saved == BusinessValueTracker.HOURS_PER_CANCEL_ALL

    def test_fallback_decision_counts_conservatively(self):
        tracker = BusinessValueTracker()
        tracker.record_decision("CANCEL_ALL", True, 50.0)
        assert tracker.total_hours_saved == BusinessValueTracker.HOURS_PER_FALLBACK

    def test_fallback_rate_calculation(self):
        tracker = BusinessValueTracker()
        tracker.record_decision("CANCEL_ALL", True, 100.0)
        tracker.record_decision("HOLD", False, 80.0)
        tracker.record_decision("PROVIDE_LIQUIDITY", False, 90.0)
        assert abs(tracker.fallback_rate - 1 / 3) < 0.01

    def test_daily_summary_present(self):
        tracker = BusinessValueTracker()
        tracker.record_decision("WIDEN_QUOTES", False, 150.0)
        summary = tracker.daily_summary()
        assert summary is not None
        assert summary.alerts_processed == 1

    def test_fte_equivalent_positive_after_records(self):
        tracker = BusinessValueTracker()
        for _ in range(10):
            tracker.record_decision("CANCEL_ALL", False, 100.0)
        assert tracker.annualized_fte_equivalent() > 0
