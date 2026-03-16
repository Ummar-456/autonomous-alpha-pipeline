"""Unit tests for Pydantic state models — 10 tests."""
from __future__ import annotations

import time
import uuid
from decimal import Decimal

import pytest
from pydantic import ValidationError

from alpha_pipeline.state import (
    ExecutionDecision,
    MarketAlert,
    MicrostructureMetrics,
    OrderBookSnapshot,
    PipelineState,
    PriceLevel,
    ResearchContext,
    ResearchDocument,
)


class TestPriceLevel:
    def test_valid_construction(self):
        pl = PriceLevel(price=Decimal("100.0"), quantity=Decimal("2.5"))
        assert pl.price == Decimal("100.0")

    def test_frozen_immutable(self):
        pl = PriceLevel(price=Decimal("100.0"), quantity=Decimal("1.0"))
        with pytest.raises(Exception):
            pl.price = Decimal("200.0")  # type: ignore


class TestMicrostructureMetrics:
    def test_valid_construction(self, sample_metrics_normal):
        assert sample_metrics_normal.obi == 0.1
        assert sample_metrics_normal.toxic_flow_detected is False

    def test_vpin_rejects_out_of_bounds(self):
        with pytest.raises(ValidationError):
            MicrostructureMetrics(
                obi=0.0, spread_bps=1.0, mid_price=Decimal("100"),
                vpin=1.5,  # > 1.0 — invalid
                toxic_flow_detected=False, volatility_spike_detected=False,
            )

    def test_frozen(self, sample_metrics_normal):
        with pytest.raises(Exception):
            sample_metrics_normal.obi = 0.9  # type: ignore


class TestMarketAlert:
    def test_valid_construction(self, toxic_alert):
        assert toxic_alert.trigger == "TOXIC_FLOW"
        assert toxic_alert.severity == "HIGH"

    def test_invalid_trigger_rejected(self, sample_metrics_toxic, sample_snapshot):
        with pytest.raises(ValidationError):
            MarketAlert(
                alert_id=str(uuid.uuid4()),
                symbol="BTCUSDT",
                severity="HIGH",
                trigger="INVALID_TRIGGER",  # not in Literal
                snapshot=sample_snapshot,
                metrics=sample_metrics_toxic,
                created_at_ns=time.time_ns(),
                pipeline_latency_ms=1.0,
            )

    def test_invalid_severity_rejected(self, sample_metrics_toxic, sample_snapshot):
        with pytest.raises(ValidationError):
            MarketAlert(
                alert_id=str(uuid.uuid4()),
                symbol="BTCUSDT",
                severity="EXTREME",   # not in Literal
                trigger="TOXIC_FLOW",
                snapshot=sample_snapshot,
                metrics=sample_metrics_toxic,
                created_at_ns=time.time_ns(),
                pipeline_latency_ms=1.0,
            )


class TestExecutionDecision:
    def test_valid_cancel_all(self):
        d = ExecutionDecision(
            action="CANCEL_ALL",
            side="NONE",
            confidence=0.99,
            reasoning="Toxic flow",
            fallback_triggered=False,
            decision_latency_ms=120.0,
        )
        assert d.action == "CANCEL_ALL"

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ExecutionDecision(
                action="HOLD",
                side="NONE",
                confidence=1.5,  # > 1.0 — invalid
                reasoning="Test",
                fallback_triggered=False,
                decision_latency_ms=100.0,
            )

    def test_reasoning_max_length_enforced(self):
        with pytest.raises(ValidationError):
            ExecutionDecision(
                action="HOLD",
                side="NONE",
                confidence=0.5,
                reasoning="x" * 2049,  # exceeds max_length=2048
                fallback_triggered=False,
                decision_latency_ms=100.0,
            )

    def test_invalid_action_rejected(self):
        with pytest.raises(ValidationError):
            ExecutionDecision(
                action="DO_NOTHING",  # not in Literal
                side="NONE",
                confidence=0.5,
                reasoning="Test",
                fallback_triggered=False,
                decision_latency_ms=100.0,
            )
