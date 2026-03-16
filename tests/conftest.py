"""conftest.py — Shared pytest fixtures for the Autonomous Alpha Pipeline test suite."""
from __future__ import annotations

import asyncio
import time
import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from alpha_pipeline.config import DecisionAgentConfig, OrchestratorConfig
from alpha_pipeline.ingestion.lob import LimitOrderBook
from alpha_pipeline.ingestion.metrics import MetricsEngine
from alpha_pipeline.rag.vector_store import MockVectorStore
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


# ── LOB Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def empty_lob() -> LimitOrderBook:
    return LimitOrderBook("BTCUSDT", depth=20)


@pytest.fixture
def populated_lob() -> LimitOrderBook:
    lob = LimitOrderBook("BTCUSDT", depth=20)
    base = Decimal("43500.00")
    bids = [[str(base - Decimal(str(i)) * Decimal("0.5")), "1.0"] for i in range(10)]
    asks = [[str(base + Decimal("1.0") + Decimal(str(i)) * Decimal("0.5")), "0.9"] for i in range(10)]
    lob.apply_snapshot(bids, asks, last_update_id=1_000_000)
    return lob


@pytest.fixture
def imbalanced_lob_bid_heavy() -> LimitOrderBook:
    """LOB with heavy bid-side volume → OBI near +1."""
    lob = LimitOrderBook("BTCUSDT", depth=20)
    bids = [[str(Decimal("43500") - Decimal(str(i)) * Decimal("0.5")), "10.0"] for i in range(10)]
    asks = [[str(Decimal("43501") + Decimal(str(i)) * Decimal("0.5")), "0.1"] for i in range(10)]
    lob.apply_snapshot(bids, asks, last_update_id=1_000_001)
    return lob


@pytest.fixture
def imbalanced_lob_ask_heavy() -> LimitOrderBook:
    """LOB with heavy ask-side volume → OBI near -1."""
    lob = LimitOrderBook("BTCUSDT", depth=20)
    bids = [[str(Decimal("43500") - Decimal(str(i)) * Decimal("0.5")), "0.1"] for i in range(10)]
    asks = [[str(Decimal("43501") + Decimal(str(i)) * Decimal("0.5")), "10.0"] for i in range(10)]
    lob.apply_snapshot(bids, asks, last_update_id=1_000_002)
    return lob


# ── Metrics Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def metrics_engine() -> MetricsEngine:
    return MetricsEngine(
        obi_levels=5,
        vpin_bucket_size=Decimal("5"),
        vpin_window=20,
        spread_window=20,
        obi_toxicity_threshold=0.7,
        vpin_toxicity_threshold=0.5,
        spread_z_threshold=2.0,
    )


# ── State Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_metrics_normal() -> MicrostructureMetrics:
    return MicrostructureMetrics(
        obi=0.1,
        spread_bps=0.8,
        mid_price=Decimal("43500.5"),
        vpin=0.2,
        toxic_flow_detected=False,
        volatility_spike_detected=False,
    )


@pytest.fixture
def sample_metrics_toxic() -> MicrostructureMetrics:
    return MicrostructureMetrics(
        obi=0.85,
        spread_bps=2.1,
        mid_price=Decimal("43500.5"),
        vpin=0.72,
        toxic_flow_detected=True,
        volatility_spike_detected=False,
    )


@pytest.fixture
def sample_metrics_volatile() -> MicrostructureMetrics:
    return MicrostructureMetrics(
        obi=0.05,
        spread_bps=5.2,
        mid_price=Decimal("43500.5"),
        vpin=0.3,
        toxic_flow_detected=False,
        volatility_spike_detected=True,
    )


@pytest.fixture
def sample_snapshot() -> OrderBookSnapshot:
    bids = [PriceLevel(price=Decimal("43500"), quantity=Decimal("1.0"))]
    asks = [PriceLevel(price=Decimal("43501"), quantity=Decimal("0.9"))]
    return OrderBookSnapshot(
        symbol="BTCUSDT",
        timestamp_ns=time.time_ns(),
        bids=bids,
        asks=asks,
        last_update_id=1_000_000,
    )


def make_alert(
    metrics: MicrostructureMetrics,
    snapshot: OrderBookSnapshot,
    trigger: str = "TOXIC_FLOW",
    severity: str = "HIGH",
) -> MarketAlert:
    return MarketAlert(
        alert_id=str(uuid.uuid4()),
        symbol="BTCUSDT",
        severity=severity,          # type: ignore[arg-type]
        trigger=trigger,            # type: ignore[arg-type]
        snapshot=snapshot,
        metrics=metrics,
        created_at_ns=time.time_ns(),
        pipeline_latency_ms=1.5,
    )


@pytest.fixture
def toxic_alert(sample_metrics_toxic, sample_snapshot) -> MarketAlert:
    return make_alert(sample_metrics_toxic, sample_snapshot, "TOXIC_FLOW", "HIGH")


@pytest.fixture
def volatile_alert(sample_metrics_volatile, sample_snapshot) -> MarketAlert:
    return make_alert(sample_metrics_volatile, sample_snapshot, "VOLATILITY_SPIKE", "MEDIUM")


@pytest.fixture
def critical_alert(sample_snapshot) -> MarketAlert:
    metrics = MicrostructureMetrics(
        obi=0.92,
        spread_bps=6.1,
        mid_price=Decimal("43500.5"),
        vpin=0.81,
        toxic_flow_detected=True,
        volatility_spike_detected=True,
    )
    return make_alert(metrics, sample_snapshot, "BOTH", "CRITICAL")


@pytest.fixture
def sample_research_context() -> ResearchContext:
    docs = [
        ResearchDocument(
            doc_id="doc_001",
            title="OBI as Price Predictor",
            content="High OBI signals upward price movement.",
            similarity_score=0.92,
            source="Test_Source",
        )
    ]
    return ResearchContext(
        query_embedding_latency_ms=3.2,
        retrieved_docs=docs,
        synthesized_context="High OBI + VPIN → CANCEL_ALL recommended.",
    )


@pytest.fixture
def pipeline_state(toxic_alert, sample_research_context) -> PipelineState:
    return {
        "alert": toxic_alert,
        "research_context": sample_research_context,
        "decision": None,
        "audit_log": [],
        "should_escalate": True,
        "total_pipeline_latency_ms": None,
    }


# ── Mock LLM Client ───────────────────────────────────────────────────────────

class MockLLMClient:
    """Synchronous mock LLM that returns configurable JSON responses."""

    def __init__(self, response: str | None = None, raise_exc: Exception | None = None, delay_s: float = 0.0) -> None:
        self._response = response or '{"action":"CANCEL_ALL","side":"NONE","confidence":0.95,"reasoning":"Test"}'
        self._raise = raise_exc
        self._delay = delay_s
        self.call_count = 0

    async def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._raise:
            raise self._raise
        return self._response


@pytest.fixture
def mock_llm_cancel_all() -> MockLLMClient:
    return MockLLMClient(
        '{"action":"CANCEL_ALL","side":"NONE","confidence":0.97,"reasoning":"Toxic flow detected"}'
    )


@pytest.fixture
def mock_llm_provide_liquidity() -> MockLLMClient:
    return MockLLMClient(
        '{"action":"PROVIDE_LIQUIDITY","side":"BOTH","confidence":0.81,"reasoning":"Normal conditions"}'
    )


@pytest.fixture
def mock_llm_timeout() -> MockLLMClient:
    return MockLLMClient(delay_s=1.0)  # longer than 500ms timeout


@pytest.fixture
def mock_llm_bad_json() -> MockLLMClient:
    return MockLLMClient("this is not json at all {{{")


@pytest.fixture
def vector_store() -> MockVectorStore:
    return MockVectorStore()
