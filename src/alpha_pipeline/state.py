"""
state.py — Canonical data contracts for the Autonomous Alpha Pipeline.

Every byte crossing a component boundary is a Pydantic model.
PipelineState TypedDict is the mutable LangGraph envelope.
"""
from __future__ import annotations

import operator
from decimal import Decimal
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


class PriceLevel(BaseModel):
    model_config = ConfigDict(frozen=True)
    price: Decimal
    quantity: Decimal


class OrderBookSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    symbol: str
    timestamp_ns: int
    bids: list[PriceLevel]
    asks: list[PriceLevel]
    last_update_id: int


class MicrostructureMetrics(BaseModel):
    """
    OBI  = (Σ_bid_qty - Σ_ask_qty) / (Σ_bid_qty + Σ_ask_qty)  top-N levels
    VPIN = simplified Easley et al. 2012 volume-bucket flow toxicity
    """
    model_config = ConfigDict(frozen=True)

    obi: float = Field(description="Order Book Imbalance ∈ [-1.0, 1.0].")
    spread_bps: float = Field(description="Bid-ask spread in basis points.")
    mid_price: Decimal = Field(description="(best_bid + best_ask) / 2.")
    vpin: float = Field(ge=0.0, le=1.0, description="Volume-sync'd prob of informed trading.")
    toxic_flow_detected: bool
    volatility_spike_detected: bool


class MarketAlert(BaseModel):
    model_config = ConfigDict(frozen=True)

    alert_id: str
    symbol: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    trigger: Literal["VOLATILITY_SPIKE", "TOXIC_FLOW", "BOTH"]
    snapshot: OrderBookSnapshot
    metrics: MicrostructureMetrics
    created_at_ns: int
    pipeline_latency_ms: float


class ResearchDocument(BaseModel):
    model_config = ConfigDict(frozen=True)
    doc_id: str
    title: str
    content: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    source: str


class ResearchContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    query_embedding_latency_ms: float
    retrieved_docs: list[ResearchDocument]
    synthesized_context: str


class ExecutionDecision(BaseModel):
    """Strictly-typed LLM output. Parse failure → fallback CANCEL_ALL."""
    model_config = ConfigDict(frozen=True)

    action: Literal[
        "PROVIDE_LIQUIDITY",
        "TAKE_LIQUIDITY",
        "WIDEN_QUOTES",
        "CANCEL_ALL",
        "HOLD",
    ]
    side: Literal["BID", "ASK", "BOTH", "NONE"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=2048)
    fallback_triggered: bool
    decision_latency_ms: float


class PipelineState(TypedDict):
    """
    Mutable envelope threaded through LangGraph nodes.
    audit_log uses operator.add so concurrent branch writes merge as concatenation.
    """
    alert: MarketAlert
    research_context: Optional[ResearchContext]
    decision: Optional[ExecutionDecision]
    audit_log: Annotated[list[str], operator.add]
    should_escalate: bool
    total_pipeline_latency_ms: Optional[float]
