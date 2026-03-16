"""config.py — Dependency-injected configuration for all pipeline components."""
from __future__ import annotations

from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str = "BTCUSDT"
    ws_endpoint: str = "ws://localhost:8765"
    lob_depth: int = Field(default=20, ge=5, le=100)
    obi_levels: int = Field(default=5, ge=1, le=20)
    vpin_bucket_size: Decimal = Field(default=Decimal("10.0"))
    vpin_window: int = Field(default=50, ge=10)
    spread_rolling_window: int = Field(default=100, ge=20)
    spread_z_threshold: float = Field(default=2.0, ge=0.0)
    obi_toxicity_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    vpin_toxicity_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    alert_queue_maxsize: int = Field(default=100, ge=1)
    max_reconnect_attempts: int = Field(default=10, ge=1)
    reconnect_base_delay_s: float = Field(default=1.0, gt=0)
    reconnect_max_delay_s: float = Field(default=30.0, gt=0)


class DecisionAgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 512
    llm_timeout_ms: float = Field(default=500.0, gt=0)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)


class TelemetryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    heartbeat_interval_s: float = Field(default=5.0, gt=0)
    prometheus_port: int = Field(default=8000, ge=1024, le=65535)
    analyst_hours_per_alert: float = Field(
        default=0.5,
        description="Estimated analyst-hours saved per automated alert resolution.",
    )
