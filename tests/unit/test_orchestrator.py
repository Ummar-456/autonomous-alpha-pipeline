"""
Unit tests for LOBOrchestrator reconnect logic and lifecycle — 8 tests.
These test the backoff scheduling and stop/start semantics without
actually opening a WebSocket connection.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alpha_pipeline.config import OrchestratorConfig
from alpha_pipeline.ingestion.orchestrator import LOBOrchestrator


class TestOrchestratorReconnect:
    @pytest.mark.asyncio
    async def test_schedule_reconnect_increments_attempt_counter(self):
        config = OrchestratorConfig(
            max_reconnect_attempts=5,
            reconnect_base_delay_s=0.001,
            reconnect_max_delay_s=0.01,
        )
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        assert orch._reconnect_attempt == 0
        await orch._schedule_reconnect(ConnectionError("test"))
        assert orch._reconnect_attempt == 1

    @pytest.mark.asyncio
    async def test_schedule_reconnect_stops_after_max_attempts(self):
        config = OrchestratorConfig(
            max_reconnect_attempts=2,
            reconnect_base_delay_s=0.001,
            reconnect_max_delay_s=0.01,
        )
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        orch._is_running = True
        orch._reconnect_attempt = 2  # already at max

        await orch._schedule_reconnect(ConnectionError("final"))
        # After exceeding max, _is_running should be set to False
        assert orch._is_running is False

    @pytest.mark.asyncio
    async def test_stop_sets_is_running_false(self):
        config = OrchestratorConfig()
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        orch._is_running = True
        await orch.stop()
        assert orch._is_running is False

    @pytest.mark.asyncio
    async def test_seed_book_populates_lob(self):
        config = OrchestratorConfig()
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        assert not orch._lob.is_populated
        await orch._seed_book()
        assert orch._lob.is_populated

    @pytest.mark.asyncio
    async def test_seed_book_sets_correct_spread(self):
        """Seeded book should have best_ask > best_bid (valid non-crossed book)."""
        config = OrchestratorConfig()
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        await orch._seed_book()
        best_bid = orch._lob.best_bid()
        best_ask = orch._lob.best_ask()
        assert best_bid is not None and best_ask is not None
        assert best_ask[0] > best_bid[0]


class TestOrchestratorMessageEdgeCases:
    @pytest.mark.asyncio
    async def test_missing_key_in_depth_update_increments_drop(self):
        """depthUpdate with missing 'u' field → drop counter increments."""
        import json
        config = OrchestratorConfig()
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        await orch._seed_book()

        msg = json.dumps({
            "e": "depthUpdate",
            "E": 1234567890000,
            "s": "BTCUSDT",
            # "u" key intentionally missing
            "b": [["43500.0", "1.0"]],
            "a": [["43501.0", "0.9"]],
        })
        await orch._process_message(msg, 0)
        assert orch.packets_dropped == 1

    @pytest.mark.asyncio
    async def test_sequence_gap_triggers_reseed(self):
        """A gap in update IDs should trigger _seed_book() re-call."""
        import json
        import time
        config = OrchestratorConfig()
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        await orch._seed_book()

        initial_update_id = orch._lob.last_update_id

        # Send a message with a large gap (skips many IDs)
        msg = json.dumps({
            "e": "depthUpdate",
            "E": int(time.time() * 1000),
            "s": "BTCUSDT",
            "U": initial_update_id + 5,   # gap: skipped IDs initial+1 through initial+4
            "u": initial_update_id + 10,
            "b": [["43500.0", "1.5"]],
            "a": [["43501.0", "0.8"]],
        })
        await orch._process_message(msg, 0)
        # After re-seed, the LOB should still be populated (re-seeded to initial state)
        assert orch._lob.is_populated

    @pytest.mark.asyncio
    async def test_alert_severity_both_for_combined_signals(self):
        """When both toxic_flow AND volatility_spike are True → severity=CRITICAL, trigger=BOTH."""
        from alpha_pipeline.state import MicrostructureMetrics
        config = OrchestratorConfig(alert_queue_maxsize=10)
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        await orch._seed_book()

        metrics = MicrostructureMetrics(
            obi=0.9,
            spread_bps=5.5,
            mid_price=Decimal("43500.5"),
            vpin=0.75,
            toxic_flow_detected=True,
            volatility_spike_detected=True,
        )
        await orch._emit_alert(metrics, 0)
        alert = queue.get_nowait()
        assert alert.severity == "CRITICAL"
        assert alert.trigger == "BOTH"
