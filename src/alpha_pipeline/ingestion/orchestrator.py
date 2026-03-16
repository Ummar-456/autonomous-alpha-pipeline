"""
orchestrator.py — LOB Data Orchestrator.

Manages: WS lifecycle, LOB state, metrics computation, alert dispatch.
Dependency-injectable for unit testing (pass lob, engine, queue).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from decimal import Decimal
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .lob import LimitOrderBook
from .metrics import MetricsEngine
from ..config import OrchestratorConfig
from ..state import MarketAlert, MicrostructureMetrics, OrderBookSnapshot

logger = logging.getLogger(__name__)


class LOBOrchestrator:
    """
    Drives the WS feed, maintains the LOB, emits structured alerts.

    Error handling:
      ConnectionClosed/WebSocketException → reconnect with exponential backoff
      Malformed JSON / missing keys        → log, count, continue
      Sequence gap                         → re-seed book
      CancelledError                       → clean shutdown (propagated)
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        alert_queue: asyncio.Queue,
        lob: Optional[LimitOrderBook] = None,
        metrics_engine: Optional[MetricsEngine] = None,
    ) -> None:
        self._config = config
        self._queue = alert_queue
        self._lob = lob or LimitOrderBook(config.symbol, depth=config.lob_depth)
        self._engine = metrics_engine or MetricsEngine(
            obi_levels=config.obi_levels,
            vpin_bucket_size=config.vpin_bucket_size,
            vpin_window=config.vpin_window,
            spread_window=config.spread_rolling_window,
            obi_toxicity_threshold=config.obi_toxicity_threshold,
            vpin_toxicity_threshold=config.vpin_toxicity_threshold,
            spread_z_threshold=config.spread_z_threshold,
        )

        self._reconnect_attempt: int = 0
        self._is_running: bool = False

        # Telemetry counters — read by SRE monitor
        self.packets_received: int = 0
        self.packets_dropped: int = 0
        self.alerts_emitted: int = 0

    async def run(self) -> None:
        self._is_running = True
        logger.info(
            "LOBOrchestrator starting | symbol=%s endpoint=%s",
            self._config.symbol, self._config.ws_endpoint,
        )
        try:
            while self._is_running:
                try:
                    await self._connect_and_consume()
                    break
                except (ConnectionClosed, WebSocketException) as exc:
                    self.packets_dropped += 1
                    await self._schedule_reconnect(exc)
                except asyncio.CancelledError:
                    logger.info("LOBOrchestrator shutting down cleanly.")
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Unexpected orchestrator error: %s", exc)
                    await self._schedule_reconnect(exc)
        finally:
            self._is_running = False
            logger.info(
                "LOBOrchestrator stopped | received=%d dropped=%d alerts=%d",
                self.packets_received, self.packets_dropped, self.alerts_emitted,
            )

    async def stop(self) -> None:
        self._is_running = False

    async def _connect_and_consume(self) -> None:
        async with websockets.connect(
            self._config.ws_endpoint,
            ping_interval=20,
            ping_timeout=10,
            max_size=2 ** 20,
            open_timeout=10,
        ) as ws:
            logger.info("WebSocket connected after %d attempt(s)", self._reconnect_attempt)
            self._reconnect_attempt = 0
            await self._seed_book()
            async for raw_message in ws:
                if not self._is_running:
                    return
                recv_ns = time.time_ns()
                self.packets_received += 1
                await self._process_message(raw_message, recv_ns)

    async def _seed_book(self) -> None:
        logger.info("Seeding LOB snapshot for %s", self._config.symbol)
        base = Decimal("43500.00")
        tick = Decimal("0.50")
        spread = Decimal("1.00")
        bids = [
            [str(base - i * tick), str(Decimal("1.0") + i * Decimal("0.12"))]
            for i in range(self._config.lob_depth)
        ]
        asks = [
            [str(base + spread + i * tick), str(Decimal("0.9") + i * Decimal("0.10"))]
            for i in range(self._config.lob_depth)
        ]
        self._lob.apply_snapshot(bids, asks, last_update_id=1_000_000)

    async def _process_message(self, raw: str | bytes, recv_ns: int) -> None:
        try:
            msg: dict = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Non-JSON message: %r", str(raw)[:200])
            self.packets_dropped += 1
            return

        if msg.get("e") != "depthUpdate":
            return

        try:
            final_update_id: int = msg["u"]
            raw_bids: list[list[str]] = msg.get("b", [])
            raw_asks: list[list[str]] = msg.get("a", [])
        except KeyError as exc:
            logger.warning("Malformed depthUpdate (missing %s): %s", exc, msg)
            self.packets_dropped += 1
            return

        if not self._lob.apply_delta(raw_bids, raw_asks, final_update_id):
            logger.warning("Update ID gap — re-seeding LOB.")
            await self._seed_book()
            return

        metrics = self._engine.compute(self._lob)
        if metrics is None:
            return

        if metrics.toxic_flow_detected or metrics.volatility_spike_detected:
            await self._emit_alert(metrics, recv_ns)

    async def _emit_alert(self, metrics: MicrostructureMetrics, recv_ns: int) -> None:
        now_ns = time.time_ns()
        latency_ms = (now_ns - recv_ns) / 1_000_000

        if metrics.toxic_flow_detected and metrics.volatility_spike_detected:
            trigger, severity = "BOTH", "CRITICAL"
        elif metrics.toxic_flow_detected:
            trigger, severity = "TOXIC_FLOW", "HIGH"
        else:
            trigger, severity = "VOLATILITY_SPIKE", "MEDIUM"

        bids, asks = self._lob.snapshot(depth=5)
        alert = MarketAlert(
            alert_id=str(uuid.uuid4()),
            symbol=self._config.symbol,
            severity=severity,          # type: ignore[arg-type]
            trigger=trigger,            # type: ignore[arg-type]
            snapshot=OrderBookSnapshot(
                symbol=self._config.symbol,
                timestamp_ns=recv_ns,
                bids=bids,
                asks=asks,
                last_update_id=self._lob.last_update_id,
            ),
            metrics=metrics,
            created_at_ns=now_ns,
            pipeline_latency_ms=latency_ms,
        )

        try:
            self._queue.put_nowait(alert)
            self.alerts_emitted += 1
            logger.info(
                "Alert emitted | id=%s severity=%s trigger=%s latency_ms=%.3f",
                alert.alert_id, severity, trigger, latency_ms,
            )
        except asyncio.QueueFull:
            logger.warning("Alert queue full — dropping | id=%s", alert.alert_id)

    async def _schedule_reconnect(self, exc: Exception) -> None:
        self._reconnect_attempt += 1
        if self._reconnect_attempt > self._config.max_reconnect_attempts:
            logger.critical(
                "Max reconnect attempts (%d) reached — halting.",
                self._config.max_reconnect_attempts,
            )
            self._is_running = False
            return
        cap = self._config.reconnect_max_delay_s
        base = self._config.reconnect_base_delay_s
        delay = random.uniform(0, min(cap, base * (2 ** self._reconnect_attempt)))
        logger.warning(
            "WS error (%s) — retry in %.1fs (attempt %d/%d)",
            exc, delay, self._reconnect_attempt, self._config.max_reconnect_attempts,
        )
        await asyncio.sleep(delay)
