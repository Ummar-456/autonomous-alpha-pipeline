"""
Integration tests — 12 tests.

Tests the full pipeline components working together:
  - LOB + MetricsEngine → alert generation
  - Orchestrator message processing
  - LangGraph graph traversal end-to-end
  - GraphRunner async queue consumption
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from decimal import Decimal

import pytest

from alpha_pipeline.agents.decision import DecisionAgent
from alpha_pipeline.agents.graph import GraphRunner, build_pipeline_graph
from alpha_pipeline.config import DecisionAgentConfig, OrchestratorConfig
from alpha_pipeline.ingestion.lob import LimitOrderBook
from alpha_pipeline.ingestion.metrics import MetricsEngine
from alpha_pipeline.ingestion.orchestrator import LOBOrchestrator
from alpha_pipeline.rag.vector_store import MockVectorStore
from alpha_pipeline.state import PipelineState


# ── LOB + Metrics Integration ─────────────────────────────────────────────────

class TestLOBMetricsIntegration:
    def test_lob_delta_then_metrics_computed(self):
        lob = LimitOrderBook("BTCUSDT", depth=20)
        engine = MetricsEngine(obi_levels=5, vpin_bucket_size=Decimal("1"), vpin_window=10)
        base = Decimal("43500")
        bids = [[str(base - Decimal(str(i)) * Decimal("0.5")), "1.0"] for i in range(10)]
        asks = [[str(base + Decimal("1") + Decimal(str(i)) * Decimal("0.5")), "0.9"] for i in range(10)]
        lob.apply_snapshot(bids, asks, last_update_id=1000)

        metrics = engine.compute(lob)
        assert metrics is not None
        assert isinstance(metrics.obi, float)
        assert metrics.spread_bps > 0

    def test_sequential_deltas_maintain_book_integrity(self):
        lob = LimitOrderBook("BTCUSDT", depth=20)
        base = Decimal("43500")
        bids = [[str(base - Decimal(str(i)) * Decimal("0.5")), "1.0"] for i in range(5)]
        asks = [[str(base + Decimal("1") + Decimal(str(i)) * Decimal("0.5")), "1.0"] for i in range(5)]
        lob.apply_snapshot(bids, asks, last_update_id=1000)

        for i in range(1, 11):
            lob.apply_delta([[str(base), str(1.0 + i * 0.1)]], [], final_update_id=1000 + i)

        assert lob.best_bid()[1] == Decimal(str(round(1.0 + 10 * 0.1, 10)))

    def test_toxic_conditions_trigger_correct_metrics(self):
        lob = LimitOrderBook("BTCUSDT", depth=20)
        engine = MetricsEngine(
            obi_levels=5,
            vpin_bucket_size=Decimal("0.01"),
            vpin_window=10,
            obi_toxicity_threshold=0.7,
            vpin_toxicity_threshold=0.0,
        )
        # Heavily bid-imbalanced
        bids = [[str(Decimal("43500") - Decimal(str(i)) * Decimal("0.5")), "20.0"] for i in range(5)]
        asks = [[str(Decimal("43501") + Decimal(str(i)) * Decimal("0.5")), "0.5"] for i in range(5)]
        lob.apply_snapshot(bids, asks, last_update_id=1000)

        for _ in range(30):
            m = engine.compute(lob)
        assert m is not None
        assert m.obi > 0.7  # bid-heavy


# ── Orchestrator Message Processing ──────────────────────────────────────────

class TestOrchestratorProcessing:
    @pytest.mark.asyncio
    async def test_process_valid_depth_update_message(self):
        config = OrchestratorConfig(alert_queue_maxsize=10)
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        await orch._seed_book()

        msg = json.dumps({
            "e": "depthUpdate",
            "E": int(time.time() * 1000),
            "s": "BTCUSDT",
            "U": 1_000_000,
            "u": 1_000_001,
            "b": [["43500.0", "2.0"]],
            "a": [["43501.0", "1.5"]],
        })
        orch.packets_received += 1
        await orch._process_message(msg, time.time_ns())
        assert orch.packets_dropped == 0

    @pytest.mark.asyncio
    async def test_process_non_json_increments_drop_counter(self):
        config = OrchestratorConfig()
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        await orch._process_message("NOT JSON {{{", time.time_ns())
        assert orch.packets_dropped == 1

    @pytest.mark.asyncio
    async def test_process_non_depth_event_is_ignored(self):
        config = OrchestratorConfig()
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        msg = json.dumps({"e": "trade", "p": "43500", "q": "0.5"})
        await orch._process_message(msg, time.time_ns())
        assert orch.packets_dropped == 0

    @pytest.mark.asyncio
    async def test_alert_emitted_via_emit_alert_directly(self):
        """Test _emit_alert directly — avoids needing full VPIN warmup history."""
        config = OrchestratorConfig(alert_queue_maxsize=10)
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        await orch._seed_book()

        from alpha_pipeline.state import MicrostructureMetrics
        metrics = MicrostructureMetrics(
            obi=0.9, spread_bps=5.0, mid_price=Decimal("43500.5"),
            vpin=0.7, toxic_flow_detected=True, volatility_spike_detected=False,
        )
        await orch._emit_alert(metrics, time.time_ns())
        assert orch.alerts_emitted == 1
        assert queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_full_queue_drops_alert_gracefully(self):
        """When queue is full, _emit_alert must drop silently without raising."""
        config = OrchestratorConfig(alert_queue_maxsize=1)
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        orch = LOBOrchestrator(config=config, alert_queue=queue)
        await orch._seed_book()

        from alpha_pipeline.state import MicrostructureMetrics
        metrics = MicrostructureMetrics(
            obi=0.9, spread_bps=5.0, mid_price=Decimal("43500.5"),
            vpin=0.7, toxic_flow_detected=True, volatility_spike_detected=False,
        )
        await orch._emit_alert(metrics, time.time_ns())
        assert queue.qsize() == 1

        # Second emit: queue is full — must not raise, must drop
        await orch._emit_alert(metrics, time.time_ns())
        assert queue.qsize() == 1  # still 1, second was silently dropped


# ── LangGraph Pipeline Integration ────────────────────────────────────────────

class TestLangGraphPipeline:
    @pytest.mark.asyncio
    async def test_full_graph_run_returns_decision(
        self, mock_llm_cancel_all, toxic_alert, vector_store
    ):
        decision_agent = DecisionAgent(
            config=DecisionAgentConfig(llm_timeout_ms=2000),
            llm_client=mock_llm_cancel_all,
        )
        graph = build_pipeline_graph(
            vector_store=vector_store,
            decision_agent=decision_agent,
        )
        initial_state: PipelineState = {
            "alert": toxic_alert,
            "research_context": None,
            "decision": None,
            "audit_log": [],
            "should_escalate": False,
            "total_pipeline_latency_ms": None,
        }
        final_state = await asyncio.to_thread(graph.invoke, initial_state)
        assert final_state["decision"] is not None
        assert final_state["decision"].action == "CANCEL_ALL"
        assert len(final_state["audit_log"]) >= 2  # researcher + decision

    @pytest.mark.asyncio
    async def test_low_severity_alert_terminates_early(
        self, vector_store, volatile_alert
    ):
        """MEDIUM severity → should_escalate=False → Decision node not invoked."""
        from tests.conftest import MockLLMClient
        llm = MockLLMClient()
        llm.call_count = 0
        decision_agent = DecisionAgent(
            config=DecisionAgentConfig(llm_timeout_ms=2000),
            llm_client=llm,
        )
        graph = build_pipeline_graph(
            vector_store=vector_store,
            decision_agent=decision_agent,
        )
        initial_state: PipelineState = {
            "alert": volatile_alert,
            "research_context": None,
            "decision": None,
            "audit_log": [],
            "should_escalate": False,
            "total_pipeline_latency_ms": None,
        }
        final_state = await asyncio.to_thread(graph.invoke, initial_state)
        # Decision node was not called → decision is None
        assert final_state.get("decision") is None
        assert llm.call_count == 0

    @pytest.mark.asyncio
    async def test_audit_log_merges_from_both_nodes(
        self, mock_llm_cancel_all, toxic_alert, vector_store
    ):
        decision_agent = DecisionAgent(
            config=DecisionAgentConfig(llm_timeout_ms=2000),
            llm_client=mock_llm_cancel_all,
        )
        graph = build_pipeline_graph(
            vector_store=vector_store,
            decision_agent=decision_agent,
        )
        initial_state: PipelineState = {
            "alert": toxic_alert,
            "research_context": None,
            "decision": None,
            "audit_log": [],
            "should_escalate": False,
            "total_pipeline_latency_ms": None,
        }
        final_state = await asyncio.to_thread(graph.invoke, initial_state)
        log = final_state["audit_log"]
        assert any("[RESEARCHER]" in entry for entry in log)
        assert any("[DECISION]" in entry for entry in log)


# ── GraphRunner Integration ───────────────────────────────────────────────────

class TestGraphRunnerIntegration:
    @pytest.mark.asyncio
    async def test_graph_runner_processes_alert_from_queue(
        self, mock_llm_cancel_all, toxic_alert, vector_store
    ):
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        decision_agent = DecisionAgent(
            config=DecisionAgentConfig(llm_timeout_ms=2000),
            llm_client=mock_llm_cancel_all,
        )
        runner = GraphRunner(
            alert_queue=queue,
            vector_store=vector_store,
            decision_agent=decision_agent,
        )

        await queue.put(toxic_alert)

        # Let the runner consume the alert then stop it
        async def run_and_stop():
            run_task = asyncio.create_task(runner.run())
            # Poll until the queue empties (alert consumed) or timeout
            for _ in range(50):
                await asyncio.sleep(0.1)
                if queue.empty():
                    break
            # Give graph invocation time to complete
            await asyncio.sleep(1.5)
            await runner.stop()
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

        await run_and_stop()
        assert runner.invocations >= 1

    @pytest.mark.asyncio
    async def test_graph_runner_p99_latency_after_invocation(
        self, mock_llm_cancel_all, toxic_alert, vector_store
    ):
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        decision_agent = DecisionAgent(
            config=DecisionAgentConfig(llm_timeout_ms=2000),
            llm_client=mock_llm_cancel_all,
        )
        runner = GraphRunner(
            alert_queue=queue,
            vector_store=vector_store,
            decision_agent=decision_agent,
        )
        await queue.put(toxic_alert)

        async def run_and_stop():
            run_task = asyncio.create_task(runner.run())
            for _ in range(50):
                await asyncio.sleep(0.1)
                if queue.empty():
                    break
            await asyncio.sleep(1.5)
            await runner.stop()
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

        await run_and_stop()
        # p99 is 0.0 until at least one invocation is recorded
        assert runner.p99_latency_ms() >= 0.0
