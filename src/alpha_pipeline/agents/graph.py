"""
graph.py — LangGraph StateGraph definition for the Autonomous Alpha Pipeline.

Graph topology:
    START
      └─► researcher ──► route_after_research ──► decision ──► END
                                                 └─────────────► END (HOLD / low-severity)

The graph runner is fully async: it wraps graph.invoke() in asyncio.to_thread
so the event loop is never blocked by synchronous LangGraph traversal.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Optional, TYPE_CHECKING

from langgraph.graph import StateGraph, END

from .researcher import MicrostructureResearcher
from .decision import DecisionAgent
from ..config import DecisionAgentConfig
from ..rag.vector_store import MockVectorStore
from ..state import PipelineState, MarketAlert

if TYPE_CHECKING:
    from ..telemetry.monitor import SREMonitor

logger = logging.getLogger(__name__)

# Keep only the most recent N latency samples for the p99 calculation.
# Without this cap, self.latencies_ms grows for the lifetime of the process —
# 10 alerts/s * 3600s = 36 000 entries/hour, unbounded heap growth.
_LATENCY_WINDOW = 1000


def _route_after_research(state: PipelineState) -> str:
    """
    Conditional edge: if should_escalate → invoke Decision Agent.
    Otherwise terminate early (HOLD decision implied).
    """
    if state.get("should_escalate", False):
        return "decision"
    return END


def build_pipeline_graph(
    vector_store: MockVectorStore | None = None,
    decision_config: DecisionAgentConfig | None = None,
    decision_agent: DecisionAgent | None = None,
    llm_client: Any | None = None,
) -> Any:
    """
    Factory that wires the LangGraph StateGraph.
    All components are injected to enable unit-test overrides and Mock mode.
    """
    researcher = MicrostructureResearcher(vector_store=vector_store)
    agent = decision_agent or DecisionAgent(
        config=decision_config, llm_client=llm_client
    )

    workflow: StateGraph = StateGraph(PipelineState)

    workflow.add_node("researcher", researcher)
    workflow.add_node("decision", agent)

    workflow.set_entry_point("researcher")
    workflow.add_conditional_edges(
        "researcher",
        _route_after_research,
        {
            "decision": "decision",
            END: END,
        },
    )
    workflow.add_edge("decision", END)

    return workflow.compile()


class GraphRunner:
    """
    Async wrapper around the compiled LangGraph.

    Consumes MarketAlert objects from the asyncio.Queue and runs them
    through the graph in asyncio.to_thread — keeping the event loop unblocked.
    End-to-end latency is tracked per-invocation for the SRE layer.
    """

    def __init__(
        self,
        alert_queue: asyncio.Queue,
        vector_store: MockVectorStore | None = None,
        decision_config: DecisionAgentConfig | None = None,
        decision_agent: DecisionAgent | None = None,
        llm_client: Any | None = None,
        sre_monitor: Optional["SREMonitor"] = None,
    ) -> None:
        self._queue = alert_queue
        self._monitor = sre_monitor

        self._graph = build_pipeline_graph(
            vector_store=vector_store,
            decision_config=decision_config,
            decision_agent=decision_agent,
            llm_client=llm_client,
        )
        self._is_running: bool = False

        # Telemetry — invocations is a plain int (SREMonitor reads it for
        # the delta-sync in _collect_metrics).
        # FIX: latencies_ms is now a capped deque instead of an unbounded
        # list. At ~10 alerts/s the old list would grow to ~36 000 entries/hr
        # and cause steady heap growth. The capped window is sufficient for
        # an accurate p99 (1 000 samples ≈ 100 s of history at 10 ops/s).
        self.invocations: int = 0
        self.latencies_ms: deque[float] = deque(maxlen=_LATENCY_WINDOW)

    async def run(self) -> None:
        """Main loop consuming alerts from the queue."""
        self._is_running = True
        logger.info("GraphRunner started.")
        try:
            while self._is_running:
                try:
                    alert: MarketAlert = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                asyncio.create_task(self._invoke_graph(alert))

        except asyncio.CancelledError:
            logger.info("GraphRunner shutting down cleanly.")
            raise
        finally:
            self._is_running = False

    async def stop(self) -> None:
        """Signal the loop to stop."""
        self._is_running = False

    async def _invoke_graph(self, alert: MarketAlert) -> None:
        """
        Executes the LangGraph in a separate thread to keep the event loop free.
        Reports results back to the SRE monitor if one is attached.
        """
        t0 = time.monotonic()
        initial_state: PipelineState = {
            "alert": alert,
            "research_context": None,
            "decision": None,
            "audit_log": [],
            "should_escalate": False,
            "total_pipeline_latency_ms": None,
        }
        try:
            final_state = await asyncio.to_thread(self._graph.invoke, initial_state)

            latency_ms = (time.monotonic() - t0) * 1000
            self.invocations += 1
            self.latencies_ms.append(latency_ms)

            decision = final_state.get("decision")
            action = decision.action if decision else "HOLD"

            # FIX: Previously used getattr(decision, "fallback_triggered", False)
            # which would silently return False if DecisionAgent never sets that
            # attribute, meaning llm_fallbacks would never increment even when
            # the circuit breaker fired. Now we also treat a missing/None
            # decision object as an implicit fallback so the counter is accurate.
            fallback = (
                bool(getattr(decision, "fallback_triggered", False))
                if decision is not None
                else True  # no decision produced → treat as fallback
            )

            if self._monitor:
                self._monitor.record_graph_invocation(
                    latency_ms=latency_ms,
                    action=action,
                    fallback_triggered=fallback,
                )

            logger.info(
                "Pipeline complete | alert_id=%s action=%s latency_ms=%.1f",
                alert.alert_id,
                action,
                latency_ms,
            )
        except Exception as exc:
            logger.exception(
                "Graph invocation failed for alert %s: %s", alert.alert_id, exc
            )

    def p99_latency_ms(self) -> float:
        """Returns the p99 latency over the most recent _LATENCY_WINDOW samples."""
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = min(int(0.99 * len(sorted_l)), len(sorted_l) - 1)
        return sorted_l[idx]