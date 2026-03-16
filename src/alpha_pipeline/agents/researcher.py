"""
researcher.py — Microstructure Researcher LangGraph node.

Responsibilities:
  1. Build a contextual query from the MarketAlert metrics
  2. Query the MockVectorStore for relevant research documents
  3. Synthesize a brief context string for the Decision Agent
  4. Write ResearchContext to PipelineState
"""
from __future__ import annotations

import logging
import time
from typing import Any

from ..rag.vector_store import MockVectorStore
from ..state import PipelineState, ResearchContext

logger = logging.getLogger(__name__)


class MicrostructureResearcher:
    """
    Stateless LangGraph node. The vector store is injected (testable).
    The node function is `__call__` so it can be passed directly to
    `graph.add_node("researcher", researcher_instance)`.
    """

    def __init__(self, vector_store: MockVectorStore | None = None) -> None:
        self._store = vector_store or MockVectorStore()

    def __call__(self, state: PipelineState) -> dict[str, Any]:
        """LangGraph node function — returns a dict of state updates."""
        alert = state["alert"]
        metrics = alert.metrics

        t0 = time.monotonic()

        # Build query from signal-rich fields
        query_parts: list[str] = []

        if metrics.toxic_flow_detected:
            query_parts.append("toxic flow VPIN informed trading adverse selection")

        if metrics.volatility_spike_detected:
            query_parts.append("volatility spike bid-ask spread widening")

        obi_dir = "bid" if metrics.obi > 0 else "ask"
        query_parts.append(f"order book imbalance {obi_dir} pressure OBI {metrics.obi:.2f}")
        query_parts.append(f"VPIN {metrics.vpin:.2f} spread {metrics.spread_bps:.1f} bps")
        query_parts.append(alert.trigger.lower().replace("_", " "))

        query = " ".join(query_parts)

        logger.info(
            "Researcher querying vector store | trigger=%s obi=%.3f vpin=%.3f",
            alert.trigger, metrics.obi, metrics.vpin,
        )

        retrieved_docs, embedding_latency_ms = self._store.query(query, top_k=3)

        # Lightweight synthesis: structured summary from top docs
        synthesis_parts: list[str] = [
            f"Alert type: {alert.trigger} (severity={alert.severity}). "
            f"OBI={metrics.obi:.3f}, VPIN={metrics.vpin:.3f}, "
            f"spread={metrics.spread_bps:.2f}bps, mid={metrics.mid_price}.\n",
            "Relevant research:\n",
        ]

        for i, doc in enumerate(retrieved_docs, 1):
            synthesis_parts.append(
                f"{i}. [{doc.source}] {doc.title} (similarity={doc.similarity_score:.3f}): "
                f"{doc.content[:300]}...\n"
            )

        synthesized_context = "".join(synthesis_parts)

        research_context = ResearchContext(
            query_embedding_latency_ms=embedding_latency_ms,
            retrieved_docs=retrieved_docs,
            synthesized_context=synthesized_context,
        )

        total_ms = (time.monotonic() - t0) * 1000
        audit_entry = (
            f"[RESEARCHER] completed in {total_ms:.1f}ms | "
            f"docs_retrieved={len(retrieved_docs)} | "
            f"top_score={retrieved_docs[0].similarity_score if retrieved_docs else 0:.4f}"
        )

        # ── TESTING OVERRIDE ──────────────────────────────────────────────────
        # should_escalate = alert.severity in ("HIGH", "CRITICAL") # Production Logic
        should_escalate = True  # TEST MODE: Force AI invocation for all alerts
        # ──────────────────────────────────────────────────────────────────────

        logger.info(
            "Researcher complete | docs=%d should_escalate=%s latency_ms=%.1f",
            len(retrieved_docs), should_escalate, total_ms,
        )

        return {
            "research_context": research_context,
            "should_escalate": should_escalate,
            "audit_log": [audit_entry],
        }