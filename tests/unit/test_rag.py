"""Unit tests for MockVectorStore and MicrostructureResearcher — 10 tests."""
from __future__ import annotations

import pytest

from alpha_pipeline.agents.researcher import MicrostructureResearcher
from alpha_pipeline.rag.corpus import get_all_documents
from alpha_pipeline.rag.vector_store import MockVectorStore
from alpha_pipeline.state import ResearchDocument


class TestMockVectorStore:
    def test_returns_correct_number_of_docs(self, vector_store):
        results, _ = vector_store.query("order book imbalance", top_k=3)
        assert len(results) == 3

    def test_returns_fewer_docs_when_k_exceeds_corpus(self):
        # Corpus has 8 documents, requesting more should return all
        vs = MockVectorStore()
        results, _ = vs.query("market microstructure", top_k=100)
        assert len(results) == len(get_all_documents())

    def test_similarity_scores_bounded(self, vector_store):
        results, _ = vector_store.query("VPIN toxic flow informed trading")
        for doc in results:
            assert 0.0 <= doc.similarity_score <= 1.0

    def test_results_sorted_by_similarity_descending(self, vector_store):
        results, _ = vector_store.query("OBI imbalance spread")
        scores = [d.similarity_score for d in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_query_returns_relevant_doc(self, vector_store):
        results, _ = vector_store.query("VPIN toxic flow cancel all liquidity", top_k=1)
        assert len(results) == 1
        # The top result should be related to VPIN / toxic flow
        top = results[0]
        assert "vpin" in top.title.lower() or "toxic" in top.title.lower() \
               or "vpin" in top.content.lower()

    def test_latency_is_positive(self, vector_store):
        _, latency_ms = vector_store.query("test query")
        assert latency_ms >= 0.0

    def test_empty_query_handled(self, vector_store):
        results, _ = vector_store.query("", top_k=3)
        assert isinstance(results, list)

    def test_results_are_research_document_instances(self, vector_store):
        results, _ = vector_store.query("order book bid ask")
        for r in results:
            assert isinstance(r, ResearchDocument)


class TestMicrostructureResearcher:
    def test_researcher_returns_research_context(self, pipeline_state):
        researcher = MicrostructureResearcher()
        update = researcher(pipeline_state)
        assert "research_context" in update
        assert update["research_context"] is not None

    def test_researcher_sets_should_escalate_for_high_severity(self, pipeline_state):
        researcher = MicrostructureResearcher()
        # toxic_alert has severity=HIGH
        update = researcher(pipeline_state)
        assert update["should_escalate"] is True

    def test_researcher_appends_audit_log(self, pipeline_state):
        researcher = MicrostructureResearcher()
        update = researcher(pipeline_state)
        assert len(update["audit_log"]) == 1
        assert "[RESEARCHER]" in update["audit_log"][0]

    def test_researcher_low_severity_no_escalation(self, volatile_alert, sample_research_context):
        # MEDIUM severity should not escalate
        state = {
            "alert": volatile_alert,
            "research_context": None,
            "decision": None,
            "audit_log": [],
            "should_escalate": False,
            "total_pipeline_latency_ms": None,
        }
        researcher = MicrostructureResearcher()
        update = researcher(state)
        assert update["should_escalate"] is False  # MEDIUM → no escalate
