"""
vector_store.py — Mocked vector database using TF-IDF-style cosine similarity.

No external vector DB dependency (no chromadb, faiss, etc.).
Uses a hand-rolled bag-of-words cosine similarity that captures keyword
overlap well enough to route microstructure alerts to relevant research.
Production replacement: swap `_embed` for a real embedding model call.
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Optional

from .corpus import CorpusDocument, get_all_documents
from ..state import ResearchDocument

logger = logging.getLogger(__name__)


class MockVectorStore:
    """
    In-memory vector store backed by TF-IDF cosine similarity.

    Each document is pre-indexed at construction time.
    Query cost is O(D) where D = number of documents — acceptable for a corpus of <100 docs.
    """

    # Common English stop-words that don't contribute to retrieval quality
    _STOP_WORDS: frozenset[str] = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "must", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "and", "or", "but", "not", "if", "this", "that", "it",
        "its", "their", "they", "we", "you", "i", "he", "she", "when",
        "where", "which", "who", "whom", "all", "each", "any", "both",
        "more", "most", "other", "than", "then", "so", "such", "up", "out",
        "about", "above", "after", "before", "between",
    })

    def __init__(self, documents: Optional[list[CorpusDocument]] = None) -> None:
        self._docs = documents or get_all_documents()
        # Pre-compute TF vectors for all documents
        self._doc_vectors: list[dict[str, float]] = [
            self._compute_tf(doc.title + " " + doc.content + " " + " ".join(doc.tags))
            for doc in self._docs
        ]
        # Compute IDF weights across corpus
        self._idf: dict[str, float] = self._compute_idf(self._doc_vectors)
        # Apply TF-IDF weighting in place
        self._tfidf_vectors: list[dict[str, float]] = [
            {term: tf * self._idf.get(term, 1.0) for term, tf in vec.items()}
            for vec in self._doc_vectors
        ]
        logger.info(
            "VectorStore indexed %d documents | vocabulary_size=%d",
            len(self._docs),
            len(self._idf),
        )

    def query(self, query_text: str, top_k: int = 3) -> tuple[list[ResearchDocument], float]:
        """
        Returns (retrieved_docs, latency_ms).
        Documents are sorted by cosine similarity descending.
        """
        t0 = time.monotonic()
        query_tf = self._compute_tf(query_text)
        query_tfidf = {term: tf * self._idf.get(term, 0.5) for term, tf in query_tf.items()}

        scored: list[tuple[float, CorpusDocument]] = []
        for doc, vec in zip(self._docs, self._tfidf_vectors):
            score = self._cosine_similarity(query_tfidf, vec)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        latency_ms = (time.monotonic() - t0) * 1000

        results: list[ResearchDocument] = [
            ResearchDocument(
                doc_id=doc.doc_id,
                title=doc.title,
                content=doc.content,
                similarity_score=round(max(score, 0.0), 4),
                source=doc.source,
            )
            for score, doc in scored[:top_k]
        ]

        logger.debug(
            "VectorStore query completed | top_score=%.4f latency_ms=%.2f",
            results[0].similarity_score if results else 0.0,
            latency_ms,
        )
        return results, latency_ms

    # ── Private helpers ───────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z_]+", text.lower())
        return [t for t in tokens if t not in self._STOP_WORDS and len(t) > 2]

    def _compute_tf(self, text: str) -> dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = len(tokens)
        return {term: count / total for term, count in counts.items()}

    def _compute_idf(self, doc_vectors: list[dict[str, float]]) -> dict[str, float]:
        N = len(doc_vectors)
        if N == 0:
            return {}
        df: Counter = Counter()
        for vec in doc_vectors:
            df.update(vec.keys())
        return {
            term: math.log((N + 1) / (count + 1)) + 1.0
            for term, count in df.items()
        }

    @staticmethod
    def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a.get(k, 0.0) * v for k, v in b.items())
        mag_a = math.sqrt(sum(v ** 2 for v in a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in b.values()))
        if mag_a < 1e-12 or mag_b < 1e-12:
            return 0.0
        return dot / (mag_a * mag_b)
