"""
decision.py — Risk & Execution Strategist LangGraph node.

Core LLM reasoning node. Enforces a hard 500ms timeout on LLM calls.
Any timeout or JSON parse failure triggers CANCEL_ALL safe-state fallback.
Circuit breaker prevents LLM call storm after repeated failures.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional, Protocol

from ..config import DecisionAgentConfig
from ..state import ExecutionDecision, PipelineState, ResearchContext
from ..utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

logger = logging.getLogger(__name__)

_DECISION_SYSTEM_PROMPT = """\
You are a quantitative market-making risk engine for BTC/USDT on Binance.
You receive structured microstructure signals and research context, then output
a single valid JSON decision object. No preamble, no explanation — only JSON.

Output schema (all fields required):
{
  "action": "PROVIDE_LIQUIDITY" | "TAKE_LIQUIDITY" | "WIDEN_QUOTES" | "CANCEL_ALL" | "HOLD",
  "side": "BID" | "ASK" | "BOTH" | "NONE",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<max 200 chars explaining the decision>"
}

Decision heuristics:
- CANCEL_ALL + NONE: VPIN > 0.6 OR (VPIN > 0.5 AND |OBI| > 0.7)
- WIDEN_QUOTES + BOTH: volatility spike with spread_bps > 2.5
- TAKE_LIQUIDITY: OBI > 0.8, VPIN < 0.4 (momentum signal, low toxic risk)
- PROVIDE_LIQUIDITY + BOTH: normal conditions, OBI < 0.3, VPIN < 0.3
- HOLD: ambiguous signals, insufficient data
"""


class LLMClientProtocol(Protocol):
    """Abstract LLM client — enables injection of mocks in tests."""

    async def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        ...


class AnthropicLLMClient:
    """Production Anthropic client adapter implementing LLMClientProtocol."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self._model = model
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic()
        except ImportError:
            raise RuntimeError("anthropic package not installed")

    async def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        import anthropic
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        return message.content[0].text


class DecisionAgent:
    """
    LangGraph node implementing the Risk & Execution Strategist.

    Timeout enforcement:
        asyncio.wait_for wraps the LLM call with timeout_s = config.llm_timeout_ms / 1000.
        On TimeoutError or any parse failure → CANCEL_ALL fallback.

    Circuit breaker:
        After `failure_threshold` consecutive LLM errors, the breaker opens
        and all subsequent calls are immediately returned as CANCEL_ALL.
        This prevents stalling the pipeline during LLM outages.
    """

    def __init__(
        self,
        config: Optional[DecisionAgentConfig] = None,
        llm_client: Optional[LLMClientProtocol] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self._config = config or DecisionAgentConfig()
        self._llm = llm_client or AnthropicLLMClient(model=self._config.model)
        self._cb = circuit_breaker or CircuitBreaker(
            failure_threshold=5,
            recovery_timeout_s=30.0,
            name="decision_agent_llm",
        )

    def __call__(self, state: PipelineState) -> dict[str, Any]:
        """
        LangGraph calls this synchronously from a worker thread (via asyncio.to_thread).
        asyncio.run() creates a fresh event loop for the thread — the correct approach.
        """
        return asyncio.run(self._async_invoke(state))

    async def invoke_async(self, state: PipelineState) -> dict[str, Any]:
        """Async entry point for direct async invocation (used in tests and main)."""
        return await self._async_invoke(state)

    async def _async_invoke(self, state: PipelineState) -> dict[str, Any]:
        alert = state["alert"]
        research = state.get("research_context")
        t0 = time.monotonic()

        user_prompt = self._build_prompt(alert, research)
        decision: ExecutionDecision

        try:
            async def _call_llm() -> str:
                return await self._llm.complete(
                    system=_DECISION_SYSTEM_PROMPT,
                    user=user_prompt,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                )

            raw_json = await asyncio.wait_for(
                self._cb.call(_call_llm),
                timeout=self._config.llm_timeout_ms / 1000,
            )
            decision = self._parse_decision(raw_json, latency_ms=(time.monotonic() - t0) * 1000)

        except asyncio.TimeoutError:
            logger.warning(
                "DecisionAgent LLM timeout (>%.0fms) — CANCEL_ALL fallback.",
                self._config.llm_timeout_ms,
            )
            decision = self._make_fallback(
                reason="LLM_TIMEOUT",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except CircuitBreakerOpenError:
            logger.warning("DecisionAgent circuit breaker OPEN — CANCEL_ALL fallback.")
            decision = self._make_fallback(
                reason="CIRCUIT_BREAKER_OPEN",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            logger.exception("DecisionAgent unexpected error: %s", exc)
            decision = self._make_fallback(
                reason=f"ERROR: {type(exc).__name__}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        total_latency = (time.monotonic() - t0) * 1000
        audit_entry = (
            f"[DECISION] action={decision.action} side={decision.side} "
            f"confidence={decision.confidence:.2f} fallback={decision.fallback_triggered} "
            f"latency_ms={total_latency:.1f}"
        )

        logger.info(
            "Decision made | action=%s side=%s confidence=%.2f latency_ms=%.1f",
            decision.action, decision.side, decision.confidence, total_latency,
        )

        return {
            "decision": decision,
            "total_pipeline_latency_ms": (
                (state["alert"].pipeline_latency_ms or 0.0) + total_latency
            ),
            "audit_log": [audit_entry],
        }

    def _build_prompt(self, alert: Any, research: Optional[ResearchContext]) -> str:
        metrics = alert.metrics
        lines = [
            f"ALERT: {alert.trigger} | Severity: {alert.severity} | Symbol: {alert.symbol}",
            f"OBI={metrics.obi:.4f} | VPIN={metrics.vpin:.4f} | "
            f"spread={metrics.spread_bps:.3f}bps | mid={metrics.mid_price}",
            f"toxic_flow={metrics.toxic_flow_detected} | "
            f"volatility_spike={metrics.volatility_spike_detected}",
        ]
        if research:
            lines.append(f"\nRESEARCH CONTEXT:\n{research.synthesized_context}")
        lines.append("\nProvide your decision as valid JSON only.")
        return "\n".join(lines)

    @staticmethod
    def _parse_decision(raw_json: str, latency_ms: float) -> ExecutionDecision:
        """
        Strict parse with fallback on any failure.
        Strips markdown code fences if the model adds them.
        """
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()
        try:
            data = json.loads(cleaned)
            return ExecutionDecision(
                action=data["action"],
                side=data["side"],
                confidence=float(data["confidence"]),
                reasoning=str(data.get("reasoning", ""))[:2048],
                fallback_triggered=False,
                decision_latency_ms=latency_ms,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Decision parse failed (%s) — falling back to CANCEL_ALL.", exc)
            return DecisionAgent._make_fallback_static(
                reason=f"PARSE_ERROR: {exc}", latency_ms=latency_ms
            )

    @staticmethod
    def _make_fallback_static(reason: str, latency_ms: float) -> ExecutionDecision:
        return ExecutionDecision(
            action="CANCEL_ALL",
            side="NONE",
            confidence=1.0,
            reasoning=f"Safe-state fallback triggered: {reason}",
            fallback_triggered=True,
            decision_latency_ms=latency_ms,
        )

    def _make_fallback(self, reason: str, latency_ms: float) -> ExecutionDecision:
        return self._make_fallback_static(reason, latency_ms)
