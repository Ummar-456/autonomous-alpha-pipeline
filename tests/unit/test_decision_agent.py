"""Unit tests for DecisionAgent — 10 tests."""
from __future__ import annotations

import asyncio

import pytest

from alpha_pipeline.agents.decision import DecisionAgent
from alpha_pipeline.config import DecisionAgentConfig
from alpha_pipeline.utils.circuit_breaker import CircuitBreaker


class TestDecisionAgentNormal:
    @pytest.mark.asyncio
    async def test_cancel_all_response_parsed_correctly(self, mock_llm_cancel_all, pipeline_state):
        config = DecisionAgentConfig(llm_timeout_ms=2000)
        agent = DecisionAgent(config=config, llm_client=mock_llm_cancel_all)
        update = await agent.invoke_async(pipeline_state)
        decision = update["decision"]
        assert decision.action == "CANCEL_ALL"
        assert decision.fallback_triggered is False

    @pytest.mark.asyncio
    async def test_provide_liquidity_response_parsed(self, mock_llm_provide_liquidity, pipeline_state):
        config = DecisionAgentConfig(llm_timeout_ms=2000)
        agent = DecisionAgent(config=config, llm_client=mock_llm_provide_liquidity)
        update = await agent.invoke_async(pipeline_state)
        decision = update["decision"]
        assert decision.action == "PROVIDE_LIQUIDITY"
        assert decision.side == "BOTH"
        assert 0.0 <= decision.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_audit_log_entry_appended(self, mock_llm_cancel_all, pipeline_state):
        config = DecisionAgentConfig(llm_timeout_ms=2000)
        agent = DecisionAgent(config=config, llm_client=mock_llm_cancel_all)
        update = await agent.invoke_async(pipeline_state)
        assert len(update["audit_log"]) == 1
        assert "[DECISION]" in update["audit_log"][0]

    @pytest.mark.asyncio
    async def test_total_latency_populated(self, mock_llm_cancel_all, pipeline_state):
        config = DecisionAgentConfig(llm_timeout_ms=2000)
        agent = DecisionAgent(config=config, llm_client=mock_llm_cancel_all)
        update = await agent.invoke_async(pipeline_state)
        assert update["total_pipeline_latency_ms"] is not None
        assert update["total_pipeline_latency_ms"] > 0


class TestDecisionAgentFallbacks:
    @pytest.mark.asyncio
    async def test_timeout_triggers_cancel_all_fallback(self, mock_llm_timeout, pipeline_state):
        config = DecisionAgentConfig(llm_timeout_ms=50)  # 50ms — mock delays 1000ms
        agent = DecisionAgent(config=config, llm_client=mock_llm_timeout)
        update = await agent.invoke_async(pipeline_state)
        decision = update["decision"]
        assert decision.action == "CANCEL_ALL"
        assert decision.fallback_triggered is True

    @pytest.mark.asyncio
    async def test_bad_json_triggers_cancel_all_fallback(self, mock_llm_bad_json, pipeline_state):
        config = DecisionAgentConfig(llm_timeout_ms=2000)
        agent = DecisionAgent(config=config, llm_client=mock_llm_bad_json)
        update = await agent.invoke_async(pipeline_state)
        decision = update["decision"]
        assert decision.action == "CANCEL_ALL"
        assert decision.fallback_triggered is True

    @pytest.mark.asyncio
    async def test_llm_exception_triggers_fallback(self, pipeline_state):
        from tests.conftest import MockLLMClient
        llm = MockLLMClient(raise_exc=RuntimeError("LLM API down"))
        config = DecisionAgentConfig(llm_timeout_ms=2000)
        agent = DecisionAgent(config=config, llm_client=llm)
        update = await agent.invoke_async(pipeline_state)
        assert update["decision"].fallback_triggered is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_triggers_fallback(self, pipeline_state):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=60.0, name="test_cb")
        # Force the breaker open
        async def broken(): raise ValueError("force open")
        try:
            await cb.call(broken)
        except ValueError:
            pass

        from tests.conftest import MockLLMClient
        config = DecisionAgentConfig(llm_timeout_ms=2000)
        agent = DecisionAgent(
            config=config,
            llm_client=MockLLMClient(),
            circuit_breaker=cb,
        )
        update = await agent.invoke_async(pipeline_state)
        # cb is OPEN after 1 failure with threshold=1
        assert update["decision"].fallback_triggered is True


class TestDecisionAgentParsing:
    @pytest.mark.asyncio
    async def test_markdown_fence_stripped(self, pipeline_state):
        from tests.conftest import MockLLMClient
        fenced = '```json\n{"action":"HOLD","side":"NONE","confidence":0.5,"reasoning":"ok"}\n```'
        llm = MockLLMClient(fenced)
        config = DecisionAgentConfig(llm_timeout_ms=2000)
        agent = DecisionAgent(config=config, llm_client=llm)
        update = await agent.invoke_async(pipeline_state)
        assert update["decision"].action == "HOLD"
        assert update["decision"].fallback_triggered is False

    @pytest.mark.asyncio
    async def test_all_valid_actions_accepted(self, pipeline_state):
        from tests.conftest import MockLLMClient
        for action in ["PROVIDE_LIQUIDITY", "TAKE_LIQUIDITY", "WIDEN_QUOTES", "CANCEL_ALL", "HOLD"]:
            resp = f'{{"action":"{action}","side":"NONE","confidence":0.5,"reasoning":"test"}}'
            llm = MockLLMClient(resp)
            config = DecisionAgentConfig(llm_timeout_ms=2000)
            agent = DecisionAgent(config=config, llm_client=llm)
            update = await agent.invoke_async(pipeline_state)
            assert update["decision"].action == action
