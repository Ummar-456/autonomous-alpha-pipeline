"""Unit tests for CircuitBreaker — 10 tests."""
from __future__ import annotations

import asyncio
import time

import pytest

from alpha_pipeline.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)


class TestCircuitBreakerClosed:
    @pytest.mark.asyncio
    async def test_passes_through_in_closed_state(self):
        cb = CircuitBreaker(failure_threshold=3, name="test")

        async def fn() -> int:
            return 42

        result = await cb.call(fn)
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_resets_failure_count_on_success(self):
        cb = CircuitBreaker(failure_threshold=3, name="test")

        async def fail():
            raise ValueError("boom")

        async def succeed():
            return "ok"

        with pytest.raises(ValueError):
            await cb.call(fail)
        # Success should reset failure count
        await cb.call(succeed)
        assert cb._failure_count == 0
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerOpens:
    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3, name="test")

        async def fail():
            raise ValueError("boom")

        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(fail)

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_rejects_calls_when_open(self):
        cb = CircuitBreaker(failure_threshold=2, name="test")

        async def fail():
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(fail)

        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(fail)

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_s=0.01, name="test")

        async def fail():
            raise ValueError("boom")

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(fail)

        await asyncio.sleep(0.05)  # wait for recovery timeout
        assert cb.state == CircuitState.HALF_OPEN


class TestCircuitBreakerHalfOpen:
    @pytest.mark.asyncio
    async def test_closes_after_success_threshold_in_half_open(self):
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout_s=0.01,
            success_threshold=2,
            name="test",
        )

        async def fail():
            raise ValueError("boom")

        async def succeed():
            return "ok"

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(fail)

        await asyncio.sleep(0.05)
        await cb.call(succeed)
        await cb.call(succeed)
        assert cb.state == CircuitState.CLOSED

    def test_manual_reset(self):
        cb = CircuitBreaker(failure_threshold=1, name="test")
        cb._state = CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0


class TestCircuitBreakerSyncFunctions:
    @pytest.mark.asyncio
    async def test_sync_function_called_correctly(self):
        cb = CircuitBreaker(failure_threshold=3, name="test")

        def sync_fn(x, y):
            return x + y

        result = await cb.call(sync_fn, 3, 4)
        assert result == 7
