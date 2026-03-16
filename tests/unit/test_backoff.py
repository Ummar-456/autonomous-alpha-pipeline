"""Unit tests for exponential_backoff_with_jitter — 8 tests."""
from __future__ import annotations

import asyncio

import pytest

from alpha_pipeline.utils.backoff import exponential_backoff_with_jitter


class TestBackoffSuccess:
    @pytest.mark.asyncio
    async def test_returns_value_on_first_success(self):
        async def fn():
            return 42

        result = await exponential_backoff_with_jitter(fn, retries=3)
        assert result == 42

    @pytest.mark.asyncio
    async def test_sync_function_supported(self):
        def sync_fn(x):
            return x * 2

        result = await exponential_backoff_with_jitter(sync_fn, 5, retries=2)
        assert result == 10

    @pytest.mark.asyncio
    async def test_succeeds_after_transient_failures(self):
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "ok"

        result = await exponential_backoff_with_jitter(
            flaky, retries=5, base_delay_s=0.001, max_delay_s=0.01
        )
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_kwargs_forwarded_to_function(self):
        async def fn(a, b=10):
            return a + b

        result = await exponential_backoff_with_jitter(fn, 5, retries=1, b=20)
        assert result == 25


class TestBackoffFailure:
    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self):
        async def always_fail():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            await exponential_backoff_with_jitter(
                always_fail, retries=3, base_delay_s=0.001
            )

    @pytest.mark.asyncio
    async def test_respects_exceptions_filter(self):
        """Only retries on specified exception types; others propagate immediately."""
        call_count = 0

        async def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            await exponential_backoff_with_jitter(
                raises_type_error,
                retries=3,
                base_delay_s=0.001,
                exceptions=(ValueError,),  # only retry on ValueError, not TypeError
            )
        # Should not retry — TypeError is not in exceptions tuple
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_immediately(self):
        """CancelledError must never be swallowed — it always propagates."""
        async def raises_cancelled():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await exponential_backoff_with_jitter(raises_cancelled, retries=5)

    @pytest.mark.asyncio
    async def test_retry_count_matches_expected_calls(self):
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await exponential_backoff_with_jitter(
                always_fail, retries=2, base_delay_s=0.001
            )
        # retries=2 means: 1 initial attempt + 2 retries = 3 total calls
        assert call_count == 3
