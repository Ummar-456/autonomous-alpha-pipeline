"""
circuit_breaker.py — Three-state circuit breaker for LLM / external API calls.

States:
    CLOSED     — normal operation, calls pass through
    OPEN       — too many failures, calls rejected immediately (fast-fail)
    HALF_OPEN  — probe state after recovery_timeout_s elapses
"""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum, auto
from typing import Callable, TypeVar, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit is OPEN and a call is attempted."""


class CircuitBreaker:
    """
    Async-compatible circuit breaker.

    Usage:
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout_s=30.0)

        async def make_llm_call():
            return await some_llm_api()

        result = await cb.call(make_llm_call)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        success_threshold: int = 2,
        name: str = "default",
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout_s = recovery_timeout_s
        self._success_threshold = success_threshold
        self._name = name

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        self._check_recovery()
        return self._state

    async def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute `fn` through the circuit breaker."""
        self._check_recovery()

        if self._state == CircuitState.OPEN:
            logger.warning("CircuitBreaker[%s] OPEN — rejecting call.", self._name)
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self._name}' is OPEN. "
                f"Retry after {self._recovery_timeout_s}s."
            )

        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
            else:
                result = fn(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerOpenError:
            raise
        except Exception as exc:
            self._on_failure(exc)
            raise

    def _check_recovery(self) -> None:
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._opened_at >= self._recovery_timeout_s
        ):
            logger.info("CircuitBreaker[%s] → HALF_OPEN (probing).", self._name)
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._success_threshold:
                logger.info("CircuitBreaker[%s] → CLOSED (recovered).", self._name)
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def _on_failure(self, exc: Exception) -> None:
        self._failure_count += 1
        logger.warning(
            "CircuitBreaker[%s] failure %d/%d: %s",
            self._name, self._failure_count, self._failure_threshold, exc,
        )
        if self._failure_count >= self._failure_threshold:
            logger.error("CircuitBreaker[%s] → OPEN (threshold reached).", self._name)
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def reset(self) -> None:
        """Manually reset to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = 0.0
