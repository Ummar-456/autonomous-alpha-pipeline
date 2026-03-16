"""backoff.py — Exponential backoff with full jitter for external API retries."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Callable, TypeVar, Any, Type

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def exponential_backoff_with_jitter(
    fn: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """
    Full-jitter exponential backoff:
        delay ~ Uniform(0, min(max_delay, base * 2^attempt))

    Avoids thundering-herd when multiple instances retry simultaneously.
    CancelledError is never caught — it always propagates.
    """
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(retries + 1):
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            return fn(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except exceptions as exc:
            last_exc = exc
            if attempt == retries:
                break
            delay = random.uniform(0, min(max_delay_s, base_delay_s * (2 ** attempt)))
            logger.warning(
                "Backoff attempt %d/%d — retrying in %.2fs (%s: %s)",
                attempt + 1, retries, delay, type(exc).__name__, exc,
            )
            await asyncio.sleep(delay)

    raise last_exc
