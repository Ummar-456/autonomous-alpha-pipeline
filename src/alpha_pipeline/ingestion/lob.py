"""
lob.py — In-memory Limit Order Book.

O(k) per delta update; effectively O(1) for typical k=1–5 changed levels.
Thread-safety: NOT thread-safe. Designed for single-threaded asyncio use.
"""
from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from typing import Optional

from ..state import OrderBookSnapshot, PriceLevel

logger = logging.getLogger(__name__)


class LimitOrderBook:
    __slots__ = ("symbol", "_bids", "_asks", "_last_update_id", "_depth")

    def __init__(self, symbol: str, depth: int = 20) -> None:
        self.symbol: str = symbol
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._last_update_id: int = 0
        self._depth: int = depth

    def apply_snapshot(
        self,
        bids: list[list[str]],
        asks: list[list[str]],
        last_update_id: int,
    ) -> None:
        """Full book replacement — called on initial connection or re-sync."""
        self._bids.clear()
        self._asks.clear()
        for level in bids[: self._depth]:
            self._upsert(self._bids, level[0], level[1])
        for level in asks[: self._depth]:
            self._upsert(self._asks, level[0], level[1])
        self._last_update_id = last_update_id
        logger.debug(
            "LOB snapshot applied | symbol=%s update_id=%d bid_levels=%d ask_levels=%d",
            self.symbol, last_update_id, len(self._bids), len(self._asks),
        )

    def apply_delta(
        self,
        bids: list[list[str]],
        asks: list[list[str]],
        final_update_id: int,
    ) -> bool:
        """
        Apply incremental depth update.
        Returns False when a sequence gap is detected — caller must re-sync.
        """
        if final_update_id <= self._last_update_id:
            return True  # stale, safe to discard

        if final_update_id > self._last_update_id + 1 and self._last_update_id != 0:
            logger.warning(
                "Sequence gap | expected=%d got=%d",
                self._last_update_id + 1, final_update_id,
            )
            return False

        for level in bids:
            self._upsert(self._bids, level[0], level[1])
        for level in asks:
            self._upsert(self._asks, level[0], level[1])
        self._last_update_id = final_update_id
        return True

    def best_bid(self) -> Optional[tuple[Decimal, Decimal]]:
        if not self._bids:
            return None
        price = max(self._bids)
        return price, self._bids[price]

    def best_ask(self) -> Optional[tuple[Decimal, Decimal]]:
        if not self._asks:
            return None
        price = min(self._asks)
        return price, self._asks[price]

    def snapshot(self, depth: int = 5) -> tuple[list[PriceLevel], list[PriceLevel]]:
        bids = sorted(self._bids.items(), key=lambda x: x[0], reverse=True)
        asks = sorted(self._asks.items(), key=lambda x: x[0])
        return (
            [PriceLevel(price=p, quantity=q) for p, q in bids[:depth]],
            [PriceLevel(price=p, quantity=q) for p, q in asks[:depth]],
        )

    def to_order_book_snapshot(self, symbol: str, timestamp_ns: int, depth: int = 5) -> OrderBookSnapshot:
        bids, asks = self.snapshot(depth)
        return OrderBookSnapshot(
            symbol=symbol,
            timestamp_ns=timestamp_ns,
            bids=bids,
            asks=asks,
            last_update_id=self._last_update_id,
        )

    @property
    def last_update_id(self) -> int:
        return self._last_update_id

    @property
    def is_populated(self) -> bool:
        return bool(self._bids and self._asks)

    @staticmethod
    def _upsert(side: dict[Decimal, Decimal], price_str: str, qty_str: str) -> None:
        try:
            price = Decimal(price_str)
            qty = Decimal(qty_str)
        except InvalidOperation:
            logger.warning("Skipping malformed level: price=%r qty=%r", price_str, qty_str)
            return
        if qty == Decimal("0"):
            side.pop(price, None)
        else:
            side[price] = qty
