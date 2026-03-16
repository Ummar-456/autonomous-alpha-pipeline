"""
metrics.py — Real-time microstructure metrics engine.

OBI   : Cont, Kukanov & Stoikov (2014)
VPIN  : Easley, Lopez de Prado & O'Hara (2012)
"""
from __future__ import annotations

import math
import logging
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Deque, Optional

from .lob import LimitOrderBook
from ..state import MicrostructureMetrics

logger = logging.getLogger(__name__)


@dataclass
class _VPINBucket:
    buy_volume: Decimal = field(default_factory=lambda: Decimal("0"))
    sell_volume: Decimal = field(default_factory=lambda: Decimal("0"))

    @property
    def total_volume(self) -> Decimal:
        return self.buy_volume + self.sell_volume

    @property
    def signed_imbalance(self) -> Decimal:
        total = self.total_volume
        if total == Decimal("0"):
            return Decimal("0")
        return abs(self.buy_volume - self.sell_volume) / total


class MetricsEngine:
    """
    Stateful metrics calculator. All computation is pure Python, runs on
    the asyncio event-loop thread without blocking I/O.
    """

    def __init__(
        self,
        obi_levels: int = 5,
        vpin_bucket_size: Decimal = Decimal("10"),
        vpin_window: int = 50,
        spread_window: int = 100,
        obi_toxicity_threshold: float = 0.7,
        vpin_toxicity_threshold: float = 0.5,
        spread_z_threshold: float = 2.0,
    ) -> None:
        self._obi_levels = obi_levels
        self._bucket_size = vpin_bucket_size
        self._vpin_window = vpin_window
        self._obi_toxicity_threshold = obi_toxicity_threshold
        self._vpin_toxicity_threshold = vpin_toxicity_threshold
        self._spread_z_threshold = spread_z_threshold

        self._spread_history: Deque[float] = deque(maxlen=spread_window)
        self._completed_buckets: Deque[_VPINBucket] = deque(maxlen=vpin_window)
        self._active_bucket: _VPINBucket = _VPINBucket()
        self._prev_mid: Optional[Decimal] = None

    def compute(self, lob: LimitOrderBook) -> Optional[MicrostructureMetrics]:
        """
        Returns None if book is empty or crossed.
        """
        best_bid = lob.best_bid()
        best_ask = lob.best_ask()

        if best_bid is None or best_ask is None:
            return None

        bid_price, _ = best_bid
        ask_price, _ = best_ask

        if ask_price <= bid_price:
            logger.warning("Crossed book — skipping metrics | bid=%s ask=%s", bid_price, ask_price)
            return None

        mid_price = (bid_price + ask_price) / Decimal("2")
        spread_bps = float((ask_price - bid_price) / mid_price * Decimal("10000"))

        obi = self._compute_obi(lob)
        self._update_vpin(mid_price, lob)
        vpin = self._compute_vpin()

        self._spread_history.append(spread_bps)
        volatility_spike = self._is_volatility_spike(spread_bps)
        toxic_flow = (
            abs(obi) > self._obi_toxicity_threshold
            and vpin > self._vpin_toxicity_threshold
        )
        self._prev_mid = mid_price

        return MicrostructureMetrics(
            obi=obi,
            spread_bps=spread_bps,
            mid_price=mid_price,
            vpin=vpin,
            toxic_flow_detected=toxic_flow,
            volatility_spike_detected=volatility_spike,
        )

    def _compute_obi(self, lob: LimitOrderBook) -> float:
        bid_levels, ask_levels = lob.snapshot(self._obi_levels)
        bid_vol = sum((lvl.quantity for lvl in bid_levels), Decimal("0"))
        ask_vol = sum((lvl.quantity for lvl in ask_levels), Decimal("0"))
        total = bid_vol + ask_vol
        if total == Decimal("0"):
            return 0.0
        return float((bid_vol - ask_vol) / total)

    def _update_vpin(self, mid_price: Decimal, lob: LimitOrderBook) -> None:
        bid_levels, ask_levels = lob.snapshot(depth=1)
        if not bid_levels or not ask_levels:
            return

        tick_vol = bid_levels[0].quantity + ask_levels[0].quantity

        if self._prev_mid is None:
            buy_frac, sell_frac = Decimal("0.5"), Decimal("0.5")
        elif mid_price > self._prev_mid:
            buy_frac, sell_frac = Decimal("1"), Decimal("0")
        elif mid_price < self._prev_mid:
            buy_frac, sell_frac = Decimal("0"), Decimal("1")
        else:
            buy_frac, sell_frac = Decimal("0.5"), Decimal("0.5")

        remaining = tick_vol
        while remaining > Decimal("0"):
            space = self._bucket_size - self._active_bucket.total_volume
            if space <= Decimal("0"):
                self._completed_buckets.append(self._active_bucket)
                self._active_bucket = _VPINBucket()
                space = self._bucket_size
            fill = min(remaining, space)
            self._active_bucket.buy_volume += fill * buy_frac
            self._active_bucket.sell_volume += fill * sell_frac
            remaining -= fill

    def _compute_vpin(self) -> float:
        if len(self._completed_buckets) < 5:
            return 0.0
        return min(
            sum(float(b.signed_imbalance) for b in self._completed_buckets)
            / len(self._completed_buckets),
            1.0,
        )

    def _is_volatility_spike(self, current_bps: float) -> bool:
        n = len(self._spread_history)
        if n < 10:
            return False
        mean = sum(self._spread_history) / n
        variance = sum((x - mean) ** 2 for x in self._spread_history) / n
        std = math.sqrt(variance) if variance > 1e-12 else 1e-9
        z = (current_bps - mean) / std
        return z > self._spread_z_threshold

    @property
    def spread_history(self) -> list[float]:
        return list(self._spread_history)

    @property
    def completed_bucket_count(self) -> int:
        return len(self._completed_buckets)
