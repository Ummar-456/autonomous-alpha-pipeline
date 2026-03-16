"""Unit tests for LimitOrderBook — 16 tests."""
from __future__ import annotations

from decimal import Decimal

import pytest

from alpha_pipeline.ingestion.lob import LimitOrderBook
from alpha_pipeline.state import PriceLevel


def _make_lob(depth: int = 10) -> LimitOrderBook:
    return LimitOrderBook("BTCUSDT", depth=depth)


def _seed(lob: LimitOrderBook, base: str = "100.0", spread: str = "1.0", n: int = 5) -> int:
    b = Decimal(base)
    s = Decimal(spread)
    bids = [[str(b - Decimal(str(i)) * Decimal("0.5")), "1.0"] for i in range(n)]
    asks = [[str(b + s + Decimal(str(i)) * Decimal("0.5")), "0.9"] for i in range(n)]
    lob.apply_snapshot(bids, asks, last_update_id=1000)
    return 1000


class TestLOBSnapshot:
    def test_snapshot_populates_bids_and_asks(self):
        lob = _make_lob()
        _seed(lob)
        assert lob.is_populated

    def test_best_bid_is_highest(self):
        lob = _make_lob()
        _seed(lob, base="100.0")
        best_bid = lob.best_bid()
        assert best_bid is not None
        assert best_bid[0] == Decimal("100.0")

    def test_best_ask_is_lowest(self):
        lob = _make_lob()
        _seed(lob, base="100.0", spread="1.0")
        best_ask = lob.best_ask()
        assert best_ask is not None
        assert best_ask[0] == Decimal("101.0")

    def test_empty_book_returns_none_best_bid(self):
        lob = _make_lob()
        assert lob.best_bid() is None

    def test_empty_book_returns_none_best_ask(self):
        lob = _make_lob()
        assert lob.best_ask() is None

    def test_snapshot_clears_previous_state(self):
        lob = _make_lob()
        _seed(lob, base="50.0")
        _seed(lob, base="200.0")
        assert lob.best_bid()[0] == Decimal("200.0")

    def test_depth_limit_respected(self):
        lob = LimitOrderBook("BTCUSDT", depth=3)
        bids = [[str(100 - i), "1.0"] for i in range(10)]
        asks = [[str(101 + i), "1.0"] for i in range(10)]
        lob.apply_snapshot(bids, asks, last_update_id=1)
        b, a = lob.snapshot(depth=10)
        assert len(b) <= 3
        assert len(a) <= 3

    def test_snapshot_returns_price_levels(self):
        lob = _make_lob()
        _seed(lob)
        bids, asks = lob.snapshot(depth=3)
        assert all(isinstance(pl, PriceLevel) for pl in bids)
        assert all(isinstance(pl, PriceLevel) for pl in asks)

    def test_bids_sorted_descending(self):
        lob = _make_lob()
        _seed(lob, n=5)
        bids, _ = lob.snapshot(depth=5)
        prices = [pl.price for pl in bids]
        assert prices == sorted(prices, reverse=True)

    def test_asks_sorted_ascending(self):
        lob = _make_lob()
        _seed(lob, n=5)
        _, asks = lob.snapshot(depth=5)
        prices = [pl.price for pl in asks]
        assert prices == sorted(prices)


class TestLOBDelta:
    def test_delta_updates_existing_level(self):
        lob = _make_lob()
        _seed(lob, base="100.0")
        lob.apply_delta([["100.0", "5.0"]], [], final_update_id=1001)
        assert lob._bids[Decimal("100.0")] == Decimal("5.0")

    def test_delta_removes_zero_quantity(self):
        lob = _make_lob()
        _seed(lob, base="100.0")
        lob.apply_delta([["100.0", "0"]], [], final_update_id=1001)
        assert Decimal("100.0") not in lob._bids

    def test_delta_discards_stale_update_id(self):
        lob = _make_lob()
        _seed(lob)
        result = lob.apply_delta([["200.0", "1.0"]], [], final_update_id=999)
        assert result is True
        assert Decimal("200.0") not in lob._bids

    def test_delta_detects_sequence_gap(self):
        lob = _make_lob()
        _seed(lob)
        result = lob.apply_delta([], [], final_update_id=1005)
        assert result is False

    def test_delta_accepts_sequential_update(self):
        lob = _make_lob()
        _seed(lob)
        result = lob.apply_delta([["90.0", "2.5"]], [], final_update_id=1001)
        assert result is True

    def test_malformed_price_level_skipped(self):
        lob = _make_lob()
        _seed(lob)
        # Should not raise; malformed level is silently dropped
        lob.apply_delta([["not_a_number", "1.0"]], [], final_update_id=1001)
        assert lob.is_populated
