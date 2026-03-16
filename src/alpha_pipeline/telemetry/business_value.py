"""
business_value.py — Analyst-hours-saved tracker.

Quantifies the business impact of the pipeline in terms of analyst time
replaced. Each automated alert resolution represents an estimated N hours
of manual analysis (configurable). The tracker maintains cumulative totals
and per-day breakdowns for the Grafana dashboard.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import NamedTuple

logger = logging.getLogger(__name__)


class DailyTotals(NamedTuple):
    date_str: str
    alerts_processed: int
    analyst_hours_saved: float
    llm_fallbacks: int
    avg_latency_ms: float


class BusinessValueTracker:
    """
    Tracks the ROI of the pipeline in analyst-hours saved.

    Analyst-time model:
        Each CANCEL_ALL decision prevents a manual review that takes
        `hours_per_cancel_all` hours (examining LOB, reading news, deciding action).
        Each PROVIDE/TAKE/WIDEN decision prevents `hours_per_routine` hours of
        routine signal-to-decision work.
        Fallback decisions are still counted — the system still acted.
    """

    HOURS_PER_CANCEL_ALL: float = 1.5   # high-urgency incident review
    HOURS_PER_ROUTINE: float = 0.25     # routine signal processing
    HOURS_PER_FALLBACK: float = 0.1     # conservative: fallback was safe-state

    def __init__(self) -> None:
        self._started_at = time.time()
        self._total_hours: float = 0.0
        self._total_alerts: int = 0
        self._total_fallbacks: int = 0
        self._daily: dict[str, dict[str, float]] = {}

    def record_decision(
        self,
        action: str,
        fallback_triggered: bool,
        latency_ms: float,
    ) -> None:
        self._total_alerts += 1

        if fallback_triggered:
            hours = self.HOURS_PER_FALLBACK
            self._total_fallbacks += 1
        elif action == "CANCEL_ALL":
            hours = self.HOURS_PER_CANCEL_ALL
        else:
            hours = self.HOURS_PER_ROUTINE

        self._total_hours += hours
        self._record_daily(hours, fallback_triggered, latency_ms)

        logger.debug(
            "BusinessValue: +%.2f hours | action=%s fallback=%s "
            "cumulative_hours=%.2f",
            hours, action, fallback_triggered, self._total_hours,
        )

    def _record_daily(self, hours: float, fallback: bool, latency_ms: float) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if day not in self._daily:
            self._daily[day] = {
                "hours": 0.0, "alerts": 0, "fallbacks": 0,
                "latency_sum": 0.0, "latency_count": 0,
            }
        d = self._daily[day]
        d["hours"] += hours
        d["alerts"] += 1
        if fallback:
            d["fallbacks"] += 1
        d["latency_sum"] += latency_ms
        d["latency_count"] += 1

    def daily_summary(self, date_str: str | None = None) -> DailyTotals | None:
        day = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        d = self._daily.get(day)
        if d is None:
            return None
        avg_lat = d["latency_sum"] / d["latency_count"] if d["latency_count"] > 0 else 0.0
        return DailyTotals(
            date_str=day,
            alerts_processed=int(d["alerts"]),
            analyst_hours_saved=d["hours"],
            llm_fallbacks=int(d["fallbacks"]),
            avg_latency_ms=avg_lat,
        )

    def annualized_fte_equivalent(self) -> float:
        """
        Project total hours saved to an annual FTE equivalent.
        FTE = (hours_saved_per_day * 252 trading_days) / 2000 working_hours_per_year
        """
        elapsed_days = max((time.time() - self._started_at) / 86400, 1 / 86400)
        daily_rate = self._total_hours / elapsed_days
        return (daily_rate * 252) / 2000

    @property
    def total_hours_saved(self) -> float:
        return self._total_hours

    @property
    def total_alerts_processed(self) -> int:
        return self._total_alerts

    @property
    def fallback_rate(self) -> float:
        if self._total_alerts == 0:
            return 0.0
        return self._total_fallbacks / self._total_alerts
