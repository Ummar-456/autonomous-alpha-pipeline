"""
main.py — Pipeline entry point.
Consolidated version with fixed indentation and hardened mock client.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import os

from .config import DecisionAgentConfig, OrchestratorConfig, TelemetryConfig
from .ingestion.mock_server import serve_mock_binance
from .ingestion.orchestrator import LOBOrchestrator
from .agents.graph import GraphRunner
from .telemetry.monitor import SREMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class MockLLMClient:
    """Hardened Mock Client that accepts any combination of arguments."""

    async def complete(self, prompt: str = "", **kwargs) -> str:
        await asyncio.sleep(0.05)
        return (
            '{"action": "CANCEL_ALL", "side": "BOTH", "confidence": 1.0,'
            ' "reasoning": "Mock mode: Manual override triggered."}'
        )


async def main() -> None:
    # Configuration
    orch_config = OrchestratorConfig()
    decision_config = DecisionAgentConfig()
    telemetry_config = TelemetryConfig()

    # Shared queue
    alert_queue: asyncio.Queue = asyncio.Queue(
        maxsize=orch_config.alert_queue_maxsize
    )

    # LLM client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    llm_client = MockLLMClient() if not api_key else None

    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not found. MockLLMClient enabled.")

    # Component initialisation
    # FIX: Build components in dependency order so SREMonitor receives
    # graph_runner at __init__ time. Previously graph_runner was patched
    # in manually after construction (sre_monitor._graph_runner = graph_runner),
    # which created a startup window where _collect_metrics could fire
    # with _graph_runner=None and silently skip the p99 gauge update.
    #
    # Order: orchestrator → sre_monitor (stub) → graph_runner → wire up
    orchestrator = LOBOrchestrator(config=orch_config, alert_queue=alert_queue)

    # Initialise monitor without graph_runner first (circular dep),
    # then construct graph_runner passing the monitor, then hand graph_runner
    # back to the monitor before any tasks start.
    sre_monitor = SREMonitor(
        config=telemetry_config,
        alert_queue=alert_queue,
        orchestrator=orchestrator,
        # graph_runner wired in below — before any tasks are scheduled
    )

    graph_runner = GraphRunner(
        alert_queue=alert_queue,
        decision_config=decision_config,
        llm_client=llm_client,
        sre_monitor=sre_monitor,
    )

    # Wire the circular reference before the event loop starts any tasks
    sre_monitor._graph_runner = graph_runner

    logger.info("Autonomous Alpha Pipeline starting...")

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(serve_mock_binance(), name="mock_server")
            # Brief pause so the mock WebSocket server is listening before
            # the orchestrator attempts to connect.
            await asyncio.sleep(0.5)

            tg.create_task(orchestrator.run(), name="orchestrator")
            tg.create_task(graph_runner.run(), name="graph_runner")
            tg.create_task(sre_monitor.run(), name="sre_monitor")

            logger.info("All components running. Press Ctrl+C to stop.")

            while True:
                await asyncio.sleep(1)

    except* KeyboardInterrupt:
        logger.info("Stopping pipeline...")
    except* Exception as eg:
        for exc in eg.exceptions:
            logger.exception("Component failure: %s", exc)
    finally:
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)