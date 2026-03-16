# Autonomous Alpha Pipeline

A production-grade, multi-agent, event-driven orchestration system for real-time market microstructure analysis. Ingests a simulated high-frequency Binance LOB feed at 10 Hz, contextualises signals via a RAG system, routes structured execution decisions through a LangGraph state machine, and surfaces everything through a live Prometheus + Grafana observability stack.

**Live dashboard results (BTC/USDT, 15-minute window):**

| Metric                        | Value   |
| ----------------------------- | ------- |
| Alerts processed autonomously | 121     |
| Analyst hours saved           | 60.5 h  |
| Pipeline P99 latency          | 99.5 ms |
| Packet drop rate              | 0 %     |
| WebSocket reconnections       | 0       |
| LLM fallback rate             | 0 %     |

---

## What this demonstrates

Built to show production AI engineering capabilities at the intersection of quantitative finance and modern LLM orchestration:

- **asyncio concurrency** — LOB ingestion, graph execution, and SRE monitoring run as independent `asyncio.Task`s sharing a bounded queue. The event loop is never blocked by LLM latency.
- **LangGraph state machines** — a typed `PipelineState` TypedDict threads through Researcher and Decision nodes with conditional routing and append-only audit logging.
- **Pydantic-strict contracts** — every byte crossing a component boundary is a frozen `BaseModel`. No raw dicts, no stringly-typed payloads.
- **Circuit-breaker resilience** — a three-state CLOSED/OPEN/HALF_OPEN FSM protects the LLM API from call storms. Any timeout exceeding 500ms automatically falls back to a safe `CANCEL_ALL` state.
- **SRE-grade telemetry** — thread-safe Counter/Histogram/Gauge primitives expose a Prometheus text endpoint scraped by Grafana every 5 seconds.
- **Business value quantification** — the pipeline tracks analyst-hours saved per decision and projects an annualised FTE equivalent.

---

## Architecture

```
WebSocket feed (mock Binance depth stream, 10 Hz)
        │
        ▼  recv_ns timestamp captured on every message
LOBOrchestrator
  · O(1) price-level upsert/delete (Decimal arithmetic — no float drift)
  · Order Book Imbalance  — Cont, Kukanov & Stoikov (2014)
  · Spread z-score spike detection
  · Simplified VPIN       — Easley, Lopez de Prado & O'Hara (2012)
  · Full-jitter exponential backoff reconnect
        │
        │  asyncio.Queue (bounded — backpressure by drop, not by blocking)
        ▼
GraphRunner ──► asyncio.to_thread(graph.invoke)
                        │
                        ▼
               LangGraph StateGraph
                 ├─ Researcher node
                 │    · TF-IDF cosine similarity retrieval
                 │    · 8-document quantitative research corpus
                 │    · Synthesises context for the LLM prompt
                 │
                 └─ Decision node  (only reached on HIGH/CRITICAL alerts)
                      · asyncio.wait_for(llm_call, timeout=0.5s)
                      · Strict JSON parse with CANCEL_ALL fallback
                      · Circuit breaker — 5-failure threshold, 30s recovery
                        │
                        ▼
               SREMonitor (background asyncio.Task)
                 · Prometheus HTTP exposition  :8000/metrics
                 · p50/p95/p99 latency tracking
                 · Analyst-hours-saved accumulator
                 · Heartbeat health log every 5 s
```

**Concurrency model in one sentence:** the LOB Orchestrator produces at 10 msg/s; `asyncio.to_thread` keeps the event loop unblocked during LangGraph traversal; the SRE monitor reads shared counters via a thread-safe lock without coordination overhead.

---

## Microstructure signals

### Order Book Imbalance (OBI)

```
OBI = (Σ_N bid_qty − Σ_N ask_qty) / (Σ_N bid_qty + Σ_N ask_qty)
```

∈ [-1, 1]. Values above 0.7 or below -0.7, combined with elevated VPIN, indicate informed directional flow. Computed over the top-5 price levels by default.

### Simplified VPIN

```
VPIN = (1/n) Σ |V_buy^τ − V_sell^τ| / V_bucket
```

Tick-rule buy/sell classification: rising mid-price → buy volume; falling → sell; flat → 50/50. Values above 0.5 signal toxic flow. Combined with `|OBI| > 0.7`, `CANCEL_ALL` is the theoretically optimal response — Easley et al. (2012) show that VPIN preceded the 2010 Flash Crash by 90 minutes.

### Spread volatility spike

Rolling z-score over the last 100 bid-ask spread observations in basis points. Spike flagged when `z > 2.0σ`.

---

## Decision output

The LLM is prompted with the structured alert payload and synthesised research context. It must respond with this exact JSON — no preamble, no markdown fences:

```json
{
  "action": "PROVIDE_LIQUIDITY | TAKE_LIQUIDITY | WIDEN_QUOTES | CANCEL_ALL | HOLD",
  "side":   "BID | ASK | BOTH | NONE",
  "confidence": 0.0–1.0,
  "reasoning": "max 200 chars"
}
```

**Fallback hierarchy** — any of the following collapses to `CANCEL_ALL, confidence=1.0, fallback_triggered=True`:

1. LLM response exceeds 500ms (`asyncio.TimeoutError`)
2. Circuit breaker is `OPEN` (≥ 5 consecutive failures)
3. JSON parse fails or a required field is missing or out of range

---

## Running it locally

### Prerequisites

- Python 3.12+

### 1. Set up the environment

```bash
cd autonomous-alpha-pipeline

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### 2. Run the test suite

No API key needed — every LLM call is replaced by an injected `MockLLMClient`.

```bash
PYTHONPATH=src pytest tests/ -v
# → 124 passed in ~8s

# With coverage
PYTHONPATH=src pytest tests/ --cov=alpha_pipeline --cov-report=term-missing
# → 124 passed · 84% coverage
```

### 3. Run the live pipeline

**Mock mode (no API key required):**

```bash
PYTHONPATH=src python -m alpha_pipeline.main
```

The pipeline runs fully — ingesting, computing metrics, routing through LangGraph, and emitting Prometheus metrics. Without an API key the Decision Agent always takes the CANCEL_ALL safe-state path. You will see structured logs:

```
10:42:01 [INFO]  Mock Binance server live at ws://localhost:8765
10:42:01 [INFO]  LOBOrchestrator starting | symbol=BTCUSDT
10:42:01 [INFO]  LOB seeded | best_bid=('43500.00', ...) best_ask=('43501.00', ...)
10:42:05 [INFO]  Alert emitted | severity=HIGH trigger=TOXIC_FLOW latency_ms=0.312
10:42:05 [INFO]  [RESEARCHER] completed in 2.1ms | docs=3 top_score=0.7821
10:42:05 [INFO]  Decision made | action=CANCEL_ALL confidence=1.0 latency_ms=0.8
10:42:10 [INFO]  SRE Heartbeat | uptime=10s | packets=100 | alerts=2 | hours_saved=0.20
```

**With live LLM decisions:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
PYTHONPATH=src python -m alpha_pipeline.main
```

Real reasoning now appears in the decision log and the LLM latency panels populate in Grafana:

```
Decision made | action=WIDEN_QUOTES side=BOTH confidence=0.84 latency_ms=312.4
```

### 4. Check the raw Prometheus metrics

While the pipeline is running:

```bash
curl http://localhost:8000/metrics
```

```
# HELP pipeline_packets_received_total Total WebSocket depth update messages received.
pipeline_packets_received_total 420.0
# HELP pipeline_graph_latency_ms_bucket End-to-end graph latency in milliseconds.
pipeline_graph_latency_ms_bucket{le="100"} 38
pipeline_graph_latency_ms_bucket{le="200"} 41
...
pipeline_analyst_hours_saved_total 20.5
```

### 5. Full observability stack with Grafana

Requires Docker Desktop.

```bash
cd docker
docker compose up --build
```

Open **http://localhost:3000** (login: `admin` / `admin`).

The "Autonomous Alpha Pipeline" dashboard auto-provisions — no manual import needed. Prometheus scrapes `:8000/metrics` every 5 seconds and you see the dashboard shown in the screenshots above updating live.

---

## Testing

```
124 tests · 0 failures · 84% coverage · ~8s
```

| File                      | Tests | What is covered                                                                        |
| ------------------------- | ----- | -------------------------------------------------------------------------------------- |
| `test_lob.py`             | 16    | Snapshot apply, delta updates, depth limits, sequence gaps, zero-qty removal           |
| `test_metrics.py`         | 13    | OBI bounds, spread BPS accuracy, VPIN warmup, volatility spike z-score                 |
| `test_state.py`           | 10    | Pydantic validation, frozen immutability, Literal enum rejection                       |
| `test_circuit_breaker.py` | 8     | CLOSED→OPEN→HALF_OPEN transitions, threshold, manual reset                             |
| `test_backoff.py`         | 8     | Success path, transient retry, exceptions filter, CancelledError propagation           |
| `test_rag.py`             | 12    | Cosine similarity ranking, top-k, relevance, researcher escalation routing             |
| `test_decision_agent.py`  | 10    | JSON parsing, 500ms timeout fallback, bad JSON fallback, markdown fence stripping      |
| `test_telemetry.py`       | 15    | Counter/Gauge/Histogram primitives, drop rate, Prometheus text format, FTE calculation |
| `test_monitor.py`         | 9     | Orchestrator counter sync, queue depth sync, HTTP 200/404 on `/metrics`                |
| `test_orchestrator.py`    | 8     | Reconnect counter, max-attempts halt, stop(), seed accuracy, CRITICAL severity mapping |
| `test_pipeline.py`        | 13    | LOB+metrics integration, message processing, full graph traversal, GraphRunner queue   |

**Isolation strategy:** all LLM calls are replaced by an injected `MockLLMClient`. Zero network I/O in any test. LangGraph graph traversal runs in `asyncio.to_thread` within integration tests exactly as in production.

---

## Project structure

```
autonomous-alpha-pipeline/
├── pyproject.toml
├── README.md
├── .env.example
│
├── src/alpha_pipeline/
│   ├── main.py                  # asyncio.TaskGroup entry point + graceful shutdown
│   ├── config.py                # Pydantic-frozen configs (injected, never global)
│   ├── state.py                 # All data contracts — frozen BaseModel throughout
│   │
│   ├── ingestion/
│   │   ├── lob.py               # LimitOrderBook — O(1) upsert/delete, gap detection
│   │   ├── metrics.py           # MetricsEngine — OBI, VPIN, spread z-score
│   │   ├── orchestrator.py      # LOBOrchestrator — WS consumer + alert dispatch
│   │   └── mock_server.py       # Regime-switching Binance depth stream simulator
│   │
│   ├── agents/
│   │   ├── graph.py             # LangGraph StateGraph + async GraphRunner
│   │   ├── researcher.py        # RAG Researcher node
│   │   └── decision.py          # LLM Decision node — timeout + circuit breaker
│   │
│   ├── rag/
│   │   ├── vector_store.py      # TF-IDF cosine similarity retrieval
│   │   └── corpus.py            # 8 quantitative research documents
│   │
│   ├── telemetry/
│   │   ├── monitor.py           # SRE monitor + Prometheus HTTP :8000/metrics
│   │   ├── metrics.py           # Thread-safe Counter / Gauge / Histogram
│   │   └── business_value.py    # Analyst-hours-saved + annualised FTE equivalent
│   │
│   └── utils/
│       ├── circuit_breaker.py   # CLOSED / OPEN / HALF_OPEN FSM
│       └── backoff.py           # Full-jitter exponential backoff
│
├── tests/
│   ├── conftest.py              # Shared fixtures + MockLLMClient
│   ├── unit/                    # 111 unit tests across 10 files
│   └── integration/             # 13 integration tests
│
├── grafana/
│   ├── dashboards/alpha_pipeline.json   # Pre-built dashboard JSON
│   └── provisioning/                    # Auto-wired Prometheus datasource
│
└── docker/
    ├── docker-compose.yml       # pipeline + prometheus + grafana
    ├── Dockerfile
    └── prometheus.yml
```

---

## Configuration reference

All configs are `BaseModel(frozen=True)` — validated at construction, immutable at runtime.

```python
OrchestratorConfig(
    symbol="BTCUSDT",
    ws_endpoint="ws://localhost:8765",
    lob_depth=20,                       # price levels retained per side
    obi_levels=5,                       # levels used in OBI calculation
    vpin_bucket_size=Decimal("10.0"),   # volume per VPIN bucket (base asset units)
    vpin_window=50,                     # rolling bucket count for VPIN average
    spread_rolling_window=100,          # ticks in spread z-score window
    spread_z_threshold=2.0,             # sigma threshold for spike detection
    obi_toxicity_threshold=0.7,         # |OBI| above this → toxic flag
    vpin_toxicity_threshold=0.5,        # VPIN above this → toxic flag
    alert_queue_maxsize=100,            # backpressure limit (drop, don't block)
    max_reconnect_attempts=10,
    reconnect_base_delay_s=1.0,         # full-jitter backoff base
    reconnect_max_delay_s=30.0,
)

DecisionAgentConfig(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    llm_timeout_ms=500.0,               # hard timeout — any overrun → CANCEL_ALL
    temperature=0.1,
)

TelemetryConfig(
    heartbeat_interval_s=5.0,
    prometheus_port=8000,
    analyst_hours_per_alert=0.5,        # hours of manual work automated per decision
)
```

---

## Design decisions

**`asyncio.Queue` with `put_nowait` between Orchestrator and Graph Runner**
Decouples ingestion throughput from LLM latency. The LOB Orchestrator processes 10 msg/s without stalling on a 400ms LLM call. When the graph runner falls behind, alerts are dropped rather than blocking — the SRE layer tracks drop rate as an explicit health signal.

**`asyncio.to_thread` for LangGraph**
`graph.invoke()` is synchronous. Running it on the event loop would block all other coroutines for the entire LLM round-trip. `to_thread` dispatches to a `ThreadPoolExecutor` worker, keeping the loop free for ingestion and telemetry.

**TF-IDF cosine similarity instead of an embedding model**
The corpus is 8 documents. A real embedding model adds a network call per alert for marginal retrieval quality improvement on a small fixed corpus. TF-IDF is deterministic, enabling reliable unit tests without mocking any external service.

**`Decimal` for all financial quantities**
IEEE-754 float accumulates error across thousands of price-level updates. `Decimal("43500.50")` is exact. OBI ratios, spread calculations in bps, and VPIN bucket fills are all exact arithmetic with no drift.

---

## References

- Cont, R., Kukanov, A. & Stoikov, S. (2014). _The Price Impact of Order Book Events._ Journal of Financial Econometrics.
- Easley, D., Lopez de Prado, M. & O'Hara, M. (2012). _Flow Toxicity and Liquidity in a High-frequency World._ Review of Financial Studies.
- Avellaneda, M. & Stoikov, S. (2008). _High-frequency trading in a limit order book._ Quantitative Finance.
