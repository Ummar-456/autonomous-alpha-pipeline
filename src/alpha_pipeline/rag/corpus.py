"""
corpus.py — Quantitative research corpus for the mocked vector store.

Each document captures a key insight from market microstructure literature,
formatted for retrieval by the Researcher agent.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    title: str
    content: str
    source: str
    tags: list[str]


CORPUS: list[CorpusDocument] = [
    CorpusDocument(
        doc_id="doc_001",
        title="Order Book Imbalance as a Price Predictor",
        content=(
            "High order book imbalance (OBI > 0.7) on the bid side is a strong "
            "short-term predictor of upward price movement. Market makers should "
            "widen bid-side quotes to avoid adverse selection. Cont, Kukanov & Stoikov "
            "(2014) show that OBI explains ~65% of short-horizon price impact across "
            "multiple equity venues. In crypto markets, this effect is amplified due "
            "to lower institutional participation and higher retail market order flow."
        ),
        source="Cont_Kukanov_Stoikov_2014",
        tags=["obi", "price_impact", "market_making", "adverse_selection"],
    ),
    CorpusDocument(
        doc_id="doc_002",
        title="VPIN and Toxic Order Flow Detection",
        content=(
            "Volume-Synchronized Probability of Informed Trading (VPIN) above 0.5 "
            "signals toxic order flow — informed traders are actively hitting resting "
            "quotes. Easley et al. (2012) demonstrate that elevated VPIN preceded the "
            "2010 Flash Crash by 90 minutes. Market makers should suspend liquidity "
            "provision or widen spreads significantly (2–5x normal) when VPIN > 0.6. "
            "Combined with OBI > 0.7, this represents a critical risk threshold requiring "
            "immediate action: cancel all outstanding limit orders."
        ),
        source="Easley_Lopez_OHara_2012",
        tags=["vpin", "toxic_flow", "flash_crash", "risk_management"],
    ),
    CorpusDocument(
        doc_id="doc_003",
        title="Bid-Ask Spread Dynamics During Volatility Spikes",
        content=(
            "Spread widening to 3+ basis points in BTC/USDT is historically correlated "
            "with upcoming 15-minute volatility of >0.8%. During spread spikes, the "
            "optimal market-making strategy shifts from narrow-quote provision to "
            "wider, asymmetric quoting. Empirical analysis of 2022–2024 Binance data "
            "shows spread spikes of 2-sigma above rolling mean precede significant "
            "directional moves in 67% of cases. Recommended response: widen quotes "
            "by 1.5–2x and reduce inventory exposure by 50%."
        ),
        source="Internal_Research_Note_v4_2024",
        tags=["spread", "volatility", "market_making", "btcusdt"],
    ),
    CorpusDocument(
        doc_id="doc_004",
        title="Liquidity Provision During Informed Trading",
        content=(
            "When both OBI and VPIN indicate informed trading, providing liquidity "
            "is highly unprofitable. The expected cost of adverse selection exceeds "
            "the bid-ask spread revenue in >80% of such episodes. The CANCEL_ALL "
            "response is the theoretically optimal action. Historical backtests on "
            "BTC/USDT from 2021–2024 show that cancelling all quotes during "
            "combined OBI>0.7 and VPIN>0.5 episodes reduces mark-to-market losses "
            "by an average of 0.34% per episode versus holding."
        ),
        source="Internal_Research_Note_v7_2024",
        tags=["adverse_selection", "cancel_all", "liquidity", "btcusdt"],
    ),
    CorpusDocument(
        doc_id="doc_005",
        title="Normal Market Conditions: Optimal Quoting Strategy",
        content=(
            "In normal market conditions (OBI < 0.3, spread < 1.5 bps, VPIN < 0.3), "
            "providing liquidity at 0.5–1.0 bps from mid is consistently profitable "
            "on BTC/USDT. Position should be balanced across bid and ask to minimize "
            "inventory risk. The Avellaneda-Stoikov (2008) framework suggests optimal "
            "spread of 2γσ²δ + (2/γ)ln(1 + γ/κ), where γ is risk aversion, σ² is "
            "variance, δ is time horizon, and κ is order arrival rate. In calm "
            "conditions, PROVIDE_LIQUIDITY on both sides is the recommended action."
        ),
        source="Avellaneda_Stoikov_2008",
        tags=["market_making", "optimal_quoting", "normal_conditions", "spread"],
    ),
    CorpusDocument(
        doc_id="doc_006",
        title="Crypto Market Microstructure: Unique Properties",
        content=(
            "Crypto markets exhibit 24/7 operation, fragmented liquidity across exchanges, "
            "and higher retail participation than traditional equity markets. This leads "
            "to more frequent but shorter-duration volatility spikes. OBI signals are "
            "more reliable on Binance than on smaller venues due to higher consolidated "
            "market share. Funding rate pressure on perpetual futures often precedes "
            "LOB imbalances on spot markets by 5–15 minutes. Always check perp funding "
            "rate context when evaluating spot LOB signals."
        ),
        source="Crypto_Microstructure_Review_2023",
        tags=["crypto", "binance", "fragmentation", "perpetuals"],
    ),
    CorpusDocument(
        doc_id="doc_007",
        title="Recovery After Toxic Flow Episode",
        content=(
            "After a CANCEL_ALL event triggered by toxic flow, the historically optimal "
            "re-entry strategy is to wait for OBI to normalize below 0.3 and VPIN to "
            "drop below 0.4 before re-posting quotes. Median recovery time is 4.2 minutes "
            "for BTC/USDT. Posting quotes while VPIN is still elevated (0.4–0.5) results "
            "in a 23% higher rate of getting adversely selected on re-entry. "
            "Recommended: 60-second mandatory quiet period post CANCEL_ALL, then "
            "gradual re-entry at 50% normal size."
        ),
        source="Internal_Research_Note_v9_2024",
        tags=["recovery", "re_entry", "vpin", "cancel_all", "post_event"],
    ),
    CorpusDocument(
        doc_id="doc_008",
        title="Asymmetric Book Imbalance: BID vs ASK Pressure",
        content=(
            "Positive OBI (bid-heavy book) suggests aggressive buy pressure. "
            "Optimal response for a market maker: reduce bid quote size (avoid "
            "getting hit by informed sellers executing against visible bids), "
            "and potentially TAKE_LIQUIDITY on the ask side if conviction is high. "
            "Negative OBI (ask-heavy) suggests sell pressure. The asymmetry of "
            "response is critical: a positive OBI > 0.5 warrants BID-side withdrawal "
            "before ASK-side. Crossing the spread aggressively should only occur "
            "with OBI > 0.8 and supporting momentum signal."
        ),
        source="Cont_Kukanov_Stoikov_2014",
        tags=["obi", "asymmetry", "directional", "take_liquidity"],
    ),
]


def get_all_documents() -> list[CorpusDocument]:
    return list(CORPUS)
