"""mock_server.py — High-stability Binance simulation with aligned price clearance."""
from __future__ import annotations
import asyncio
import json
import logging
import random
import time
from decimal import Decimal
import websockets

logger = logging.getLogger(__name__)

_BASE_PRICE = Decimal("43500.00")
_TICK = Decimal("0.50")
_FEED_HZ = 5 

async def _handle_client(ws):
    update_id = 1_000_000
    mid_price = _BASE_PRICE
    logger.info("Mock Binance: client connected")
    
    try:
        while True:
            update_id += 1
            now_ms = int(time.time() * 1000)
            
            # 1. Price movement (Small increments)
            mid_price += Decimal(str(round(random.uniform(-0.1, 0.1), 2)))
            
            # 2. Force spread alignment
            best_bid = (mid_price - Decimal("1.00")).quantize(Decimal("0.01"))
            best_ask = (mid_price + Decimal("1.00")).quantize(Decimal("0.01"))

            # 3. GLOBAL CLEARANCE
            # We clear a massive range of 100 ticks to ensure no 'ghost' prices survive.
            # We use 0.01 increments to catch any unaligned prices.
            clearance_b = [[str((best_bid + (Decimal(i) * Decimal("0.01"))).quantize(Decimal("0.01"))), "0.0000"] for i in range(1, 200)]
            clearance_a = [[str((best_ask - (Decimal(i) * Decimal("0.01"))).quantize(Decimal("0.01"))), "0.0000"] for i in range(1, 200)]

            # 4. Active Levels
            bids = [[str((best_bid - i * _TICK).quantize(Decimal("0.01"))), "2.0000"] for i in range(5)]
            asks = [[str((best_ask + i * _TICK).quantize(Decimal("0.01"))), "2.0000"] for i in range(5)]

            payload = json.dumps({
                "e": "depthUpdate",
                "E": now_ms,
                "s": "BTCUSDT",
                "U": update_id - 1,
                "u": update_id,
                "b": clearance_b + bids,
                "a": clearance_a + asks,
            })
            
            await ws.send(payload)
            await asyncio.sleep(1 / _FEED_HZ)
    except Exception as e:
        logger.info(f"Mock Binance: client disconnected ({e})")
        
async def serve_mock_binance(host="localhost", port=8765):
    async with websockets.serve(_handle_client, host, port):
        await asyncio.Future()