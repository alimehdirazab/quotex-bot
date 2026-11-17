"""Diagnostic script to inspect what Quotex sends back for history requests."""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from pyquotex.stable_api import Quotex

async def test_history():
    q = Quotex(email=settings.QUOTEX_EMAIL, password=settings.QUOTEX_PASSWORD)
    ok, reason = await q.connect()
    print("Connect:", ok, reason)
    if not ok:
        return

    # Check live candle stream
    print("Subscribing to candles stream for EURUSD_otc...")
    await q.start_candles_stream("EURUSD_otc", 60)
    await asyncio.sleep(3)
    candles = await q.get_realtime_candles("EURUSD_otc")
    print(f"Realtime candles in memory: {len(candles)}")
    if candles:
        print("Latest realtime candle:", candles[-1])

    # Check get_candles
    print("Calling get_candles with offset=100...")
    c_batch = await q.get_candles("EURUSD_otc", end_from_time=time.time(), offset=100, period=60, timeout=10)
    print(f"get_candles returned: {len(c_batch) if c_batch else 0} candles")
    if c_batch:
        print("Sample candle:", c_batch[0])

    await q.close()

if __name__ == "__main__":
    asyncio.run(test_history())
