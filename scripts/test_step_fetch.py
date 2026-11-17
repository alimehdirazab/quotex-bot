"""Test stepping backward with get_candles."""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from pyquotex.stable_api import Quotex

async def fetch_test():
    q = Quotex(email=settings.QUOTEX_EMAIL, password=settings.QUOTEX_PASSWORD)
    await q.connect()
    t = time.time()
    all_c = []
    for step in range(5):
        batch = await q.get_candles('EURUSD_otc', end_from_time=t, offset=100, period=60, timeout=10)
        if not batch:
            print(f"step {step}: empty batch")
            break
        print(f"step {step}: got {len(batch)} candles, oldest: {batch[0]['time']}, newest: {batch[-1]['time']}")
        all_c.extend(batch)
        t = batch[0]['time'] - 60
        await asyncio.sleep(0.2)
    print(f"Total collected: {len(all_c)} candles")
    await q.close()

if __name__ == "__main__":
    asyncio.run(fetch_test())

