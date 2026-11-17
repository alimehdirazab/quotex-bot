"""Verify real historical and live market candle retrieval from Quotex."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from config import settings
from data_layer import QuotexDataClient


async def verify_real_data():
    logger.info("Initializing QuotexDataClient for real data verification...")
    client = QuotexDataClient()
    connected = await client.connect()
    if not connected:
        logger.error("Authentication failed during real data check.")
        sys.exit(1)

    asset = "EURUSD_otc"
    days = 30
    timeframe = 60
    logger.info(f"Calling get_historical_candles('{asset}', days={days}, timeframe={timeframe})...")
    
    try:
        df = await client.get_historical_candles(asset, days=days, timeframe=timeframe, allow_synthetic=False)
        print("\n" + "=" * 60)
        print(f"REAL HISTORICAL CANDLE DATA VERIFICATION: {asset}")
        print("=" * 60)
        print(f"Total Rows Returned: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"First Timestamp: {df['time'].iloc[0]} ({pd.to_datetime(df['time'].iloc[0], unit='s')})")
        print(f"Last Timestamp : {df['time'].iloc[-1]} ({pd.to_datetime(df['time'].iloc[-1], unit='s')})")
        print("\nHead (First 5 Rows):")
        print(df.head(5))
        print("\nTail (Last 5 Rows):")
        print(df.tail(5))
        
        # Sanity check: price range
        mean_close = df['close'].mean()
        min_close = df['close'].min()
        max_close = df['close'].max()
        print(f"\nPrice Sanity Check: min={min_close:.5f}, max={max_close:.5f}, mean={mean_close:.5f}")
        
        if 1.00 <= mean_close <= 1.25:
            print("SANITY_CHECK_PASSED: EURUSD_otc prices are in expected realistic forex range.")
        else:
            print(f"SANITY_CHECK_WARNING: Price range {mean_close:.5f} is outside standard 1.00-1.25 range.")

        # Test payout
        payout = await client.get_payout(asset)
        print(f"\nCurrent Broker Payout for {asset}: {payout * 100:.1f}%")

        print("=" * 60 + "\n")
        
    except Exception as exc:
        logger.exception(f"Real data verification encountered an error: {exc}")
        sys.exit(1)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    import pandas as pd
    asyncio.run(verify_real_data())

