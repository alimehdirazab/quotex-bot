"""Diagnostic script to verify Quotex authentication using Email & Password.

Runs without fallback to synthetic data and checks demo account balance.
Exits 0 on success, 1 on failure.
"""

import asyncio
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from config import settings
from data_layer import QuotexDataClient


async def check_login() -> int:
    logger.info("Starting authentication check using QUOTEX_EMAIL and QUOTEX_PASSWORD...")
    email = settings.QUOTEX_EMAIL
    password = settings.QUOTEX_PASSWORD
    
    if not email or not password:
        print("ERROR: QUOTEX_EMAIL and QUOTEX_PASSWORD must be configured in .env", file=sys.stderr)
        return 1
        
    masked_email = email[:3] + "***" + email[email.find("@"):] if "@" in email else "***"
    logger.info(f"Target Account Email: {masked_email}")

    # Initialize client explicitly with empty SSID to force email/password path
    client = QuotexDataClient(
        email=email,
        password=password,
        ssid=""
    )

    try:
        ok = await client.connect()
        if not ok:
            print("AUTH_FAILED: Quotex client connection returned False.")
            return 1

        # Check balance and profile if available
        balance_info = None
        if client.client and hasattr(client.client, "get_balance"):
            try:
                balance_info = await client.client.get_balance()
            except Exception as e:
                logger.warning(f"Could not retrieve balance: {e}")

        # Check account mode
        account_mode = "DEMO"
        if client.client and hasattr(client.client, "account_is_demo"):
            account_mode = "DEMO" if client.client.account_is_demo == 1 else "REAL"

        print("AUTH_OK")
        print(f"Account Mode: {account_mode}")
        print(f"Account Balance: {balance_info}")
        
        await client.disconnect()
        return 0

    except Exception as exc:
        print(f"AUTH_EXCEPTION: {type(exc).__name__}: {exc}", file=sys.stderr)
        logger.exception("Login check failed with an unhandled exception")
        return 1


def main():
    exit_code = asyncio.run(check_login())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
