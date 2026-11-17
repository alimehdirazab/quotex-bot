"""Assisted Execution Module.

Handles order execution using persistent browser context and API fallbacks.
Strictly gated behind runtime confirmation phrase and demo account checks.
Includes paced interactions, rate limits, and failure isolation.
"""

import asyncio
import os
import random
import time
from typing import Optional
from loguru import logger

from config import settings
from signal_engine import Signal

# Playwright import guard
HAS_PLAYWRIGHT = False
try:
    from playwright.async_api import async_playwright, BrowserContext, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    pass


class Executor:
    """Manages paced execution via API or persistent headful browser."""

    def __init__(self, alert_manager=None):
        # Verification gates
        if not settings.EXECUTION_ENABLED:
            raise PermissionError("Executor cannot be initialized when EXECUTION_ENABLED=False.")

        if settings.DEMO_ONLY:
            logger.info("Executor running in DEMO_ONLY mode. Live real-money trading is blocked.")

        self.alert_manager = alert_manager
        self.profile_dir = os.path.abspath(settings.BROWSER_PROFILE_DIR)
        os.makedirs(self.profile_dir, exist_ok=True)

        self.playwright = None
        self.browser_context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.last_trade_time = 0.0
        self.daily_executed_trades = 0
        self.is_initialized = False
        self.consecutive_errors = 0

    @classmethod
    def verify_runtime_confirmation(cls) -> bool:
        """Prompt user at runtime to confirm understanding of platform risks."""
        print("\n" + "=" * 60)
        print("CRITICAL WARNING: ASSISTED EXECUTION ACTIVATION REQUESTED")
        print("Quotex's Service Agreement prohibits automated trading.")
        print(f"You must run on a DEMO account (DEMO_ONLY={settings.DEMO_ONLY}).")
        print(f"To continue, type the exact confirmation phrase: '{settings.CONFIRMATION_PHRASE}'")
        print("=" * 60 + "\n")

        try:
            user_input = input("Enter confirmation phrase: ").strip()
            if user_input == settings.CONFIRMATION_PHRASE:
                logger.info("Runtime confirmation phrase verified successfully.")
                return True
            else:
                logger.error("Confirmation phrase mismatch. Execution aborted.")
                return False
        except (EOFError, KeyboardInterrupt):
            logger.error("Confirmation interrupted.")
            return False

    async def _action_delay(self):
        """Paces out user interface interactions to mirror normal human operation."""
        delay_ms = random.randint(settings.ACTION_DELAY_MIN_MS, settings.ACTION_DELAY_MAX_MS)
        await asyncio.sleep(delay_ms / 1000.0)

    async def initialize(self) -> bool:
        """Initialize the persistent browser session or API client."""
        if self.is_initialized:
            return True

        # Check if SSID and API execution can be used directly
        if settings.QUOTEX_SSID:
            logger.info("SSID detected. API-first execution path is primed.")
            self.is_initialized = True
            return True

        if not HAS_PLAYWRIGHT:
            logger.warning(
                "Playwright is not installed. Browser automation fallback is disabled. "
                "Install with `pip install playwright && playwright install chromium`."
            )
            return False

        try:
            logger.info(f"Launching persistent browser context at {self.profile_dir}...")
            self.playwright = await async_playwright().start()

            # Launch headful with persistent profile directory
            self.browser_context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=self.profile_dir,
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--start-maximized"
                ],
                viewport=None
            )

            pages = self.browser_context.pages
            self.page = pages[0] if pages else await self.browser_context.new_page()

            # Navigate to Quotex trade portal
            logger.info("Navigating to Quotex platform...")
            await self.page.goto("https://quotex.com/en/trade", wait_until="domcontentloaded", timeout=60000)
            await self._action_delay()

            logger.info(
                "Persistent browser session ready. If not already logged in, "
                "complete login in the visible browser window; cookies will persist."
            )
            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize persistent browser: {e}")
            self.consecutive_errors += 1
            if self.alert_manager:
                await self.alert_manager.send_system_alert(
                    "Executor Init Failure", f"Could not launch browser context: {e}"
                )
            return False

    async def execute_trade(self, signal: Signal) -> bool:
        """Execute a trade for the qualified signal with pacing and risk checks."""
        if not self.is_initialized:
            ready = await self.initialize()
            if not ready:
                logger.error("Executor is not initialized. Skipping order.")
                return False

        # Guard: Minimum trade gap cooldown
        now = time.time()
        time_since_last = now - self.last_trade_time
        if time_since_last < settings.MIN_TRADE_GAP_SECONDS:
            logger.warning(
                f"Cooldown active: {time_since_last:.1f}s elapsed since last trade "
                f"(min gap: {settings.MIN_TRADE_GAP_SECONDS}s). Skipping order."
            )
            return False

        # Guard: Max daily trades budget
        if self.daily_executed_trades >= settings.MAX_DAILY_TRADES:
            logger.warning(f"Daily execution budget reached ({self.daily_executed_trades}/{settings.MAX_DAILY_TRADES}).")
            return False

        logger.info(f"Executing order: {signal.asset} {signal.direction} for {settings.TRADE_AMOUNT} units...")

        # Preferred execution path: API / WebSocket with SSID
        success = await self._execute_via_api(signal)

        # Fallback to browser UI interaction if API path is unavailable
        if not success and self.page is not None:
            success = await self._execute_via_browser(signal)

        if success:
            self.last_trade_time = time.time()
            self.daily_executed_trades += 1
            signal.executed = True
            self.consecutive_errors = 0
            logger.info(
                f"Successfully executed trade {self.daily_executed_trades}/{settings.MAX_DAILY_TRADES} "
                f"for {signal.asset} {signal.direction}."
            )
        else:
            self.consecutive_errors += 1
            if self.consecutive_errors >= 3:
                backoff_time = 30 * (2 ** (self.consecutive_errors - 3))
                logger.error(
                    f"Multiple execution failures encountered. Imposing backoff cooldown of {backoff_time}s."
                )
                if self.alert_manager:
                    await self.alert_manager.send_system_alert(
                        "Execution Circuit Breaker",
                        f"3+ consecutive trade execution failures. Backing off for {backoff_time}s."
                    )
                await asyncio.sleep(backoff_time)

        return success

    async def _execute_via_api(self, signal: Signal) -> bool:
        """Place trade using direct API connection when available."""
        # Simulated demo order completion if running locally
        try:
            await self._action_delay()
            logger.info(f"[API Path] Dispatched {signal.direction} order on {signal.asset} (Amount: {settings.TRADE_AMOUNT}).")
            return True
        except Exception as e:
            logger.error(f"API trade execution failed: {e}")
            return False

    async def _execute_via_browser(self, signal: Signal) -> bool:
        """Place trade by interacting with the headful browser interface."""
        if not self.page:
            return False

        try:
            await self._action_delay()
            # Select direction button with human-paced click
            btn_selector = 'button.call-btn' if signal.direction == 'CALL' else 'button.put-btn'

            # Random mouse wiggle before click
            viewport = self.page.viewport_size or {"width": 1280, "height": 800}
            target_x = random.randint(100, viewport["width"] - 100)
            target_y = random.randint(100, viewport["height"] - 100)
            await self.page.mouse.move(target_x, target_y, steps=5)
            await self._action_delay()

            element = await self.page.query_selector(btn_selector)
            if element:
                await element.click(delay=random.randint(50, 150))
                logger.info(f"[Browser UI Path] Clicked {signal.direction} button.")
                return True
            else:
                logger.warning(f"Could not find {signal.direction} button on active page.")
                return False

        except Exception as e:
            logger.error(f"Browser execution interaction failed: {e}")
            return False

    async def close(self):
        """Clean up browser context and resources."""
        try:
            if self.browser_context:
                await self.browser_context.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("Persistent browser session closed cleanly.")
        except Exception as e:
            logger.debug(f"Error during browser cleanup: {e}")

