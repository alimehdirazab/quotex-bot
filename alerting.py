"""Notification and Alerting Module.

Supports Telegram Bot API and Meta WhatsApp Cloud API with clear mode indicators,
rich formatting, system event broadcasts, and deduplication.
"""

import asyncio
from datetime import datetime
import json
import time
from typing import Optional, Set
import urllib.request
import urllib.parse
from loguru import logger

from config import settings
from signal_engine import Signal

# Try importing pyTelegramBotAPI
HAS_TELEBOT = False
try:
    import telebot  # type: ignore
    HAS_TELEBOT = True
except ImportError:
    pass

# Try importing whatsapp-cloud-api
HAS_WHATSAPP_SDK = False
try:
    from whatsapp import WhatsApp  # type: ignore
    HAS_WHATSAPP_SDK = True
except ImportError:
    pass


class AlertManager:
    """Manages outbound alert dispatching to Telegram and WhatsApp."""

    def __init__(self):
        self.telegram_bot = None
        self.whatsapp_client = None
        self.sent_signals_cache: Set[str] = set()

        if settings.TELEGRAM_BOT_TOKEN and HAS_TELEBOT:
            try:
                self.telegram_bot = telebot.TeleBot(settings.TELEGRAM_BOT_TOKEN, parse_mode="Markdown")
                logger.info("Telegram Bot alerting client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram Bot: {e}")

        if settings.WHATSAPP_API_TOKEN and settings.WHATSAPP_PHONE_NUMBER_ID and HAS_WHATSAPP_SDK:
            try:
                self.whatsapp_client = WhatsApp(
                    token=settings.WHATSAPP_API_TOKEN,
                    phone_number_id=settings.WHATSAPP_PHONE_NUMBER_ID
                )
                logger.info("WhatsApp Cloud API client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize WhatsApp SDK: {e}")

    def _get_signal_key(self, signal: Signal) -> str:
        """Create unique deduplication key for a signal."""
        return f"{signal.asset}_{signal.direction}_{int(signal.timestamp)}"

    def format_signal_message(self, signal: Signal) -> str:
        """Create a cleanly formatted markdown notification message."""
        mode_badge = "🟢 [ASSISTED EXECUTED]" if signal.executed else "🔵 [SIGNAL ONLY]"
        dir_emoji = "🔼 CALL (BUY)" if signal.direction == "CALL" else "🔽 PUT (SELL)"

        msg = (
            f"*{mode_badge}*\n\n"
            f"*Asset:* `{signal.asset}`\n"
            f"*Action:* *{dir_emoji}*\n"
            f"*Confidence:* `{signal.probability * 100:.1f}%`\n"
            f"*Expected Value (EV):* `{signal.ev * 100:+.1f}%`\n"
            f"*Broker Payout:* `{signal.payout * 100:.0f}%`\n"
            f"*Expiry:* `{signal.expiry}s`\n"
            f"*Time:* `{signal.formatted_time}`\n"
        )
        if signal.metadata:
            meta_str = " | ".join(f"{k.upper()}: {v}" for k, v in signal.metadata.items())
            msg += f"\n_Technicals:_ `{meta_str}`\n"

        if not signal.executed:
            msg += "\n⚠️ _Educational / Demo signal only. No order executed._"
        else:
            msg += "\n✅ _Assisted execution order dispatched._"

        return msg

    async def send_signal(self, signal: Signal):
        """Dispatch signal notification to configured channels."""
        sig_key = self._get_signal_key(signal)
        if sig_key in self.sent_signals_cache:
            logger.debug(f"Signal {sig_key} already sent, skipping deduplicated alert.")
            return
        self.sent_signals_cache.add(sig_key)

        # Retain only recent 200 keys in cache
        if len(self.sent_signals_cache) > 200:
            self.sent_signals_cache = set(list(self.sent_signals_cache)[-100:])

        text = self.format_signal_message(signal)
        logger.info(f"Alert Dispatch: {signal.asset} {signal.direction} ({signal.mode})")

        # Run channel sends concurrently
        await asyncio.gather(
            self.send_telegram(text),
            self.send_whatsapp(text),
            return_exceptions=True
        )

    async def send_telegram(self, message: str):
        """Send message via Telegram Bot API."""
        if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
            return

        def _sync_telegram():
            try:
                if self.telegram_bot:
                    self.telegram_bot.send_message(
                        chat_id=settings.TELEGRAM_CHAT_ID,
                        text=message,
                        parse_mode="Markdown"
                    )
                else:
                    # Direct HTTP API fallback
                    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
                    data = urllib.parse.urlencode({
                        "chat_id": settings.TELEGRAM_CHAT_ID,
                        "text": message,
                        "parse_mode": "Markdown"
                    }).encode("utf-8")
                    req = urllib.request.Request(url, data=data)
                    with urllib.request.urlopen(req, timeout=5) as response:
                        response.read()
            except Exception as e:
                logger.error(f"Telegram dispatch error: {e}")

        await asyncio.to_thread(_sync_telegram)

    async def send_whatsapp(self, message: str):
        """Send message via Meta WhatsApp Cloud API."""
        if not settings.WHATSAPP_API_TOKEN or not settings.WHATSAPP_RECIPIENT:
            return

        def _sync_whatsapp():
            try:
                if self.whatsapp_client:
                    self.whatsapp_client.send_message(
                        message=message,
                        recipient_id=settings.WHATSAPP_RECIPIENT
                    )
                elif settings.WHATSAPP_PHONE_NUMBER_ID:
                    # Direct Cloud API Graph POST fallback
                    url = f"https://graph.facebook.com/v18.0/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
                    headers = {
                        "Authorization": f"Bearer {settings.WHATSAPP_API_TOKEN}",
                        "Content-Type": "application/json"
                    }
                    payload = json.dumps({
                        "messaging_product": "whatsapp",
                        "to": settings.WHATSAPP_RECIPIENT,
                        "type": "text",
                        "text": {"body": message}
                    }).encode("utf-8")
                    req = urllib.request.Request(url, data=payload, headers=headers)
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        resp.read()
            except Exception as e:
                logger.error(f"WhatsApp dispatch error: {e}")

        await asyncio.to_thread(_sync_whatsapp)

    async def send_system_alert(self, title: str, details: str):
        """Broadcast high-priority system events (circuit breaker, retrains, errors)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"🔔 *[SYSTEM EVENT]*: *{title}*\n\n`{details}`\n\n_Timestamp:_ `{timestamp}`"
        logger.warning(f"SYSTEM ALERT: {title} - {details}")
        await asyncio.gather(
            self.send_telegram(msg),
            self.send_whatsapp(msg),
            return_exceptions=True
        )

