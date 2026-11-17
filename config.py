"""Configuration module for Quotex Signal & Assisted-Trading Bot.

Uses pydantic-settings to provide type-safe, environment-variable-backed
configuration with sensible defaults and risk guardrails.
"""

from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for the bot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ---------------------------------------------------------
    # SAFETY & EXECUTION GATES
    # ---------------------------------------------------------
    DEMO_ONLY: bool = Field(
        default=True,
        description="Enforces demo mode. Real money trading is blocked unless explicitly False."
    )
    EXECUTION_ENABLED: bool = Field(
        default=False,
        description="Whether automated/assisted execution is enabled. Default False (Signal Only)."
    )
    CONFIRMATION_PHRASE: str = Field(
        default="I_ACCEPT_THE_RISK",
        description="Exact phrase user must type at session launch to unlock assisted execution."
    )
    USE_SYNTHETIC: bool = Field(
        default=False,
        description="Gated flag: False by default. When False, synthetic data fallback is hard-blocked."
    )

    # ---------------------------------------------------------
    # QUOTEX AUTHENTICATION & CONNECTION
    # ---------------------------------------------------------
    QUOTEX_EMAIL: str = Field(default="", description="Account email for Quotex")
    QUOTEX_PASSWORD: str = Field(default="", description="Account password for Quotex")
    QUOTEX_SSID: str = Field(
        default="",
        description="Session SSID extracted from browser DevTools cookies (recommended)"
    )
    ASSETS: Union[List[str], str] = Field(
        default=["EURUSD_otc"],
        description="List of assets to monitor and trade"
    )
    SUPPORTED_ASSETS: List[str] = Field(
        default=[
            "EURUSD_otc", "GBPUSD_otc", "USDJPY_otc", "AUDUSD_otc", "NZDUSD_otc", "USDCAD_otc",
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"
        ],
        description="All cataloged assets (OTC and standard Forex pairs)"
    )
    TIMEFRAME_SECONDS: int = Field(
        default=60,
        description="Candle timeframe in seconds (e.g. 60 for 1M, 300 for 5M, 900 for 15M)"
    )
    CANDLE_HISTORY_DAYS: int = Field(
        default=30,
        description="Number of historical days to pull for initial training"
    )
    MAX_CONCURRENT_STREAMS: int = Field(
        default=3,
        le=5,
        description="Maximum simultaneous asset WebSocket streams (never exceed 5)"
    )
    DEFAULT_PAYOUT: float = Field(
        default=0.85,
        description="Default payout ratio if live payout query is temporarily unavailable"
    )

    # ---------------------------------------------------------
    # REAL FOREX & MARKET HOURS (PHASE 2)
    # ---------------------------------------------------------
    MARKET_HOURS_ONLY: bool = Field(
        default=True,
        description="Refuse to stream or trade real forex pairs outside standard market hours"
    )
    PREFER_SESSION_OVERLAP: bool = Field(
        default=True,
        description="Prioritize London/New York overlap window (12:00-16:00 UTC) for real forex"
    )
    ALLOW_OTC_FALLBACK: bool = Field(
        default=False,
        description="Whether to fall back to _otc counterpart if real forex market is closed"
    )
    ALLOW_LOW_LIQUIDITY: bool = Field(
        default=False,
        description="Whether to allow signals on real forex during Asia low-liquidity session"
    )

    # ---------------------------------------------------------
    # SIGNAL & ML THRESHOLDS & IMPROVEMENTS (PHASE 3)
    # ---------------------------------------------------------
    SIGNAL_CONFIDENCE_THRESHOLD: float = Field(
        default=0.65,
        ge=0.50,
        le=1.0,
        description="Minimum ML predicted probability required to generate a signal"
    )
    MIN_EXPECTED_VALUE: float = Field(
        default=0.02,
        description="Minimum expected value (EV) percentage required (e.g., 0.02 = 2% EV)"
    )
    RETRAIN_INTERVAL_HOURS: int = Field(
        default=6,
        description="Interval in hours between adaptive rolling model retrains"
    )
    MODEL_PURGE_KEEP: int = Field(
        default=5,
        description="Number of timestamped model checkpoints to retain on disk"
    )
    MODEL_DIR: str = Field(
        default="./models",
        description="Directory where trained model artifacts are stored"
    )
    USE_ENSEMBLE: bool = Field(
        default=True,
        description="Ensemble LightGBM + HistGradientBoosting + ExtraTrees"
    )
    CALIBRATE_PROBABILITIES: bool = Field(
        default=True,
        description="Use Isotonic CalibratedClassifierCV for accurate EV estimation"
    )
    USE_MULTI_TIMEFRAME: bool = Field(
        default=True,
        description="Extract 5m and 15m aggregated features"
    )
    USE_REGIME_FILTER: bool = Field(
        default=True,
        description="Detect ATR volatility regimes (low, mid, high)"
    )
    CROSS_ASSET_CONFIRMATION: bool = Field(
        default=False,
        description="Require directional agreement from correlated pairs (e.g. EURUSD & GBPUSD)"
    )
    CORRELATED_ASSET: str = Field(
        default="GBPUSD_otc",
        description="Correlated reference asset for cross-asset confirmation"
    )

    # ---------------------------------------------------------
    # RISK MANAGEMENT & PACING
    # ---------------------------------------------------------
    TRADE_AMOUNT: float = Field(
        default=1.0,
        gt=0,
        description="Amount in currency units per executed trade"
    )
    MAX_DAILY_TRADES: int = Field(
        default=20,
        description="Maximum executed trades allowed in a 24-hour period"
    )
    MAX_DAILY_LOSS_PCT: float = Field(
        default=0.05,
        description="Maximum daily loss percentage before circuit breaker halts operations"
    )
    ACTION_DELAY_MIN_MS: int = Field(
        default=150,
        description="Minimum millisecond delay between consecutive automated UI actions"
    )
    ACTION_DELAY_MAX_MS: int = Field(
        default=400,
        description="Maximum millisecond delay between consecutive automated UI actions"
    )
    MIN_TRADE_GAP_SECONDS: int = Field(
        default=30,
        description="Minimum cooldown period in seconds between consecutive trade orders"
    )

    # ---------------------------------------------------------
    # ALERT CHANNELS
    # ---------------------------------------------------------
    TELEGRAM_BOT_TOKEN: str = Field(default="", description="Telegram Bot API Token")
    TELEGRAM_CHAT_ID: str = Field(default="", description="Telegram Chat ID for notifications")
    WHATSAPP_API_TOKEN: str = Field(default="", description="Meta WhatsApp Cloud API Access Token")
    WHATSAPP_PHONE_NUMBER_ID: str = Field(
        default="",
        description="Meta WhatsApp Cloud API Sender Phone Number ID"
    )
    WHATSAPP_RECIPIENT: str = Field(
        default="",
        description="Recipient phone number with country code (e.g. +1234567890)"
    )

    # ---------------------------------------------------------
    # ENVIRONMENT & PERSISTENCE
    # ---------------------------------------------------------
    LOG_LEVEL: str = Field(default="INFO", description="Loguru logging level")
    LOG_FILE: str = Field(default="./logs/quotex_bot.log", description="Rotating log file path")
    BROWSER_PROFILE_DIR: str = Field(
        default="./browser_profile",
        description="Persistent browser profile directory to retain cookies and session state"
    )

    @field_validator("ASSETS", mode="before")
    @classmethod
    def parse_assets_list(cls, value):
        if isinstance(value, str):
            if value.startswith("[") and value.endswith("]"):
                import json
                try:
                    return json.loads(value)
                except Exception:
                    pass
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


# Global settings instance
settings = Settings()
