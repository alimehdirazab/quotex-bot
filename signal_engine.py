"""Signal Generation Engine.

Fuses technical confluence indicators with adaptive ML probabilities and
Expected Value calculations. Enforces risk guardrails and signal deduplication.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
import time
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

from config import settings
from feature_engine import build_features
from model_pipeline import AdaptiveModel
from data_layer import is_real_forex, get_market_status


@dataclass
class Signal:
    """Represents a trading signal produced by the SignalEngine."""
    asset: str
    direction: str  # "CALL" or "PUT"
    probability: float
    ev: float
    payout: float
    timestamp: float
    expiry: int
    confluence: bool
    mode: str = "SIGNAL_ONLY"
    executed: bool = False
    metadata: Dict = field(default_factory=dict)

    @property
    def formatted_time(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")


class SignalEngine:
    """Evaluates incoming market data against technical confluence and ML models."""

    def __init__(self, models: Optional[Dict[str, AdaptiveModel]] = None):
        self.models = models or {}
        self.candle_buffers: Dict[str, pd.DataFrame] = {}
        self.last_signal_times: Dict[str, float] = {}

        # Daily Risk Tracking State
        self.current_date: date = date.today()
        self.daily_trades_count: int = 0
        self.daily_pnl: float = 0.0
        self.daily_starting_balance: float = 100.0  # Normalized starting unit balance
        self.circuit_breaker_active: bool = False

    def reset_daily_metrics_if_new_day(self):
        """Reset daily trade counters at midnight UTC."""
        today = date.today()
        if today != self.current_date:
            logger.info(f"New trading day detected ({today}). Resetting daily risk metrics.")
            self.current_date = today
            self.daily_trades_count = 0
            self.daily_pnl = 0.0
            self.circuit_breaker_active = False

    def check_risk_limits(self) -> bool:
        """Check if trading is permissible under the daily risk limits.

        Returns:
            bool: True if safe to emit/trade, False if circuit breaker is tripped.
        """
        self.reset_daily_metrics_if_new_day()

        if self.circuit_breaker_active:
            logger.warning("Risk Limit: Circuit breaker is currently active. Operation halted.")
            return False

        if self.daily_trades_count >= settings.MAX_DAILY_TRADES:
            logger.warning(
                f"Risk Limit: Maximum daily trades ({settings.MAX_DAILY_TRADES}) reached."
            )
            self.circuit_breaker_active = True
            return False

        max_allowed_loss = self.daily_starting_balance * settings.MAX_DAILY_LOSS_PCT
        if self.daily_pnl <= -max_allowed_loss:
            logger.warning(
                f"Risk Limit: Daily loss limit hit (P&L: {self.daily_pnl:.2f}, limit: -{max_allowed_loss:.2f})."
            )
            self.circuit_breaker_active = True
            return False

        return True

    def record_trade_outcome(self, pnl: float):
        """Update daily counters with the outcome of an executed trade."""
        self.reset_daily_metrics_if_new_day()
        self.daily_trades_count += 1
        self.daily_pnl += pnl
        logger.info(
            f"Updated daily metrics: Trades: {self.daily_trades_count}/{settings.MAX_DAILY_TRADES}, "
            f"Daily P&L: {self.daily_pnl:.2f}"
        )
        self.check_risk_limits()

    def update_buffer(self, asset: str, candle: dict, max_buffer_size: int = 300) -> pd.DataFrame:
        """Append incoming candle to asset's rolling buffer."""
        candle_df = pd.DataFrame([candle])
        if asset not in self.candle_buffers or self.candle_buffers[asset].empty:
            self.candle_buffers[asset] = candle_df
        else:
            buf = self.candle_buffers[asset]
            # Replace candle if same timestamp (in-progress update), else append
            if buf.iloc[-1]['time'] == candle['time']:
                buf.iloc[-1] = candle_df.iloc[0]
            else:
                self.candle_buffers[asset] = pd.concat([buf, candle_df], ignore_index=True)

        if len(self.candle_buffers[asset]) > max_buffer_size:
            self.candle_buffers[asset] = self.candle_buffers[asset].iloc[-max_buffer_size:].reset_index(drop=True)

        return self.candle_buffers[asset]

    def set_initial_buffer(self, asset: str, df: pd.DataFrame):
        """Seed the rolling buffer with historical candles."""
        self.candle_buffers[asset] = df.copy()

    def process_candle(
        self,
        asset: str,
        candle: dict,
        payout: float = 0.85,
        expiry: int = 60,
        mode: str = "SIGNAL_ONLY"
    ) -> Optional[Signal]:
        """Process a candle update, calculate features, evaluate models, and produce a Signal if qualified."""
        if not self.check_risk_limits():
            return None

        # Real forex market hours & session checks
        candle_time = float(candle.get('time', time.time()))
        if is_real_forex(asset):
            market_status = get_market_status(candle_time)
            if not market_status["is_open"]:
                logger.debug(f"Signal suppressed: Real forex market closed for {asset} ({market_status['reason']}).")
                return None
            if not getattr(settings, "ALLOW_LOW_LIQUIDITY", False):
                if market_status.get("session") == "Asia":
                    logger.debug(f"Signal suppressed: Asian low-liquidity session for real forex {asset}.")
                    return None
            if getattr(settings, "PREFER_SESSION_OVERLAP", False):
                if not market_status.get("is_overlap", False):
                    logger.debug(f"Signal suppressed: PREFER_SESSION_OVERLAP active and not in overlap session.")
                    return None

        # Deduplication check: prevent multiple signals in the same candle window
        last_time = self.last_signal_times.get(asset, 0.0)
        if candle_time - last_time < (expiry * 0.8):
            return None

        # Update historical buffer
        df = self.update_buffer(asset, candle)
        if len(df) < 30:
            return None

        # Compute technical indicators and features
        features_df = build_features(df, drop_na=True)
        if features_df.empty:
            return None

        latest_row = features_df.iloc[-1]

        # Volatility regime filter
        if getattr(settings, "USE_REGIME_FILTER", False):
            regime = latest_row.get("volatility_regime", 0)
            if regime == -1:
                logger.debug(f"Signal suppressed for {asset}: Low volatility / compression regime.")
                return None

        # ML Probability Inference
        model = self.models.get(asset)
        if not model:
            model = AdaptiveModel(asset=asset)
            model.load_latest()
            self.models[asset] = model

        proba_up = model.predict_proba(features_df)
        direction, confidence, ev = model.calculate_expected_value(proba_up, payout)

        # Technical Confluence Check
        has_bullish_confluence = bool(latest_row.get('bullish_confluence', False))
        has_bearish_confluence = bool(latest_row.get('bearish_confluence', False))

        is_confluent = (
            (direction == "CALL" and has_bullish_confluence) or
            (direction == "PUT" and has_bearish_confluence)
        )

        # Fusion Decision Criteria:
        # 1. Directional alignment between ML and technicals (or exceptional ML confidence)
        # 2. Confidence >= SIGNAL_CONFIDENCE_THRESHOLD
        # 3. Expected Value >= MIN_EXPECTED_VALUE
        threshold_met = confidence >= settings.SIGNAL_CONFIDENCE_THRESHOLD
        ev_met = ev >= settings.MIN_EXPECTED_VALUE

        signal_qualified = (is_confluent and threshold_met and ev_met) or (
            confidence >= 0.75 and ev >= 0.05
        )

        if signal_qualified:
            self.last_signal_times[asset] = candle_time
            sig = Signal(
                asset=asset,
                direction=direction,
                probability=round(confidence, 4),
                ev=round(ev, 4),
                payout=round(payout, 2),
                timestamp=candle_time,
                expiry=expiry,
                confluence=is_confluent,
                mode=mode,
                executed=False,
                metadata={
                    "rsi": round(float(latest_row.get('rsi_14', 50)), 1),
                    "bb_percent": round(float(latest_row.get('bb_percent', 0.5)), 3),
                    "close": float(latest_row.get('close', 0.0))
                }
            )
            logger.info(
                f"Generated qualified signal: [{sig.mode}] {sig.asset} {sig.direction} "
                f"| Conf: {sig.probability*100:.1f}% | EV: {sig.ev*100:.1f}% | Payout: {sig.payout*100:.0f}%"
            )
            return sig

        return None

