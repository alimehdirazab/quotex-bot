"""Automated Test Suite for Quotex Signal & Trading Bot.

Verifies feature calculation integrity, EV mathematical properties,
signal fusion logic, risk circuit breakers, and backtesting metrics.
"""

import unittest
import numpy as np
import pandas as pd

from config import settings
from feature_engine import build_features, get_feature_columns
from model_pipeline import AdaptiveModel
from signal_engine import SignalEngine, Signal
from backtester import Backtester


def create_mock_candles(count: int = 120, trend: float = 0.0001) -> pd.DataFrame:
    """Generate deterministic mock candlestick data for testing."""
    times = np.arange(1700000000, 1700000000 + (count * 60), 60)
    prices = 1.0800 + np.cumsum(np.full(count, trend) + np.sin(np.linspace(0, 10, count)) * 0.0005)
    opens = prices
    closes = prices + 0.0001
    highs = closes + 0.0002
    lows = opens - 0.0002
    volumes = np.full(count, 500.0)

    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })


class TestFeatureEngine(unittest.TestCase):
    """Test suite for feature engineering."""

    def setUp(self):
        self.df = create_mock_candles(150)

    def test_feature_columns_presence(self):
        features_df = build_features(self.df, drop_na=True)
        self.assertFalse(features_df.empty, "Feature dataframe should not be empty.")
        expected_cols = get_feature_columns()
        for col in expected_cols:
            self.assertIn(col, features_df.columns, f"Expected feature column '{col}' missing.")

    def test_target_creation(self):
        features_df = build_features(self.df, drop_na=True)
        self.assertIn("target", features_df.columns)
        unique_targets = features_df["target"].unique()
        self.assertTrue(set(unique_targets).issubset({0, 1}))


class TestModelPipeline(unittest.TestCase):
    """Test suite for adaptive ML and Expected Value calculations."""

    def setUp(self):
        self.model = AdaptiveModel(asset="TEST_ASSET")

    def test_expected_value_call(self):
        # 60% probability of up with 85% payout
        # EV = 0.60 * 0.85 - 0.40 * 1.0 = 0.51 - 0.40 = +0.11 (+11%)
        direction, conf, ev = self.model.calculate_expected_value(proba_up=0.60, payout=0.85)
        self.assertEqual(direction, "CALL")
        self.assertAlmostEqual(conf, 0.60, places=4)
        self.assertAlmostEqual(ev, 0.11, places=4)

    def test_expected_value_put(self):
        # 30% probability of up -> 70% probability of down with 80% payout
        # EV_put = 0.70 * 0.80 - 0.30 * 1.0 = 0.56 - 0.30 = +0.26 (+26%)
        direction, conf, ev = self.model.calculate_expected_value(proba_up=0.30, payout=0.80)
        self.assertEqual(direction, "PUT")
        self.assertAlmostEqual(conf, 0.70, places=4)
        self.assertAlmostEqual(ev, 0.26, places=4)

    def test_break_even_point(self):
        # At 80% payout, break-even is 1 / (1 + 0.80) = 5/9 ~ 0.5555...
        break_even_p = 1.0 / 1.80
        _, _, ev = self.model.calculate_expected_value(proba_up=break_even_p, payout=0.80)
        self.assertAlmostEqual(ev, 0.0, places=4)


class TestSignalEngine(unittest.TestCase):
    """Test suite for signal generation and risk limits."""

    def setUp(self):
        self.engine = SignalEngine()

    def test_risk_limits_daily_trades(self):
        self.engine.daily_trades_count = settings.MAX_DAILY_TRADES
        can_trade = self.engine.check_risk_limits()
        self.assertFalse(can_trade, "Circuit breaker should halt when max daily trades reached.")

    def test_risk_limits_daily_loss(self):
        max_allowed_loss = self.engine.daily_starting_balance * settings.MAX_DAILY_LOSS_PCT
        self.engine.daily_pnl = -(max_allowed_loss + 1.0)
        can_trade = self.engine.check_risk_limits()
        self.assertFalse(can_trade, "Circuit breaker should halt when max loss % exceeded.")

    def test_deduplication(self):
        candle = {
            "time": 1700000060,
            "open": 1.0810,
            "high": 1.0820,
            "low": 1.0805,
            "close": 1.0815,
            "volume": 200.0
        }
        self.engine.last_signal_times["EURUSD_otc"] = 1700000050
        # Within same 60s candle window, should be blocked by deduplication
        sig = self.engine.process_candle("EURUSD_otc", candle, payout=0.85, expiry=60)
        self.assertIsNone(sig, "Duplicate signals within same candle window must be suppressed.")


class TestBacktester(unittest.TestCase):
    """Test suite for event-driven backtesting engine."""

    def test_backtest_execution(self):
        df = create_mock_candles(200)
        backtester = Backtester(default_payout=0.85, trade_amount=1.0)
        results = backtester.run(df, asset="TEST_PAIR", confidence_threshold=0.55, min_ev=0.01)

        self.assertIn("total_trades", results)
        self.assertIn("win_rate", results)
        self.assertIn("net_pnl", results)
        self.assertIn("break_even_win_rate", results)
        self.assertAlmostEqual(results["break_even_win_rate"], 1.0 / 1.85, places=3)


if __name__ == "__main__":
    unittest.main()

