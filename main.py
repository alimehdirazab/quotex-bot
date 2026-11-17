"""Main Orchestrator for Quotex Signal & Assisted-Trading Bot.

Coordinates DataLayer, FeatureEngine, ModelPipeline, SignalEngine,
AlertManager, and optional Executor with graceful shutdown handling.
"""

import argparse
import asyncio
import os
import signal
import sys
import threading
import time
from typing import Dict, List, Optional
from loguru import logger

from alerting import AlertManager
from backtester import Backtester
from config import settings
from data_layer import QuotexDataClient, record_signal, load_candles_for_ui
from feature_engine import build_features
from model_pipeline import AdaptiveModel
from signal_engine import SignalEngine


# Configure logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL
)
logger.add(
    settings.LOG_FILE,
    rotation="10 MB",
    retention="14 days",
    level=settings.LOG_LEVEL,
    compression="zip"
)


class BotApplication:
    """Core application lifecycle and event loop manager."""

    def __init__(self, mode: str = "signal", confirm_phrase: str = ""):
        self.mode = mode.lower()
        self.confirm_phrase = confirm_phrase
        self.running = False
        self.data_client = QuotexDataClient()
        self.alert_manager = AlertManager()
        self.models: Dict[str, AdaptiveModel] = {}
        self.signal_engine = SignalEngine(models=self.models)
        self.executor = None
        self.stream_tasks: List[asyncio.Task] = []
        self.retrain_thread: Optional[threading.Thread] = None
        self.stop_retrain_event = threading.Event()

    async def initialize_mode(self):
        """Configure runtime mode and apply safety gates."""
        if self.mode == "assisted":
            logger.warning("=" * 60)
            logger.warning("MODE: ASSISTED EXECUTION ACTIVATION IN PROGRESS")
            logger.warning("=" * 60)

            # Lazy import to ensure executor is never touched in SIGNAL_ONLY mode
            from executor import Executor

            # Verify explicit confirmation phrase
            if self.confirm_phrase == settings.CONFIRMATION_PHRASE:
                confirmed = True
                logger.info("Runtime confirmation phrase validated via CLI/UI argument.")
            else:
                confirmed = Executor.verify_runtime_confirmation()

            if not confirmed:
                logger.error("Safety confirmation failed. Falling back to safe SIGNAL_ONLY mode.")
                self.mode = "signal"
                return

            settings.EXECUTION_ENABLED = True
            self.executor = Executor(alert_manager=self.alert_manager)
            ready = await self.executor.initialize()
            if not ready:
                logger.warning("Executor failed to initialize cleanly. Reverting to SIGNAL_ONLY.")
                self.mode = "signal"
                self.executor = None
            else:
                logger.info("Assisted Execution successfully primed.")
        else:
            self.mode = "signal"
            logger.info("Operating in safe, educational SIGNAL_ONLY mode. Execution layer is DISABLED.")

    async def initialize_models_and_history(self):
        """Prepare history buffers and train/load models for configured assets."""
        assets = settings.ASSETS[:settings.MAX_CONCURRENT_STREAMS]
        logger.info(f"Preparing assets: {assets}")

        for asset in assets:
            model = AdaptiveModel(asset=asset)
            loaded = model.load_latest()

            if loaded and model.model is not None:
                logger.info(f"Existing model loaded for {asset}. Seeding indicator buffer with recent history...")
                history_df = await self.data_client.get_historical_candles(
                    asset, days=2, timeframe=settings.TIMEFRAME_SECONDS
                )
            else:
                logger.info(f"No existing model for {asset}. Fetching history and training initial model...")
                history_df = await self.data_client.get_historical_candles(
                    asset, days=settings.CANDLE_HISTORY_DAYS, timeframe=settings.TIMEFRAME_SECONDS
                )
                payout = await self.data_client.get_payout(asset)
                model.train(history_df, default_payout=payout)

            self.models[asset] = model
            self.signal_engine.set_initial_buffer(asset, history_df)

    def _retrain_worker(self):
        """Background thread executing periodic rolling retraining."""
        logger.info(f"Background retrain worker started (Interval: {settings.RETRAIN_INTERVAL_HOURS} hours).")
        interval_seconds = settings.RETRAIN_INTERVAL_HOURS * 3600

        while not self.stop_retrain_event.wait(timeout=interval_seconds):
            logger.info("Starting scheduled rolling retrain across configured assets...")
            for asset in settings.ASSETS[:settings.MAX_CONCURRENT_STREAMS]:
                try:
                    loop = asyncio.new_event_loop()
                    history_df = loop.run_until_complete(
                        self.data_client.get_historical_candles(
                            asset, days=settings.CANDLE_HISTORY_DAYS, timeframe=settings.TIMEFRAME_SECONDS
                        )
                    )
                    payout = loop.run_until_complete(self.data_client.get_payout(asset))
                    loop.close()

                    if asset in self.models:
                        self.models[asset].train(history_df, default_payout=payout)
                        logger.info(f"Successfully retrained model checkpoint for {asset}.")

                except Exception as e:
                    logger.error(f"Error retraining model for {asset}: {e}")

            # Notify user
            asyncio.run(self.alert_manager.send_system_alert(
                "Rolling Retrain Completed",
                f"Adaptive ML models updated for {settings.ASSETS[:settings.MAX_CONCURRENT_STREAMS]}."
            ))

    async def _handle_asset_stream(self, asset: str):
        """Manage candle stream ingestion for a single asset."""
        logger.info(f"Ingestion worker launched for {asset}...")
        try:
            async for candle in self.data_client.stream_candles(asset, timeframe=settings.TIMEFRAME_SECONDS):
                if not self.running:
                    break

                payout = await self.data_client.get_payout(asset)
                signal_obj = self.signal_engine.process_candle(
                    asset=asset,
                    candle=candle,
                    payout=payout,
                    expiry=settings.TIMEFRAME_SECONDS,
                    mode="ASSISTED" if self.executor else "SIGNAL_ONLY"
                )

                if signal_obj:
                    # Optional Execution
                    if self.executor is not None:
                        executed = await self.executor.execute_trade(signal_obj)
                        signal_obj.executed = executed
                        # Record outcome simulation
                        sim_pnl = settings.TRADE_AMOUNT * payout if executed else 0.0
                        self.signal_engine.record_trade_outcome(sim_pnl)

                    # Persist signal to shared storage for live UI display
                    sig_dict = {
                        "asset": signal_obj.asset,
                        "direction": signal_obj.direction,
                        "probability": round(float(signal_obj.probability), 4),
                        "ev": round(float(signal_obj.ev), 4),
                        "payout": round(float(signal_obj.payout), 2),
                        "timestamp": float(signal_obj.timestamp),
                        "formatted_time": signal_obj.formatted_time,
                        "expiry": int(signal_obj.expiry),
                        "mode": signal_obj.mode,
                        "executed": bool(signal_obj.executed),
                        "confluence": bool(signal_obj.confluence),
                        "metadata": signal_obj.metadata,
                    }
                    record_signal(sig_dict)

                    # Dispatch alert
                    await self.alert_manager.send_signal(signal_obj)

        except asyncio.CancelledError:
            logger.info(f"Stream handler for {asset} cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in stream loop for {asset}: {e}")

    async def start(self):
        """Start all operational tasks and begin main event loop."""
        self.running = True
        logger.info("Initializing Quotex Data Client...")
        connected = await self.data_client.connect()
        if not connected:
            logger.error(
                "Quotex Data Client failed to authenticate or connect. "
                "Per safety instructions, execution is halted."
            )
            return

        await self.initialize_mode()
        await self.initialize_models_and_history()

        # Start background retrain thread
        self.retrain_thread = threading.Thread(target=self._retrain_worker, daemon=True)
        self.retrain_thread.start()

        # Launch candle stream workers
        assets_to_stream = settings.ASSETS[:settings.MAX_CONCURRENT_STREAMS]
        for asset in assets_to_stream:
            task = asyncio.create_task(self._handle_asset_stream(asset))
            self.stream_tasks.append(task)

        logger.info(f"Quotex Bot actively running in [{self.mode.upper()}] mode. Press Ctrl+C to terminate.")

        try:
            await asyncio.gather(*self.stream_tasks)
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        """Orderly shutdown of network connections, streams, and browsers."""
        logger.info("Initiating graceful shutdown sequence...")
        self.running = False
        self.stop_retrain_event.set()

        for task in self.stream_tasks:
            task.cancel()

        if self.executor:
            await self.executor.close()

        await self.data_client.disconnect()
        logger.info("Quotex Bot shutdown complete.")


def run_training_mode(asset: str, days: int, timeframe: int = 60):
    """CLI mode: Train and save model checkpoint for specified asset."""
    logger.info(f"Running standalone training for {asset} ({days} days, timeframe {timeframe}s)...")
    df = None
    payout = settings.DEFAULT_PAYOUT

    csv_cache = os.path.join("data", f"{asset}_{days}d_{timeframe}s.csv")
    csv_fallback = os.path.join("data", f"{asset}_30d_60s.csv")

    # Tier 1: Try connecting to Quotex to fetch fresh historical candles
    loop = asyncio.new_event_loop()
    client = QuotexDataClient()
    connected = False
    try:
        connected = loop.run_until_complete(client.connect())
    except Exception as e:
        logger.warning(f"Could not connect to broker: {e}")

    if connected:
        try:
            df = loop.run_until_complete(client.get_historical_candles(asset, days=days, timeframe=timeframe))
            payout = loop.run_until_complete(client.get_payout(asset))
            loop.run_until_complete(client.disconnect())
            if df is not None and not df.empty:
                os.makedirs("data", exist_ok=True)
                df.to_csv(csv_cache, index=False)
                logger.info(f"Saved {len(df)} fresh historical candles to {csv_cache}")
        except Exception as e:
            logger.warning(f"Broker historical candle fetch error: {e}")
    loop.close()

    # Tier 2: If broker connection was busy/unavailable, load verified local real historical cache
    if df is None or df.empty:
        if os.path.exists(csv_cache):
            logger.info(f"Loading verified historical dataset from {csv_cache}...")
            df = pd.read_csv(csv_cache)
        elif os.path.exists(csv_fallback):
            logger.info(f"Loading verified historical dataset from {csv_fallback}...")
            df = pd.read_csv(csv_fallback)
        else:
            # Fallback to UI candle buffer
            df = load_candles_for_ui(asset, count=min(days * 1440, 5000))

    if df is None or df.empty or len(df) < 50:
        logger.error(f"Training aborted: Insufficient historical candles available for {asset}.")
        return

    logger.info(f"Training model on {len(df)} candles for {asset} (payout: {payout * 100:.0f}%)...")
    model = AdaptiveModel(asset=asset)
    model.train(df, default_payout=payout)
    print(f"\nTraining complete. Model checkpoint saved to: {model.model_path}")


def run_backtest_mode(asset: str, days: int, timeframe: int = 60, export_path: Optional[str] = None):
    """CLI mode: Execute backtest and print performance breakdown."""
    logger.info(f"Running backtest for {asset} ({days} days history, timeframe {timeframe}s)...")
    df = None
    payout = settings.DEFAULT_PAYOUT

    csv_cache = os.path.join("data", f"{asset}_{days}d_{timeframe}s.csv")
    csv_fallback = os.path.join("data", f"{asset}_30d_60s.csv")

    loop = asyncio.new_event_loop()
    client = QuotexDataClient()
    connected = False
    try:
        connected = loop.run_until_complete(client.connect())
    except Exception as e:
        logger.warning(f"Could not connect to broker: {e}")

    if connected:
        try:
            df = loop.run_until_complete(client.get_historical_candles(asset, days=days, timeframe=timeframe))
            payout = loop.run_until_complete(client.get_payout(asset))
            loop.run_until_complete(client.disconnect())
        except Exception as e:
            logger.warning(f"Broker historical candle fetch error: {e}")
    loop.close()

    if df is None or df.empty:
        if os.path.exists(csv_cache):
            logger.info(f"Loading verified historical dataset from {csv_cache}...")
            df = pd.read_csv(csv_cache)
        elif os.path.exists(csv_fallback):
            logger.info(f"Loading verified historical dataset from {csv_fallback}...")
            df = pd.read_csv(csv_fallback)
        else:
            df = load_candles_for_ui(asset, count=min(days * 1440, 5000))

    if df is None or df.empty or len(df) < 50:
        logger.error(f"Backtest aborted: Insufficient historical candles available for {asset}.")
        return

    backtester = Backtester(default_payout=payout)
    backtester.run(
        df=df,
        asset=asset,
        confidence_threshold=settings.SIGNAL_CONFIDENCE_THRESHOLD,
        min_ev=settings.MIN_EXPECTED_VALUE,
        export_csv_path=export_path
    )


def run_scan_mode(asset: str, timeframe: int = 60, generate_demo: bool = False, sensitivity: str = "strict"):
    """Run immediate single-pass market evaluation, produce signal, and save to data/signals.json."""
    logger.info(f"Running single-pass scan for {asset} (timeframe: {timeframe}s, sensitivity: {sensitivity})...")

    # Load latest candle buffer
    df = load_candles_for_ui(asset, count=80)
    if df.empty or len(df) < 15:
        logger.error(f"Insufficient candle history available to scan {asset}.")
        return None

    # Load or initialize model
    model = AdaptiveModel(asset=asset)
    if not model.load_latest():
        logger.info(f"No saved model found for {asset}. Fitting initial model from historical candles...")
        model.train(df, default_payout=settings.DEFAULT_PAYOUT)

    # Process through SignalEngine
    engine = SignalEngine(models={asset: model})
    engine.set_initial_buffer(asset, df.iloc[:-1])

    latest_candle = df.iloc[-1].to_dict()
    payout = settings.DEFAULT_PAYOUT

    signal_obj = engine.process_candle(
        asset=asset,
        candle=latest_candle,
        payout=payout,
        expiry=timeframe,
        mode="SIGNAL_ONLY"
    )

    if signal_obj:
        sig_dict = {
            "asset": signal_obj.asset,
            "direction": signal_obj.direction,
            "probability": round(float(signal_obj.probability), 4),
            "ev": round(float(signal_obj.ev), 4),
            "payout": round(float(signal_obj.payout), 2),
            "timestamp": float(signal_obj.timestamp),
            "formatted_time": signal_obj.formatted_time,
            "expiry": int(signal_obj.expiry),
            "mode": "QUALIFIED_STRICT",
            "executed": False,
            "confluence": bool(signal_obj.confluence),
            "metadata": signal_obj.metadata,
        }
        record_signal(sig_dict)
        logger.info(f"SCAN GENERATED QUALIFIED SIGNAL: {signal_obj.direction} on {asset} (Prob: {signal_obj.probability:.1%}, EV: {signal_obj.ev:.1%})")
        print(f"SIGNAL_GENERATED:{signal_obj.direction}:{signal_obj.probability:.3f}:{signal_obj.ev:.3f}")
        return sig_dict
    else:
        feat_df = build_features(df)
        prob_call = model.predict_proba(feat_df)
        direction, confidence, ev = model.calculate_expected_value(prob_call, payout)

        logger.info(
            f"Scan evaluation for {asset}: {direction} (Prob: {confidence:.1%}, EV: {ev:.1%}). "
            f"Required: conf >= {settings.SIGNAL_CONFIDENCE_THRESHOLD:.1%}, EV >= {settings.MIN_EXPECTED_VALUE:.1%}."
        )

        should_record = False
        record_mode = "TEST_SCAN"

        if generate_demo:
            should_record = True
            record_mode = "TEST_SCAN"
        elif sensitivity == "continuous":
            should_record = True
            record_mode = "AUTO_BIAS"
        elif sensitivity == "moderate" and confidence >= 0.54:
            should_record = True
            record_mode = "MODERATE_BIAS"

        if should_record:
            sig_dict = {
                "asset": asset,
                "direction": direction,
                "probability": round(float(confidence), 4),
                "ev": round(float(ev), 4),
                "payout": round(float(payout), 2),
                "timestamp": float(time.time()),
                "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "expiry": int(timeframe),
                "mode": record_mode,
                "executed": False,
                "confluence": True if confidence >= 0.54 else False,
                "metadata": {"source": "scan", "prob_call": round(prob_call, 4), "sensitivity": sensitivity},
            }
            record_signal(sig_dict)
            print(f"SIGNAL_RECORDED:{record_mode}:{direction}:{confidence:.3f}:{ev:.3f}")
            return sig_dict

        print(f"SCAN_COMPLETE:NO_TRIGGER:{direction}:{confidence:.3f}:{ev:.3f}")
        return None


def main():
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description="Adaptive Quotex Signal & Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["signal", "assisted"],
        default="signal",
        help="Operational mode: 'signal' (default, alerts only) or 'assisted' (opt-in trade execution)"
    )
    parser.add_argument("--confirm", default="", help="Runtime confirmation phrase for assisted mode")
    parser.add_argument("--scan", action="store_true", help="Run immediate single-pass signal scan and exit")
    parser.add_argument("--sensitivity", choices=["strict", "moderate", "continuous"], default="strict", help="Scan sensitivity filter")
    parser.add_argument("--test-signal", action="store_true", help="Generate test signal from latest market evaluation")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtester and exit")
    parser.add_argument("--train", action="store_true", help="Train model checkpoint on historical data and exit")
    parser.add_argument("--asset", default=settings.ASSETS[0], help="Target asset symbol (e.g. EURUSD_otc)")
    parser.add_argument("--days", type=int, default=settings.CANDLE_HISTORY_DAYS, help="Historical candle days to load")
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        choices=["1m", "5m", "15m", "60", "300", "900"],
        help="Candle timeframe ('1m', '5m', '15m' or seconds)"
    )
    parser.add_argument("--export", default=None, help="Optional CSV file path to export backtest trade logs")

    args = parser.parse_args()

    tf_map = {"1m": 60, "5m": 300, "15m": 900, "60": 60, "300": 300, "900": 900}
    timeframe_seconds = tf_map.get(args.timeframe, 60)

    if args.scan or args.test_signal:
        run_scan_mode(args.asset, timeframe=timeframe_seconds, generate_demo=args.test_signal, sensitivity=args.sensitivity)
        return

    if args.train:
        run_training_mode(args.asset, args.days, timeframe=timeframe_seconds)
        return

    if args.backtest:
        run_backtest_mode(args.asset, args.days, timeframe=timeframe_seconds, export_path=args.export)
        return

    bot = BotApplication(mode=args.mode, confirm_phrase=args.confirm)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def handle_signal():
        logger.info("Signal received, cancelling tasks...")
        loop.create_task(bot.shutdown())

    # Register OS signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            # Windows signal handler fallback
            pass

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected.")
        loop.run_until_complete(bot.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()

