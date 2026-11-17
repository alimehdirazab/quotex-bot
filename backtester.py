"""Event-Driven Backtesting Engine for Quotex Binary Options.

Emulates fixed-payout and fixed-expiry dynamics. Evaluates predictive performance,
win rates, drawdown, profit factor, and breaks down results across confidence
and Expected Value (EV) buckets with break-even threshold analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from feature_engine import build_features, get_feature_columns
from model_pipeline import AdaptiveModel


@dataclass
class TradeRecord:
    timestamp: float
    asset: str
    direction: str
    entry_price: float
    exit_price: float
    probability: float
    ev: float
    payout: float
    win: bool
    tie: bool
    pnl: float
    cumulative_pnl: float


class Backtester:
    """Simulates binary options trade outcomes across historical candle series."""

    def __init__(
        self,
        default_payout: float = 0.85,
        trade_amount: float = 1.0,
        initial_balance: float = 100.0
    ):
        self.default_payout = default_payout
        self.trade_amount = trade_amount
        self.initial_balance = initial_balance
        self.feature_columns = get_feature_columns()

    def run(
        self,
        df: pd.DataFrame,
        asset: str = "EURUSD_otc",
        model: Optional[AdaptiveModel] = None,
        confidence_threshold: float = 0.65,
        min_ev: float = 0.02,
        export_csv_path: Optional[str] = None
    ) -> Dict:
        """Execute historical replay backtest."""
        logger.info(f"Starting event-driven backtest for {asset} ({len(df)} candles)...")
        features_df = build_features(df, drop_na=True)
        if features_df.empty or len(features_df) < 50:
            raise ValueError("Insufficient candles for reliable backtesting.")

        # Train model if not supplied
        if model is None:
            model = AdaptiveModel(asset=asset)
            # Use first 70% for training and remaining 30% for out-of-sample backtest
            split_idx = int(len(features_df) * 0.7)
            train_df = df.iloc[:split_idx]
            test_features = features_df.iloc[split_idx:].copy()
            model.train(train_df, default_payout=self.default_payout)
        else:
            test_features = features_df.copy()

        trades: List[TradeRecord] = []
        balance = self.initial_balance
        cumulative_pnl = 0.0

        for i in range(len(test_features) - 1):
            row = test_features.iloc[i]
            next_row = test_features.iloc[i + 1]

            # Model inference
            proba_up = model.predict_proba(test_features.iloc[:i + 1])
            direction, conf, ev = model.calculate_expected_value(proba_up, self.default_payout)

            # Confluence check
            has_bullish_confluence = bool(row.get('bullish_confluence', False))
            has_bearish_confluence = bool(row.get('bearish_confluence', False))
            is_confluent = (
                (direction == "CALL" and has_bullish_confluence) or
                (direction == "PUT" and has_bearish_confluence)
            )

            # Check signal criteria
            qualified = (is_confluent and conf >= confidence_threshold and ev >= min_ev) or (
                conf >= 0.75 and ev >= 0.05
            )

            if not qualified:
                continue

            entry_price = float(row['close'])
            exit_price = float(next_row['close'])

            # Determine binary option outcome
            if direction == "CALL":
                win = exit_price > entry_price
                tie = exit_price == entry_price
            else:  # PUT
                win = exit_price < entry_price
                tie = exit_price == entry_price

            if tie:
                pnl = 0.0
            elif win:
                pnl = self.trade_amount * self.default_payout
            else:
                pnl = -self.trade_amount

            balance += pnl
            cumulative_pnl += pnl

            record = TradeRecord(
                timestamp=float(row.get('time', i)),
                asset=asset,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                probability=round(conf, 4),
                ev=round(ev, 4),
                payout=self.default_payout,
                win=win,
                tie=tie,
                pnl=round(pnl, 4),
                cumulative_pnl=round(cumulative_pnl, 4)
            )
            trades.append(record)

        results = self._compute_performance_metrics(trades, balance)
        self.print_report(results)

        if export_csv_path and trades:
            try:
                import os
                os.makedirs(os.path.dirname(os.path.abspath(export_csv_path)), exist_ok=True)
                export_df = pd.DataFrame([t.__dict__ for t in trades])
                export_df['datetime'] = pd.to_datetime(export_df['timestamp'], unit='s', errors='coerce')
                export_df.to_csv(export_csv_path, index=False)
                logger.info(f"Exported {len(trades)} backtest trade records to {export_csv_path}")
            except Exception as e:
                logger.error(f"Failed to export trade records to CSV: {e}")

        return results

    def _compute_performance_metrics(self, trades: List[TradeRecord], final_balance: float) -> Dict:
        """Compute comprehensive trading performance metrics and breakdowns."""
        total_trades = len(trades)
        if total_trades == 0:
            return {"total_trades": 0, "message": "No trades satisfied the signal filter criteria."}

        wins = sum(1 for t in trades if t.win)
        losses = sum(1 for t in trades if not t.win and not t.tie)
        ties = sum(1 for t in trades if t.tie)
        decided_trades = wins + losses
        win_rate = (wins / decided_trades) if decided_trades > 0 else 0.0

        pnl_series = [t.pnl for t in trades]
        net_pnl = sum(pnl_series)
        gross_profit = sum(p for p in pnl_series if p > 0)
        gross_loss = abs(sum(p for p in pnl_series if p < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # Peak to trough drawdown
        cum_pnl = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Break-even win rate calculation: 1 / (1 + payout)
        break_even_win_rate = 1.0 / (1.0 + self.default_payout)

        # Sharpe ratio
        if len(pnl_series) > 1 and np.std(pnl_series) > 0:
            sharpe = (np.mean(pnl_series) / np.std(pnl_series)) * np.sqrt(252 * 1440)
        else:
            sharpe = 0.0

        # Confidence Bucket Breakdown
        conf_buckets = {
            "50%-55%": [t for t in trades if 0.50 <= t.probability < 0.55],
            "55%-60%": [t for t in trades if 0.55 <= t.probability < 0.60],
            "60%-65%": [t for t in trades if 0.60 <= t.probability < 0.65],
            "65%-70%": [t for t in trades if 0.65 <= t.probability < 0.70],
            ">70%":    [t for t in trades if t.probability >= 0.70]
        }
        confidence_breakdown = {}
        for bucket, b_trades in conf_buckets.items():
            b_wins = sum(1 for t in b_trades if t.win)
            b_total = len(b_trades)
            b_wr = (b_wins / b_total) if b_total > 0 else 0.0
            b_pnl = sum(t.pnl for t in b_trades)
            confidence_breakdown[bucket] = {
                "trades": b_total,
                "win_rate": round(b_wr, 4),
                "net_pnl": round(b_pnl, 2)
            }

        # EV Bucket Breakdown
        ev_buckets = {
            "<0.0":     [t for t in trades if t.ev < 0.0],
            "0.0%-2.0%": [t for t in trades if 0.0 <= t.ev < 0.02],
            "2.0%-5.0%": [t for t in trades if 0.02 <= t.ev < 0.05],
            "5.0%-10%":  [t for t in trades if 0.05 <= t.ev < 0.10],
            ">10%":      [t for t in trades if t.ev >= 0.10]
        }
        ev_breakdown = {}
        for bucket, b_trades in ev_buckets.items():
            b_wins = sum(1 for t in b_trades if t.win)
            b_total = len(b_trades)
            b_wr = (b_wins / b_total) if b_total > 0 else 0.0
            b_pnl = sum(t.pnl for t in b_trades)
            ev_breakdown[bucket] = {
                "trades": b_total,
                "win_rate": round(b_wr, 4),
                "net_pnl": round(b_pnl, 2)
            }

        # Critical EV / Failure Assessment
        is_ev_negative = net_pnl < 0
        high_wr_negative_ev_trap = (win_rate > 0.50 and win_rate < break_even_win_rate)

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": round(win_rate, 4),
            "break_even_win_rate": round(break_even_win_rate, 4),
            "net_pnl": round(net_pnl, 2),
            "initial_balance": self.initial_balance,
            "final_balance": round(final_balance, 2),
            "return_pct": round(((final_balance - self.initial_balance) / self.initial_balance) * 100, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2),
            "confidence_breakdown": confidence_breakdown,
            "ev_breakdown": ev_breakdown,
            "high_wr_negative_ev_trap": high_wr_negative_ev_trap,
            "is_ev_negative": is_ev_negative
        }

    def print_report(self, r: Dict):
        """Display formatted console report."""
        if r.get("total_trades", 0) == 0:
            print(f"\n[BACKTEST REPORT] {r.get('message')}")
            return

        print("\n" + "=" * 65)
        print("          QUOTEX BINARY OPTIONS BACKTEST REPORT          ")
        print("=" * 65)
        print(f"Total Trades Taken  : {r['total_trades']}")
        print(f"Record              : {r['wins']} Wins | {r['losses']} Losses | {r['ties']} Ties")
        print(f"Observed Win Rate   : {r['win_rate'] * 100:.2f}%")
        print(f"Break-Even Win Rate : {r['break_even_win_rate'] * 100:.2f}% (at {self.default_payout*100:.0f}% payout)")
        print(f"Net P&L             : {r['net_pnl']:+0.2f} units ({r['return_pct']:+0.2f}%)")
        print(f"Profit Factor       : {r['profit_factor']}")
        print(f"Max Drawdown        : {r['max_drawdown']:.2f} units")
        print(f"Sharpe Ratio        : {r['sharpe_ratio']}")
        print("-" * 65)

        print("PERFORMANCE BY CONFIDENCE BUCKET:")
        for bucket, metrics in r['confidence_breakdown'].items():
            print(
                f"  [{bucket:9s}] Trades: {metrics['trades']:3d} | "
                f"Win Rate: {metrics['win_rate']*100:5.1f}% | "
                f"Net P&L: {metrics['net_pnl']:+6.2f}"
            )
        print("-" * 65)

        print("PERFORMANCE BY EXPECTED VALUE (EV) BUCKET:")
        for bucket, metrics in r['ev_breakdown'].items():
            print(
                f"  [{bucket:9s}] Trades: {metrics['trades']:3d} | "
                f"Win Rate: {metrics['win_rate']*100:5.1f}% | "
                f"Net P&L: {metrics['net_pnl']:+6.2f}"
            )
        print("=" * 65)

        if r['high_wr_negative_ev_trap']:
            print("⚠️ CRITICAL WARNING: NEGATIVE EXPECTED VALUE TRAP DETECTED!")
            print(
                f"  Observed win rate ({r['win_rate']*100:.1f}%) exceeds 50%, BUT is below the "
                f"break-even threshold ({r['break_even_win_rate']*100:.1f}%) for this payout."
            )
            print("  This strategy WILL lose money over time despite appearing 'accurate'.")
            print("=" * 65 + "\n")
        elif r['is_ev_negative']:
            print("❌ STRATEGY EVALUATION: FAILED (Net negative returns)")
            print("=" * 65 + "\n")
        else:
            print("✅ STRATEGY EVALUATION: POSITIVE EXPECTED VALUE CONFIRMED")
            print("=" * 65 + "\n")

