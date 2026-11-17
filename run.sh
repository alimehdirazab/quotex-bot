#!/usr/bin/env bash
set -e

echo "========================================================"
echo " Adaptive Quotex Signal & Assisted-Trading Bot"
echo "========================================================"
echo ""
echo "Select an option:"
echo "[1] Run Backtest (EURUSD_otc)"
echo "[2] Train Adaptive Model (EURUSD_otc)"
echo "[3] Run Automated Test Suite"
echo "[4] Start Bot in SIGNAL_ONLY Mode (Default, Safe)"
echo "[5] Start Bot in ASSISTED Mode (Requires Confirmation)"
echo "[6] Exit"
echo ""

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo "Running Backtester..."
        python main.py --backtest --asset EURUSD_otc --days 30
        ;;
    2)
        echo "Training Model..."
        python main.py --train --asset EURUSD_otc --days 60
        ;;
    3)
        echo "Running Test Suite..."
        python -m unittest discover tests
        ;;
    4)
        echo "Starting SIGNAL_ONLY Bot..."
        python main.py --mode signal
        ;;
    5)
        echo "Starting ASSISTED Bot..."
        python main.py --mode assisted
        ;;
    *)
        echo "Exiting."
        ;;
esac

