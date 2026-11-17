@echo off
setlocal

echo ========================================================
echo  Adaptive Quotex Signal & Assisted-Trading Bot
echo ========================================================
echo.
echo Select an option:
echo [1] Run Backtest (EURUSD_otc)
echo [2] Train Adaptive Model (EURUSD_otc)
echo [3] Run Automated Test Suite
echo [4] Start Bot in SIGNAL_ONLY Mode (Default, Safe)
echo [5] Start Bot in ASSISTED Mode (Requires Confirmation)
echo [6] Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo Running Backtester...
    python main.py --backtest --asset EURUSD_otc --days 30
    pause
) else if "%choice%"=="2" (
    echo.
    echo Training Model...
    python main.py --train --asset EURUSD_otc --days 60
    pause
) else if "%choice%"=="3" (
    echo.
    echo Running Test Suite...
    python -m unittest discover tests
    pause
) else if "%choice%"=="4" (
    echo.
    echo Starting SIGNAL_ONLY Bot...
    python main.py --mode signal
) else if "%choice%"=="5" (
    echo.
    echo Starting ASSISTED Bot...
    python main.py --mode assisted
) else (
    echo Exiting.
)

endlocal

