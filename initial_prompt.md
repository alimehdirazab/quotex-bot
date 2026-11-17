# PROJECT: Quotex Signal & Assisted-Trading Bot with Optional Execution

## ROLE
You are a senior Python engineer specialising in algorithmic trading systems, browser automation, and adaptive machine learning pipelines. Build the complete, production-ready codebase described below. Deliver every file, every function, and a working end-to-end system.

## CONTEXT & SCOPE
Quotex is an unregulated offshore binary options broker. Its Service Agreement prohibits automated trading. This project is built for **educational, research, and demo-account use**. The system must default to a SIGNAL-ONLY configuration that never places a trade. An optional execution mode is provided for users who accept the platform risk on a demo account. Include a README section explaining these constraints plainly.

## ARCHITECTURE — TWO MODES, SHARED CORE

The bot supports two runtime modes selected by configuration:

1.  **SIGNAL_ONLY (default):**
    - Connects to Quotex market data.
    - Runs feature engineering and ML inference.
    - Sends formatted alerts to Telegram and/or WhatsApp.
    - Never instantiates any execution component.

2.  **ASSISTED_EXECUTION (opt-in):**
    - Same data and signal pipeline.
    - Adds a browser-automation executor that places a trade only after the user confirms.
    - Requires `DEMO_ONLY=False` AND a typed confirmation phrase at session start.

Both modes share `DataLayer` and `SignalEngine`. Mode switching only toggles whether `Executor` is constructed.

## TECHNOLOGY STACK

| Layer | Library | Reason |
|---|---|---|
| Language | Python 3.10+ | Required. |
| Quotex market data | `pyquotex` (cleitonleonel/pyquotex) | Provides WebSocket candle streaming and account info. Uses `curl_cffi` for TLS compatibility with modern Chrome. |
| Alternative data client | `ChipaDevTeam/QuotexAPI` | Async WebSocket fallback. |
| Browser automation (execution mode only) | `playwright` with a persistent user profile | Standard, well-supported automation; persistent profiles preserve session state and reduce repeated logins. |
| Technical indicators | `pandas-ta` | Pure Python, no compiled dependencies. |
| ML | `lightgbm` (primary), `xgboost` (comparison) | Fast, accurate on tabular binary classification. |
| Alerts — Telegram | `pyTelegramBotAPI` | Simple, reliable. |
| Alerts — WhatsApp | `whatsapp-cloud-api` (Meta Cloud API) | Official API; supports text and image messages. |
| Config | `pydantic-settings` + `.env` | Type-safe configuration. |
| Logging | `loguru` | Structured, rotating logs. |
| Backtesting | Custom event-driven backtester | Must replicate Quotex payout and expiry rules exactly. |

## MODULES — DELIVER EVERY FILE

### 1. `config.py` — Pydantic Settings
- `DEMO_ONLY: bool = True`
- `EXECUTION_ENABLED: bool = False`
- `CONFIRMATION_PHRASE: str = "I_ACCEPT_THE_RISK"`
- `QUOTEX_EMAIL`, `QUOTEX_PASSWORD`, `QUOTEX_SSID` (SSID preferred)
- `ASSETS: list[str]` (e.g., `["EURUSD_otc", "GBPUSD_otc"]`)
- `TIMEFRAME_SECONDS: int = 60`
- `CANDLE_HISTORY_DAYS: int = 60`
- `MAX_CONCURRENT_STREAMS: int = 3`  # honours documented library limits
- `SIGNAL_CONFIDENCE_THRESHOLD: float = 0.65`
- `MIN_EXPECTED_VALUE: float = 0.02`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `WHATSAPP_API_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_RECIPIENT`
- `RETRAIN_INTERVAL_HOURS: int = 6`
- `MODEL_PURGE_KEEP: int = 5`
- `MAX_DAILY_TRADES: int = 20`
- `MAX_DAILY_LOSS_PCT: float = 0.05`
- `ACTION_DELAY_MIN_MS: int = 150`, `ACTION_DELAY_MAX_MS: int = 400`

### 2. `data_layer.py`
Class `QuotexDataClient`:
- Initialise with SSID (preferred) or email/password.
- `stream_candles(asset, timeframe)` → async generator of OHLCV dicts.
- `get_payout(asset)` → current payout percentage.
- `get_historical_candles(asset, days)` → pandas DataFrame.
- **Concurrency guard:** cap active streams at `MAX_CONCURRENT_STREAMS` (default 3), matching the library's documented limits.
- **Resilient reconnect:** exponential backoff with jitter on disconnect; log every reconnect.
- **Request pacing:** minimum interval between API calls, enforced centrally.

### 3. `feature_engine.py`
`build_features(df)`:
- RSI(14), MACD(12,26,9), Bollinger Bands(20,2), EMA(9,21), ATR(14).
- Candle body ratio, upper/lower wick ratios, rolling volatility, time-of-day sine/cosine.
- `triple_confluence` boolean = (RSI<30 or RSI>70) AND MACD crossover AND price outside Bollinger Band.
- All features shifted to prevent look-ahead bias.

### 4. `model_pipeline.py` — Adaptive ML
Class `AdaptiveModel`:
- `train(df)` → LightGBM binary classifier; target = next-candle direction.
- **Walk-forward validation:** train on months 1–6, validate on 7; retrain on 2–7, validate on 8; etc. Report precision, recall, and **expected value (EV)**.
- **EV:** for CALL, `EV = p_up * payout − (1 − p_up)`; for PUT, `EV = p_down * payout − (1 − p_down)`. Emit signals only when `EV > MIN_EXPECTED_VALUE`.
- **Rolling retrain:** every `RETRAIN_INTERVAL_HOURS`, fetch fresh history, retrain, save with timestamp. Purge older models beyond `MODEL_PURGE_KEEP`.
- Persist via `joblib` in `./models/`.
- `predict_proba(features) -> float`.

### 5. `signal_engine.py`
Class `SignalEngine`:
- Loads latest `AdaptiveModel`.
- On each candle: build features → predict → combine with `triple_confluence` → emit `Signal {asset, direction, probability, ev, payout, timestamp, expiry}` when all thresholds pass.
- Deduplicate signals within the same candle period.
- Respect daily trade and loss limits.

### 6. `executor.py` — Assisted Execution (constructed only when EXECUTION_ENABLED)
Class `Executor`:
- **Not imported or instantiated in SIGNAL_ONLY mode.**
- Playwright with a **persistent user profile directory** so the Quotex session is reused across runs (fewer logins = more stable sessions and fewer rate-limit hits).
- **Login:** open a normal visible browser window; if SSID is absent, prompt the user to log in once, then persist cookies.
- **Paced interactions:** every click and scroll waits a random interval between `ACTION_DELAY_MIN_MS` and `ACTION_DELAY_MAX_MS`. This respects platform timing and mirrors normal usage.
- **Session budget:** at most `MAX_DAILY_TRADES` per day and a minimum gap between trades.
- **Confirmation gate:** at the start of each session the user must type `CONFIRMATION_PHRASE`. Refuse otherwise.
- **Preferred execution path:** place trades via `ChipaDevTeam/QuotexAPI` using SSID when available; fall back to the browser only when necessary.
- **Backoff:** on any HTTP 429, 403, or repeated timeout, pause activity for an exponentially increasing interval and notify the user via alert channel.

### 7. `alerting.py`
Class `AlertManager`:
- `send_telegram(signal)` and `send_whatsapp(signal)` — format asset, direction, confidence, EV, payout, expiry.
- `send_system_alert(message)` — for reconnect events, retrain completions, risk-limit triggers.
- Every signal is tagged `[SIGNAL]` or `[EXECUTED]` so the user always knows which mode acted.
- Deduplicate identical signals.

### 8. `backtester.py`
Class `Backtester`:
- Replays historical candles through `SignalEngine`.
- Emulates Quotex payout (configurable, default 80%) and expiry (default 60s).
- Reports win rate, net P&L, Sharpe (when N is sufficient), max drawdown.
- Breaks results down **by confidence bucket and by EV threshold**; flags configurations where win rate is high but EV is negative.

### 9. `main.py`
- CLI: `--mode signal|assisted`, `--backtest`, `--train`.
- Construct `DataLayer`, `SignalEngine`, `AlertManager`; construct `Executor` only in assisted mode with confirmations satisfied.
- Start retrain thread and candle-stream loop.
- Graceful shutdown on SIGINT.

### 10. `requirements.txt`
Pinned versions of: `pyquotex`, `ChipaDevTeam-QuotexAPI`, `pandas`, `pandas-ta`, `lightgbm`, `xgboost`, `scikit-learn`, `pyTelegramBotAPI`, `whatsapp-cloud-api`, `playwright`, `pydantic-settings`, `loguru`, `joblib`, `python-dotenv`.

### 11. `README.md`
- Setup (SSID extraction via browser DevTools).
- Plain-spoken warning about Quotex's terms and the risk of account closure.
- How to run in SIGNAL_ONLY mode (default).
- How to enable assisted execution (with confirmation phrase).
- How to backtest and read the EV report.
- Telegram and WhatsApp setup.

## ENGINEERING STANDARDS FOR THE EXECUTION LAYER
- **Session stability:** persistent browser profile; reuse the same Chrome user-data directory between runs.
- **Pacing:** randomised delays between actions (configurable range); minimum gap between trades.
- **Rate limits:** never exceed the library's documented worker limit; respect Retry-After headers.
- **Backoff & alerting:** on 429/403/network errors, pause and notify the user; never hammer endpoints.
- **Configurability:** every timing and limit value lives in `config.py`.
- **Failure isolation:** execution errors never crash the signal pipeline; they emit a system alert and disable execution for the session.

## DELIVERABLES
- All Python files listed.
- `requirements.txt`, `.env.example`, `README.md`.
- Optional `docker-compose.yml` mounting a persistent profile volume.

## FINAL INSTRUCTION
Build the complete system. Default to SIGNAL_ONLY. Gate assisted execution behind explicit confirmations. Every alert states which mode acted. Prioritise clarity, robust error handling, and the README warnings.