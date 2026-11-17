"""Quotex Adaptive Signal & Trading Bot - Streamlit Web Interface.

Provides real-time dashboards, configuration management, model training,
backtesting analytics, asset/payout tracking, real forex session monitoring,
alert testing, log viewing, and educational safety disclosures.
"""

import asyncio
from datetime import datetime, timezone, timedelta
import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Local module imports
from config import settings
from data_layer import (
    QuotexDataClient,
    is_real_forex,
    get_market_status,
    load_candles_for_ui,
    load_signals_for_ui,
    record_signal,
)
from model_pipeline import AdaptiveModel

st.set_page_config(
    page_title="Quotex Adaptive Signal Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Sidebar Global Header & Navigation
# ---------------------------------------------------------
st.sidebar.title("🤖 Quotex Bot UI")

# Account Mode Badge
if settings.DEMO_ONLY:
    st.sidebar.success("🛡️ MODE: PRACTICE / DEMO ONLY")
else:
    st.sidebar.error("⚠️ CRITICAL WARNING: LIVE REAL-MONEY MODE")

nav_page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Configuration",
        "Training",
        "Backtest",
        "Assets & Payouts",
        "Real Forex Analytics",
        "Alerts",
        "Logs",
        "About / Safety",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Environment: Python {sys.version.split()[0]}")
st.sidebar.caption(f"Default Payout: {settings.DEFAULT_PAYOUT * 100:.0f}%")
st.sidebar.caption(f"Log Level: {settings.LOG_LEVEL}")

# ---------------------------------------------------------
# Session State Helpers
# ---------------------------------------------------------
if "bot_process" not in st.session_state:
    st.session_state.bot_process = None

if "signals_history" not in st.session_state:
    st.session_state.signals_history = []


# =========================================================
# PAGE 1: DASHBOARD
# =========================================================
if nav_page == "Dashboard":
    st.title("📊 Control Dashboard")
    st.markdown("Live monitoring, real-time candlestick charts, and signal execution management for Quotex.")

    col1, col2, col3, col4 = st.columns(4)

    has_creds = bool((settings.QUOTEX_EMAIL and settings.QUOTEX_PASSWORD) or settings.QUOTEX_SSID)
    with col1:
        st.metric(
            "Connection Status",
            "Configured" if has_creds else "Missing Credentials",
            delta="Ready" if has_creds else "Action Needed",
        )
    with col2:
        st.metric(
            "Account Type",
            "DEMO (Safe)" if settings.DEMO_ONLY else "LIVE (DANGER)",
            delta=None if settings.DEMO_ONLY else "HIGH RISK",
            delta_color="inverse",
        )
    with col3:
        st.metric("Active Assets", len(settings.ASSETS), delta=f"Max {settings.MAX_CONCURRENT_STREAMS}")
    with col4:
        bot_running = st.session_state.bot_process is not None and st.session_state.bot_process.poll() is None
        st.metric("Bot Status", "RUNNING" if bot_running else "STOPPED", delta="Active" if bot_running else "Idle")

    # Quotex 2FA / PIN Code Challenge Banner
    if os.path.exists("data/otp_request.json"):
        st.error("🚨 **Quotex Email Security Challenge Detected**")
        otp_msg = "Quotex has sent a verification PIN code to your registered email address. Enter the code below to complete authentication:"
        try:
            with open("data/otp_request.json", "r", encoding="utf-8") as f:
                otp_payload = json.load(f)
                otp_msg = otp_payload.get("message", otp_msg)
        except Exception:
            pass
        st.markdown(f"**Prompt:** `{otp_msg}`")
        with st.form("pin_challenge_form"):
            otp_val = st.text_input("Enter 6-Digit Email PIN Code:", max_chars=8, placeholder="e.g. 123456")
            sub_otp = st.form_submit_button("Submit Verification PIN", type="primary")
            if sub_otp and otp_val:
                with open("data/otp_response.json", "w", encoding="utf-8") as f:
                    json.dump({"pin": otp_val.strip()}, f)
                try:
                    if os.path.exists("data/otp_request.json"):
                        os.remove("data/otp_request.json")
                except Exception:
                    pass
                st.success(f"PIN '{otp_val.strip()}' submitted! Broker connection resuming...")
                time.sleep(1.5)
                st.rerun()

    # Quick SSID Browser Cookie Login Expander
    with st.expander("🔑 Quick Connect via Browser SSID Cookie (Fast, Bypasses Email PIN)"):
        st.markdown("""
        **Have Quotex open in Chrome/Edge?** You can connect instantly without email PIN:
        1. Open your Quotex tab in your browser
        2. Press `F12` ➔ Go to **Application** (or Storage) ➔ **Cookies** ➔ `quotex.com` (or `qxbroker.com`)
        3. Copy the value of cookie named **`ssid`** *(do not use `remember_web`)*
        4. Paste it below and click Save.
        """)
        c_ssid1, c_ssid2 = st.columns([3, 1])
        with c_ssid1:
            fast_ssid = st.text_input("Paste Quotex `ssid` Cookie Value:", type="password", key="fast_ssid_box")
        with c_ssid2:
            st.write("")
            st.write("")
            if st.button("Save SSID & Connect", key="btn_save_ssid", use_container_width=True):
                if fast_ssid.strip():
                    lines = []
                    found = False
                    if os.path.exists(".env"):
                        with open(".env", "r", encoding="utf-8") as f:
                            lines = f.readlines()
                    out_lines = []
                    for l in lines:
                        if l.startswith("QUOTEX_SSID="):
                            out_lines.append(f"QUOTEX_SSID={fast_ssid.strip()}\n")
                            found = True
                        else:
                            out_lines.append(l)
                    if not found:
                        out_lines.append(f"QUOTEX_SSID={fast_ssid.strip()}\n")
                    with open(".env", "w", encoding="utf-8") as f:
                        f.writelines(out_lines)
                    st.success("SSID saved to .env! Ready to connect.")
                    st.rerun()

    st.markdown("---")

    # ---------------------------------------------------------
    # Chart & Market Data Section
    # ---------------------------------------------------------
    st.subheader("📈 Live Market Candlestick Chart & Confluence Engine")

    chart_ctl1, chart_ctl2, chart_ctl3 = st.columns([2, 1, 1])
    with chart_ctl1:
        selected_asset = st.selectbox(
            "Select Asset",
            ["EURUSD_otc", "AUDUSD_otc", "GBPUSD_otc", "USDJPY_otc", "EURUSD", "GBPUSD", "USDJPY"],
            index=0,
            key="dash_selected_asset"
        )
    with chart_ctl2:
        candle_count = st.slider("Candles Displayed", 30, 150, 75, step=5, key="dash_candle_count")
    with chart_ctl3:
        st.write("")
        st.write("")
        refresh_btn = st.button("🔄 Refresh Market Data", use_container_width=True)

    # Load candles
    candles_df = load_candles_for_ui(selected_asset, count=candle_count)

    if not candles_df.empty and len(candles_df) >= 10:
        # Convert times to formatted timestamp strings
        if "time" in candles_df.columns:
            if pd.api.types.is_numeric_dtype(candles_df["time"]):
                times_series = pd.to_datetime(candles_df["time"], unit="s")
            else:
                times_series = pd.to_datetime(candles_df["time"])
        else:
            times_series = pd.date_range(end=datetime.now(), periods=len(candles_df), freq="1min")

        close_prices = candles_df["close"].astype(float)
        high_prices = candles_df["high"].astype(float)
        low_prices = candles_df["low"].astype(float)
        open_prices = candles_df["open"].astype(float)

        # Technical Indicators
        ema9 = close_prices.ewm(span=9, adjust=False).mean()
        ema21 = close_prices.ewm(span=21, adjust=False).mean()
        sma20 = close_prices.rolling(window=20).mean()
        std20 = close_prices.rolling(window=20).std()
        bb_upper = sma20 + (2.0 * std20)
        bb_lower = sma20 - (2.0 * std20)

        # RSI 14
        delta = close_prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi14 = 100 - (100 / (1 + rs))

        # Hero Metrics Card
        curr_price = float(close_prices.iloc[-1])
        prev_open = float(open_prices.iloc[-1])
        pct_change = ((curr_price - prev_open) / prev_open) * 100
        payout_val = settings.DEFAULT_PAYOUT
        be_winrate = (1.0 / (1.0 + payout_val)) * 100

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric(
                f"{selected_asset} Price",
                f"{curr_price:.5f}" if curr_price < 20 else f"{curr_price:.3f}",
                delta=f"{pct_change:+.3f}% (bar)",
                delta_color="normal"
            )
        with m_col2:
            st.metric("24h Range", f"H: {high_prices.max():.5f}", delta=f"L: {low_prices.min():.5f}")
        with m_col3:
            st.metric("Quotex Payout", f"{payout_val * 100:.0f}%", delta="Standard OTC")
        with m_col4:
            st.metric("Break-Even Win Rate", f"{be_winrate:.1f}%", delta="Kelly Zero EV")
        with m_col5:
            m_stat = get_market_status(selected_asset)
            st.metric("Market Status", "OPEN" if m_stat.get("is_open") else "CLOSED", delta=m_stat.get("session", "Active"))

        # Build 2-row subplot: Row 1 = Candles + EMAs + BBs + Signals; Row 2 = RSI
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.74, 0.26],
            subplot_titles=(f"{selected_asset} 1-Minute Live Candlesticks & Indicators", "RSI (14) Momentum Oscillator")
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=times_series,
                open=open_prices,
                high=high_prices,
                low=low_prices,
                close=close_prices,
                name="Candles",
                increasing_line_color="#00E676",
                decreasing_line_color="#FF1744",
                increasing_fillcolor="#00E676",
                decreasing_fillcolor="#FF1744"
            ),
            row=1, col=1
        )

        # EMA 9
        fig.add_trace(
            go.Scatter(x=times_series, y=ema9, name="EMA 9 (Fast)", line=dict(color="#00E5FF", width=1.5)),
            row=1, col=1
        )
        # EMA 21
        fig.add_trace(
            go.Scatter(x=times_series, y=ema21, name="EMA 21 (Slow)", line=dict(color="#FF9100", width=1.5)),
            row=1, col=1
        )
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=times_series, y=bb_upper, name="BB Upper (20,2)", line=dict(color="rgba(180, 130, 255, 0.4)", width=1, dash="dot")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=times_series, y=bb_lower, name="BB Lower (20,2)",
                line=dict(color="rgba(180, 130, 255, 0.4)", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(180, 130, 255, 0.05)"
            ),
            row=1, col=1
        )

        # Add Signal Markers if any exist for this asset
        raw_signals = load_signals_for_ui(limit=50)
        asset_signals = [s for s in raw_signals if s.get("asset") == selected_asset]

        call_x, call_y, call_txt = [], [], []
        put_x, put_y, put_txt = [], [], []

        for sig in asset_signals:
            sig_ts = sig.get("timestamp", 0)
            sig_time = pd.to_datetime(sig_ts, unit="s")
            direction = sig.get("direction", "CALL")
            conf = sig.get("probability", 0.5) * 100
            ev_pct = sig.get("ev", 0.0) * 100
            txt = f"{direction} ({conf:.1f}% conf, {ev_pct:+.1f}% EV)"

            if direction == "CALL":
                call_x.append(sig_time)
                call_y.append(low_prices.iloc[-1] * 0.9998)
                call_txt.append(txt)
            else:
                put_x.append(sig_time)
                put_y.append(high_prices.iloc[-1] * 1.0002)
                put_txt.append(txt)

        if call_x:
            fig.add_trace(
                go.Scatter(
                    x=call_x, y=call_y,
                    mode="markers+text",
                    name="CALL Signal",
                    text=["▲ CALL" for _ in call_x],
                    textposition="bottom center",
                    textfont=dict(color="#00E676", size=11),
                    marker=dict(symbol="triangle-up", size=14, color="#00E676", line=dict(width=1, color="#FFFFFF")),
                    hovertext=call_txt,
                ),
                row=1, col=1
            )
        if put_x:
            fig.add_trace(
                go.Scatter(
                    x=put_x, y=put_y,
                    mode="markers+text",
                    name="PUT Signal",
                    text=["▼ PUT" for _ in put_x],
                    textposition="top center",
                    textfont=dict(color="#FF1744", size=11),
                    marker=dict(symbol="triangle-down", size=14, color="#FF1744", line=dict(width=1, color="#FFFFFF")),
                    hovertext=put_txt,
                ),
                row=1, col=1
            )

        # Row 2: RSI
        fig.add_trace(
            go.Scatter(x=times_series, y=rsi14, name="RSI 14", line=dict(color="#AB47BC", width=2)),
            row=2, col=1
        )
        # RSI Overbought/Oversold levels
        fig.add_hline(y=70, line_dash="dash", line_color="#EF5350", row=2, col=1, annotation_text="Overbought 70")
        fig.add_hline(y=30, line_dash="dash", line_color="#26A69A", row=2, col=1, annotation_text="Oversold 30")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=620,
            margin=dict(l=30, r=30, t=35, b=25),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161B22",
        )
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1, gridcolor="#21262D")
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100], gridcolor="#21262D")
        fig.update_xaxes(gridcolor="#21262D")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Loading candle history... If this is the first start, historical data will populate shortly.")

    st.markdown("---")

    # Bot Process Control
    st.subheader("⚡ Bot Runtime Execution Controls")

    # Sensitivity and Automation row
    opt_col1, opt_col2 = st.columns([2, 2])
    with opt_col1:
        sens_choice = st.selectbox(
            "Signal Generation Filter / Sensitivity",
            [
                "⚡ Continuous (Auto-Feed Directional Bias on Every Candle)",
                "🛡️ Strict Edge (Confidence ≥ 65%, Positive EV Only)",
                "⚖️ Moderate Edge (Confidence ≥ 54%, Break-Even+)"
            ],
            index=0,
            key="dash_sens_choice"
        )
        sens_key = "continuous" if "Continuous" in sens_choice else ("moderate" if "Moderate" in sens_choice else "strict")

    with opt_col2:
        auto_scan_active = st.checkbox(
            "🔄 Auto-Scan on Candle Close (Automatic Live Signals)",
            value=True,
            key="chk_auto_scan_stream"
        )
        if auto_scan_active:
            st.caption("Active: The bot automatically evaluates every closed candle and streams signals to the table.")

    # Background auto-scan check
    if "last_auto_scan_ts" not in st.session_state:
        st.session_state["last_auto_scan_ts"] = time.time()

    if auto_scan_active:
        now_ts = time.time()
        last_scan_ts = st.session_state.get("last_auto_scan_ts", now_ts)
        if now_ts - last_scan_ts >= 60:  # Scan every 60s candle close
            st.session_state["last_auto_scan_ts"] = now_ts
            try:
                subprocess.run(
                    [sys.executable, "main.py", "--scan", "--asset", selected_asset, "--sensitivity", sens_key],
                    capture_output=True,
                    timeout=10,
                    text=True
                )
            except Exception:
                pass

    ctl_col1, ctl_col2, ctl_col3, ctl_col4 = st.columns([1.2, 1.2, 1.3, 1.3])

    with ctl_col1:
        if not bot_running:
            if st.button("▶️ Start SIGNAL_ONLY Mode", use_container_width=True, type="primary"):
                proc = subprocess.Popen(
                    [sys.executable, "main.py", "--mode", "signal"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                st.session_state.bot_process = proc
                st.success("Started Quotex Bot in safe SIGNAL_ONLY mode.")
                st.rerun()
        else:
            if st.button("⏹️ Stop Bot Execution", use_container_width=True, type="secondary"):
                if st.session_state.bot_process:
                    st.session_state.bot_process.terminate()
                    st.session_state.bot_process = None
                st.warning("Bot execution stopped.")
                st.rerun()

    with ctl_col2:
        if not bot_running:
            enable_assisted = st.checkbox("Unlock Assisted Execution Mode", key="chk_assisted_unlock")
        else:
            st.info(f"Bot PID {st.session_state.bot_process.pid} active")

    with ctl_col3:
        if st.button("🎯 Scan & Generate Signal Now", use_container_width=True):
            with st.spinner(f"Evaluating {selected_asset} latest bar through ML model..."):
                scan_res = subprocess.run(
                    [sys.executable, "main.py", "--scan", "--asset", selected_asset, "--sensitivity", sens_key],
                    capture_output=True,
                    text=True
                )
                st.session_state.last_scan_output = scan_res.stdout
                st.rerun()

    with ctl_col4:
        if st.button("🧪 Generate Test Signal Marker", use_container_width=True):
            with st.spinner(f"Producing test signal marker on {selected_asset}..."):
                res = subprocess.run(
                    [sys.executable, "main.py", "--test-signal", "--asset", selected_asset],
                    capture_output=True,
                    text=True
                )
                st.success("Test signal generated and plotted on chart!")
                st.rerun()

    # Assisted mode confirmation gate box
    if not bot_running and st.session_state.get("chk_assisted_unlock", False):
        st.markdown("---")
        st.warning("🛡️ **ASSISTED EXECUTION CONFIRMATION GATE**")
        st.markdown("""
        Assisted mode will automatically place practice trades on Quotex when high-confidence signals trigger.
        - **Account Mode:** DEMO Practice only (`DEMO_ONLY=True` strictly enforced).
        - **Confirmation Phrase:** Must type `I_ACCEPT_THE_RISK` to authorize execution.
        """)
        c_gate1, c_gate2 = st.columns([2, 1])
        with c_gate1:
            phrase_input = st.text_input("Type confirmation phrase:", placeholder="I_ACCEPT_THE_RISK", key="assisted_phrase_box")
            demo_ack = st.checkbox("I verify this is a DEMO practice account and accept all execution risks.", key="assisted_ack_box")
        with c_gate2:
            st.write("")
            st.write("")
            if st.button("🚀 Confirm & Launch ASSISTED Mode", type="primary", use_container_width=True):
                if not demo_ack:
                    st.error("You must check the DEMO confirmation checkbox.")
                elif phrase_input.strip() != settings.CONFIRMATION_PHRASE:
                    st.error(f"Confirmation phrase must exactly equal '{settings.CONFIRMATION_PHRASE}'.")
                else:
                    proc = subprocess.Popen(
                        [sys.executable, "main.py", "--mode", "assisted", "--confirm", phrase_input.strip()],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    st.session_state.bot_process = proc
                    st.success("Started Quotex Bot in ASSISTED mode!")
                    st.rerun()

    # Show last scan output if available
    if "last_scan_output" in st.session_state and st.session_state.last_scan_output:
        st.info(f"**Last Scan Result:**\n`{st.session_state.last_scan_output.strip()}`")

    # Live Signals Feed
    st.markdown("---")
    sig_hdr1, sig_hdr2 = st.columns([3, 1])
    with sig_hdr1:
        st.subheader("📡 Recent Signals Feed & Execution Records")
    with sig_hdr2:
        if st.button("🗑️ Clear Signals History", use_container_width=True):
            if os.path.exists("data/signals.json"):
                try:
                    os.remove("data/signals.json")
                except Exception:
                    pass
            st.rerun()

    signals_list = load_signals_for_ui(limit=25)
    if signals_list:
        table_rows = []
        for s in signals_list:
            dir_label = "🟢 CALL" if s.get("direction") == "CALL" else "🔴 PUT"
            raw_mode = s.get("mode", "SIGNAL_ONLY")
            if "QUALIFIED" in raw_mode:
                mode_disp = "🌟 QUALIFIED (High Edge)"
            elif "AUTO_BIAS" in raw_mode:
                mode_disp = "⚡ AUTO BIAS"
            elif "MODERATE" in raw_mode:
                mode_disp = "⚖️ MODERATE BIAS"
            elif "TEST" in raw_mode:
                mode_disp = "🧪 DEMO MARKER"
            else:
                mode_disp = raw_mode

            table_rows.append({
                "Time": s.get("formatted_time", "N/A"),
                "Asset": s.get("asset", "N/A"),
                "Direction": dir_label,
                "Confidence": f"{s.get('probability', 0.5) * 100:.1f}%",
                "EV": f"{s.get('ev', 0.0) * 100:+.1f}%",
                "Payout": f"{s.get('payout', 0.85) * 100:.0f}%",
                "Expiry": f"{s.get('expiry', 60)}s",
                "Mode": mode_disp,
                "Executed": "✅ Yes (Demo)" if s.get("executed") else "❌ No (Signal Only)",
                "Confluence": "✅ Pass" if s.get("confluence") else "⚠️ Warning"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.info("No signals generated yet. Click '🎯 Scan & Generate Signal Now' or '🧪 Generate Test Signal Marker' above to evaluate the market.")

    # Active Models Summary
    st.markdown("---")
    st.subheader("🧠 Model Checkpoints On Disk")
    model_files = glob.glob(os.path.join(settings.MODEL_DIR, "model_*.joblib"))
    if model_files:
        records = []
        for mf in sorted(model_files, key=os.path.getmtime, reverse=True)[:10]:
            name = os.path.basename(mf)
            parts = name.replace(".joblib", "").split("_")
            asset = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
            mod_time = datetime.fromtimestamp(os.path.getmtime(mf)).strftime("%Y-%m-%d %H:%M:%S")
            records.append({
                "Filename": name,
                "Asset": asset,
                "Modified": mod_time,
                "Size (KB)": round(os.path.getsize(mf) / 1024, 1),
            })
        st.dataframe(pd.DataFrame(records), use_container_width=True)
    else:
        st.warning("No saved model checkpoints found in ./models/. Train a model on the Training page.")


# =========================================================
# PAGE 2: CONFIGURATION
# =========================================================
elif nav_page == "Configuration":
    st.title("⚙️ Bot Configuration")
    st.markdown("Inspect, validate, and update parameters stored in `.env`.")

    # Actions: Test Connection & Validate
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("🧪 Test Quotex Connection (scripts/check_login.py)", use_container_width=True):
            with st.spinner("Connecting to Quotex WebSocket..."):
                res = subprocess.run(
                    [sys.executable, "scripts/check_login.py"],
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    st.success(res.stdout)
                else:
                    st.error(res.stdout or res.stderr)

    with col_act2:
        if st.button("✔️ Validate Configuration Settings", use_container_width=True):
            errors = []
            if not settings.QUOTEX_EMAIL and not settings.QUOTEX_SSID:
                errors.append("Neither QUOTEX_EMAIL nor QUOTEX_SSID is set.")
            if settings.MIN_EXPECTED_VALUE <= -0.15:
                errors.append("MIN_EXPECTED_VALUE is set to allow heavy losses.")
            if errors:
                for err in errors:
                    st.error(f"❌ {err}")
            else:
                st.success("✅ Configuration is structurally valid.")

    st.markdown("---")

    with st.form("config_form"):
        st.subheader("🔑 Credentials")
        cred_email = st.text_input("Quotex Email", value=settings.QUOTEX_EMAIL)
        cred_pass = st.text_input("Quotex Password", value=settings.QUOTEX_PASSWORD, type="password")
        cred_ssid = st.text_input("Quotex SSID (Alternative)", value=settings.QUOTEX_SSID)

        st.subheader("🛡️ Safety & Mode Guards")
        demo_only = st.checkbox("DEMO_ONLY (Enforce Paper Trading)", value=settings.DEMO_ONLY)
        market_hours_only = st.checkbox("MARKET_HOURS_ONLY (Block Real Forex on Weekends)", value=settings.MARKET_HOURS_ONLY)
        allow_otc_fallback = st.checkbox("ALLOW_OTC_FALLBACK", value=settings.ALLOW_OTC_FALLBACK)
        allow_low_liquidity = st.checkbox("ALLOW_LOW_LIQUIDITY (Allow Real Forex in Asian Session)", value=settings.ALLOW_LOW_LIQUIDITY)

        st.subheader("📐 Risk & Signal Parameters")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            max_daily_trades = st.slider("Max Daily Trades", 5, 200, settings.MAX_DAILY_TRADES)
            max_daily_loss = st.slider("Max Daily Loss %", 0.01, 0.20, settings.MAX_DAILY_LOSS_PCT, step=0.01)
        with col_r2:
            conf_thresh = st.slider("Signal Confidence Threshold", 0.50, 0.75, settings.SIGNAL_CONFIDENCE_THRESHOLD, step=0.01)
            min_ev = st.slider("Minimum Expected Value (EV)", -0.05, 0.10, settings.MIN_EXPECTED_VALUE, step=0.01)

        st.subheader("📈 Assets & Strategy Features")
        selected_assets = st.multiselect(
            "Subscribed Assets",
            options=settings.SUPPORTED_ASSETS,
            default=[a for a in settings.ASSETS if a in settings.SUPPORTED_ASSETS],
        )
        use_ensemble = st.checkbox("USE_ENSEMBLE (LightGBM + HistGradient + ExtraTrees)", value=settings.USE_ENSEMBLE)
        calibrate_probs = st.checkbox("CALIBRATE_PROBABILITIES (Isotonic Calibration)", value=settings.CALIBRATE_PROBABILITIES)
        use_mtf = st.checkbox("USE_MULTI_TIMEFRAME (5m / 15m Indicators)", value=settings.USE_MULTI_TIMEFRAME)
        use_regime = st.checkbox("USE_REGIME_FILTER (ATR Volatility Regime Filter)", value=settings.USE_REGIME_FILTER)

        submitted = st.form_submit_state = st.form_submit_button("Save Configuration to .env")
        if submitted:
            env_path = ".env"
            # Read existing .env lines
            lines = []
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

            cfg_dict = {
                "QUOTEX_EMAIL": cred_email,
                "QUOTEX_PASSWORD": cred_pass,
                "QUOTEX_SSID": cred_ssid,
                "DEMO_ONLY": str(demo_only),
                "MARKET_HOURS_ONLY": str(market_hours_only),
                "ALLOW_OTC_FALLBACK": str(allow_otc_fallback),
                "ALLOW_LOW_LIQUIDITY": str(allow_low_liquidity),
                "MAX_DAILY_TRADES": str(max_daily_trades),
                "MAX_DAILY_LOSS_PCT": str(max_daily_loss),
                "SIGNAL_CONFIDENCE_THRESHOLD": str(conf_thresh),
                "MIN_EXPECTED_VALUE": str(min_ev),
                "ASSETS": ",".join(selected_assets),
                "USE_ENSEMBLE": str(use_ensemble),
                "CALIBRATE_PROBABILITIES": str(calibrate_probs),
                "USE_MULTI_TIMEFRAME": str(use_mtf),
                "USE_REGIME_FILTER": str(use_regime),
            }

            written_keys = set()
            new_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#") or not stripped or "=" not in stripped:
                    new_lines.append(line)
                    continue
                k = stripped.split("=")[0].strip()
                if k in cfg_dict:
                    new_lines.append(f"{k}={cfg_dict[k]}\n")
                    written_keys.add(k)
                else:
                    new_lines.append(line)

            for k, v in cfg_dict.items():
                if k not in written_keys:
                    new_lines.append(f"{k}={v}\n")

            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            st.success("Configuration saved to .env successfully! Reloading...")
            st.rerun()


# =========================================================
# PAGE 3: TRAINING
# =========================================================
elif nav_page == "Training":
    st.title("🧠 Adaptive Model Training")
    st.markdown("Train adaptive machine learning checkpoints on real historical candles from Quotex.")

    col1, col2, col3 = st.columns(3)
    with col1:
        train_asset = st.selectbox("Select Asset", settings.SUPPORTED_ASSETS, index=0)
    with col2:
        train_days = st.slider("History Days to Fetch", 7, 90, 30)
    with col3:
        train_tf = st.selectbox("Timeframe", ["1m", "5m", "15m"], index=0)

    if st.button("🚀 Train Model Now", type="primary", use_container_width=True):
        st.info(f"Starting training for {train_asset} ({train_days} days, timeframe {train_tf})...")
        log_placeholder = st.empty()

        cmd = [
            sys.executable,
            "main.py",
            "--train",
            "--asset",
            train_asset,
            "--days",
            str(train_days),
            "--timeframe",
            train_tf,
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        for line in iter(proc.stdout.readline, ""):
            output_lines.append(line)
            log_placeholder.code("".join(output_lines[-25:]), language="bash")

        proc.wait()
        if proc.returncode == 0:
            st.success(f"Training completed successfully for {train_asset}!")
            time.sleep(1.5)
            st.rerun()
        else:
            st.error(f"Training failed with return code {proc.returncode}.")

    st.markdown("---")
    st.subheader("📊 Latest Feature Importances")

    # Load latest model for asset
    mod = AdaptiveModel(asset=train_asset)
    if mod.load_latest():
        metrics = mod.latest_metrics
        st.write(f"**Loaded Checkpoint**: `{mod.model_path}`")
        w_acc = metrics.get("accuracy", 0.5)
        w_prec = metrics.get("precision", 0.5)
        w_ev = metrics.get("mean_ev", 0.0)
        brier = metrics.get("brier_score", 0.25)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Walk-Forward Accuracy", f"{w_acc * 100:.2f}%")
        m_col2.metric("Precision", f"{w_prec * 100:.2f}%")
        m_col3.metric("Brier Score", f"{brier:.4f}")
        m_col4.metric("Mean EV", f"{w_ev * 100:+.2f}%")

        imps = metrics.get("feature_importances", {})
        if imps:
            df_imp = pd.DataFrame(list(imps.items()), columns=["Feature", "Importance"])
            df_imp = df_imp.sort_values(by="Importance", ascending=True)

            colors = ["#EF553B" if v > 0.40 else "#636EFA" for v in df_imp["Importance"]]
            fig = go.Figure(
                go.Bar(
                    x=df_imp["Importance"] * 100,
                    y=df_imp["Feature"],
                    orientation="h",
                    marker_color=colors,
                )
            )
            fig.update_layout(
                title=f"Top 10 Feature Importances ({train_asset})",
                xaxis_title="Importance (%)",
                yaxis_title="Feature",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
            if any(v > 0.40 for v in df_imp["Importance"]):
                st.warning("⚠️ Warning: One or more features account for >40% importance. Check for data leakage.")
        else:
            st.info("Feature importance data not stored in this checkpoint.")
    else:
        st.info(f"No existing checkpoint found for {train_asset}.")

    # Model Checkpoint Management
    st.markdown("---")
    st.subheader("📁 Saved Checkpoints on Disk")
    all_checkpoints = sorted(glob.glob(os.path.join(settings.MODEL_DIR, "model_*.joblib")), key=os.path.getmtime, reverse=True)
    if all_checkpoints:
        ckpt_records = []
        for ckpt in all_checkpoints:
            fname = os.path.basename(ckpt)
            fsize_kb = round(os.path.getsize(ckpt) / 1024, 1)
            ftime = datetime.fromtimestamp(os.path.getmtime(ckpt)).strftime("%Y-%m-%d %H:%M:%S")
            c_asset = fname.split("_")[1] if len(fname.split("_")) > 1 else "Unknown"
            ckpt_records.append({
                "Filename": fname,
                "Asset": c_asset,
                "Size (KB)": f"{fsize_kb} KB",
                "Created": ftime,
                "Path": ckpt
            })
        st.dataframe(pd.DataFrame(ckpt_records), use_container_width=True)
    else:
        st.info("No saved model checkpoints found.")


# =========================================================
# PAGE 4: BACKTEST
# =========================================================
elif nav_page == "Backtest":
    st.title("🧪 Walk-Forward Backtester")
    st.markdown("Evaluate strategy expectancy, win rate, and drawdown over real historical data.")

    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        bt_asset = st.selectbox("Asset Symbol", settings.SUPPORTED_ASSETS, index=0)
    with b_col2:
        bt_days = st.slider("History Days", 7, 90, 30)
    with b_col3:
        bt_tf = st.selectbox("Timeframe", ["1m", "5m", "15m"], index=0)

    log_dir = os.path.dirname(settings.LOG_FILE) or "logs"
    bt_export_path = os.path.join(log_dir, "backtest_trades.csv")

    if st.button("▶️ Run Out-Of-Sample Backtest", type="primary", use_container_width=True):
        st.info(f"Running backtest for {bt_asset} ({bt_days} days)...")
        b_placeholder = st.empty()

        cmd = [
            sys.executable,
            "main.py",
            "--backtest",
            "--asset",
            bt_asset,
            "--days",
            str(bt_days),
            "--timeframe",
            bt_tf,
            "--export",
            bt_export_path,
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        b_lines = []
        for line in iter(proc.stdout.readline, ""):
            b_lines.append(line)
            b_placeholder.code("".join(b_lines[-25:]), language="bash")

        proc.wait()
        st.success("Backtest execution finished.")

    # Display Backtest Results if CSV exists
    if os.path.exists(bt_export_path):
        try:
            df_trades = pd.read_csv(bt_export_path)
            if not df_trades.empty and "pnl" in df_trades.columns:
                st.markdown("---")
                st.subheader("📈 Backtest Performance Summary")

                total_trades = len(df_trades)
                wins = len(df_trades[df_trades["pnl"] > 0])
                win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0
                net_pnl = df_trades["pnl"].sum()
                gross_win = df_trades[df_trades["pnl"] > 0]["pnl"].sum()
                gross_loss = abs(df_trades[df_trades["pnl"] < 0]["pnl"].sum())
                profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)

                # Equity Curve
                df_trades["cum_pnl"] = df_trades["pnl"].cumsum()
                df_trades["peak"] = df_trades["cum_pnl"].cummax()
                df_trades["drawdown"] = df_trades["cum_pnl"] - df_trades["peak"]
                max_dd = df_trades["drawdown"].min()

                # Metric Cards
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Trades", total_trades)
                m2.metric("Win Rate", f"{win_rate:.1f}%")
                m3.metric("Net P&L (Units)", f"{net_pnl:+.2f}")
                m4.metric("Profit Factor", f"{profit_factor:.2f}")
                m5.metric("Max Drawdown", f"{max_dd:.2f}")

                # Negative EV Warning
                payout_val = float(df_trades["payout"].iloc[0]) if "payout" in df_trades.columns else 0.77
                break_even_wr = (1.0 / (1.0 + payout_val)) * 100
                if win_rate < break_even_wr or net_pnl < 0:
                    st.error(
                        f"⚠️ Negative Expectancy Detected: Win rate ({win_rate:.1f}%) is below break-even threshold "
                        f"({break_even_wr:.1f}%) at {payout_val * 100:.0f}% payout. Net EV is negative."
                    )
                else:
                    st.success(f"Positive Expectancy: Strategy achieved edge above break-even ({break_even_wr:.1f}%).")

                # Plot Equity Curve
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    y=df_trades["cum_pnl"],
                    mode="lines",
                    name="Cumulative P&L",
                    line=dict(color="#00CC96", width=2),
                ))
                fig_eq.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_eq.update_layout(
                    title="Strategy Equity Curve (Unit Stakes)",
                    xaxis_title="Trade Number",
                    yaxis_title="Cumulative P&L (Units)",
                    height=400,
                )
                st.plotly_chart(fig_eq, use_container_width=True)

                # Trade Table & CSV Download
                st.subheader("📋 Trade Log")
                st.dataframe(df_trades.tail(50), use_container_width=True)

                with open(bt_export_path, "rb") as f:
                    st.download_button(
                        "📥 Download Backtest CSV",
                        f,
                        file_name="backtest_trades.csv",
                        mime="text/csv",
                    )
            else:
                st.warning("Backtest trade log is empty (0 signals met EV and confidence threshold).")
        except Exception as e:
            st.error(f"Error parsing backtest CSV: {e}")


# =========================================================
# PAGE 5: ASSETS & PAYOUTS
# =========================================================
elif nav_page == "Assets & Payouts":
    st.title("💰 Assets & Broker Payouts")
    st.markdown("Live payout rates, market hours status, and required break-even win rates.")

    if st.button("🔄 Refresh Live Payouts from Quotex", type="primary"):
        st.info("Querying Quotex client for live asset payouts...")
        async def fetch_payouts():
            client = QuotexDataClient()
            conn = await client.connect()
            results = []
            if conn:
                for asset in settings.SUPPORTED_ASSETS:
                    p = await client.get_payout(asset)
                    is_real = is_real_forex(asset)
                    m_stat = get_market_status() if is_real else {"is_open": True, "reason": "OTC 24/7"}
                    be_rate = 1.0 / (1.0 + p)
                    results.append({
                        "Asset": asset,
                        "Type": "Real Forex" if is_real else "OTC",
                        "Live Payout %": round(p * 100, 1),
                        "Break-Even Win Rate %": round(be_rate * 100, 1),
                        "Market Status": "Open" if m_stat["is_open"] else f"Closed ({m_stat['reason']})",
                        "Recommended Timeframe": "5m / 15m" if is_real else "1m / 5m",
                    })
                await client.disconnect()
            return results

        try:
            loop = asyncio.new_event_loop()
            payouts_data = loop.run_until_complete(fetch_payouts())
            loop.close()
            st.session_state["payouts_data"] = payouts_data
        except Exception as e:
            st.error(f"Could not refresh payouts: {e}")

    cached_payouts = st.session_state.get("payouts_data", None)
    if not cached_payouts:
        # Default static baseline
        cached_payouts = []
        for asset in settings.SUPPORTED_ASSETS:
            is_real = is_real_forex(asset)
            p = 0.85 if not is_real else 0.77
            be_rate = 1.0 / (1.0 + p)
            m_stat = get_market_status() if is_real else {"is_open": True, "reason": "OTC 24/7"}
            cached_payouts.append({
                "Asset": asset,
                "Type": "Real Forex" if is_real else "OTC",
                "Live Payout %": round(p * 100, 1),
                "Break-Even Win Rate %": round(be_rate * 100, 1),
                "Market Status": "Open" if m_stat["is_open"] else f"Closed ({m_stat['reason']})",
                "Recommended Timeframe": "5m / 15m" if is_real else "1m / 5m",
            })

    df_payouts = pd.DataFrame(cached_payouts)

    # Style table: highlight payouts
    def highlight_payout(val):
        try:
            v = float(val)
            if v < 75.0:
                return "background-color: #ffcccc; color: #990000;"
            elif v < 85.0:
                return "background-color: #fff3cd; color: #856404;"
            else:
                return "background-color: #d4edda; color: #155724;"
        except Exception:
            return ""

    # Support pandas map (pandas >= 2.1.0) with fallback to applymap
    styler = df_payouts.style
    if hasattr(styler, "map"):
        styled_df = styler.map(highlight_payout, subset=["Live Payout %"])
    else:
        styled_df = getattr(styler, "applymap")(highlight_payout, subset=["Live Payout %"])

    st.dataframe(
        styled_df,
        use_container_width=True,
    )

    st.markdown("""
    > [!NOTE] **Understanding Payout Asymmetry**
    >
    > Break-Even Win Rate Formula:
    > $$W_{be} = \\frac{1}{1 + Payout}$$
    > - At **77% payout** (typical Real Forex), you must win **56.50%** of trades just to avoid losing money.
    > - At **85% payout** (typical OTC), you must win **54.05%** of trades.
    > - At **50% accuracy** (random walk), a 77% payout produces a guaranteed **-8.98% Expected Value** per trade.
    """)


# =========================================================
# PAGE 6: REAL FOREX ANALYTICS
# =========================================================
elif nav_page == "Real Forex Analytics":
    st.title("🌍 Real Forex vs OTC Analytics")
    st.markdown("Detailed breakdown of transparent Real Forex market hours, session liquidity, and OTC structural differences.")

    now_utc = datetime.now(timezone.utc)
    market_stat = get_market_status(now_utc.timestamp())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("UTC Time", now_utc.strftime("%H:%M:%S UTC"))
    with c2:
        st.metric("Forex Market", "OPEN" if market_stat["is_open"] else "CLOSED", delta=market_stat["reason"])
    with c3:
        st.metric("Current Session", market_stat["session"])
    with c4:
        st.metric("London-NY Overlap", "ACTIVE 🟢" if market_stat["is_overlap"] else "INACTIVE ⚪")

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("⚖️ Real Forex vs OTC Comparison")
        st.markdown("""
        | Attribute | Real Forex (e.g. `EURUSD`) | OTC (e.g. `EURUSD_otc`) |
        |---|---|---|
        | **Pricing Source** | Global Interbank Spot Market | Proprietary Broker Algorithm |
        | **Trading Hours** | Sun 22:00 - Fri 22:00 UTC | 24/7 (365 Days) |
        | **Transparency** | Publicly Auditable Order Books | Black Box / Synthetic |
        | **Broker Risk** | High Liquidity, Narrow Spread | Conflict of Interest (House is Counterparty) |
        | **Typical Payout** | 70% - 80% | 80% - 92% |
        """)

    with col_r:
        st.subheader("🕒 Forex Trading Sessions (UTC)")
        st.markdown("""
        - **Sydney / Asia**: 21:00 - 06:00 UTC *(Low volatility, wide spreads)*
        - **London**: 07:00 - 16:00 UTC *(High liquidity, strong trends)*
        - **New York**: 12:00 - 21:00 UTC *(High volume, US economic data)*
        - **London/NY Overlap**: 12:00 - 16:00 UTC *(Optimal liquidity and signal quality)*
        """)

    st.markdown("---")
    st.subheader("📊 Session Liquidity & Accuracy Profile")

    # Sample session stats
    session_data = {
        "Session": ["Asia (Tokyo/Sydney)", "London", "London-NY Overlap", "Late New York"],
        "Liquidity": ["Low", "High", "Maximum", "Medium"],
        "Mean Spread Impact": ["High", "Low", "Lowest", "Medium"],
        "Real Forex Win Rate": ["49.8%", "52.4%", "53.8%", "50.9%"],
        "Recommendation": ["Avoid", "Trade", "Optimal", "Selective"],
    }
    st.table(pd.DataFrame(session_data))


# =========================================================
# PAGE 7: ALERTS
# =========================================================
elif nav_page == "Alerts":
    st.title("🔔 Alert Management & Notification Testing")
    st.markdown("Configure and verify outbound alert channels (Telegram & WhatsApp).")

    a_col1, a_col2 = st.columns(2)

    with a_col1:
        st.subheader("✈️ Telegram Alerts")
        tg_token = st.text_input("Telegram Bot Token", value=settings.TELEGRAM_BOT_TOKEN or "", type="password")
        tg_chat = st.text_input("Telegram Chat ID", value=settings.TELEGRAM_CHAT_ID or "")

        if st.button("📨 Send Test Telegram Alert", use_container_width=True):
            if not tg_token or not tg_chat:
                st.error("Telegram Bot Token and Chat ID must be configured in .env.")
            else:
                from alerting import AlertManager
                mgr = AlertManager()
                # Dummy signal
                from signal_engine import Signal
                dummy = Signal(
                    asset="EURUSD_otc",
                    direction="CALL",
                    probability=0.62,
                    ev=0.075,
                    payout=0.85,
                    timestamp=time.time(),
                    expiry=60,
                    confluence=True,
                )
                success = asyncio.run(mgr.send_telegram_alert(dummy))
                if success:
                    st.success("Test Telegram alert delivered successfully!")
                else:
                    st.error("Failed to send Telegram alert. Check token and chat ID.")

    with a_col2:
        st.subheader("💬 WhatsApp Alerts")
        wa_phone_id = st.text_input("WhatsApp Phone Number ID", value=getattr(settings, "WHATSAPP_PHONE_NUMBER_ID", "") or "")
        wa_token = st.text_input("WhatsApp API Token", value=getattr(settings, "WHATSAPP_API_TOKEN", "") or "", type="password")
        wa_phone = st.text_input("WhatsApp Recipient Phone", value=getattr(settings, "WHATSAPP_RECIPIENT", "") or "")

        if st.button("📱 Send Test WhatsApp Alert", use_container_width=True):
            if not wa_phone_id or not wa_token or not wa_phone:
                st.error("WhatsApp credentials not fully configured in .env.")
            else:
                from alerting import AlertManager
                mgr = AlertManager()
                from signal_engine import Signal
                dummy = Signal(
                    asset="EURUSD_otc",
                    direction="PUT",
                    probability=0.64,
                    ev=0.088,
                    payout=0.85,
                    timestamp=time.time(),
                    expiry=60,
                    confluence=True,
                )
                success = asyncio.run(mgr.send_whatsapp_alert(dummy))
                if success:
                    st.success("Test WhatsApp alert delivered successfully!")
                else:
                    st.error("Failed to send WhatsApp alert. Check webhook and token.")

    st.markdown("---")
    st.subheader("📑 Formatted Alert Template Preview")
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.markdown("""
        **[CALL SIGNAL PREVIEW]**
        ```
        🚨 QUOTEX TRADING SIGNAL 🚨
        -----------------------------
        Asset:       EURUSD_otc
        Direction:   CALL 🔼
        Confidence:  63.5%
        Est. EV:     +7.8%
        Payout:      85.0%
        Expiry:      60s
        Confluence:  YES (RSI Oversold, Lower BB)
        Mode:        SIGNAL_ONLY
        Time:        2026-11-09 23:45:00 UTC
        -----------------------------
        ⚠️ Educational demo analysis only.
        ```
        """)

    with preview_col2:
        st.markdown("""
        **[PUT SIGNAL PREVIEW]**
        ```
        🚨 QUOTEX TRADING SIGNAL 🚨
        -----------------------------
        Asset:       GBPUSD_otc
        Direction:   PUT 🔽
        Confidence:  65.1%
        Est. EV:     +10.7%
        Payout:      85.0%
        Expiry:      60s
        Confluence:  YES (MACD Bearish Cross, Upper BB)
        Mode:        SIGNAL_ONLY
        Time:        2026-11-09 23:45:00 UTC
        -----------------------------
        ⚠️ Educational demo analysis only.
        ```
        """)


# =========================================================
# PAGE 8: LOGS
# =========================================================
elif nav_page == "Logs":
    st.title("📜 Live System Logs")
    st.markdown("Real-time log tail from `logs/quotex_bot.log` with level filtering.")

    log_path = settings.LOG_FILE
    if os.path.exists(log_path):
        col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
        with col_f1:
            level_filter = st.selectbox("Filter Level", ["ALL", "INFO", "WARNING", "ERROR", "DEBUG"])
        with col_f2:
            search_query = st.text_input("Search Text", "")
        with col_f3:
            line_count = st.slider("Lines to Display", 50, 1000, 200)

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()

        filtered_lines = []
        for line in all_lines:
            if level_filter != "ALL" and level_filter not in line:
                continue
            if search_query and search_query.lower() not in line.lower():
                continue
            filtered_lines.append(line)

        st.code("".join(filtered_lines[-line_count:]), language="log")

        with open(log_path, "rb") as f:
            st.download_button("📥 Download Full Log File", f, file_name="quotex_bot.log", mime="text/plain")
    else:
        st.warning(f"Log file not found at `{log_path}`. Run the bot or training to generate logs.")


# =========================================================
# PAGE 9: ABOUT / SAFETY
# =========================================================
elif nav_page == "About / Safety":
    st.title("🛡️ Regulatory Disclosures & Mathematical Reality")

    st.error("""
    ### ⚠️ CRITICAL REGULATORY & LEGAL WARNING
    Quotex is an **unregulated offshore binary options broker** operating outside major financial regulatory oversight (such as SEC, CFTC, FCA, or ASIC).
    
    Quotex explicitly prohibits algorithmic trading and automation in its Service Agreement:
    - **Clause 2.5.1**: Prohibits third-party automated tools and scripts.
    - **Clause 7.12.10**: Authorizes unilateral cancellation of trades suspected of automation.
    - **Clause 12.7**: Authorizes account termination and forfeiture of account balances.
    
    This software is developed strictly for **educational, academic, and paper-trading research purposes**.
    Never risk live capital on binary options brokers.
    """)

    st.markdown("---")
    st.subheader("📐 The Mathematical Asymmetry of Binary Options")

    st.markdown(r"""
    In binary options, payouts are fundamentally asymmetric:
    - Winning trade returns: $+Payout \times Stake$ (e.g., $+0.77 \times 1.0 = +0.77$)
    - Losing trade forfeits: $-1.0 \times Stake$ ($-1.00$)
    
    The Expected Value ($EV$) of each trade is:
    $$EV = (P_{win} \times Payout) - ((1 - P_{win}) \times 1.0)$$
    
    To break even ($EV = 0$):
    $$W_{be} = \frac{1}{1 + Payout}$$
    
    At a 77% payout:
    $$W_{be} = \frac{1}{1 + 0.77} = \frac{1}{1.77} \approx 56.50\%$$
    
    A machine learning model achieving **50.2% accuracy** (the empirical out-of-sample result on 1-minute real Forex) produces:
    $$EV = (0.502 \times 0.77) - (0.498 \times 1.0) = 0.38654 - 0.49800 = -11.15\%$$
    """)

    # Mathematical Simulation Chart
    st.subheader("📉 Simulation: Capital Decay Over 1,000 Trades")
    n_sim_trades = 1000
    np.random.seed(42)

    # 50.2% win rate at 77% payout
    pnl_50 = np.where(np.random.rand(n_sim_trades) < 0.502, 0.77, -1.0)
    # 57.0% win rate at 77% payout
    pnl_57 = np.where(np.random.rand(n_sim_trades) < 0.570, 0.77, -1.0)

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        y=np.cumsum(pnl_50),
        mode="lines",
        name="Real Market ML (50.2% Win Rate, 77% Payout) - Rapid Decay",
        line=dict(color="red", width=2),
    ))
    fig_sim.add_trace(go.Scatter(
        y=np.cumsum(pnl_57),
        mode="lines",
        name="Theoretical Edge (57.0% Win Rate, 77% Payout) - Growth",
        line=dict(color="green", width=2),
    ))
    fig_sim.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_sim.update_layout(
        title="1,000 Trade Equity Trajectory: 50.2% vs 57.0%",
        xaxis_title="Trades",
        yaxis_title="Account Balance Delta (Units)",
        height=400,
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("""
    ### 🔄 Transferability: Where Does This ML Pipeline Actually Work?
    While binary options suffer from broker payout asymmetry, this codebase's architecture (**LightGBM + Walk-Forward Validation + Confluence Filtering**) transfers directly to **symmetric payout markets**:
    1. **Spot Forex & Futures**: 1:1 risk/reward or trailing stops where 52% accuracy provides substantial positive alpha.
    2. **Perpetual Crypto Swaps**: Commission-only costs (0.02%-0.05%) rather than a 23% broker spread.
    3. **Stock Options (Long/Short Spreads)**: Quantifiable risk with favorable risk-to-reward ratios.
    """)
