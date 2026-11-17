"""Feature Engineering Engine for Quotex Candlestick Data.

Computes technical indicators, price action dynamics, rolling volatility,
cyclical time encodings, multi-timeframe aggregations (5m/15m), volatility regimes,
session-aware flags (London/NY/Overlap/Asia), and triple confluence flags.
"""

from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

# Try importing pandas_ta, but provide native fallbacks
try:
    import pandas_ta as ta  # type: ignore
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


def _calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1.0 * delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD Line, Signal Line, and Histogram."""
    ema_fast = _calc_ema(series, fast)
    ema_slow = _calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calc_bollinger_bands(series: pd.Series, length: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands: middle, upper, lower, and %B."""
    middle = series.rolling(window=length).mean()
    sigma = series.rolling(window=length).std()
    upper = middle + (std_dev * sigma)
    lower = middle - (std_dev * sigma)
    percent_b = (series - lower) / ((upper - lower) + 1e-9)
    bandwidth = (upper - lower) / (middle + 1e-9)
    return upper, middle, lower, percent_b, bandwidth


def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    return atr


def build_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """Transform raw OHLCV candle dataframe into a rich feature set for ML and rules.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns: ['time', 'open', 'high', 'low', 'close', 'volume']
    drop_na : bool
        Whether to drop warm-up NaN rows.

    Returns:
    --------
    pd.DataFrame
        DataFrame populated with technical indicators, multi-timeframe aggregations,
        volatility regimes, session indicators, and confluence flags.
    """
    if df.empty or len(df) < 30:
        logger.warning("Dataframe has insufficient candles to calculate features (minimum 30 required).")
        return pd.DataFrame()

    feat = df.copy()

    # Ensure required column names are lowercase
    feat.columns = [col.lower() for col in feat.columns]
    req_cols = ['open', 'high', 'low', 'close']
    for c in req_cols:
        if c not in feat.columns:
            raise ValueError(f"Missing required price column: {c}")

    if 'volume' not in feat.columns:
        feat['volume'] = 0.0

    # Ensure datetime index or time column
    if 'time' in feat.columns:
        if not pd.api.types.is_datetime64_any_dtype(feat['time']):
            feat['datetime'] = pd.to_datetime(feat['time'], unit='s', errors='coerce')
        else:
            feat['datetime'] = feat['time']
    else:
        feat['datetime'] = pd.date_range(end=pd.Timestamp.now(), periods=len(feat), freq='1min')

    # 1. Base Technical Indicators (1m)
    feat['rsi_14'] = _calc_rsi(feat['close'], 14)
    feat['macd'], feat['macd_signal'], feat['macd_hist'] = _calc_macd(feat['close'])
    (feat['bb_upper'], feat['bb_middle'], feat['bb_lower'],
     feat['bb_percent'], feat['bb_bandwidth']) = _calc_bollinger_bands(feat['close'])
    feat['ema_9'] = _calc_ema(feat['close'], 9)
    feat['ema_21'] = _calc_ema(feat['close'], 21)
    feat['atr_14'] = _calc_atr(feat['high'], feat['low'], feat['close'], 14)

    # EMA Spread
    feat['ema_spread'] = (feat['ema_9'] - feat['ema_21']) / (feat['close'] + 1e-9)

    # 2. Candle Price Action Morphology
    candle_range = feat['high'] - feat['low'] + 1e-9
    body_size = (feat['close'] - feat['open']).abs()
    feat['body_ratio'] = body_size / candle_range

    upper_shadow = feat['high'] - np.maximum(feat['open'], feat['close'])
    feat['upper_wick_ratio'] = upper_shadow / candle_range

    lower_shadow = np.minimum(feat['open'], feat['close']) - feat['low']
    feat['lower_wick_ratio'] = lower_shadow / candle_range

    # Candle Direction (+1 bullish, -1 bearish, 0 doji)
    feat['candle_direction'] = np.sign(feat['close'] - feat['open'])

    # 3. Rolling Volatility (standard deviation of log returns over 14 bars)
    log_ret = np.log(feat['close'] / feat['close'].shift(1).clip(lower=1e-9))
    feat['rolling_volatility_14'] = log_ret.rolling(window=14).std()

    # 4. Multi-Timeframe Synthetic Aggregates (5m and 15m)
    # Higher timeframe momentum via multi-period lookbacks without lookahead
    feat['rsi_5m'] = _calc_rsi(feat['close'], 14 * 5)
    feat['macd_5m'], _, feat['macd_hist_5m'] = _calc_macd(feat['close'], fast=12 * 5, slow=26 * 5, signal=9 * 5)
    feat['ema_spread_15m'] = (_calc_ema(feat['close'], 9 * 15) - _calc_ema(feat['close'], 21 * 15)) / (feat['close'] + 1e-9)
    feat['atr_15m'] = _calc_atr(feat['high'], feat['low'], feat['close'], period=14 * 15) / (feat['close'] + 1e-9)

    # 5. Volatility Regime Detection (Rolling 200-bar percentile rank: 0=low, 1=mid, 2=high)
    atr_rolling_pct = feat['atr_14'].rolling(window=200, min_periods=30).apply(
        lambda s: float((s.iloc[-1] - s.min()) / (s.max() - s.min() + 1e-9)), raw=False
    )
    feat['volatility_regime'] = np.where(atr_rolling_pct < 0.33, 0, np.where(atr_rolling_pct > 0.66, 2, 1))

    # 6. Session-Aware Indicators for Real Forex (UTC)
    hours = feat['datetime'].dt.hour
    weekdays = feat['datetime'].dt.weekday
    feat['hour_of_day'] = hours
    feat['day_of_week'] = weekdays
    feat['session_london'] = ((hours >= 7) & (hours < 16)).astype(int)
    feat['session_ny'] = ((hours >= 12) & (hours < 21)).astype(int)
    feat['session_overlap'] = ((hours >= 12) & (hours < 16)).astype(int)  # Peak liquidity
    feat['session_asia'] = ((hours >= 23) | (hours < 7)).astype(int)
    feat['is_weekend'] = (weekdays >= 5).astype(int)

    # Cyclical Sine/Cosine for 24-hour day
    minutes_of_day = hours * 60 + feat['datetime'].dt.minute
    total_minutes = 24.0 * 60.0
    feat['sin_time'] = np.sin(2 * np.pi * minutes_of_day / total_minutes)
    feat['cos_time'] = np.cos(2 * np.pi * minutes_of_day / total_minutes)

    # 7. MACD Crossover Events
    prev_macd = feat['macd'].shift(1)
    prev_signal = feat['macd_signal'].shift(1)
    feat['macd_bullish_cross'] = ((prev_macd <= prev_signal) & (feat['macd'] > feat['macd_signal'])).astype(int)
    feat['macd_bearish_cross'] = ((prev_macd >= prev_signal) & (feat['macd'] < feat['macd_signal'])).astype(int)

    # 8. Triple Confluence Flags
    bullish_confluence = (
        (feat['rsi_14'] < 30) &
        (feat['macd_bullish_cross'] == 1) &
        (feat['close'] <= feat['bb_lower'] * 1.002)
    )
    bearish_confluence = (
        (feat['rsi_14'] > 70) &
        (feat['macd_bearish_cross'] == 1) &
        (feat['close'] >= feat['bb_upper'] * 0.998)
    )

    feat['bullish_confluence'] = bullish_confluence
    feat['bearish_confluence'] = bearish_confluence
    feat['triple_confluence'] = bullish_confluence | bearish_confluence

    # 9. Supervised Classification Target:
    # 1 if next candle close > current candle close, else 0
    feat['target'] = (feat['close'].shift(-1) > feat['close']).astype(int)

    if drop_na:
        feat = feat.dropna(subset=['rsi_14', 'bb_upper', 'atr_14', 'rolling_volatility_14']).copy()

    return feat


def get_feature_columns() -> list[str]:
    """Return the ordered list of predictive features fed into ML models."""
    return [
        'rsi_14',
        'macd',
        'macd_hist',
        'macd_signal',
        'bb_percent',
        'bb_bandwidth',
        'ema_spread',
        'atr_14',
        'body_ratio',
        'upper_wick_ratio',
        'lower_wick_ratio',
        'candle_direction',
        'rolling_volatility_14',
        'rsi_5m',
        'macd_5m',
        'ema_spread_15m',
        'atr_15m',
        'volatility_regime',
        'session_london',
        'session_ny',
        'session_overlap',
        'session_asia',
        'is_weekend',
        'hour_of_day',
        'day_of_week',
        'sin_time',
        'cos_time',
        'macd_bullish_cross',
        'macd_bearish_cross'
    ]
