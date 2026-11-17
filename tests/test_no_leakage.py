"""Tests for Feature Engineering Integrity and Zero Lookahead Bias / Leakage."""

import numpy as np
import pandas as pd
import pytest

from feature_engine import build_features, get_feature_columns


def test_feature_columns_contain_no_raw_targets_or_prices():
    """Verify raw unshifted price columns and targets are not in ML feature list."""
    feature_cols = get_feature_columns()
    banned = {"target", "close", "high", "low", "open", "time", "timestamp"}
    for col in feature_cols:
        assert col.lower() not in banned, f"Banned column '{col}' found in feature list!"


def test_no_future_leakage_in_features():
    """Verify that features at time t depend strictly on data up to time t, not future candles."""
    np.random.seed(42)
    n = 200
    prices = 1.1000 + np.cumsum(np.random.randn(n) * 0.0005)
    df = pd.DataFrame({
        "time": np.arange(1700000000, 1700000000 + n * 60, 60),
        "open": prices + np.random.randn(n) * 0.0001,
        "high": prices + 0.0005,
        "low": prices - 0.0005,
        "close": prices,
        "volume": np.random.randint(100, 1000, n),
    })

    feat1 = build_features(df, drop_na=False)

    # Now modify candle at index 150 (simulate future change)
    df_modified = df.copy()
    df_modified.loc[150, "close"] = 2.5000
    df_modified.loc[150, "high"] = 2.5100
    df_modified.loc[150, "low"] = 2.4900

    feat2 = build_features(df_modified, drop_na=False)

    feature_cols = get_feature_columns()

    # Features at index 140 (prior to index 150) MUST be strictly identical
    for col in feature_cols:
        val1 = feat1.loc[140, col]
        val2 = feat2.loc[140, col]
        if np.isnan(val1) and np.isnan(val2):
            continue
        assert np.isclose(val1, val2, rtol=1e-5, atol=1e-5), (
            f"Feature '{col}' changed at index 140 when future index 150 was altered! Lookahead leakage detected."
        )


def test_no_unrealistic_correlation_with_target():
    """Ensure no feature has an unnaturally high correlation (|r| > 0.40) with binary target."""
    np.random.seed(42)
    n = 1000
    prices = 1.1000 + np.cumsum(np.random.randn(n) * 0.0005)
    df = pd.DataFrame({
        "time": np.arange(1700000000, 1700000000 + n * 60, 60),
        "open": prices + np.random.randn(n) * 0.0001,
        "high": prices + 0.0005,
        "low": prices - 0.0005,
        "close": prices,
        "volume": np.random.randint(100, 1000, n),
    })

    feat = build_features(df, drop_na=True)
    target = feat["target"].values
    feature_cols = get_feature_columns()

    for col in feature_cols:
        arr = feat[col].values
        # Skip constant / zero-variance columns where correlation is undefined (nan)
        if np.std(arr) == 0 or np.isnan(arr).any():
            continue
        r = np.corrcoef(arr, target)[0, 1]
        if not np.isnan(r):
            assert abs(r) < 0.40, f"Suspiciously high correlation (|r|={abs(r):.3f}) between feature '{col}' and target!"
