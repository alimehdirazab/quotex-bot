"""Tests for Real Forex Classification, Market Hours, and Real Data Enforcement."""

from datetime import datetime, timezone
import pytest

from config import settings
from data_layer import is_real_forex, get_market_status, QuotexDataClient


def test_is_real_forex():
    """Verify distinction between real forex pairs and OTC synthetic instruments."""
    assert is_real_forex("EURUSD") is True
    assert is_real_forex("GBPUSD") is True
    assert is_real_forex("USDJPY") is True
    assert is_real_forex("EURUSD_otc") is False
    assert is_real_forex("GBPUSD_otc") is False
    assert is_real_forex("AUDCAD_otc") is False


def test_market_status_schedule():
    """Verify forex market schedule rules across days of week and trading sessions."""
    # Wednesday 14:00 UTC (Market should be Open, London-NY Overlap active)
    dt_wed_overlap = datetime(2026, 11, 4, 14, 0, tzinfo=timezone.utc).timestamp()
    stat = get_market_status(dt_wed_overlap)
    assert stat["is_open"] is True
    assert stat["is_overlap"] is True
    assert stat["session"] == "London-NY Overlap"

    # Saturday 15:00 UTC (Market should be Closed)
    dt_sat = datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc).timestamp()
    stat_sat = get_market_status(dt_sat)
    assert stat_sat["is_open"] is False
    assert "Weekend" in stat_sat["reason"]

    # Sunday 12:00 UTC (Market should be Closed until 22:00 UTC)
    dt_sun_noon = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc).timestamp()
    stat_sun = get_market_status(dt_sun_noon)
    assert stat_sun["is_open"] is False

    # Sunday 23:00 UTC (Market should be Open, Asia session)
    dt_sun_night = datetime(2026, 11, 8, 23, 0, tzinfo=timezone.utc).timestamp()
    stat_sun_open = get_market_status(dt_sun_night)
    assert stat_sun_open["is_open"] is True
    assert stat_sun_open["session"] == "Asia"


def test_synthetic_data_strictly_disabled():
    """Confirm global settings enforce real market data and prohibit synthetic fallback."""
    assert settings.USE_SYNTHETIC is False, "USE_SYNTHETIC must be False for honest validation!"

