"""Unit tests for Expected Value (EV) and Break-even mathematical formulations."""

import pytest
from model_pipeline import AdaptiveModel


def test_ev_calculation():
    """Verify Expected Value math for CALL and PUT options."""
    model = AdaptiveModel(asset="EURUSD_otc")

    # Case 1: 50% probability at 77% payout (0.77)
    # EV_call = 0.5 * 0.77 - 0.5 * 1.0 = 0.385 - 0.5 = -0.115 (-11.5%)
    direction, conf, ev = model.calculate_expected_value(proba_up=0.5, payout=0.77)
    assert round(ev, 4) == -0.1150
    assert conf == 0.5

    # Case 2: Break-even win rate at 77% payout: W_be = 1 / (1 + 0.77) = 0.56497
    be_wr = 1.0 / (1.0 + 0.77)
    direction, conf, ev = model.calculate_expected_value(proba_up=be_wr, payout=0.77)
    assert abs(ev) < 1e-4

    # Case 3: 65% bullish probability at 85% payout
    # EV_call = 0.65 * 0.85 - 0.35 * 1.0 = 0.5525 - 0.35 = +0.2025
    direction, conf, ev = model.calculate_expected_value(proba_up=0.65, payout=0.85)
    assert direction == "CALL"
    assert round(conf, 2) == 0.65
    assert round(ev, 4) == 0.2025

    # Case 4: 25% bullish probability (75% bearish) at 85% payout
    # EV_put = 0.75 * 0.85 - 0.25 * 1.0 = 0.6375 - 0.25 = +0.3875
    direction, conf, ev = model.calculate_expected_value(proba_up=0.25, payout=0.85)
    assert direction == "PUT"
    assert round(conf, 2) == 0.75
    assert round(ev, 4) == 0.3875

