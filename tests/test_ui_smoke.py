"""UI Smoke Tests using streamlit.testing.v1.AppTest.

Verifies that all 9 pages of the Streamlit application render without exceptions.
"""

import os
import pytest
from streamlit.testing.v1 import AppTest


def test_ui_all_pages_render():
    """Test that all 9 navigation pages load cleanly."""
    app_path = os.path.abspath("app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    assert not at.exception, f"Dashboard threw exception: {at.exception}"

    pages = [
        "Configuration",
        "Training",
        "Backtest",
        "Assets & Payouts",
        "Real Forex Analytics",
        "Alerts",
        "Logs",
        "About / Safety",
        "Dashboard",
    ]

    for page in pages:
        # Find radio selector for navigation
        at.sidebar.radio[0].set_value(page)
        at.run(timeout=30)
        assert not at.exception, f"Page '{page}' threw exception: {at.exception}"
