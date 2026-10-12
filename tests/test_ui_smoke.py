"""Headless smoke tests: each Streamlit page must render without raising (streamlit.testing AppTest)."""
import os
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
PAGES = ["src/ui/training_monitor.py", "src/ui/app.py", "src/ui/annotate_events.py", "src/ui/label_tips.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page, monkeypatch):
    monkeypatch.setenv("PRISM_MONITOR_ONCE", "1")  # the monitor otherwise sleeps and reruns forever
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
