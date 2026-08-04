#!/usr/bin/env python
"""Standalone verification that every app/ page loads without raising --
uses Streamlit's own streamlit.testing.v1.AppTest rather than pytest
(matches this repo's runnable-script + printed pass/fail convention,
despite AppTest itself being new here). This checks "doesn't crash", not
"looks/works right" -- see the golden-path browser check before calling a
UI change done.

Usage: python tests/test_app_pages.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    REPO_ROOT / "app" / "Home.py",
    REPO_ROOT / "app" / "pages" / "1_Label_Validation_Points.py",
    REPO_ROOT / "app" / "pages" / "2_Island_Map.py",
    REPO_ROOT / "app" / "pages" / "3_Subzone_Breakdown.py",
    REPO_ROOT / "app" / "pages" / "4_Counterfactual_Greening.py",
    REPO_ROOT / "app" / "pages" / "5_Validation_Dashboard.py",
]


def test_page_loads_without_exception(page_path: Path):
    at = AppTest.from_file(str(page_path), default_timeout=120)
    at.run()
    if at.exception:
        for exc in at.exception:
            print(f"  {exc.value}")
        raise AssertionError(f"{page_path.name} raised an exception on load — see printed traceback above.")
    print(f"PASS: {page_path.name} loads without exception")


def main():
    for page_path in PAGES:
        test_page_loads_without_exception(page_path)
    print("\nAll app page checks passed.")


if __name__ == "__main__":
    main()
