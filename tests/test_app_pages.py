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


def test_all_places_view_loads():
    """The map and breakdown pages in the all-places view. The breakdown page is
    opened on TUAS NORTH, which the residents view does not rank (30 residents)
    -- it must appear here and show no sensitivity bar."""
    for name in ("2_Island_Map.py", "3_Subzone_Breakdown.py"):
        at = AppTest.from_file(str(REPO_ROOT / "app" / "pages" / name), default_timeout=120)
        at.session_state["priority_lens"] = "all_places"
        at.session_state["selected_subzone_id"] = "TUAS NORTH"
        at.run()
        assert not at.exception, f"{name} (all-places view): {[e.value for e in at.exception]}"
        assert at.radio[0].value == "all_places", f"{name}: expected the all-places view, got {at.radio[0].value!r}"
    assert at.selectbox[0].value == "TUAS NORTH", f"breakdown page should offer TUAS NORTH, got {at.selectbox[0].value!r}"
    print("PASS: map and breakdown pages load in the all-places view, and TUAS NORTH is selectable there")


def test_island_map_weight_sliders():
    """Turning on the weight sliders and moving everything onto heat must
    re-score the map and report how many top-N subzones changed."""
    at = AppTest.from_file(str(REPO_ROOT / "app" / "pages" / "2_Island_Map.py"), default_timeout=120)
    at.session_state["priority_lens"] = "residents"
    at.run()
    at.toggle[0].set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.slider) == 3, f"expected 3 weight sliders, got {len(at.slider)}"
    changed = at.metric[0].value
    assert changed == "0", f"default slider weights should leave the top-N unchanged, got {changed}"
    at.slider[1].set_value(0.0).run()
    at.slider[2].set_value(0.0).run()
    assert not at.exception, [e.value for e in at.exception]
    assert int(at.metric[0].value) > 0, "heat-only weights should change the top-N"
    print(f"PASS: weight sliders load, default weights change 0 of the top-N, heat-only changes {at.metric[0].value}")

    color_radio = next(r for r in at.radio if r.label == "Color by")
    color_radio.set_value("Top 20 only").run()
    assert not at.exception, [e.value for e in at.exception]
    at.toggle[0].set_value(False).run()
    assert not at.exception, [e.value for e in at.exception]
    assert next(r for r in at.radio if r.label == "Color by").value == "Top 20 only", "turning the sliders off must keep 'Top 20 only'"
    print("PASS: 'Top 20 only' colour option draws with and without custom weights, and survives the slider toggle")

    at = AppTest.from_file(str(REPO_ROOT / "app" / "pages" / "2_Island_Map.py"), default_timeout=120)
    at.session_state["priority_lens"] = "all_places"
    at.run()
    at.toggle[0].set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.slider) == 2, f"all-places view should have 2 weight sliders, got {len(at.slider)}"
    assert at.metric[0].value == "0", f"default all-places weights should change nothing, got {at.metric[0].value}"
    at.slider[1].set_value(0.0).run()
    assert not at.exception, [e.value for e in at.exception]
    print(f"PASS: all-places view has 2 sliders, defaults change 0, heat-only changes {at.metric[0].value}")


def main():
    for page_path in PAGES:
        test_page_loads_without_exception(page_path)
    test_all_places_view_loads()
    test_island_map_weight_sliders()
    print("\nAll app page checks passed.")


if __name__ == "__main__":
    main()
