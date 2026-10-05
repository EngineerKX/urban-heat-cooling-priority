#!/usr/bin/env python
"""Build the evaluation reports from saved results.

Each chapter module (ch_*.py) has TITLE, STANDALONE_TITLE, SUBTITLE and
build(doc, repo). The same code builds a standalone report per chapter and the
main report, where each chapter becomes a numbered chapter (sections renumber
automatically). Output goes to docs/reports/.

Usage (from the repo root, with the project's Python):
  python docs/report_build/build.py landcover      # one standalone report
  python docs/report_build/build.py all            # every standalone report
  python docs/report_build/build.py main           # the combined main report
"""

import importlib
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from helpers import Doc  # noqa: E402

OUT_DIR = REPO / "docs" / "reports"
# Main-report order. Chapters not written yet are skipped with a note.
CHAPTERS = [
    ("strategy", "evaluation_strategy.docx"),
    ("downscaling", "downscaling_evaluation.docx"),
    ("landcover", "landcover_evaluation.docx"),
    ("typology", "heat_typology_evaluation.docx"),
    ("priority", "cooling_priority_evaluation.docx"),
    ("scenario", "scenario_modelling_evaluation.docx"),
]
MAIN_TITLE = "Evaluation Strategy and Ablation Testing"
MAIN_FILE = "evaluation_and_ablation_report.docx"
TODAY = date.today().strftime("%d %B %Y").lstrip("0")


def _module(key):
    try:
        return importlib.import_module(f"ch_{key}")
    except ModuleNotFoundError as e:
        if e.name == f"ch_{key}":
            return None
        raise


def build_standalone(key, filename):
    mod = _module(key)
    if mod is None:
        print(f"skip {key}: ch_{key}.py not written yet")
        return
    doc = Doc(date=TODAY)
    mod.build(doc, REPO)
    doc.save(OUT_DIR / filename, mod.STANDALONE_TITLE)


def build_main():
    doc = Doc(date=TODAY)
    doc.para(MAIN_TITLE, "Title")
    doc.para(f"Urban heat cooling-priority project | {TODAY}", "Subtitle")
    number = 0
    for key, _ in CHAPTERS:
        mod = _module(key)
        if mod is None:
            print(f"main report: chapter '{key}' not written yet, skipped")
            continue
        number += 1
        if number > 1:
            doc.page_break()
        chapter = Doc(chapter_number=number)
        mod.build(chapter, REPO)
        doc.body.extend(chapter.body)
    doc.save(OUT_DIR / MAIN_FILE, MAIN_TITLE)


def main(target):
    if target == "main":
        build_main()
    elif target == "all":
        for key, filename in CHAPTERS:
            build_standalone(key, filename)
    else:
        filename = dict(CHAPTERS).get(target)
        if filename is None:
            raise SystemExit(f"unknown target '{target}'; use one of {[k for k, _ in CHAPTERS]}, all, main")
        build_standalone(target, filename)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
