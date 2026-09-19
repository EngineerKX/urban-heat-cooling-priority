#!/usr/bin/env python
"""Standalone verification for src/utils/caching.py::is_stale -- no network or
data required. Same runnable-script + printed pass/fail style as the rest of
tests/.

Usage: python tests/test_caching.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.caching import is_stale


def test_is_stale_follows_modified_times():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        output, source = tmp / "out.csv", tmp / "in.csv"
        source.write_text("x")
        assert is_stale(output, [source]), "a missing output is always stale"

        output.write_text("y")
        now = time.time()
        os.utime(source, (now - 100, now - 100))
        os.utime(output, (now, now))
        assert not is_stale(output, [source]), "an output newer than its input is fresh"

        os.utime(source, (now + 100, now + 100))
        assert is_stale(output, [source]), "an input newer than the output makes it stale — the failure mode this exists for"

        assert not is_stale(output, [tmp / "does_not_exist.csv"]), "inputs that don't exist are ignored"
        print("PASS: is_stale is true for a missing output or a newer input, false otherwise")


if __name__ == "__main__":
    test_is_stale_follows_modified_times()
    print("\nAll caching checks passed.")
