"""Generic "fetch once, reuse forever" helper.

This is the fix for the notebooks' main inconsistency: almost every one of
them re-hit data.gov.sg / the NEA API / GEE on every single run, even though
the pulled data (subzone boundaries, WorldCover class table, NEA readings,
SingStat population) doesn't change between runs. `load_or_fetch` checks a
local cache file first and only calls the network/GEE-side `fetch_fn` on a
cache miss.
"""

import json
from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd

T = TypeVar("T")


def load_or_fetch(
    path: Path,
    fetch_fn: Callable[[], T],
    loader: Callable[[Path], T],
    saver: Callable[[T, Path], None],
    force: bool = False,
) -> T:
    """Return `loader(path)` if `path` exists (and `force` is False),
    otherwise call `fetch_fn()`, persist the result via `saver`, and return it.
    """
    if path.exists() and not force:
        return loader(path)
    result = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    saver(result, path)
    return result


def is_stale(output: Path, inputs) -> bool:
    """True if `output` is missing or older than any input that exists.

    The plain "skip if the output file exists" check used by most build
    scripts can't notice that an upstream file was rebuilt, so they keep
    serving results computed from the old input (this left the priority score
    weeks behind a rebuilt land-cover raster). A modified-time check catches
    exactly that case; scripts that adopt it recompute on their own when an
    input is newer, and still take `--force` to override.
    """
    output = Path(output)
    if not output.exists():
        return True
    output_mtime = output.stat().st_mtime
    return any(Path(p).exists() and Path(p).stat().st_mtime > output_mtime for p in inputs)


def load_or_fetch_json(path: Path, fetch_fn: Callable[[], dict], force: bool = False) -> dict:
    def _load(p: Path) -> dict:
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def _save(data: dict, p: Path) -> None:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)

    return load_or_fetch(path, fetch_fn, _load, _save, force=force)


def load_or_fetch_csv(path: Path, fetch_fn: Callable[[], pd.DataFrame], force: bool = False) -> pd.DataFrame:
    def _load(p: Path) -> pd.DataFrame:
        return pd.read_csv(p)

    def _save(df: pd.DataFrame, p: Path) -> None:
        df.to_csv(p, index=False)

    return load_or_fetch(path, fetch_fn, _load, _save, force=force)
