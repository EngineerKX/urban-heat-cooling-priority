"""NEA real-time air-temperature API (data.gov.sg), used both as a Week-1
feasibility gate and as the LST held-out validation source.

Station metadata is cached locally (it barely changes run to run). Reading
pulls are NOT cached the way subzones/WorldCover are: they're inherently
"give me the last N sampled days relative to today," a moving target that a
permanent cache would just go stale against — re-fetching those on every run
is the correct behaviour, not an inconsistency to fix.
"""

import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from config.settings import RAW_NEA_STATIONS_DIR
from src.utils.caching import load_or_fetch_json

BASE_URL = "https://api-open.data.gov.sg/v2/real-time/api/air-temperature"
STATION_CACHE_PATH = RAW_NEA_STATIONS_DIR / "stations.json"


def _headers(api_key: str = "") -> dict:
    return {"x-api-key": api_key} if api_key else {}


def fetch_air_temp(date_str: str = None, api_key: str = "") -> requests.Response:
    params = {"date": date_str} if date_str else {}
    return requests.get(BASE_URL, params=params, headers=_headers(api_key))


def fetch_station_metadata(api_key: str = "", force: bool = False) -> pd.DataFrame:
    """Station id/name/lat/lon, cached at data/raw/nea_stations/stations.json."""

    def _fetch() -> dict:
        resp = fetch_air_temp(api_key=api_key)
        if resp.status_code == 429:
            raise RuntimeError(
                "Rate-limited (HTTP 429). Get a free key at https://data.gov.sg "
                "(sign in -> Create API Key -> Developer), pass it as api_key, and retry."
            )
        resp.raise_for_status()
        return resp.json()

    latest_json = load_or_fetch_json(STATION_CACHE_PATH, _fetch, force=force)
    stations = latest_json.get("data", {}).get("stations", [])
    if not stations:
        raise RuntimeError("Zero stations returned — API response shape may have changed.")

    return pd.DataFrame(
        [
            {
                "station_id": s["id"],
                "station_name": s.get("name", ""),
                "latitude": s["location"]["latitude"],
                "longitude": s["location"]["longitude"],
            }
            for s in stations
        ]
    )


def sample_readings(
    n_sample_days: int,
    n_months_back: int,
    seed: int,
    api_key: str = "",
) -> pd.DataFrame:
    """Pool (station_id, value) readings across `n_sample_days` random dates
    spread over the last `n_months_back` months. Not cached — see module
    docstring."""
    random.seed(seed)
    today = datetime.now()
    candidate_days = [(today - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(1, n_months_back * 30)]
    sample_days = sorted(random.sample(candidate_days, min(n_sample_days, len(candidate_days))))
    print(f"Sampling {len(sample_days)} days: {sample_days[0]} .. {sample_days[-1]}")

    all_readings = []
    n_days_ok = 0
    for d in sample_days:
        r = fetch_air_temp(date_str=d, api_key=api_key)
        if r.status_code != 200:
            print(f"  {d}: HTTP {r.status_code}, skipped")
            time.sleep(0.3)
            continue
        payload = r.json()
        for batch in payload.get("data", {}).get("readings", []):
            for entry in batch.get("data", []):
                all_readings.append((entry["stationId"], entry["value"]))
        n_days_ok += 1
        time.sleep(0.3)  # be polite to the API

    print(f"Days successfully fetched: {n_days_ok} / {len(sample_days)}")
    if not all_readings:
        raise RuntimeError("No readings collected — check API status/rate limiting before proceeding.")

    return pd.DataFrame(all_readings, columns=["station_id", "value"])
