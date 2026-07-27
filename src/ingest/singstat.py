"""SingStat resident population by subzone (data.gov.sg CKAN datastore),
cached locally. Paginates explicitly — CKAN's default page size is small
enough that this ~390-row table can get silently truncated without a loop.
"""

import pandas as pd
import requests

from config.settings import POP_DATASET_ID, RAW_SINGSTAT_DIR
from src.utils.caching import load_or_fetch_csv

CACHE_PATH = RAW_SINGSTAT_DIR / "population_by_subzone.csv"


def _fetch_datastore_all(resource_id: str, page_size: int = 500) -> pd.DataFrame:
    base_url = "https://data.gov.sg/api/action/datastore_search"
    records = []
    offset = 0
    while True:
        r = requests.get(base_url, params={"resource_id": resource_id, "limit": page_size, "offset": offset})
        r.raise_for_status()
        payload = r.json()
        if not payload.get("success"):
            raise RuntimeError(f"data.gov.sg API error: {payload}")
        batch = payload["result"]["records"]
        records.extend(batch)
        print(f"  fetched {len(batch)} records (offset {offset}, total so far {len(records)})")
        if len(batch) < page_size:
            break
        offset += page_size
    return pd.DataFrame(records)


def fetch_population_by_subzone(force: bool = False) -> pd.DataFrame:
    """Raw SingStat population-by-subzone table, cached at
    data/raw/singstat/population_by_subzone.csv. Cleaning (dropping planning-
    area totals, coercing suppressed cells) happens in
    src/priority_score/pillars.py, not here."""

    def _fetch():
        print("No cache found — fetching SingStat population data from data.gov.sg (one-time).")
        return _fetch_datastore_all(POP_DATASET_ID)

    return load_or_fetch_csv(CACHE_PATH, _fetch, force=force)
