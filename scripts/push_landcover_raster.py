#!/usr/bin/env python
"""Push the production land-cover map to GCS for the CNN Colab notebook, which
can't regenerate it (it needs local U-Net inference).

Uploads config.settings.LANDCOVER_RASTER_PATH plus two sidecars:
  <prefix>.sha256  -- the file's hash, checked after download in Colab
  <prefix>.json    -- which model and training labels produced it, and when
The CNN notebook refuses to train if the downloaded map's hash doesn't match,
or if its model / label source differ from the current settings -- so the CNN
can never be trained on a stale or wrong land-cover map without it showing.

Run after `python scripts/run_landcover_unet_inference.py --with-probabilities`
(or after rebuilding whichever map LANDCOVER_PRODUCTION_MODEL points at).

Usage: python scripts/push_landcover_raster.py
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (
    GCS_MODEL_BUCKET, LANDCOVER_PRODUCTION_MODEL, LANDCOVER_RASTER_GCS_PREFIX, LANDCOVER_RASTER_PATH,
)
from src.landcover.labels import label_source_tag
from src.utils import gcs


def main():
    if not LANDCOVER_RASTER_PATH.exists():
        raise FileNotFoundError(f"{LANDCOVER_RASTER_PATH} not found — build the production land-cover map first.")

    sha = hashlib.sha256(LANDCOVER_RASTER_PATH.read_bytes()).hexdigest()
    meta = {
        "landcover_model": LANDCOVER_PRODUCTION_MODEL,
        "label_source": label_source_tag(),
        "file": LANDCOVER_RASTER_PATH.name,
        "raster_built": datetime.fromtimestamp(LANDCOVER_RASTER_PATH.stat().st_mtime).isoformat(timespec="minutes"),
        "sha256": sha,
    }
    gcs.upload_file(LANDCOVER_RASTER_PATH, GCS_MODEL_BUCKET, f"{LANDCOVER_RASTER_GCS_PREFIX}.tif")
    gcs.upload_text(sha, GCS_MODEL_BUCKET, f"{LANDCOVER_RASTER_GCS_PREFIX}.sha256")
    gcs.upload_text(json.dumps(meta, indent=1), GCS_MODEL_BUCKET, f"{LANDCOVER_RASTER_GCS_PREFIX}.json")
    print(f"Pushed {LANDCOVER_RASTER_PATH.name} -> gs://{GCS_MODEL_BUCKET}/{LANDCOVER_RASTER_GCS_PREFIX}.tif")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
