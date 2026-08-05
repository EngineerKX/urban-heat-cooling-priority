#!/usr/bin/env python
"""Push trained model artifacts (U-Net, CNN heat model) to GCS -- called
from the Colab training notebooks (notebooks/colab_training/) right after
training finishes. Skips the upload if the local file's hash already
matches what's already in GCS; --force re-uploads regardless.

Usage:
  python scripts/push_models.py --model unet
  python scripts/push_models.py --model cnn --force
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (
    CNN_MODEL_GCS_PREFIX,
    CNN_MODEL_SAVE_PATH,
    GCS_MODEL_BUCKET,
    UNET_MODEL_GCS_PREFIX,
    UNET_MODEL_SAVE_PATH,
)
from src.utils import gcs

MODELS = {
    "unet": (UNET_MODEL_SAVE_PATH, UNET_MODEL_GCS_PREFIX),
    "cnn": (CNN_MODEL_SAVE_PATH, CNN_MODEL_GCS_PREFIX),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def push_model(name: str, force: bool = False) -> bool:
    """Returns True if an upload happened, False if skipped (remote already matches)."""
    local_path, gcs_prefix = MODELS[name]
    local_path = Path(local_path)

    if not local_path.exists():
        print(f"[{name}] {local_path} doesn't exist locally — nothing to push.")
        return False

    local_hash = _sha256_file(local_path)
    remote_hash = gcs.download_text(GCS_MODEL_BUCKET, f"{gcs_prefix}.sha256")

    if not force and remote_hash == local_hash:
        print(f"[{name}] Remote copy already matches this hash — skipping push (pass --force to redo anyway).")
        return False

    print(f"[{name}] Uploading {local_path} -> gs://{GCS_MODEL_BUCKET}/{gcs_prefix}.pt ...")
    gcs.upload_file(local_path, GCS_MODEL_BUCKET, f"{gcs_prefix}.pt")
    gcs.upload_text(local_hash, GCS_MODEL_BUCKET, f"{gcs_prefix}.sha256")
    print(f"[{name}] Pushed OK.")
    return True


def main(model: str, force: bool):
    names = list(MODELS) if model == "all" else [model]
    for name in names:
        push_model(name, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.model, args.force)
