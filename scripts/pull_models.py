#!/usr/bin/env python
"""Pull trained model artifacts (U-Net, CNN heat model) from GCS -- the
source of truth once training moves to Colab (see notebooks/colab_training/).
Skips the download if the local copy's hash already matches the remote
sha256 sidecar; --force re-downloads regardless. Also imports any new
Colab training-run summaries into the local MLflow store (see
src/utils/experiment_tracking.py::import_remote_runs) so U-Net/CNN runs
show up alongside RF/XGBoost's.

Usage:
  python scripts/pull_models.py --model unet
  python scripts/pull_models.py --model all --force
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
from src.utils.experiment_tracking import import_remote_runs

MODELS = {
    "unet": (UNET_MODEL_SAVE_PATH, UNET_MODEL_GCS_PREFIX),
    "cnn": (CNN_MODEL_SAVE_PATH, CNN_MODEL_GCS_PREFIX),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar_path(local_path: Path) -> Path:
    return local_path.parent / f"{local_path.stem}.sha256"


def pull_model(name: str, force: bool = False) -> bool:
    """Returns True if a download happened, False if skipped (already up to date)."""
    local_path, gcs_prefix = MODELS[name]
    local_path = Path(local_path)
    remote_hash = gcs.download_text(GCS_MODEL_BUCKET, f"{gcs_prefix}.sha256")

    if remote_hash is None:
        print(f"[{name}] No model found at gs://{GCS_MODEL_BUCKET}/{gcs_prefix}.pt yet — train it in Colab first.")
        return False

    sidecar_path = _sidecar_path(local_path)
    local_hash = sidecar_path.read_text().strip() if sidecar_path.exists() else None

    if not force and local_path.exists() and local_hash == remote_hash:
        print(f"[{name}] Already up to date ({local_path}) — skipping (pass --force to redo anyway).")
        return False

    print(f"[{name}] Downloading gs://{GCS_MODEL_BUCKET}/{gcs_prefix}.pt -> {local_path} ...")
    gcs.download_blob(GCS_MODEL_BUCKET, f"{gcs_prefix}.pt", local_path)

    downloaded_hash = _sha256_file(local_path)
    if downloaded_hash != remote_hash:
        raise RuntimeError(
            f"[{name}] Downloaded file hash ({downloaded_hash}) doesn't match the remote sha256 sidecar "
            f"({remote_hash}) — download likely truncated/corrupted. Try again."
        )

    sidecar_path.write_text(downloaded_hash)
    print(f"[{name}] Pulled and verified OK.")
    return True


def main(model: str, force: bool):
    names = list(MODELS) if model == "all" else [model]
    for name in names:
        pull_model(name, force=force)

    print("\nChecking for new Colab training-run summaries to import into the local MLflow store ...")
    import_remote_runs(GCS_MODEL_BUCKET)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.model, args.force)
