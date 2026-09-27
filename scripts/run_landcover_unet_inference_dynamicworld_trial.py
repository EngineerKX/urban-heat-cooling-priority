#!/usr/bin/env python
"""U-Net trial (checkpoint_DynamicWorld branch): pull the Dynamic-World-
labeled U-Net trained by
notebooks/colab_training/train_unet_dynamicworld_trial.ipynb, run it over all
of Singapore, and score it against the same 300 hand-labeled points.

Downloads directly from this trial's OWN GCS prefix (not `pull_models.py`,
which only knows the production `--model unet` prefix) into its OWN local
path, and reuses `export_inference_patches`'s SHARED cache -- inference
patches are just the feature image tiled, label-source-independent, so
reusing the cache the production U-Net already paid for is correct, not a
collision. Writes its own raster(s) and evaluates into its own output
folder; `models/unet_landcover.keras`, `data/processed/landcover/
unet_landcover.tif`, and `comparison_table.csv` are never touched.

Usage: python scripts/run_landcover_unet_inference_dynamicworld_trial.py [--force-export] [--with-probabilities] [--force-pull]
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import pandas as pd

from config.settings import (
    DRY_SEASON_MONTHS,
    GCS_MODEL_BUCKET,
    INTERIM_DIR,
    PROCESSED_DIR,
    S2_CLOUD_PROB_MAX,
    SG_BBOX,
    YEARS,
)
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, dissolve_boundary, fetch_subzones_geojson
from src.landcover.rf_baseline import build_feature_image
from src.landcover.unet_data import export_inference_patches
from src.landcover.unet_infer import load_unet, run_inference_and_reconstruct
from src.utils import gcs
from src.utils.seed import set_all_seeds
from validation.landcover_validation.classifier_evaluation import (
    compare_classifiers,
    evaluate_classifier,
    save_evaluation_outputs,
)

VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_300_labeled.csv"
UNET_MODEL_SAVE_PATH_TRIAL = Path("models/unet_landcover_dw_trial.keras")
UNET_MODEL_GCS_PREFIX_TRIAL = "models/unet_landcover_dw_trial"
UNET_RASTER_PATH_DW = PROCESSED_DIR / "landcover" / "unet_landcover_dw_trial.tif"
UNET_PROB_RASTER_PATH_DW = PROCESSED_DIR / "landcover" / "unet_landcover_dw_trial_prob.tif"
EVAL_DIR_DW = PROCESSED_DIR / "landcover" / "evaluation" / "dw_trial"
PRODUCTION_COMPARISON_CSV = PROCESSED_DIR / "landcover" / "evaluation" / "comparison_table.csv"
TRIAL_COMPARISON_CSV = EVAL_DIR_DW / "comparison_table.csv"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pull_trial_weights(force: bool = False) -> Path:
    """Mirrors scripts/pull_models.py's own hash-verified download, against
    this trial's own GCS prefix instead of the production one."""
    remote_hash = gcs.download_text(GCS_MODEL_BUCKET, f"{UNET_MODEL_GCS_PREFIX_TRIAL}.sha256")
    if remote_hash is None:
        raise FileNotFoundError(
            f"No model found at gs://{GCS_MODEL_BUCKET}/{UNET_MODEL_GCS_PREFIX_TRIAL}.keras yet — "
            "train it via notebooks/colab_training/train_unet_dynamicworld_trial.ipynb first."
        )

    sidecar = UNET_MODEL_SAVE_PATH_TRIAL.parent / f"{UNET_MODEL_SAVE_PATH_TRIAL.stem}.sha256"
    local_hash = sidecar.read_text().strip() if sidecar.exists() else None
    if not force and UNET_MODEL_SAVE_PATH_TRIAL.exists() and local_hash == remote_hash:
        print(f"[unet_dynamicworld_trial] Already up to date ({UNET_MODEL_SAVE_PATH_TRIAL}) — skipping (pass --force-pull to redo).")
        return UNET_MODEL_SAVE_PATH_TRIAL

    print(f"[unet_dynamicworld_trial] Downloading gs://{GCS_MODEL_BUCKET}/{UNET_MODEL_GCS_PREFIX_TRIAL}.keras -> {UNET_MODEL_SAVE_PATH_TRIAL} ...")
    gcs.download_blob(GCS_MODEL_BUCKET, f"{UNET_MODEL_GCS_PREFIX_TRIAL}.keras", UNET_MODEL_SAVE_PATH_TRIAL)

    downloaded_hash = _sha256_file(UNET_MODEL_SAVE_PATH_TRIAL)
    if downloaded_hash != remote_hash:
        raise RuntimeError(
            f"Downloaded file hash ({downloaded_hash}) doesn't match the remote sha256 sidecar "
            f"({remote_hash}) — download likely truncated/corrupted. Try again."
        )
    sidecar.write_text(downloaded_hash)
    print("[unet_dynamicworld_trial] Pulled and verified OK.")
    return UNET_MODEL_SAVE_PATH_TRIAL


def main(force_export: bool = False, with_probabilities: bool = False, force_pull: bool = False):
    _pull_trial_weights(force=force_pull)

    set_all_seeds()
    init_ee()
    subzones_fc = as_ee_feature_collection(fetch_subzones_geojson())
    boundary = dissolve_boundary(subzones_fc)
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))

    feature_image, _valid_mask = build_feature_image(sg_bbox, boundary, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX)

    # Label-source-independent (just the tiled feature image) -- correctly
    # SHARES the same cache the production U-Net's inference already paid for.
    inference_patch_dir = export_inference_patches(feature_image, boundary, force_export=force_export)

    model = load_unet(UNET_MODEL_SAVE_PATH_TRIAL)
    raster_path, _crs_str, prob_path = run_inference_and_reconstruct(
        model, inference_patch_dir, boundary, out_path=UNET_RASTER_PATH_DW,
        also_write_probabilities=with_probabilities, prob_out_path=UNET_PROB_RASTER_PATH_DW,
    )
    print(f"\nDynamic-World-trained U-Net raster: {raster_path}")
    if prob_path:
        print(f"Probability raster (needed for the hybrid rebuild): {prob_path}")

    if not VALIDATION_CSV.exists():
        return

    validation_df = pd.read_csv(VALIDATION_CSV)
    results = {"unet_dynamicworld": evaluate_classifier(raster_path, validation_df, "unet_dynamicworld")}
    comparison_df = compare_classifiers(results)

    # Fold in whatever's already been scored for this trial (e.g. the RF
    # trial's row) plus the production numbers, for one side-by-side view.
    existing = [p for p in (TRIAL_COMPARISON_CSV, PRODUCTION_COMPARISON_CSV) if p.exists()]
    if existing:
        prior = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
        prior = prior[prior["model"] != "unet_dynamicworld"]  # replace a stale re-run's own row
        comparison_df = pd.concat([prior, comparison_df], ignore_index=True).drop_duplicates("model", keep="last")

    print("\n" + comparison_df.to_string(index=False))
    save_evaluation_outputs(results, comparison_df, out_dir=EVAL_DIR_DW)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-export", action="store_true", help="Re-export inference patches even if cached locally/in GCS.")
    parser.add_argument("--with-probabilities", action="store_true",
                         help="Also write a per-class probability raster (needed for the hybrid rebuild).")
    parser.add_argument("--force-pull", action="store_true", help="Re-download the trial weights even if the local copy's hash already matches.")
    args = parser.parse_args()
    main(force_export=args.force_export, with_probabilities=args.with_probabilities, force_pull=args.force_pull)
