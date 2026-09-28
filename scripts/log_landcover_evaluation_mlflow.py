#!/usr/bin/env python
"""Log the full Dynamic World land-cover evaluation to MLflow, one run per
model, grouped under a single parent run so they sit together in the UI:

  rf_sweep       RF variants (raw, texture features, 3x3 smoothing) -- the
                 3x3 smoothed RF is the selected RF. Wider smoothing windows
                 are logged too, but tagged `diagnostic`: their size was
                 compared against the same 300 validation points, so they are
                 not eligible for selection (see src/landcover/spatial_smoothing.py).
  unet           U-Net trained on Dynamic World labels -- the selected
                 production land-cover model.
  hybrid_sweep   RF + U-Net combinations: raw-RF and smoothed-RF soft-voting
                 hybrids, plus a sweep of the RF/U-Net blend weight.
  worldcover_baseline
                 The WorldCover-trained RF, U-Net and hybrid, for comparison.

Every run is scored by the same code (validation/landcover_validation/
classifier_evaluation.py) against the same 300 hand-labelled points, and gets
accuracy, macro/weighted F1, per-class F1 and n_scored as metrics, plus its
confusion matrix and per-class table as artifacts.

Nothing is retrained: every raster already exists (the blend-weight and
diagnostic-window rasters are rebuilt locally from existing ones into a temp
folder). Re-running is skipped if this batch is already logged; pass --force
to log it again.

Usage: python scripts/log_landcover_evaluation_mlflow.py [--force]
Browse: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import numpy as np
import pandas as pd
import rasterio

from config.settings import INTERIM_DIR, PROCESSED_DIR
from src.ingest.worldcover import BUCKET_NAMES
from src.landcover.hybrid import load_prob_raster
from src.landcover.spatial_smoothing import majority_smooth
from src.utils.experiment_tracking import EXPERIMENT_NAME, log_artifact_safe, start_run
from validation.landcover_validation.classifier_evaluation import evaluate_classifier

BATCH = "dynamicworld_eval_2026-09-29"
VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_300_labeled.csv"
LC = PROCESSED_DIR / "landcover"

DIAGNOSTIC_WINDOWS = [5, 7, 11, 21]
BLEND_UNET_WEIGHTS = [0.9, 0.8, 0.7, 0.6]  # 0.5 is the standard hybrid, logged separately

# (run name, group, raster, tags). Tags say what the run is and whether it was chosen.
STATIC_RUNS = [
    ("rf_dw_raw", "rf_sweep", LC / "rf_landcover_dw_trial.tif",
     {"label_source": "dynamicworld", "model_family": "rf", "rf_variant": "raw"}),
    ("rf_dw_texture", "rf_sweep", LC / "rf_landcover_dw_trial_texture.tif",
     {"label_source": "dynamicworld", "model_family": "rf", "rf_variant": "texture_3x3_mean_std"}),
    ("rf_dw_smooth3x3", "rf_sweep", LC / "rf_landcover_dw_trial_smoothed.tif",
     {"label_source": "dynamicworld", "model_family": "rf", "rf_variant": "smooth_3x3",
      "smoothing_window": "3", "selected": "best_rf",
      "selection_note": "window fixed in advance (30 m), not tuned on the validation points"}),
    ("unet_dw", "unet", LC / "unet_landcover_dw_trial.tif",
     {"label_source": "dynamicworld", "model_family": "unet", "selected": "production_landcover",
      "selection_note": "highest accuracy and macro F1; gap to the hybrid is 6 vs 2 discordant points (McNemar p~0.29)"}),
    ("hybrid_dw_rfraw", "hybrid_sweep", LC / "hybrid_landcover_dw_trial.tif",
     {"label_source": "dynamicworld", "model_family": "hybrid", "rf_input": "raw", "rf_weight": "0.5"}),
    ("hybrid_dw_rfsmooth3x3", "hybrid_sweep", LC / "hybrid_dynamicworld_rfsmoothed.tif",
     {"label_source": "dynamicworld", "model_family": "hybrid", "rf_input": "smooth_3x3", "rf_weight": "0.5"}),
    ("rf_worldcover", "worldcover_baseline", LC / "rf_landcover.tif",
     {"label_source": "worldcover", "model_family": "rf"}),
    ("unet_worldcover", "worldcover_baseline", LC / "unet_landcover.tif",
     {"label_source": "worldcover", "model_family": "unet"}),
    ("hybrid_worldcover", "worldcover_baseline", LC / "hybrid_landcover.tif",
     {"label_source": "worldcover", "model_family": "hybrid", "rf_input": "raw", "rf_weight": "0.5"}),
]


def _write_label_raster(path: Path, label: np.ndarray, profile: dict) -> Path:
    profile = {**profile, "count": 1, "dtype": label.dtype}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(label, 1)
    return path


def _diagnostic_window_runs(tmp: Path) -> list:
    """Wider majority-vote windows on the raw RF. Logged for completeness, but
    tagged diagnostic: comparing window sizes against the validation points
    and keeping the best would be tuning on the test set."""
    with rasterio.open(LC / "rf_landcover_dw_trial.tif") as src:
        label, profile = src.read(1), src.profile
    class_ids = sorted(BUCKET_NAMES)
    runs = []
    for window in DIAGNOSTIC_WINDOWS:
        smoothed, _ = majority_smooth(label, class_ids, window=window)
        path = _write_label_raster(tmp / f"rf_smooth{window}.tif", smoothed, profile)
        runs.append((f"rf_dw_smooth{window}x{window}", "rf_sweep", path,
                     {"label_source": "dynamicworld", "model_family": "rf",
                      "rf_variant": f"smooth_{window}x{window}", "smoothing_window": str(window),
                      "diagnostic": "window_compared_on_validation_points"}))
    return runs


def _blend_weight_runs(tmp: Path) -> list:
    """RF/U-Net probability blends at fixed U-Net weights, on the raw RF."""
    rf_prob_path = LC / "rf_landcover_dw_trial_prob.tif"
    unet_prob_path = LC / "unet_landcover_dw_trial_prob.tif"
    with rasterio.open(rf_prob_path) as ref:
        transform, crs, shape = ref.transform, ref.crs, (ref.height, ref.width)
    rf_prob, rf_valid = load_prob_raster(rf_prob_path, transform, crs, shape)
    unet_prob, unet_valid = load_prob_raster(unet_prob_path, transform, crs, shape)
    valid = rf_valid & unet_valid
    profile = {"driver": "GTiff", "height": shape[0], "width": shape[1], "crs": crs, "transform": transform}

    runs = []
    for w in BLEND_UNET_WEIGHTS:
        label = (np.argmax(w * unet_prob + (1 - w) * rf_prob, axis=0) + 1).astype(np.uint8)
        label[~valid] = 0
        path = _write_label_raster(tmp / f"blend_{w:.1f}.tif", label, profile)
        runs.append((f"hybrid_dw_unetweight{w:.1f}", "hybrid_sweep", path,
                     {"label_source": "dynamicworld", "model_family": "hybrid", "rf_input": "raw",
                      "rf_weight": f"{1 - w:.1f}"}))
    return runs


def _already_logged() -> bool:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return False
    runs = mlflow.search_runs([experiment.experiment_id], filter_string=f"tags.eval_batch = '{BATCH}'")
    return len(runs) > 0


def _log_run(name, group, raster, tags, validation_df, tmp: Path):
    result = evaluate_classifier(raster, validation_df, name)
    with start_run(name, experiment_name=EXPERIMENT_NAME, nested=True, stage="evaluation",
                   eval_batch=BATCH, eval_group=group, evaluated_model=name, **tags):
        mlflow.log_param("raster", Path(raster).name)
        mlflow.log_param("n_validation_points", len(validation_df))
        mlflow.log_metrics({
            "accuracy": result["accuracy"], "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"], "n_scored": result["n_scored"],
        })
        per_class = result["per_class_metrics"].set_index("class")
        mlflow.log_metrics({f"{cls}_f1": f1 for cls, f1 in per_class["f1"].items()})
        mlflow.log_metrics({f"{cls}_recall": r for cls, r in per_class["recall"].items()})

        cm_path = tmp / f"confusion_matrix_{name}.csv"
        pc_path = tmp / f"per_class_metrics_{name}.csv"
        result["confusion_matrix"].to_csv(cm_path)
        result["per_class_metrics"].to_csv(pc_path, index=False)
        log_artifact_safe(cm_path)
        log_artifact_safe(pc_path)

    return {"run": name, "group": group, "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"], "n_scored": result["n_scored"],
            "selected": tags.get("selected", ""), "diagnostic": "yes" if "diagnostic" in tags else ""}


def main(force: bool = False):
    if not force and _already_logged():
        print(f"Batch '{BATCH}' is already in MLflow — skipping (pass --force to log it again).")
        return
    missing = [str(r) for _, _, r, _ in STATIC_RUNS if not Path(r).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raster(s): {missing}")

    validation_df = pd.read_csv(VALIDATION_CSV)
    rows = []
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        runs = STATIC_RUNS + _diagnostic_window_runs(tmp) + _blend_weight_runs(tmp)
        with start_run("dynamicworld_landcover_evaluation", experiment_name=EXPERIMENT_NAME,
                       stage="evaluation", eval_batch=BATCH, eval_group="parent"):
            mlflow.log_param("validation_csv", VALIDATION_CSV.name)
            for name, group, raster, tags in runs:
                rows.append(_log_run(name, group, raster, tags, validation_df, tmp))
            summary = pd.DataFrame(rows)
            summary_path = tmp / "evaluation_summary.csv"
            summary.to_csv(summary_path, index=False)
            log_artifact_safe(summary_path)

    print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nLogged {len(rows)} runs to MLflow experiment '{EXPERIMENT_NAME}' under parent run "
          f"'dynamicworld_landcover_evaluation' (tag eval_batch={BATCH}).")
    print("Browse: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    main(force=parser.parse_args().force)
