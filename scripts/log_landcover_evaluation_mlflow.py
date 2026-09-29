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

Every model here was trained with the 15 m exclusion buffer around the SAME
300 validation points it is scored on (tag `trained_n_validation_points`).
The production WorldCover RF was trained against an older 200-point sample,
so the WorldCover RF and hybrid baselines come from a retrain against the 300
(scripts/train_landcover_rf_worldcover_baseline.py), not the production files.

Every run is scored by the same code (validation/landcover_validation/
classifier_evaluation.py) and gets accuracy with a 95% bootstrap interval,
macro/weighted F1, per-class F1/recall and n_scored, plus its confusion matrix
and per-class table. The parent run holds a summary table and paired McNemar
tests for the key comparisons, so it is clear which differences are real and
which are within noise.

Nothing is retrained here: every raster already exists (blend-weight and
diagnostic-window rasters are rebuilt locally from existing ones in a temp
folder). Re-running is skipped if this batch is already logged (--force logs
it again). Earlier batches are tagged `superseded`, never deleted.

Usage: python scripts/log_landcover_evaluation_mlflow.py [--force]
Browse: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import binomtest

from config.settings import DW_TRAIN_END, DW_TRAIN_START, INTERIM_DIR, PROCESSED_DIR, RANDOM_SEED
from src.ingest.worldcover import BUCKET_NAMES
from src.landcover.hybrid import build_hybrid, load_prob_raster
from src.landcover.spatial_smoothing import majority_smooth, write_smoothed_outputs
from src.utils.experiment_tracking import EXPERIMENT_NAME, log_artifact_safe, start_run
from validation.landcover_validation.classifier_evaluation import evaluate_classifier, sample_raster_at_points

BATCH = "landcover_eval_final_2026-09-29c"
SUPERSEDED_BATCHES = {
    "dynamicworld_eval_2026-09-29": "WorldCover RF/hybrid baselines used a model trained against the old "
                                    "200-point validation sample; replaced by a retrain against the current 300.",
    "dynamicworld_eval_2026-09-29b": "Trial models trained on an open-ended Dynamic World window (every image up "
                                     "to the training day), so not reproducible; replaced by the final models on "
                                     "the fixed window (config.settings.DATA_END_DATE).",
}
VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_300_labeled.csv"
LC = PROCESSED_DIR / "landcover"

# Final production rasters (Dynamic World labels, fixed window) and their probabilities.
RF_FINAL, RF_FINAL_PROB = LC / "rf_landcover.tif", LC / "rf_landcover_prob.tif"
UNET_FINAL, UNET_FINAL_PROB = LC / "unet_landcover.tif", LC / "unet_landcover_prob.tif"

DIAGNOSTIC_WINDOWS = [5, 7, 11, 21]
BLEND_UNET_WEIGHTS = [0.9, 0.8, 0.7, 0.6]  # 0.5 is the standard hybrid, logged separately
N_BOOT = 4000

_TRAIN_300 = {"trained_n_validation_points": "300"}
_DW = {"label_source": "dynamicworld", "label_window": f"{DW_TRAIN_START}..{DW_TRAIN_END}", **_TRAIN_300}
_WC = {"label_source": "worldcover", **_TRAIN_300}
# (run name, group, raster, tags). Tags say what the run is, where it came from, and whether it was chosen.
STATIC_RUNS = [
    ("rf_dw_raw", "rf_sweep", RF_FINAL,
     {"model_family": "rf", "rf_variant": "raw",
      "producing_script": "scripts/train_landcover_rf.py --with-probabilities", **_DW}),
    ("rf_dw_texture", "rf_sweep", LC / "rf_landcover_texture.tif",
     {"model_family": "rf", "rf_variant": "texture_3x3_mean_std",
      "producing_script": "scripts/train_landcover_rf_dynamicworld_texture.py", **_DW}),
    ("unet_dw", "unet", UNET_FINAL,
     {"model_family": "unet", "selected": "production_landcover",
      "selection_note": "highest accuracy and macro F1 of the final models; the gap to the hybrid is within "
                        "noise (see parent run)",
      "producing_script": "notebooks/colab_training/train_unet.ipynb + "
                          "scripts/run_landcover_unet_inference.py --with-probabilities", **_DW}),
    ("hybrid_dw_rfraw", "hybrid_sweep", LC / "hybrid_landcover.tif",
     {"model_family": "hybrid", "rf_input": "raw", "rf_weight": "0.5",
      "producing_script": "scripts/build_landcover_hybrid.py --force", **_DW}),
    ("rf_worldcover", "worldcover_baseline", LC / "rf_landcover_wc300.tif",
     {"model_family": "rf", "producing_script": "scripts/train_landcover_rf_worldcover_baseline.py", **_WC}),
    ("unet_worldcover", "worldcover_baseline", LC / "unet_landcover_worldcover.tif",
     {"model_family": "unet",
      "producing_script": "notebooks/colab_training/train_unet.ipynb (2026-09-18 WorldCover run) + "
                          "scripts/run_landcover_unet_inference.py", **_WC}),
    ("hybrid_worldcover", "worldcover_baseline", LC / "hybrid_landcover_wc300.tif",
     {"model_family": "hybrid", "rf_input": "raw", "rf_weight": "0.5",
      "producing_script": "scripts/train_landcover_rf_worldcover_baseline.py", **_WC}),
]

# Paired comparisons whose significance is logged on the parent run.
MCNEMAR_PAIRS = [
    ("hybrid_worldcover", "hybrid_dw_rfraw", "worldcover_vs_dynamicworld_hybrid"),
    ("unet_worldcover", "unet_dw", "worldcover_vs_dynamicworld_unet"),
    ("rf_worldcover", "rf_dw_raw", "worldcover_vs_dynamicworld_rf"),
    ("rf_dw_raw", "unet_dw", "rf_vs_unet"),
    ("rf_dw_raw", "rf_dw_smooth3x3", "rf_raw_vs_smooth3x3"),
    ("rf_dw_raw", "rf_dw_texture", "rf_raw_vs_texture"),
    ("hybrid_dw_rfraw", "unet_dw", "hybrid_vs_unet"),
    ("hybrid_dw_rfraw", "hybrid_dw_rfsmooth3x3", "hybrid_rfraw_vs_rfsmooth"),
]


def _write_label_raster(path: Path, label: np.ndarray, profile: dict) -> Path:
    profile = {**profile, "count": 1, "dtype": label.dtype}
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(label, 1)
    return path


def _smoothed_runs(tmp: Path) -> list:
    """The selected 3x3-smoothed RF and the smoothed-RF hybrid, rebuilt here
    from the final RF so they can't drift from it."""
    label_path, prob_path = write_smoothed_outputs(
        RF_FINAL, sorted(BUCKET_NAMES), BUCKET_NAMES, window=3,
        label_out_path=tmp / "rf_smooth3x3.tif", prob_out_path=tmp / "rf_smooth3x3_prob.tif",
    )
    hybrid_path, _ = build_hybrid(prob_path, UNET_FINAL_PROB, label_out_path=tmp / "hybrid_rfsmooth3x3.tif",
                                  prob_out_path=tmp / "hybrid_rfsmooth3x3_prob.tif")
    return [
        ("rf_dw_smooth3x3", "rf_sweep", label_path,
         {"model_family": "rf", "rf_variant": "smooth_3x3", "smoothing_window": "3", "selected": "best_rf",
          "selection_note": "window fixed in advance (30 m), not tuned on the validation points",
          "producing_script": "built in this script from rf_landcover.tif (src/landcover/spatial_smoothing.py)",
          **_DW}),
        ("hybrid_dw_rfsmooth3x3", "hybrid_sweep", hybrid_path,
         {"model_family": "hybrid", "rf_input": "smooth_3x3", "rf_weight": "0.5",
          "producing_script": "built in this script from the smoothed RF and unet_landcover_prob.tif", **_DW}),
    ]


def _diagnostic_window_runs(tmp: Path) -> list:
    """Wider majority-vote windows on the raw RF. Logged for completeness, but
    tagged diagnostic: comparing window sizes against the validation points
    and keeping the best would be tuning on the test set."""
    with rasterio.open(RF_FINAL) as src:
        label, profile = src.read(1), src.profile
    class_ids = sorted(BUCKET_NAMES)
    runs = []
    for window in DIAGNOSTIC_WINDOWS:
        smoothed, _ = majority_smooth(label, class_ids, window=window)
        path = _write_label_raster(tmp / f"rf_smooth{window}.tif", smoothed, profile)
        runs.append((f"rf_dw_smooth{window}x{window}", "rf_sweep", path,
                     {"model_family": "rf", "rf_variant": f"smooth_{window}x{window}",
                      "smoothing_window": str(window), "diagnostic": "window_compared_on_validation_points",
                      "producing_script": "built in this script from rf_landcover.tif", **_DW}))
    return runs


def _blend_weight_runs(tmp: Path) -> list:
    """RF/U-Net probability blends at fixed U-Net weights, on the raw RF."""
    with rasterio.open(RF_FINAL_PROB) as ref:
        transform, crs, shape = ref.transform, ref.crs, (ref.height, ref.width)
    rf_prob, rf_valid = load_prob_raster(RF_FINAL_PROB, transform, crs, shape)
    unet_prob, unet_valid = load_prob_raster(UNET_FINAL_PROB, transform, crs, shape)
    valid = rf_valid & unet_valid
    profile = {"driver": "GTiff", "height": shape[0], "width": shape[1], "crs": crs, "transform": transform}

    runs = []
    for w in BLEND_UNET_WEIGHTS:
        label = (np.argmax(w * unet_prob + (1 - w) * rf_prob, axis=0) + 1).astype(np.uint8)
        label[~valid] = 0
        path = _write_label_raster(tmp / f"blend_{w:.1f}.tif", label, profile)
        runs.append((f"hybrid_dw_unetweight{w:.1f}", "hybrid_sweep", path,
                     {"model_family": "hybrid", "rf_input": "raw", "rf_weight": f"{1 - w:.1f}",
                      "producing_script": "built in this script from rf_landcover_prob.tif and "
                                          "unet_landcover_prob.tif", **_DW}))
    return runs


def _batch_runs(batch: str) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return pd.DataFrame()
    return mlflow.search_runs([experiment.experiment_id], filter_string=f"tags.eval_batch = '{batch}'")


def _mark_superseded():
    """Tag (never delete) every run from earlier batches so the UI shows which
    numbers were replaced and why."""
    client = mlflow.tracking.MlflowClient()
    for batch, reason in SUPERSEDED_BATCHES.items():
        runs = _batch_runs(batch)
        for run_id in runs.get("run_id", []):
            client.set_tag(run_id, "superseded", "true")
            client.set_tag(run_id, "superseded_by", BATCH)
            client.set_tag(run_id, "superseded_reason", reason)
        if len(runs):
            print(f"Tagged {len(runs)} run(s) from batch '{batch}' as superseded.")


def _per_point_correct(raster, labelled: pd.DataFrame) -> pd.Series:
    """True/False per validation point (index = point_id), only where the
    raster has a real prediction -- the same points evaluate_classifier scores."""
    sampled = sample_raster_at_points(raster, labelled)
    scored = sampled[sampled["pred_bucket"].notna() & (sampled["pred_bucket"] != 0)]
    return pd.Series((scored["pred_bucket"] == scored["true_bucket"]).values, index=scored["point_id"].values)


def _log_run(name, group, raster, tags, validation_df, labelled, tmp: Path, rng):
    result = evaluate_classifier(raster, validation_df, name)
    correct = _per_point_correct(raster, labelled)
    boot = [rng.choice(correct.values, len(correct)).mean() for _ in range(N_BOOT)]
    built = datetime.fromtimestamp(Path(raster).stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    with start_run(name, experiment_name=EXPERIMENT_NAME, nested=True, stage="evaluation",
                   eval_batch=BATCH, eval_group=group, evaluated_model=name, raster_built=built, **tags):
        mlflow.log_param("raster", Path(raster).name)
        mlflow.log_param("n_validation_points", len(validation_df))
        mlflow.log_metrics({
            "accuracy": result["accuracy"], "accuracy_ci95_low": float(np.percentile(boot, 2.5)),
            "accuracy_ci95_high": float(np.percentile(boot, 97.5)),
            "macro_f1": result["macro_f1"], "weighted_f1": result["weighted_f1"], "n_scored": result["n_scored"],
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

    row = {"run": name, "group": group, "accuracy": result["accuracy"],
           "ci95_low": float(np.percentile(boot, 2.5)), "ci95_high": float(np.percentile(boot, 97.5)),
           "macro_f1": result["macro_f1"], "n_scored": result["n_scored"], "raster_built": built,
           "selected": tags.get("selected", ""), "diagnostic": "yes" if "diagnostic" in tags else ""}
    return row, correct


def _mcnemar(correct: dict) -> pd.DataFrame:
    rows = []
    for a, b, label in MCNEMAR_PAIRS:
        common = correct[a].index.intersection(correct[b].index)
        ca, cb = correct[a].loc[common], correct[b].loc[common]
        only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
        n = only_a + only_b
        p = binomtest(only_b, n, 0.5).pvalue if n else 1.0
        rows.append({"comparison": label, "model_a": a, "model_b": b, "n_common": len(common),
                     "accuracy_a": ca.mean(), "accuracy_b": cb.mean(),
                     "only_a_correct": only_a, "only_b_correct": only_b, "p_value": p,
                     "verdict": "real difference" if p < 0.05 else "within noise"})
    return pd.DataFrame(rows)


def main(force: bool = False):
    if not force and len(_batch_runs(BATCH)):
        print(f"Batch '{BATCH}' is already in MLflow — skipping (pass --force to log it again).")
        return
    missing = [str(r) for _, _, r, _ in STATIC_RUNS if not Path(r).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raster(s): {missing}")

    validation_df = pd.read_csv(VALIDATION_CSV)
    name_to_id = {v: k for k, v in BUCKET_NAMES.items()}
    labelled = validation_df[validation_df["agreed_label"].isin(name_to_id)].copy()
    labelled["true_bucket"] = labelled["agreed_label"].map(name_to_id)
    rng = np.random.default_rng(RANDOM_SEED)

    rows, correct = [], {}
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        runs = STATIC_RUNS + _smoothed_runs(tmp) + _diagnostic_window_runs(tmp) + _blend_weight_runs(tmp)
        with start_run("landcover_evaluation_final", experiment_name=EXPERIMENT_NAME,
                       stage="evaluation", eval_batch=BATCH, eval_group="parent"):
            mlflow.log_param("validation_csv", VALIDATION_CSV.name)
            mlflow.log_param("bootstrap_draws", N_BOOT)
            for name, group, raster, tags in runs:
                row, correct[name] = _log_run(name, group, raster, tags, validation_df, labelled, tmp, rng)
                rows.append(row)

            summary = pd.DataFrame(rows)
            significance = _mcnemar(correct)
            mlflow.log_metrics({f"p_{r.comparison}": r.p_value for r in significance.itertuples()})
            for df, fname in ((summary, "evaluation_summary.csv"), (significance, "significance_mcnemar.csv")):
                df.to_csv(tmp / fname, index=False)
                log_artifact_safe(tmp / fname)

    _mark_superseded()
    fmt = lambda x: f"{x:.3f}"
    print("\n" + summary.to_string(index=False, float_format=fmt))
    print("\n" + significance.to_string(index=False, float_format=fmt))
    print(f"\nLogged {len(rows)} runs to MLflow experiment '{EXPERIMENT_NAME}' under parent run "
          f"'landcover_evaluation_final' (tag eval_batch={BATCH}).")
    print("Browse: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    main(force=parser.parse_args().force)
