"""Formal RF-vs-U-Net-vs-ensemble land-cover evaluation: confusion matrix,
per-class precision/recall/F1, macro/weighted F1 -- scored identically for
all three classifiers against the same hand-labeled validation points.

Distinct from both models' own `informal_accuracy_check()` (RF via GEE
server-side sampleRegions, U-Net via local rasterio indexing -- two
different code paths for the same operation, and neither computes anything
beyond plain accuracy). `sample_raster_at_points` here is the single,
symmetric method used for all three rasters, so the comparison isn't
confounded by how each model happens to be sampled.
"""

from pathlib import Path

import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from config.settings import PROCESSED_DIR
from src.ingest.worldcover import BUCKET_NAMES

EVAL_DIR = PROCESSED_DIR / "landcover" / "evaluation"


def sample_raster_at_points(raster_path, validation_df: pd.DataFrame, band: int = 1) -> pd.DataFrame:
    """Point-sample a single-band raster at each validation point's lon/lat.
    Reads the CRS directly off the raster file (unlike
    unet.py::informal_accuracy_check, which needs it passed in separately
    because it's called right after reconstruction) -- this is what lets one
    function sample RF's, U-Net's, and the ensemble's already-written raster
    files identically."""
    with rasterio.open(raster_path) as src:
        raster_data = src.read(band)
        raster_transform = src.transform
        crs_str = src.crs.to_string()

    transformer = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)

    def _lonlat_to_pixel(lon, lat):
        x, y = transformer.transform(lon, lat)
        col, row = ~raster_transform * (x, y)
        return int(row), int(col)

    preds = []
    for row in validation_df.itertuples():
        r, c = _lonlat_to_pixel(row.lon, row.lat)
        if 0 <= r < raster_data.shape[0] and 0 <= c < raster_data.shape[1]:
            preds.append(int(raster_data[r, c]))
        else:
            preds.append(None)

    out = validation_df.copy()
    out["pred_bucket"] = preds
    return out


def score_predictions(df: pd.DataFrame, true_col: str, pred_col: str, class_names: dict = BUCKET_NAMES) -> dict:
    """Confusion matrix (as a labeled DataFrame, following
    validation/input_validation/labeling_agreement.py's cm_df pattern) +
    per-class precision/recall/F1 + accuracy + macro/weighted F1."""
    class_values = sorted(class_names.keys())
    y_true = df[true_col].astype(int)
    y_pred = df[pred_col].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=class_values)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{class_names[v]}" for v in class_values],
        columns=[f"pred:{class_names[v]}" for v in class_values],
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_values, average=None, zero_division=0,
    )
    per_class = pd.DataFrame({
        "class": [class_names[v] for v in class_values],
        "precision": precision, "recall": recall, "f1": f1, "support": support,
    })

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=class_values, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=class_values, average="weighted", zero_division=0)),
        "confusion_matrix": cm_df,
        "per_class_metrics": per_class,
    }


def evaluate_classifier(raster_path, validation_df: pd.DataFrame, model_name: str, class_names: dict = BUCKET_NAMES) -> dict:
    validation_df = validation_df.copy()
    validation_df["agreed_label"] = validation_df["agreed_label"].fillna("").astype(str)
    valid_rows = validation_df[validation_df["agreed_label"].isin(class_names.values())].copy()
    n_skipped = len(validation_df) - len(valid_rows)
    if n_skipped:
        print(f"[{model_name}] Skipping {n_skipped} validation point(s) with blank/uncertain labels.")

    sampled = sample_raster_at_points(raster_path, valid_rows)
    bucket_name_to_id = {v: k for k, v in class_names.items()}
    sampled["true_bucket"] = sampled["agreed_label"].map(bucket_name_to_id)

    scored = sampled.dropna(subset=["pred_bucket", "true_bucket"])
    scored = scored[scored["pred_bucket"] != 0]  # nodata pixel at a validation point -> unscoreable
    n_dropped = len(sampled) - len(scored)
    if n_dropped:
        print(f"[{model_name}] Dropping {n_dropped} point(s) with no prediction (nodata/out-of-raster).")

    result = score_predictions(scored, "true_bucket", "pred_bucket", class_names)
    result["model_name"] = model_name
    result["n_scored"] = len(scored)
    print(
        f"[{model_name}] accuracy={result['accuracy'] * 100:.1f}%  macro_f1={result['macro_f1']:.3f}  "
        f"weighted_f1={result['weighted_f1']:.3f}  (n={len(scored)})"
    )
    return result


def compare_classifiers(results: dict) -> pd.DataFrame:
    """results: {model_name: evaluate_classifier(...) dict}. One row per
    model: accuracy, macro/weighted F1, n_scored, per-class F1."""
    rows = []
    for model_name, r in results.items():
        row = {
            "model": model_name, "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"], "n_scored": r["n_scored"],
        }
        per_class_f1 = r["per_class_metrics"].set_index("class")["f1"]
        for cls_name in BUCKET_NAMES.values():
            row[f"{cls_name}_f1"] = per_class_f1.get(cls_name, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def save_evaluation_outputs(results: dict, comparison_df: pd.DataFrame, out_dir=EVAL_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(out_dir / "comparison_table.csv", index=False)
    for model_name, r in results.items():
        slug = model_name.lower().replace(" ", "_")
        r["confusion_matrix"].to_csv(out_dir / f"confusion_matrix_{slug}.csv")
        r["per_class_metrics"].to_csv(out_dir / f"per_class_metrics_{slug}.csv", index=False)
    print(f"Evaluation outputs written to {out_dir}")
