#!/usr/bin/env python
"""Supplementary land-cover results for the evaluation report, all computed
from the final rasters in data/processed/landcover/ against the same 300
hand-labelled points as scripts/log_landcover_evaluation_mlflow.py:

  mcnemar_counts.csv        points correct for only one model of each pair,
                            with the exact McNemar p-value (the MLflow batch
                            kept only the p-values; these counts reproduce them)
  unet_confusion_matrix.csv the production U-Net's confusion matrix
  unet_per_class.csv        its per-class precision / recall / F1
  vegetation_binary.csv     vegetation vs everything else, per model -- the
                            only distinction that reaches the priority score
                            (the greenery pillar uses the vegetation share)
  ../../landcover_label_source_swap.csv
                            decision impact of the training-label source:
                            priority score rebuilt with greenery from each
                            WorldCover-trained and Dynamic-World-trained map,
                            vs the production U-Net (validation/score_validation/
                            landcover_impact.py)

The two 3x3-smoothed-RF comparisons are not recomputed here (their rasters
are built in a temp folder by the MLflow logging script).

Usage: python scripts/evaluate_landcover_supplementary.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import binomtest

from config.settings import INTERIM_DIR, PROCESSED_DIR, SUBZONE_ID_PROPERTY
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from src.ingest.worldcover import BUCKET_NAMES
from src.landcover.zonal import zonal_class_fractions
from src.priority_score.io import load_and_join
from validation.landcover_validation.classifier_evaluation import evaluate_classifier, sample_raster_at_points
from validation.score_validation.landcover_impact import run_landcover_swap

LC = PROCESSED_DIR / "landcover"
OUT = LC / "evaluation"
VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_300_labeled.csv"
MLFLOW_P = OUT / "final_batch_mcnemar_p.csv"
SWAP_OUT = PROCESSED_DIR / "landcover_label_source_swap.csv"

# Run names follow the MLflow batch so the two tables join.
RASTERS = {
    "rf_dw_raw": LC / "rf_landcover.tif",
    "unet_dw": LC / "unet_landcover.tif",
    "hybrid_dw_rfraw": LC / "hybrid_landcover.tif",
    "rf_dw_texture": LC / "rf_landcover_texture.tif",
    "rf_worldcover": LC / "rf_landcover_wc300.tif",
    "unet_worldcover": LC / "unet_landcover_worldcover.tif",
    "hybrid_worldcover": LC / "hybrid_landcover_wc300.tif",
}
PAIRS = [  # (model_a, model_b, comparison name in the MLflow batch)
    ("hybrid_worldcover", "hybrid_dw_rfraw", "worldcover_vs_dynamicworld_hybrid"),
    ("unet_worldcover", "unet_dw", "worldcover_vs_dynamicworld_unet"),
    ("rf_worldcover", "rf_dw_raw", "worldcover_vs_dynamicworld_rf"),
    ("rf_dw_raw", "unet_dw", "rf_vs_unet"),
    ("rf_dw_raw", "rf_dw_texture", "rf_raw_vs_texture"),
    ("hybrid_dw_rfraw", "unet_dw", "hybrid_vs_unet"),
]
SWAP_SOURCES = {  # label -> raster; the first is the reference (production)
    "U-Net, Dynamic World (production)": "unet_dw",
    "Hybrid, Dynamic World": "hybrid_dw_rfraw",
    "RF, Dynamic World": "rf_dw_raw",
    "U-Net, WorldCover": "unet_worldcover",
    "Hybrid, WorldCover": "hybrid_worldcover",
    "RF, WorldCover": "rf_worldcover",
}


def _labelled():
    v = pd.read_csv(VALIDATION_CSV)
    name_to_id = {name: k for k, name in BUCKET_NAMES.items()}
    v = v[v["agreed_label"].isin(name_to_id)].copy()
    v["true_bucket"] = v["agreed_label"].map(name_to_id)
    return v


def _predictions(labelled):
    """point_id -> predicted bucket per model, only where the raster has a prediction
    (same scoring rule as classifier_evaluation.evaluate_classifier)."""
    preds = {}
    for name, path in RASTERS.items():
        s = sample_raster_at_points(path, labelled)
        s = s[s["pred_bucket"].notna() & (s["pred_bucket"] != 0)]
        preds[name] = s.set_index("point_id")["pred_bucket"].astype(int)
    return preds


def mcnemar_counts(labelled, preds):
    truth = labelled.set_index("point_id")["true_bucket"]
    mlflow_p = pd.read_csv(MLFLOW_P).set_index("comparison")["p_value"] if MLFLOW_P.exists() else None
    rows = []
    for a, b, label in PAIRS:
        common = preds[a].index.intersection(preds[b].index)
        ca, cb = preds[a][common] == truth[common], preds[b][common] == truth[common]
        only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p = binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        rows.append({"comparison": label, "model_a": a, "model_b": b, "n_common": len(common),
                     "accuracy_a": ca.mean(), "accuracy_b": cb.mean(),
                     "only_a_correct": only_a, "only_b_correct": only_b, "p_value": p,
                     "p_value_mlflow": mlflow_p.get(label) if mlflow_p is not None else None})
    return pd.DataFrame(rows)


def vegetation_binary(labelled, preds):
    veg = {name: k for k, name in BUCKET_NAMES.items()}["vegetation"]
    truth = labelled.set_index("point_id")["true_bucket"] == veg
    rows = []
    for name, p in preds.items():
        t, y = truth[p.index], p == veg
        tp, fp, fn = int((t & y).sum()), int((~t & y).sum()), int((t & ~y).sum())
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        rows.append({"model": name, "n_scored": len(p), "accuracy": float((t == y).mean()),
                     "vegetation_precision": precision, "vegetation_recall": recall,
                     "vegetation_f1": 2 * precision * recall / (precision + recall),
                     "predicted_vegetation_share": float(y.mean()), "true_vegetation_share": float(t.mean())})
    return pd.DataFrame(rows)


def label_source_swap():
    df, _ = load_and_join(toy_mode=False)
    subzones = as_geodataframe(fetch_subzones_geojson())
    columns = {}
    for label, run in SWAP_SOURCES.items():
        frac = zonal_class_fractions(RASTERS[run], subzones, SUBZONE_ID_PROPERTY)[["subzone_id", "fraction_vegetation"]]
        col = f"greenery_{run}"
        df = df.merge(frac.rename(columns={"fraction_vegetation": col}), on="subzone_id", how="left")
        columns[label] = col
    return run_landcover_swap(df, columns, reference_label=next(iter(SWAP_SOURCES)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labelled = _labelled()
    preds = _predictions(labelled)

    mc = mcnemar_counts(labelled, preds)
    mc.to_csv(OUT / "mcnemar_counts.csv", index=False)
    print(mc.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    unet = evaluate_classifier(RASTERS["unet_dw"], pd.read_csv(VALIDATION_CSV), "unet_dw")
    unet["confusion_matrix"].to_csv(OUT / "unet_confusion_matrix.csv")
    unet["per_class_metrics"].to_csv(OUT / "unet_per_class.csv", index=False)
    print("\n" + unet["confusion_matrix"].to_string())

    vb = vegetation_binary(labelled, preds)
    vb.to_csv(OUT / "vegetation_binary.csv", index=False)
    print("\n" + vb.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Training-label source: decision impact ===")
    swap = label_source_swap()
    swap.to_csv(SWAP_OUT, index=False)
    print(f"\nSaved: {OUT} (4 files), {SWAP_OUT}")


if __name__ == "__main__":
    main()
