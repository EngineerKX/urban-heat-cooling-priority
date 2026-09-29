#!/usr/bin/env python
"""Train S5's XGBoost tabular half (C2) -- subzone-level LST regression
from land-cover / seasonal-index / population features (the S4 hotspot
cluster label is deliberately not a feature -- see
src/heat_model/tabular.py::XGB_FEATURE_COLUMNS). New pipeline stage:
src/heat_model/ was empty, no source notebook to migrate. Runs natively on
Windows, CPU-only, no GPU needed (unlike the CNN half, which trains in
Colab -- see notebooks/colab_training/train_heat_cnn.ipynb).

Usage: python scripts/train_heat_model_xgboost.py [--force-retrain]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import pandas as pd

from config.settings import (
    INTERIM_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SUBSAMPLE,
)
from src.heat_model.tabular import (
    XGB_FEATURE_COLUMNS,
    XGB_TARGET_COLUMN,
    build_xgb_training_table,
    fit_index_landcover_model,
    train_xgb_model,
)
from src.utils.experiment_tracking import HEAT_MODEL_EXPERIMENT_NAME, log_artifact_safe, start_run

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
HOTSPOT_CLUSTERS_CSV_PATH = PROCESSED_DIR / "hotspot_clusters.csv"
SENSITIVITY_CSV_PATH = INTERIM_DIR / "sensitivity_pillar.csv"

OUT_DIR = PROCESSED_DIR / "heat_model"
MODEL_OUT_PATH = OUT_DIR / "xgb_model.pkl"
PREDICTIONS_OUT_PATH = OUT_DIR / "xgb_predictions.csv"
FEATURE_IMPORTANCE_OUT_PATH = OUT_DIR / "xgb_feature_importance.csv"
# The spectral-index ~ land-cover fit the greening counterfactual uses to keep
# an edited subzone's NDVI/NDBI consistent with its new land cover.
INDEX_MODEL_OUT_PATH = OUT_DIR / "xgb_index_landcover_model.json"


def main(force_retrain: bool = False):
    if PREDICTIONS_OUT_PATH.exists() and not force_retrain:
        print(f"{PREDICTIONS_OUT_PATH} already exists — skipping retrain (pass --force-retrain to redo).")
        return PREDICTIONS_OUT_PATH

    for label, path in [
        ("heat variants", HEAT_CSV_PATH),
        ("S4 hotspot clusters", HOTSPOT_CLUSTERS_CSV_PATH),
        ("sensitivity pillar", SENSITIVITY_CSV_PATH),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found ({label}) — run its build script first.")

    df = build_xgb_training_table(HEAT_CSV_PATH, HOTSPOT_CLUSTERS_CSV_PATH, SENSITIVITY_CSV_PATH)

    with start_run("xgboost", experiment_name=HEAT_MODEL_EXPERIMENT_NAME):
        mlflow.log_params({
            "n_estimators": XGB_N_ESTIMATORS, "max_depth": XGB_MAX_DEPTH,
            "learning_rate": XGB_LEARNING_RATE, "subsample": XGB_SUBSAMPLE,
            "n_subzones": len(df), "target_column": XGB_TARGET_COLUMN,
        })

        model, metrics = train_xgb_model(df, seed=RANDOM_SEED)
        mlflow.log_metrics(metrics)

        index_model = fit_index_landcover_model(df)
        mlflow.log_metrics({f"index_fit_r2_{col}": r2 for col, r2 in index_model["r2"].items()})
        mlflow.log_metrics({f"index_coef_{col}_{frac.replace('fraction_', '')}": coef
                            for col, coefs in index_model["coefs"].items() for frac, coef in coefs.items()})

        importances = pd.Series(
            model.feature_importances_, index=XGB_FEATURE_COLUMNS, name="importance"
        ).sort_values(ascending=False)
        print("\nFeature importances:")
        print(importances.to_string())

        df["lst_predicted"] = model.predict(df[XGB_FEATURE_COLUMNS])
        df["residual"] = df["lst_predicted"] - df[XGB_TARGET_COLUMN]

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_OUT_PATH, "wb") as f:
            pickle.dump(model, f)
        log_artifact_safe(MODEL_OUT_PATH)

        df[["subzone_id", XGB_TARGET_COLUMN, "lst_predicted", "residual"]].rename(
            columns={XGB_TARGET_COLUMN: "lst_actual"}
        ).to_csv(PREDICTIONS_OUT_PATH, index=False)
        importances.to_csv(FEATURE_IMPORTANCE_OUT_PATH)
        INDEX_MODEL_OUT_PATH.write_text(json.dumps(index_model, indent=1))

    print(f"\nSaved: {MODEL_OUT_PATH}")
    print(f"Saved: {PREDICTIONS_OUT_PATH}")
    print(f"Saved: {FEATURE_IMPORTANCE_OUT_PATH}")
    return PREDICTIONS_OUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    main(force_retrain=args.force_retrain)
