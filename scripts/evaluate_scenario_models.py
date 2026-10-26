#!/usr/bin/env python
"""Evaluate the scenario-modelling component (the two heat models and the
greening what-if) for the evaluation report. Uses the production models:
data/processed/heat_model/xgb_model.pkl and models/heat_cnn.keras. Writes to
data/processed/diagnostics/:

  scenario_xgb_cv.csv             XGBoost vs baselines, random and spatial CV
  scenario_feature_ablation.csv   XGBoost with feature groups removed / alone
  scenario_mechanism.csv          what-if at +15 points vegetation, three edit mechanisms
  scenario_dose_response.csv      what-if at several vegetation increases
  scenario_association.csv        cross-subzone temperature-vegetation slope
  scenario_cnn_validation.csv     CNN on its own held-out patches vs baselines
  scenario_model_agreement.csv    XGBoost vs CNN direction of change, per subzone

--skip-cnn runs only the subzone (XGBoost) part, which needs no patch data.

Usage: python scripts/evaluate_scenario_models.py [--skip-cnn]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import (
    BUCKET_VEGETATION, CNN_MODEL_SAVE_PATH, CNN_TRAIN_VAL_SPLIT, DIAGNOSTICS_DIR, INTERIM_DIR, LANDCOVER_RASTER_PATH,
    PROCESSED_DIR, RANDOM_SEED, S2_UTM_CRS, SUBZONE_ID_PROPERTY, TARGET_SCALE_M, TOP_N, UNET_PATCH_SIZE,
)
from src.heat_model.tabular import XGB_FEATURE_COLUMNS, build_xgb_training_table, predict_counterfactual_subzone
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from validation.heat_model_validation.scenario_evaluation import (
    cross_subzone_slope, cross_validate, feature_group_ablation, mahalanobis_reference, subset_index_model,
    summarise_what_if, what_if,
)

HEAT_MODEL_DIR = PROCESSED_DIR / "heat_model"
INFERENCE_PATCH_DIR = INTERIM_DIR / "unet_patches" / "inference"
MIXER_JSON_PATH = INFERENCE_PATCH_DIR / "unet_inference.json"
LST_BICUBIC10_PATH = INTERIM_DIR / "lst_bicubic10_full.tif"
DELTA = 0.15            # +15 percentage points of vegetation, the app's and the progress log's reference edit
DOSES = [0.05, 0.10, 0.15, 0.20, 0.30]
CNN_RADIUS_M = 50.0     # same edit as scripts/diagnose_heat_model.py
FRACTIONS = ["fraction_vegetation", "fraction_built_up", "fraction_bare", "fraction_water"]
INDICES = ["ndvi_dry", "ndvi_wet", "ndbi_dry", "ndbi_wet"]
POPULATION = ["population_total", "elderly_proportion"]
FEATURE_GROUPS = {
    "All ten (production)": XGB_FEATURE_COLUMNS,
    "Without land-cover fractions": INDICES + POPULATION,
    "Without spectral indices": FRACTIONS + POPULATION,
    "Without population": FRACTIONS + INDICES,
    "Land-cover fractions only": FRACTIONS,
    "Spectral indices only": INDICES,
}


def _save(df, name):
    df.to_csv(DIAGNOSTICS_DIR / name, index=False)
    print(f"\n{name}\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def subzone_part(df, model, index_model):
    # The production model on the held-out 20% it was trained without (same split as train_xgb_model).
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from src.heat_model.tabular import XGB_TARGET_COLUMN
    _, X_te, _, y_te = train_test_split(df[XGB_FEATURE_COLUMNS], df[XGB_TARGET_COLUMN], test_size=0.2,
                                        random_state=RANDOM_SEED)
    p = model.predict(X_te)
    _save(pd.DataFrame([{"r2": r2_score(y_te, p), "rmse_c": float(np.sqrt(mean_squared_error(y_te, p))),
                         "n_test": len(y_te), "n_train": len(df) - len(y_te)}]), "scenario_xgb_production_split.csv")

    gdf = as_geodataframe(fetch_subzones_geojson())
    planning_area = df["subzone_id"].map(pd.Series(gdf["PLN_AREA_N"].values, index=gdf[SUBZONE_ID_PROPERTY].values))
    cv = pd.concat([cross_validate(df).assign(validation="Random 5-fold, repeated 5 times"),
                    cross_validate(df, groups=planning_area).assign(validation="Spatial: whole planning areas held out")])
    _save(cv, "scenario_xgb_cv.csv")
    _save(feature_group_ablation(df, FEATURE_GROUPS), "scenario_feature_ablation.csv")

    dist, cutoff = mahalanobis_reference(df)
    real_outside = float((dist(df[XGB_FEATURE_COLUMNS].values) > cutoff).mean())
    mechanisms = {
        "Land cover only (indices unchanged)": subset_index_model(index_model, []),
        "Land cover + NDVI": subset_index_model(index_model, ["ndvi_dry", "ndvi_wet"]),
        "Land cover + NDVI + NDBI (production)": index_model,
    }
    rows = [{"mechanism": name, "delta_vegetation": DELTA,
             **summarise_what_if(what_if(df, model, im, DELTA, dist, cutoff))} for name, im in mechanisms.items()]
    _save(pd.DataFrame(rows).assign(real_subzones_outside_range=real_outside), "scenario_mechanism.csv")

    rows = []
    for d in DOSES:
        per = what_if(df, model, index_model, d, dist, cutoff)
        rows.append({"delta_vegetation": d, "mean_actual_delta": per["actual_delta_vegetation"].mean(),
                     **summarise_what_if(per)})
    _save(pd.DataFrame(rows), "scenario_dose_response.csv")

    slope = cross_subzone_slope(df)
    _save(pd.DataFrame([{"slope_c_per_unit_vegetation": slope, "implied_c_for_delta": slope * DELTA,
                         "delta_vegetation": DELTA}]), "scenario_association.csv")


def cnn_part(df, model, index_model):
    import tensorflow as tf  # noqa: F401  (loaded by cnn_infer; imported here so a missing install fails early)
    from src.heat_model.cnn_data import build_local_feature_target_patches
    from src.heat_model.cnn_infer import load_cnn_regressor, locate_patch_and_pixel, run_cnn_inference
    from src.heat_model.counterfactual import class_mean_feature_vectors, run_patch_counterfactual

    print("\nBuilding the CNN patch stack ...")
    X, y, valid = build_local_feature_target_patches(INFERENCE_PATCH_DIR, MIXER_JSON_PATH, LANDCOVER_RASTER_PATH,
                                                     LST_BICUBIC10_PATH)
    cnn = load_cnn_regressor(CNN_MODEL_SAVE_PATH)

    # The training split exactly as src/heat_model/cnn_train.py::train_cnn_regressor draws it.
    idx = np.random.default_rng(RANDOM_SEED).permutation(X.shape[0])
    cut = int(X.shape[0] * CNN_TRAIN_VAL_SPLIT)
    tr, va = idx[:cut], idx[cut:]
    pred = run_cnn_inference(cnn, X[va])  # (n, H, W)
    v = valid[va]
    truth, cnn_pred = y[va][v], pred[v]

    train_mean = y[tr][valid[tr]].mean()
    rng = np.random.default_rng(RANDOM_SEED)
    tr_px = np.argwhere(valid[tr])
    pick = tr_px[rng.choice(len(tr_px), size=min(300_000, len(tr_px)), replace=False)]
    Xtr = X[tr][pick[:, 0], pick[:, 1], pick[:, 2]]
    ytr = y[tr][pick[:, 0], pick[:, 1], pick[:, 2]]
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    beta = np.linalg.lstsq(A, ytr, rcond=None)[0]
    lin = np.column_stack([np.ones(v.sum()), X[va][v]]) @ beta

    def score(p):
        """RMSE and R², plus what an RMSE alone hides: bias (mean error),
        correlation with the truth (does it get the spatial pattern?) and the
        spread ratio (sd of predictions / sd of truth; above 1 = exaggerates
        contrasts)."""
        err = p - truth
        return {"rmse_c": float(np.sqrt((err ** 2).mean())),
                "r2": float(1 - (err ** 2).sum() / ((truth - truth.mean()) ** 2).sum()),
                "bias_c": float(err.mean()),
                "correlation": float(np.corrcoef(p, truth)[0, 1]) if np.std(p) > 0 else float("nan"),
                "spread_ratio": float(np.std(p) / truth.std())}
    _save(pd.DataFrame([
        {"model": "CNN (production)", **score(cnn_pred)},
        {"model": "Per-pixel linear regression on the same input channels", **score(lin)},
        {"model": "Predict the training mean", **score(np.full_like(truth, train_mean))},
    ]).assign(n_val_patches=len(va), n_val_pixels=int(v.sum()), target_sd_c=float(truth.std())),
        "scenario_cnn_validation.csv")

    # XGBoost vs CNN: direction of the greening effect, per subzone.
    class_means = class_mean_feature_vectors(X, valid_mask=valid)
    gdf = as_geodataframe(fetch_subzones_geojson())
    utm = gdf.to_crs(S2_UTM_CRS)
    ranked = pd.read_csv(PROCESSED_DIR / "priority_score.csv").nlargest(TOP_N, "priority_score")["subzone_id"].tolist()
    demo = ["TUAS NORTH", "GUL CIRCLE", "CHIN BEE"]  # scripts/diagnose_heat_model.py's demo subzones
    yy, xx = np.mgrid[0:UNET_PATCH_SIZE, 0:UNET_PATCH_SIZE]
    rows = []
    for group, ids in (("Residents top 20", ranked), ("Diagnostic demo subzones", demo)):
        for sid in ids:
            row = df[df["subzone_id"] == sid]
            geom = utm[utm[SUBZONE_ID_PROPERTY] == sid]
            if row.empty or geom.empty:
                continue
            centroid = gpd_centroid_lonlat(geom)
            patch, r, c = locate_patch_and_pixel(*centroid, MIXER_JSON_PATH, UNET_PATCH_SIZE)
            if not 0 <= patch < X.shape[0]:
                continue
            mask = ((yy - r) ** 2 + (xx - c) ** 2) <= (CNN_RADIUS_M / TARGET_SCALE_M) ** 2
            cnn_res = run_patch_counterfactual(cnn, X[patch], mask, BUCKET_VEGETATION, class_means)
            xgb_res = predict_counterfactual_subzone(model, row, DELTA, index_model)
            rows.append({"group": group, "subzone_id": sid, "xgb_delta_c": xgb_res["delta_lst"],
                         "cnn_delta_in_edit_area_c": cnn_res["mean_delta_lst_in_edit_area"],
                         "same_direction": bool(np.sign(xgb_res["delta_lst"]) == np.sign(cnn_res["mean_delta_lst_in_edit_area"]))})
    agree = pd.DataFrame(rows)
    _save(agree, "scenario_model_agreement.csv")
    top = agree[agree["group"] == "Residents top 20"]
    if len(top) > 2:
        print(f"\nTop 20: same direction {top['same_direction'].mean():.0%}; Spearman of magnitudes "
              f"{spearmanr(top['xgb_delta_c'], top['cnn_delta_in_edit_area_c'])[0]:.2f}")


def gpd_centroid_lonlat(geom_utm):
    import geopandas as gpd
    c = geom_utm.geometry.centroid.iloc[0]
    p = gpd.GeoSeries([c], crs=S2_UTM_CRS).to_crs("EPSG:4326").iloc[0]
    return p.x, p.y


def main(skip_cnn: bool):
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    df = build_xgb_training_table(INTERIM_DIR / "heat_variants_subzone.csv", PROCESSED_DIR / "hotspot_clusters.csv",
                                  INTERIM_DIR / "sensitivity_pillar.csv")
    with open(HEAT_MODEL_DIR / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    index_model = json.loads((HEAT_MODEL_DIR / "xgb_index_landcover_model.json").read_text())
    subzone_part(df, model, index_model)
    if not skip_cnn:
        cnn_part(df, model, index_model)
    print(f"\nSaved to {DIAGNOSTICS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cnn", action="store_true")
    main(skip_cnn=parser.parse_args().skip_cnn)
