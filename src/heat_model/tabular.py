"""S5 (C2) subzone-level tabular half: XGBoost regression from land-cover /
seasonal-index / population / hotspot-cluster features to raw LST, plus a
concrete counterfactual mechanism ("what if this subzone had more green
cover?"). Target is `lst_native30` (least-processed LST variant) rather
than `lst_regress10`, since the regression variant was itself fit on
NDVI/NDBI/NDWI -- reusing those as XGBoost inputs against that target would
be circular.

Pure join of already-built per-subzone CSVs (heat variants, S4 hotspot
clusters -- which already carries seasonal indices AND land-cover
fractions from Item 1 -- and the sensitivity pillar's population table).
No new fetch of any kind.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config.settings import RANDOM_SEED, XGB_LEARNING_RATE, XGB_MAX_DEPTH, XGB_N_ESTIMATORS, XGB_SUBSAMPLE

XGB_TARGET_COLUMN = "lst_native30"
XGB_FEATURE_COLUMNS = [
    "fraction_vegetation", "fraction_built_up", "fraction_bare", "fraction_water",
    "ndvi_dry", "ndvi_wet", "ndbi_dry", "ndbi_wet",
    "population_total", "elderly_proportion", "primary_cluster",
]
XGB_CATEGORICAL_COLUMNS = ["primary_cluster"]


def build_xgb_training_table(
    heat_csv_path: Path, hotspot_clusters_csv_path: Path, sensitivity_csv_path: Path,
    feature_columns=XGB_FEATURE_COLUMNS, target_column: str = XGB_TARGET_COLUMN,
) -> pd.DataFrame:
    """Inner-joins the 3 source CSVs on subzone_id and casts categorical
    feature columns to pandas 'category' dtype (XGBoost's native
    categorical support, `enable_categorical=True` in train_xgb_model --
    avoids implying a false ordinal relationship between cluster ids)."""
    heat = pd.read_csv(heat_csv_path)[["subzone_id", target_column]]
    hotspot = pd.read_csv(hotspot_clusters_csv_path)
    sensitivity = pd.read_csv(sensitivity_csv_path)

    hotspot_cols = ["subzone_id"] + [c for c in feature_columns if c in hotspot.columns]
    sensitivity_cols = ["subzone_id"] + [c for c in feature_columns if c in sensitivity.columns]
    missing = set(feature_columns) - (set(hotspot_cols) | set(sensitivity_cols))
    if missing:
        raise ValueError(f"Feature column(s) {missing} not found in either source CSV.")

    df = heat.merge(hotspot[hotspot_cols], on="subzone_id", how="inner").merge(
        sensitivity[sensitivity_cols], on="subzone_id", how="inner"
    )
    n_dropped = len(heat) - len(df)
    print(f"XGBoost training table: {len(heat)} -> {len(df)} subzones after join.")
    if n_dropped:
        print(f"⚠️  {n_dropped} subzones dropped — check subzone_id alignment across the 3 source CSVs.")

    for col in XGB_CATEGORICAL_COLUMNS:
        if col in df.columns:
            # Plain int -> category, NOT via pandas' nullable "Int64" extension
            # dtype first -- that intermediate cast produces category values
            # XGBoost's categorical encoder chokes on (TypeError: object of
            # type 'int' has no len(), confirmed empirically). No NaNs are
            # expected here (inner join on complete S4 cluster output).
            df[col] = df[col].astype(int).astype("category")
    return df


def train_xgb_model(
    df: pd.DataFrame, feature_columns=XGB_FEATURE_COLUMNS, target_column: str = XGB_TARGET_COLUMN,
    test_size: float = 0.2, seed: int = RANDOM_SEED, **xgb_params,
):
    X, y = df[feature_columns], df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    params = dict(
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE, random_state=seed, enable_categorical=True,
    )
    params.update(xgb_params)
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_r2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print(f"XGBoost test RMSE={metrics['test_rmse']:.3f}, R²={metrics['test_r2']:.3f} "
          f"(n_train={metrics['n_train']}, n_test={metrics['n_test']})")
    return model, metrics


def fit_ndvi_vegetation_slope(df: pd.DataFrame, vegetation_col: str = "fraction_vegetation",
                               ndvi_col: str = "ndvi_dry") -> float:
    """OLS slope of ndvi_col on vegetation_col -- a one-off coefficient
    predict_counterfactual_subzone uses to keep NDVI internally consistent
    with a hypothetically edited vegetation fraction, instead of leaving
    NDVI stale while only the fraction changes underneath it."""
    result = linregress(df[vegetation_col], df[ndvi_col])
    print(f"Fitted {ndvi_col}~{vegetation_col} slope: {result.slope:.3f} (R²={result.rvalue ** 2:.3f})")
    return float(result.slope)


def redistribute_vegetation_fraction(veg0: float, built0: float, bare0: float, delta: float) -> dict:
    """Pure fraction-arithmetic core of the counterfactual mechanism,
    factored out so it's independently testable: apply `delta` to
    fraction_vegetation (clamped to [0, 1]), pulling the change out of
    built-up/bare proportionally to their current share of the non-
    vegetation area. fraction_water is deliberately not a parameter here --
    it never changes (greening a carpark doesn't plausibly convert water).
    Returns veg1/built1/bare1 that sum to exactly veg0+built0+bare0 (the
    invariant callers should check), plus actual_delta (may differ from
    the requested `delta` if clamping kicked in)."""
    non_veg0 = built0 + bare0
    veg1 = min(max(veg0 + delta, 0.0), 1.0)
    actual_delta = veg1 - veg0

    if non_veg0 > 1e-9:
        built1 = max(built0 - actual_delta * (built0 / non_veg0), 0.0)
        bare1 = max(bare0 - actual_delta * (bare0 / non_veg0), 0.0)
    else:
        built1, bare1 = built0, bare0

    return {"fraction_vegetation": veg1, "fraction_built_up": built1, "fraction_bare": bare1,
            "actual_delta_vegetation": actual_delta}


def predict_counterfactual_subzone(
    model, row_df: pd.DataFrame, delta_fraction_vegetation: float, ndvi_slope: float,
    feature_columns=XGB_FEATURE_COLUMNS,
) -> dict:
    """`row_df` is a single-row DataFrame (e.g. `df[df.subzone_id == X]`) so
    column dtypes (notably primary_cluster's 'category' dtype) pass through
    unchanged -- a bare pd.Series round-trip can silently lose that.

    Mechanism: redistribute the vegetation-fraction delta proportionally
    out of built-up/bare (see redistribute_vegetation_fraction), bump
    ndvi_dry/ndvi_wet via the fitted slope so they stay consistent with the
    edited fraction, then re-predict with the same model."""
    if len(row_df) != 1:
        raise ValueError(f"row_df must have exactly one row, got {len(row_df)}")

    original = row_df[feature_columns].copy()
    edited = original.copy()

    redistributed = redistribute_vegetation_fraction(
        float(original["fraction_vegetation"].iloc[0]),
        float(original["fraction_built_up"].iloc[0]),
        float(original["fraction_bare"].iloc[0]),
        delta_fraction_vegetation,
    )
    actual_delta = redistributed.pop("actual_delta_vegetation")
    for col, value in redistributed.items():
        edited[col] = value
    # fraction_water intentionally untouched.

    for ndvi_col in ("ndvi_dry", "ndvi_wet"):
        if ndvi_col in edited.columns:
            edited[ndvi_col] = edited[ndvi_col].astype(float) + actual_delta * ndvi_slope

    original_pred = float(model.predict(original)[0])
    edited_pred = float(model.predict(edited)[0])

    return {
        "original_lst": original_pred,
        "counterfactual_lst": edited_pred,
        "delta_lst": edited_pred - original_pred,
        "requested_delta_vegetation": delta_fraction_vegetation,
        "actual_delta_vegetation": actual_delta,
    }
