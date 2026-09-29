"""S5 (C2) subzone-level tabular half: XGBoost regression from land-cover /
seasonal-index / population features to raw LST, plus a concrete
counterfactual mechanism ("what if this subzone had more green cover?").
Target is `lst_native30` (least-processed LST variant) rather than
`lst_regress10`, since the regression variant was itself fit on
NDVI/NDBI/NDWI -- reusing those as XGBoost inputs against that target would
be circular.

Pure join of already-built per-subzone CSVs (heat variants, the S4 hotspot
feature table -- which carries seasonal indices AND land-cover fractions from
Item 1 -- and the sensitivity pillar's population table). No new fetch of any
kind. The S4 cluster label itself is NOT a feature, see XGB_FEATURE_COLUMNS.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config.settings import RANDOM_SEED, XGB_LEARNING_RATE, XGB_MAX_DEPTH, XGB_N_ESTIMATORS, XGB_SUBSAMPLE

XGB_TARGET_COLUMN = "lst_native30"
# The S4 hotspot-typology label (`primary_cluster`) was a feature until 2026-09-21 and is
# deliberately excluded now: the clustering takes `lst_dry` as an input, and `lst_dry` is
# identical to this model's target `lst_native30`, so the label carried part of the answer
# (indirect target leakage). Dropping it cost ~0.03 test R2 (about +0.08 C RMSE) -- the
# model does not depend on it.
XGB_FEATURE_COLUMNS = [
    "fraction_vegetation", "fraction_built_up", "fraction_bare", "fraction_water",
    "ndvi_dry", "ndvi_wet", "ndbi_dry", "ndbi_wet",
    "population_total", "elderly_proportion",
]
XGB_CATEGORICAL_COLUMNS = []  # none at present; kept so a categorical feature can be re-added cleanly


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


# The land-cover fractions the counterfactual edits (fraction_water never
# changes, so it is the implicit reference category), and the spectral-index
# features that have to move with them to keep an edited subzone realistic.
EDITED_FRACTION_COLUMNS = ["fraction_vegetation", "fraction_built_up", "fraction_bare"]
SPECTRAL_INDEX_COLUMNS = ["ndvi_dry", "ndvi_wet", "ndbi_dry", "ndbi_wet"]


def fit_index_landcover_model(df: pd.DataFrame, index_columns=SPECTRAL_INDEX_COLUMNS,
                              fraction_columns=EDITED_FRACTION_COLUMNS) -> dict:
    """For each spectral index, an OLS fit on the land-cover fractions across
    real subzones: index ~ a + b_veg*veg + b_built*built + b_bare*bare.
    predict_counterfactual_subzone moves every index by b . (change in the
    fractions), so a greened subzone's NDVI rises AND its NDBI falls the way
    real greener subzones' do.

    Replaced a single NDVI-on-vegetation slope (2026-09-30). That version left
    NDBI -- XGBoost's second most important feature -- untouched, so an edited
    subzone had the land cover of a greener place with the NDBI of the old
    one. Result: only 56% of subzones cooled under +15 pts vegetation (median
    -0.10 C), i.e. mostly noise. Moving all four indices: 89% cool, median
    -0.99 C, in line with the cross-subzone association (about -1.1 C per
    +15 pts), and 1.8% of edited subzones fall outside the range of real ones
    (vs 5% for real subzones themselves). Fitted on the fractions only -- the
    LST target plays no part in it.

    Returns {"coefs": {index: {fraction: coef}}, "r2": {index: r2}}."""
    X = df[fraction_columns].astype(float).values
    design = np.column_stack([np.ones(len(X)), X])
    coefs, r2 = {}, {}
    for col in index_columns:
        y = df[col].astype(float).values
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        r2[col] = float(1 - (residual ** 2).sum() / ((y - y.mean()) ** 2).sum())
        coefs[col] = {frac: float(b) for frac, b in zip(fraction_columns, beta[1:])}
        print(f"Fitted {col} ~ land-cover fractions (R²={r2[col]:.3f}): " +
              ", ".join(f"{f.replace('fraction_', '')} {b:+.3f}" for f, b in coefs[col].items()))
    return {"coefs": coefs, "r2": r2}


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
    model, row_df: pd.DataFrame, delta_fraction_vegetation: float, index_model: dict,
    feature_columns=XGB_FEATURE_COLUMNS,
) -> dict:
    """`row_df` is a single-row DataFrame (e.g. `df[df.subzone_id == X]`) so
    column dtypes (notably any 'category' feature) pass through unchanged --
    a bare pd.Series round-trip can silently lose that.

    Mechanism: redistribute the vegetation-fraction delta proportionally
    out of built-up/bare (see redistribute_vegetation_fraction), move every
    spectral index by its fitted land-cover coefficients times the change in
    each fraction (`index_model`, from fit_index_landcover_model) so NDVI and
    NDBI stay consistent with the edited land cover, then re-predict with the
    same model."""
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
    fraction_change = {col: value - float(original[col].iloc[0]) for col, value in redistributed.items()}
    for col, value in redistributed.items():
        edited[col] = value
    # fraction_water intentionally untouched.

    for index_col, coefs in index_model["coefs"].items():
        if index_col in edited.columns:
            shift = sum(coefs[frac] * change for frac, change in fraction_change.items())
            edited[index_col] = edited[index_col].astype(float) + shift

    original_pred = float(model.predict(original)[0])
    edited_pred = float(model.predict(edited)[0])

    return {
        "original_lst": original_pred,
        "counterfactual_lst": edited_pred,
        "delta_lst": edited_pred - original_pred,
        "requested_delta_vegetation": delta_fraction_vegetation,
        "actual_delta_vegetation": actual_delta,
    }
