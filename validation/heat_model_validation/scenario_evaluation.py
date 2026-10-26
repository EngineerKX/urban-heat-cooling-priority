"""Scenario-modelling evaluation: how well the two heat models predict
temperature, and whether the greening what-if built on them gives plausible,
consistent answers. There is no observed before-and-after greening data, so
the what-if itself cannot be checked against truth; these tests check the
models' accuracy and the what-if's internal consistency instead.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from config.settings import RANDOM_SEED, XGB_LEARNING_RATE, XGB_MAX_DEPTH, XGB_N_ESTIMATORS, XGB_SUBSAMPLE
from src.heat_model.tabular import XGB_FEATURE_COLUMNS, XGB_TARGET_COLUMN, predict_counterfactual_subzone


def _xgb():
    return xgb.XGBRegressor(n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LEARNING_RATE,
                            subsample=XGB_SUBSAMPLE, random_state=RANDOM_SEED)


MODELS = {
    "XGBoost (production settings)": _xgb,
    "Linear regression, same features": LinearRegression,
    "Linear regression, vegetation share only": "veg_only",
    "Predict the training mean": "mean",
}


def _filled(X):
    """Missing elderly share (subzones with no residents) as 0, the sensitivity
    pillar's convention. Only the linear baselines and the distance check need
    it; XGBoost handles missing values itself, as in production."""
    return X.fillna(0)


def _fit_predict(name, X_tr, y_tr, X_te):
    kind = MODELS[name]
    if kind == "mean":
        return np.full(len(X_te), y_tr.mean())
    if kind == "veg_only":
        m = LinearRegression().fit(X_tr[["fraction_vegetation"]], y_tr)
        return m.predict(X_te[["fraction_vegetation"]])
    if kind is LinearRegression:
        return LinearRegression().fit(_filled(X_tr), y_tr).predict(_filled(X_te))
    return kind().fit(X_tr, y_tr).predict(X_te)


def cross_validate(df, groups=None, features=XGB_FEATURE_COLUMNS, models=MODELS, n_repeats=5, n_folds=5) -> pd.DataFrame:
    """R² and RMSE from cross-validation. `groups=None`: repeated random
    5-fold. With `groups` (e.g. planning area): grouped folds, so every
    subzone of a held-out group is predicted by a model that saw none of its
    neighbours -- a stricter, spatial test."""
    X, y = df[features], df[XGB_TARGET_COLUMN]
    rows = []
    for name in models:
        r2s, rmses = [], []
        splits = ([split for rep in range(n_repeats)
                   for split in KFold(n_folds, shuffle=True, random_state=RANDOM_SEED + rep).split(X)]
                  if groups is None else list(GroupKFold(n_folds).split(X, y, groups)))
        for tr, te in splits:
            pred = _fit_predict(name, X.iloc[tr], y.iloc[tr], X.iloc[te])
            r2s.append(r2_score(y.iloc[te], pred))
            rmses.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))
        rows.append({"model": name, "r2_mean": np.mean(r2s), "r2_sd": np.std(r2s),
                     "rmse_mean": np.mean(rmses), "rmse_sd": np.std(rmses), "n_folds": len(r2s)})
    return pd.DataFrame(rows)


def feature_group_ablation(df, groups: dict, n_repeats=5) -> pd.DataFrame:
    """Cross-validated XGBoost R² with each feature group removed or kept alone."""
    rows = []
    for label, cols in groups.items():
        res = cross_validate(df, features=cols, models={"XGBoost (production settings)": _xgb}, n_repeats=n_repeats)
        rows.append({"features": label, "n_features": len(cols), **res.iloc[0].drop("model").to_dict()})
    return pd.DataFrame(rows)


def mahalanobis_reference(df, features=XGB_FEATURE_COLUMNS, q=0.95):
    """Distance function and cut-off for the plausibility check: an edited
    subzone is 'outside the range of real subzones' if its feature vector is
    further from the average real subzone (Mahalanobis distance) than 95% of
    real subzones are. By construction 5% of real subzones exceed it."""
    X = _filled(df[features].astype(float)).values
    mean, cov_inv = X.mean(axis=0), np.linalg.pinv(np.cov(X, rowvar=False))

    def dist(Z):
        d = np.nan_to_num(np.asarray(Z, dtype=float), nan=0.0) - mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, cov_inv, d))
    return dist, float(np.quantile(dist(X), q))


def what_if(df, model, index_model, delta, dist, cutoff, features=XGB_FEATURE_COLUMNS) -> pd.DataFrame:
    """Apply the greening edit (+`delta` vegetation share) to every subzone
    and return per-subzone temperature change plus the plausibility flag."""
    from src.heat_model.tabular import redistribute_vegetation_fraction

    rows = []
    for i in range(len(df)):
        row = df.iloc[[i]]
        res = predict_counterfactual_subzone(model, row, delta, index_model, features)
        # Rebuild the edited feature vector the same way, for the plausibility check.
        edited = row[features].astype(float).copy()
        red = redistribute_vegetation_fraction(*(float(row[c].iloc[0]) for c in
                                                 ("fraction_vegetation", "fraction_built_up", "fraction_bare")), delta)
        change = {}
        for col in ("fraction_vegetation", "fraction_built_up", "fraction_bare"):
            change[col] = red[col] - float(row[col].iloc[0])
            edited[col] = red[col]
        for idx_col, coefs in index_model["coefs"].items():
            edited[idx_col] = edited[idx_col] + sum(coefs[f] * c for f, c in change.items())
        rows.append({"subzone_id": row["subzone_id"].iloc[0], "delta_lst": res["delta_lst"],
                     "actual_delta_vegetation": res["actual_delta_vegetation"],
                     "outside_real_range": bool(dist(edited[features].values)[0] > cutoff)})
    return pd.DataFrame(rows)


def summarise_what_if(per_subzone: pd.DataFrame) -> dict:
    d = per_subzone["delta_lst"]
    return {"share_cooling": float((d < 0).mean()), "median_delta_c": float(d.median()),
            "p10_delta_c": float(d.quantile(0.1)), "p90_delta_c": float(d.quantile(0.9)),
            "share_outside_real_range": float(per_subzone["outside_real_range"].mean())}


def subset_index_model(index_model, keep):
    """The production index model restricted to the index columns in `keep`
    (an empty `keep` leaves every spectral index unchanged by the edit)."""
    return {"coefs": {k: v for k, v in index_model["coefs"].items() if k in keep}, "r2": index_model.get("r2", {})}


def cross_subzone_slope(df) -> float:
    """Ordinary least-squares slope of temperature on vegetation share across
    real subzones, in °C per unit share: the association the what-if should
    roughly match in size."""
    return float(np.polyfit(df["fraction_vegetation"], df[XGB_TARGET_COLUMN], 1)[0])
