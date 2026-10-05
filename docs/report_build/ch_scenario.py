"""Scenario-modelling chapter: the two heat models (XGBoost per subzone, CNN
per 10 m patch) and the greening what-if built on them. Numbers come from
data/processed/diagnostics/scenario_*.csv (scripts/evaluate_scenario_models.py)
and data/processed/heat_model/xgb_index_landcover_model.json.
"""

import json

import pandas as pd

from helpers import pct, signed

TITLE = "Scenario Modelling"
STANDALONE_TITLE = "Scenario Modelling Evaluation and Ablation Tests"
SUBTITLE = "XGBoost and CNN heat models and the greening what-if"


def build(doc, repo):
    D = repo / "data/processed/diagnostics"
    prod = pd.read_csv(D / "scenario_xgb_production_split.csv").iloc[0]
    cv = pd.read_csv(D / "scenario_xgb_cv.csv")
    feat = pd.read_csv(D / "scenario_feature_ablation.csv").set_index("features")
    mech = pd.read_csv(D / "scenario_mechanism.csv").set_index("mechanism")
    dose = pd.read_csv(D / "scenario_dose_response.csv")
    assoc = pd.read_csv(D / "scenario_association.csv").iloc[0]
    cnn = pd.read_csv(D / "scenario_cnn_validation.csv").set_index("model")
    agree = pd.read_csv(D / "scenario_model_agreement.csv")
    index_model = json.loads((repo / "data/processed/heat_model/xgb_index_landcover_model.json").read_text())

    rnd = cv[cv["validation"].str.startswith("Random")].set_index("model")
    spa = cv[cv["validation"].str.startswith("Spatial")].set_index("model")
    XG, LIN, VEG, MEAN = ("XGBoost (production settings)", "Linear regression, same features",
                          "Linear regression, vegetation share only", "Predict the training mean")
    PROD_MECH = "Land cover + NDVI + NDBI (production)"
    m15 = mech.loc[PROD_MECH]
    c, lin_px, mean_px = (cnn.loc["CNN (production)"], cnn.loc["Per-pixel linear regression on the same input channels"],
                          cnn.loc["Predict the training mean"])
    top = agree[agree["group"] == "Residents top 20"]
    demo = agree[agree["group"] != "Residents top 20"]
    from scipy.stats import spearmanr
    mag_corr = spearmanr(top["xgb_delta_c"], top["cnn_delta_in_edit_area_c"])[0]
    disagree = ", ".join(top.loc[~top["same_direction"], "subzone_id"].str.title())
    cnn_larger = int((top["cnn_delta_in_edit_area_c"].abs() > top["xgb_delta_c"].abs()).sum())
    flat = list(top.loc[top["xgb_delta_c"] > -0.1, "subzone_id"].str.title())
    xgb_flat = " and ".join([", ".join(flat[:-1]), flat[-1]]) if len(flat) > 1 else "".join(flat)

    def f2(x):
        return f"{0.0 if abs(x) < 0.005 else x:.2f}"
    r2_idx = index_model.get("r2", {})

    doc.title(TITLE if doc.chapter else STANDALONE_TITLE, SUBTITLE)

    # --- Summary ------------------------------------------------------------------
    doc.section("Summary", numbered=False)
    doc.para(
        "Scenario modelling answers \"what if this place were greener?\". A subzone-level XGBoost model predicts "
        "surface temperature from land cover, spectral indices and population; a 10 m CNN predicts temperature "
        "patch by patch. The greening what-if edits a subzone's or a patch's inputs and reads off the change in "
        "predicted temperature. No greening intervention was observed before and after, so the what-if cannot "
        "be checked against truth; the models' accuracy and the what-if's consistency can."
    )
    doc.para(
        f"XGBoost is accurate at subzone level (R² {prod['r2']:.2f} on held-out subzones, "
        f"{spa.loc[XG, 'r2_mean']:.2f} with whole planning areas held out), but only slightly better than a "
        f"linear regression on the same inputs ({spa.loc[LIN, 'r2_mean']:.2f}). Its what-if is consistent: adding "
        f"15 percentage points of vegetation cools {pct(m15['share_cooling'], 0)} of subzones by a median "
        f"{abs(m15['median_delta_c']):.2f} °C, close to the cross-subzone association "
        f"({abs(assoc['implied_c_for_delta']):.2f} °C), and only because the edit moves the spectral indices with "
        f"the land cover. The CNN follows the spatial pattern of temperature (correlation {c['correlation']:.2f}) "
        f"but exaggerates its contrasts by about {pct(c['spread_ratio'] - 1, 0)}, so its held-out error "
        f"({c['rmse_c']:.2f} °C) is worse than predicting the average. The two models agree on the direction of "
        f"the greening effect for {int(top['same_direction'].sum())} of the top-20 subzones but not on its size. "
        "What-if results are indicative directions, not forecasts."
    )

    # --- Setup ------------------------------------------------------------------------
    doc.section("Setup")
    doc.table(
        ["Item", "Setting"],
        [
            ["XGBoost", f"One row per subzone ({int(prod['n_train'] + prod['n_test'])}). Ten features: land-cover "
                        "fractions (vegetation, built-up, bare, water), NDVI and NDBI in two seasons, resident "
                        "count and elderly share. Target: Landsat 30 m surface temperature. Production model "
                        f"trained on {int(prod['n_train'])} subzones, tested on {int(prod['n_test'])}."],
            ["CNN", f"{int(c['n_val_patches'])} of 1,066 patches (128 × 128 pixels at 10 m) held out for "
                    "validation (85/15 split by patch). Inputs: Sentinel-2 bands, spectral indices and the "
                    "production land-cover map. Target: the 10 m bicubic temperature layer."],
            ["Subzone what-if (XGBoost)", "Vegetation share raised by a set amount, taken proportionally from "
                                          "built-up and bare land; water unchanged. NDVI and NDBI move with "
                                          "the land cover by regressions fitted across real subzones "
                                          f"(R² {min(r2_idx.values()):.2f} to {max(r2_idx.values()):.2f})."],
            ["Patch what-if (CNN)", "A 50 m-radius disc around the subzone centre is set to the average "
                                    "vegetation pixel, and the CNN is rerun on the edited patch."],
            ["Validation", "Repeated 5-fold cross-validation (25 fits) and spatial cross-validation holding out "
                           "whole planning areas; baselines: linear regression and predicting the mean."],
        ],
        [2400, 7500],
    )

    # --- Evaluation ------------------------------------------------------------------
    doc.section("Evaluation")
    doc.subsection("XGBoost accuracy")
    rows = []
    for name, label in ((XG, "XGBoost (production settings)"), (LIN, "Linear regression, same ten features"),
                        (VEG, "Linear regression, vegetation share only"), (MEAN, "Predict the training mean")):
        rows.append([label, f"{f2(rnd.loc[name, 'r2_mean'])} ± {rnd.loc[name, 'r2_sd']:.2f}",
                     f"{rnd.loc[name, 'rmse_mean']:.2f}", f"{f2(spa.loc[name, 'r2_mean'])} ± {spa.loc[name, 'r2_sd']:.2f}",
                     f"{spa.loc[name, 'rmse_mean']:.2f}"])
    doc.table(["Model", "R², random CV", "RMSE (°C)", "R², planning areas held out", "RMSE (°C)"], rows,
              [3300, 1700, 1200, 2400, 1300])
    doc.para(
        f"The production model scores R² {prod['r2']:.3f} (RMSE {prod['rmse_c']:.2f} °C) on its own held-out "
        f"subzones; cross-validation gives a similar {rnd.loc[XG, 'r2_mean']:.2f}. Holding out whole planning "
        "areas barely lowers it, so the accuracy does not come from neighbouring subzones leaking into the "
        f"test set. XGBoost halves the error of predicting the mean, but a linear regression on the same "
        f"features does almost as well ({rnd.loc[LIN, 'r2_mean']:.2f}): the relationship is largely linear, "
        "and the choice of XGBoost is not what makes the model accurate."
    )
    doc.subsection("CNN accuracy")
    doc.table(
        ["Model (validation patches)", "RMSE (°C)", "R²", "Bias (°C)", "Correlation", "Spread ratio"],
        [["CNN (production)", f"{c['rmse_c']:.2f}", f"{c['r2']:.2f}", signed(c["bias_c"]),
          f"{c['correlation']:.2f}", f"{c['spread_ratio']:.2f}"],
         ["Per-pixel linear regression, same inputs", f"{lin_px['rmse_c']:.2f}", f"{lin_px['r2']:.2f}",
          signed(lin_px["bias_c"]), f"{lin_px['correlation']:.2f}", f"{lin_px['spread_ratio']:.2f}"],
         ["Predict the training mean", f"{mean_px['rmse_c']:.2f}", f2(mean_px["r2"]), signed(mean_px["bias_c"]),
          "—", "0"]],
        [3300, 1200, 1100, 1300, 1500, 1500],
    )
    doc.para(
        f"On {int(c['n_val_pixels']):,} validation pixels the CNN's error ({c['rmse_c']:.2f} °C) is larger than "
        f"the spread of the temperatures themselves ({c['target_sd_c']:.2f} °C), so it does worse than predicting "
        f"the average everywhere. Its overall bias is small ({signed(c['bias_c'])} °C) and it does follow the "
        f"pattern (correlation {c['correlation']:.2f}), but its predictions vary {c['spread_ratio']:.2f} times as "
        "much as the real temperatures: hot areas come out too hot and cool areas too cool. A per-pixel linear "
        f"regression on the same inputs is far more accurate ({lin_px['rmse_c']:.2f} °C). The CNN's temperature "
        "values should not be read as estimates, and its what-if effects are likely to be too large."
    )
    doc.subsection("Consistency of the greening what-if")
    doc.table(
        ["Vegetation added", "Subzones that cool", "Median change (°C)", "10th–90th percentile (°C)",
         "Edited subzones outside the real range"],
        [[f"+{int(round(r.delta_vegetation * 100))} points", pct(r.share_cooling, 0), signed(r.median_delta_c),
          f"{signed(r.p10_delta_c)} to {signed(r.p90_delta_c)}", pct(r.share_outside_real_range, 1)]
         for r in dose.itertuples()],
        [1800, 1700, 1800, 2400, 2200],
    )
    doc.para(
        "Cooling grows steadily with the amount of greening and becomes nearly universal. At +15 points the "
        f"median cooling ({abs(m15['median_delta_c']):.2f} °C) is close to what the cross-subzone association "
        f"implies ({abs(assoc['implied_c_for_delta']):.2f} °C, from {assoc['slope_c_per_unit_vegetation']:.1f} °C "
        "per unit of vegetation share). An edited subzone counts as outside the real range when its features are "
        "further from the average subzone than 95% of real subzones are; edited subzones fall outside about as "
        f"often as real ones do ({pct(m15['share_outside_real_range'], 1)} against "
        f"{pct(m15['real_subzones_outside_range'], 1)}), so the edits stay within the data the model learned from."
    )
    doc.subsection("Agreement between the two models")
    doc.para(
        f"For each of the residents' top-20 subzones, XGBoost predicted the effect of +15 points of vegetation "
        "and the CNN the effect of greening a 50 m disc at the subzone centre. They agree on the direction for "
        f"{int(top['same_direction'].sum())} of {len(top)} (not for {disagree}) and for all "
        f"{len(demo)} diagnostic demo subzones. They do not agree on size: the rank correlation of the two "
        f"estimates is {mag_corr:.2f}, and the CNN's effect is the larger in {cnn_larger} of {len(top)}. XGBoost predicts essentially no cooling (above −0.1 °C) for {xgb_flat}."
    )

    # --- Ablations ----------------------------------------------------------------------
    doc.section("Ablations and controlled comparisons")
    doc.subsection("How the what-if edit moves the inputs")
    doc.table(
        ["Edit mechanism (+15 points vegetation)", "Subzones that cool", "Median change (°C)",
         "Edited subzones outside the real range"],
        [[name, pct(r.share_cooling, 0), signed(r.median_delta_c), pct(r.share_outside_real_range, 1)]
         for name, r in mech.iterrows()],
        [4300, 1800, 1800, 2000],
    )
    lc_only, ndvi = mech.iloc[0], mech.iloc[1]
    doc.para(
        f"Changing land cover alone cools only {pct(lc_only['share_cooling'], 0)} of subzones, by a median "
        f"{abs(lc_only['median_delta_c']):.2f} °C, so barely more than half; moving NDVI as well barely helps. Only "
        f"when NDBI moves too does the effect become consistent ({pct(m15['share_cooling'], 0)} cool). The model "
        "relies on the spectral indices more than on the land-cover fractions (next table), so an edit that "
        "changes the fractions but leaves the indices of the original place describes an unrealistic subzone, "
        "which the plausibility check also shows."
    )
    doc.subsection("XGBoost feature groups")
    doc.table(
        ["Features", "Number", "R², random CV", "RMSE (°C)"],
        [[f, str(int(r.n_features)), f"{r.r2_mean:.2f} ± {r.r2_sd:.2f}", f"{r.rmse_mean:.2f}"] for f, r in feat.iterrows()],
        [4500, 1300, 2300, 1800],
    )
    f_ = feat["r2_mean"]
    doc.para(
        f"The spectral indices carry most of the information (alone, R² {f_['Spectral indices only']:.2f}); the "
        f"land-cover fractions alone reach {f_['Land-cover fractions only']:.2f}. Removing population changes "
        f"almost nothing ({f_['Without population']:.2f} against {f_['All ten (production)']:.2f})."
    )
    doc.subsection("XGBoost against linear regression")
    doc.para(
        f"With the same features, linear regression comes within {rnd.loc[XG, 'r2_mean'] - rnd.loc[LIN, 'r2_mean']:.2f} "
        "in R² of XGBoost under both validation schemes (first table). The gradient-boosted model adds little "
        "accuracy; its value is flexibility, not a large gain in fit."
    )

    # --- Relation to the ranking ------------------------------------------------------------
    doc.section("Relation to the priority ranking")
    doc.para(
        "Scenario modelling does not enter the priority score, so it has no decision-impact test. It answers a "
        "follow-up question for the subzones the score selects: how much greening might help there."
    )

    # --- Limitations --------------------------------------------------------------------------
    doc.section("Limitations")
    for lead, text in [
        ("Association, not causation. ", "XGBoost learned how temperature differs between subzones with "
                                         "different greenery; it was never shown a place that was greened. "
                                         "What-if results are what that association implies, not forecasts."),
        ("Agreement with the association is partly built in. ", "The edit moves the indices along relationships "
                                                                "fitted across subzones, so agreeing with the "
                                                                "cross-subzone slope shows consistency, not "
                                                                "external validity."),
        ("CNN target. ", "The CNN learns the 10 m bicubic temperature layer, which is interpolated from Landsat's "
                         "30 m product (itself about 100 m measurement), so it cannot learn real 10 m detail."),
        ("CNN validation split. ", "Patches are split at random, so validation patches have training neighbours. "
                                   "This would flatter the CNN, which still does poorly."),
        ("Reproducibility. ", "Retraining XGBoost on another machine, or with a different number of processor "
                              "threads, moves individual predictions by a few tenths of a degree; the reported "
                              "model is the production one."),
        ("Surface temperature. ", "Both models predict daytime land surface temperature, not air temperature or "
                                  "thermal comfort."),
    ]:
        doc.lead(lead, text)

    doc.section("Sources", numbered=False)
    doc.para(
        "Models: scripts/train_heat_model_xgboost.py, src/heat_model/tabular.py, "
        "notebooks/colab_training/train_heat_cnn.ipynb, src/heat_model/cnn_*.py, src/heat_model/counterfactual.py. "
        "Evaluation: scripts/evaluate_scenario_models.py, validation/heat_model_validation/scenario_evaluation.py "
        "(outputs in data/processed/diagnostics/scenario_*.csv)."
    )
