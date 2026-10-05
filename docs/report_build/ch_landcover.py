"""Land-cover chapter: RF, U-Net and hybrid classifiers on Dynamic World labels.

Built on the teammate's report (docs/landcover_model_evaluation_report.docx,
30 September 2026). Numbers come from:
  data/processed/landcover/evaluation/final_batch_metrics.csv, final_batch_mcnemar_p.csv
      (MLflow batch landcover_eval_final_2026-09-29c, exported)
  data/processed/landcover/evaluation/mcnemar_counts.csv, unet_confusion_matrix.csv,
      unet_per_class.csv, vegetation_binary.csv   (scripts/evaluate_landcover_supplementary.py)
  data/processed/landcover_swap_comparison.csv, landcover_label_source_swap.csv
  data/processed/priority_score_confidence_bands.csv (noise floor)
  data/interim/validation_sample/validation_sample_300_labeled.csv
The map-comparison figures (section on label-source evaluation) are quoted from
docs/dynamic_world_vs_worldcover_2026-09-27.md, which has no saved table.
"""

import pandas as pd

from helpers import pct

TITLE = "Land-cover Classification"
STANDALONE_TITLE = "Land-cover Model Evaluation and Ablation Tests"
SUBTITLE = "RF, U-Net and RF + U-Net hybrid, trained on Dynamic World labels"


def build(doc, repo):
    lc_eval = repo / "data/processed/landcover/evaluation"
    m = pd.read_csv(lc_eval / "final_batch_metrics.csv").set_index("run")
    mc = pd.read_csv(lc_eval / "mcnemar_counts.csv").set_index("comparison")
    p_all = pd.read_csv(lc_eval / "final_batch_mcnemar_p.csv").set_index("comparison")["p_value"]
    cm = pd.read_csv(lc_eval / "unet_confusion_matrix.csv", index_col=0)
    per_class = pd.read_csv(lc_eval / "unet_per_class.csv").set_index("class")
    veg = pd.read_csv(lc_eval / "vegetation_binary.csv").set_index("model")
    swap = pd.read_csv(repo / "data/processed/landcover_label_source_swap.csv")
    # The NDVI-threshold row comes from the production swap table (same reference, same score).
    ndvi = pd.read_csv(repo / "data/processed/landcover_swap_comparison.csv")
    ndvi = ndvi[ndvi["greenery_source"] == "NDVI-threshold proxy"].assign(greenery_source="NDVI threshold (no classifier)")
    swap = pd.concat([swap, ndvi], ignore_index=True)
    bands = pd.read_csv(repo / "data/processed/priority_score_confidence_bands.csv")
    labels = pd.read_csv(repo / "data/interim/validation_sample/validation_sample_300_labeled.csv")

    top = bands.sort_values("priority_score_point", ascending=False).head(20)
    floor = 20 - top["p_top20"].sum()
    counts = labels["agreed_label"].value_counts()

    def acc(run):
        return pct(m.loc[run, "accuracy"])

    def gain(a, b):
        """Accuracy change in points, from the rounded values shown beside it."""
        return f"{round(m.loc[b, 'accuracy'] * 100, 1) - round(m.loc[a, 'accuracy'] * 100, 1):+.1f} points"

    def ci(run):
        return f"{m.loc[run, 'accuracy_ci95_low'] * 100:.1f}–{m.loc[run, 'accuracy_ci95_high'] * 100:.1f}"

    def p_fmt(p):
        return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"

    def mcn(name):
        r = mc.loc[name]
        return f"{int(r.only_b_correct)} vs {int(r.only_a_correct)}; p {p_fmt(r.p_value)}"

    doc.title(TITLE if doc.chapter else STANDALONE_TITLE, SUBTITLE)

    # --- Executive summary -------------------------------------------------
    doc.section("Summary", numbered=False)
    doc.para(
        "Land cover feeds the priority score through one number per subzone: its vegetation share, which sets "
        "the greenery (adaptive-capacity) pillar. Three classifiers were trained on Dynamic World labels and "
        "scored against 300 hand-labelled points: a random forest (RF), a U-Net, and a hybrid that averages "
        "their class probabilities."
    )
    doc.para(
        f"The U-Net has the highest accuracy ({acc('unet_dw')}, 95% interval {ci('unet_dw')}%) and is the "
        f"production model. It is significantly better than RF ({acc('rf_dw_raw')}) but not distinguishable from "
        f"the hybrid ({acc('hybrid_dw_rfraw')}). Switching the training labels from ESA WorldCover to Dynamic "
        "World improved all three models significantly. Texture features, smoothing RF's output and adding RF "
        "to the U-Net showed no significant benefit. For the planning output, no land-cover choice changes more "
        f"of the top-20 subzones than measurement noise alone (noise floor {floor:.1f}); among classifiers, the "
        "label-source switch has the largest effect."
    )

    # --- Setup ---------------------------------------------------------------
    doc.section("Setup")
    doc.table(
        ["Item", "Setting"],
        [
            ["Test points",
             f"300 points labelled by hand on high-resolution imagery: {counts.get('built_up', 0)} built-up, "
             f"{counts.get('vegetation', 0)} vegetation, {counts.get('water', 0)} water, {counts.get('bare', 0)} "
             f"bare; {300 - counts.reindex(['built_up', 'vegetation', 'water', 'bare']).sum()} flagged uncertain and "
             "excluded."],
            ["Sampling", "Drawn by WorldCover class, with vegetation over-sampled. Accuracies are unweighted "
                         "over the sample, not area-weighted estimates for the whole island."],
            ["Independence", "Every model's training data excludes a 15 m buffer around every test point."],
            ["Label quality", "Two labellers agreed with kappa 0.676 on a 20-point trial."],
            ["Points scored", f"{int(m.loc['rf_dw_raw', 'n_scored'])} for RF; "
                              f"{int(m.loc['unet_dw', 'n_scored'])} for the U-Net and hybrid (3 points fall "
                              "outside the U-Net's mapped area)."],
            ["Statistics", "95% accuracy intervals from 4,000 bootstrap resamples; exact McNemar tests on "
                           "paired predictions."],
            ["Record", "MLflow experiment landcover_classifiers, batch landcover_eval_final_2026-09-29c. "
                       "Production map: U-Net, SHA-256 dc33f409…"],
        ],
        [2300, 7600],
    )

    # --- Evaluation ----------------------------------------------------------
    doc.section("Evaluation of the final models")
    doc.table(
        ["Model (Dynamic World labels)", "Accuracy", "95% interval", "Macro F1", "Role"],
        [["U-Net", acc("unet_dw"), ci("unet_dw"), f"{m.loc['unet_dw', 'macro_f1']:.3f}", "Production model"],
         ["Hybrid (50% RF, 50% U-Net)", acc("hybrid_dw_rfraw"), ci("hybrid_dw_rfraw"),
          f"{m.loc['hybrid_dw_rfraw', 'macro_f1']:.3f}", "Comparison"],
         ["RF", acc("rf_dw_raw"), ci("rf_dw_raw"), f"{m.loc['rf_dw_raw', 'macro_f1']:.3f}", "Baseline"]],
        [3000, 1400, 1700, 1400, 2400],
    )
    doc.para(
        f"U-Net against RF: {mcn('rf_vs_unet')} (points correct only for the U-Net vs only for RF). U-Net against "
        f"the hybrid: {mcn('hybrid_vs_unet')}, not significant. The U-Net was chosen for its higher observed "
        "accuracy and simpler single-model deployment; the test set cannot separate it from the hybrid."
    )
    doc.para(
        "The U-Net's 89.8% validation accuracy during training measures agreement with Dynamic World labels on "
        "held-out patches, not with human labels, and is not reported as accuracy."
    )
    doc.subsection("Per-class results for the production U-Net")
    order = ["vegetation", "built_up", "water", "bare"]
    names = {"vegetation": "Vegetation", "built_up": "Built-up", "water": "Water", "bare": "Bare"}
    doc.table(
        ["Class", "Precision", "Recall", "F1", "Test points"],
        [[names[c], f"{per_class.loc[c, 'precision']:.2f}", f"{per_class.loc[c, 'recall']:.2f}",
          f"{per_class.loc[c, 'f1']:.2f}", str(int(per_class.loc[c, "support"]))] for c in order],
        [2900, 1750, 1750, 1750, 1750],
    )
    doc.table(
        ["True class \\ predicted"] + [names[c] for c in order],
        [[names[t]] + [str(int(cm.loc[f"true:{t}", f"pred:{p}"])) for p in order] for t in order],
        [2900, 1750, 1750, 1750, 1750],
    )
    bu_as_veg = int(cm.loc["true:built_up", "pred:vegetation"])
    doc.para(
        f"The largest error is built-up land read as vegetation ({bu_as_veg} of "
        f"{int(per_class.loc['built_up', 'support'])} built-up points). It matters downstream because it "
        "inflates the greenery pillar."
    )
    doc.subsection("Vegetation against everything else")
    doc.para(
        "Only the vegetation share of each subzone reaches the priority score, so a classifier's errors among "
        "built-up, bare and water do not affect the ranking. This table scores that distinction alone."
    )
    vrows = [("U-Net, Dynamic World", "unet_dw"), ("Hybrid, Dynamic World", "hybrid_dw_rfraw"),
             ("RF, Dynamic World", "rf_dw_raw"), ("U-Net, WorldCover", "unet_worldcover"),
             ("Hybrid, WorldCover", "hybrid_worldcover"), ("RF, WorldCover", "rf_worldcover")]
    doc.table(
        ["Model", "Accuracy", "Vegetation precision", "Vegetation recall", "Vegetation F1", "Points called vegetation"],
        [[lab, pct(veg.loc[r, "accuracy"]), f"{veg.loc[r, 'vegetation_precision']:.2f}",
          f"{veg.loc[r, 'vegetation_recall']:.2f}", f"{veg.loc[r, 'vegetation_f1']:.2f}",
          pct(veg.loc[r, "predicted_vegetation_share"], 0)] for lab, r in vrows],
        [2500, 1300, 1600, 1500, 1400, 1600],
    )
    dw = veg.loc[["unet_dw", "hybrid_dw_rfraw", "rf_dw_raw"], "predicted_vegetation_share"]
    wc = veg.loc[["unet_worldcover", "hybrid_worldcover", "rf_worldcover"], "predicted_vegetation_share"]
    doc.para(
        f"About {pct(veg.loc['unet_dw', 'true_vegetation_share'], 0)} of the test points are vegetation. The "
        f"Dynamic World models call {pct(dw.min(), 0)} to {pct(dw.max(), 0)} of points vegetation; the WorldCover "
        f"models call {pct(wc.min(), 0)} to {pct(wc.max(), 0)}, which is why their greenery estimates are much higher."
    )

    # --- Label-source evaluation (map comparison) --------------------------
    doc.section("Label-source evaluation (reference maps)")
    doc.para(
        "Before retraining, the two candidate label maps were scored directly against the hand labels. This is a "
        "map-quality comparison, not a model ablation. Agreement is weighted to Singapore's real class areas "
        "(the sample was drawn by WorldCover class), on the 293 points both maps cover, with 95% intervals from "
        "resampling within each stratum."
    )
    doc.table(
        ["Reference map", "Agreement with hand labels", "95% interval"],
        [["ESA WorldCover 2021", "69.5%", "65–74%"],
         ["Dynamic World, around 2021", "81.9%", "77–86%"]],
        [4400, 3000, 2500],
    )
    doc.para(
        "Dynamic World agrees 12.4 points better (interval 8 to 17). The gain is concentrated in built-up land: "
        "a separate check found WorldCover labelled only 47% of hand-labelled built-up points correctly, against "
        "78% for Dynamic World. The comparison used a window around 2021; final training uses the fixed window "
        "1 January 2021 to 1 July 2026. Map agreement and model accuracy use different scoring and samples, so "
        "each is read within its own comparison."
    )

    # --- Ablations ---------------------------------------------------------------
    doc.section("Ablations and controlled comparisons")
    doc.para(
        "Each comparison changes one component and scores both versions on the same points. McNemar counts are "
        "the points correct only under the second version against only under the first. A non-significant "
        "result does not establish equivalence."
    )
    doc.subsection("Training-label source")
    lrows = [("RF", "rf_worldcover", "rf_dw_raw", "worldcover_vs_dynamicworld_rf"),
             ("U-Net", "unet_worldcover", "unet_dw", "worldcover_vs_dynamicworld_unet"),
             ("Hybrid (50/50)", "hybrid_worldcover", "hybrid_dw_rfraw", "worldcover_vs_dynamicworld_hybrid")]
    doc.table(
        ["Model", "WorldCover", "Dynamic World", "Change", "McNemar (counts; p)"],
        [[lab, acc(a), acc(b), gain(a, b), mcn(c)]
         for lab, a, b, c in lrows],
        [1900, 1500, 1700, 1700, 3100],
    )
    doc.para(
        f"All three favour Dynamic World significantly. For the U-Net, built-up F1 rose from "
        f"{m.loc['unet_worldcover', 'built_up_f1']:.2f} to {m.loc['unet_dw', 'built_up_f1']:.2f} and bare-land "
        f"F1 from {m.loc['unet_worldcover', 'bare_f1']:.2f} to {m.loc['unet_dw', 'bare_f1']:.2f}; the WorldCover "
        "U-Net never predicted bare land correctly."
    )
    doc.subsection("Features, post-processing and model components")
    doc.table(
        ["Component tested", "Comparison", "Accuracy", "p", "Conclusion"],
        [
            ["Texture features", "RF vs RF + 3 × 3 mean and standard deviation of NDVI, NDBI, NDWI",
             f"{acc('rf_dw_raw')} vs {acc('rf_dw_texture')}", p_fmt(p_all["rf_raw_vs_texture"]),
             "No detected improvement."],
            ["RF smoothing", "RF vs RF with 3 × 3 majority smoothing (30 m), window fixed in advance",
             f"{acc('rf_dw_raw')} vs {acc('rf_dw_smooth3x3')}", p_fmt(p_all["rf_raw_vs_smooth3x3"]),
             f"{gain('rf_dw_raw', 'rf_dw_smooth3x3')}, not significant."],
            ["RF in the hybrid", "Hybrid vs U-Net alone",
             f"{acc('hybrid_dw_rfraw')} vs {acc('unet_dw')}", p_fmt(p_all["hybrid_vs_unet"]),
             "No demonstrated benefit from adding RF."],
            ["Smoothing in the hybrid", "Hybrid with raw RF vs with smoothed RF",
             f"{acc('hybrid_dw_rfraw')} vs {acc('hybrid_dw_rfsmooth3x3')}", p_fmt(p_all["hybrid_rfraw_vs_rfsmooth"]),
             "No significant difference."],
        ],
        [1800, 3200, 1700, 900, 2300],
    )
    doc.para("No U-Net architecture ablation or input-band removal experiment was run.")

    # --- Decision impact --------------------------------------------------------
    doc.section("Decision impact")
    doc.para(
        "The priority score was rebuilt with the greenery pillar taken from each classifier's map, everything "
        "else fixed, and compared with the production score (U-Net, Dynamic World)."
    )
    doc.table(
        ["Greenery source", "Top-20 subzones changed", "Rank correlation with production", "Mean greenery share"],
        [[r.greenery_source, str(int(r.top20_changed)), f"{r.score_spearman_vs_reference:.3f}",
          pct(r.greenery_mean, 1)] for r in swap.itertuples()],
        [3700, 2000, 2400, 1800],
    )
    doc.para(
        f"Every alternative changes fewer top-20 subzones than measurement noise alone ({floor:.1f}). The "
        "replacement of the classifier by a simple NDVI threshold has the largest effect; among classifiers, every "
        "WorldCover-trained map moves 6 subzones and more than doubles the mean greenery share, while the choice "
        "among Dynamic World models moves at most 4. Two "
        "properties of the score limit the impact: only the vegetation share reaches it, and a constant "
        "over- or under-estimate of greenery is removed by the score's rescaling, so only errors that vary "
        "between subzones can move a rank."
    )

    # --- Exploratory ---------------------------------------------------------------
    doc.section("Exploratory sensitivity analyses")
    doc.para(
        "These sweeps compared settings on the same test points, so choosing the best-scoring setting would tune "
        "the method to the test set. They show sensitivity only."
    )
    windows = [("3 × 3 (fixed in advance)", "rf_dw_smooth3x3"), ("5 × 5", "rf_dw_smooth5x5"),
               ("7 × 7", "rf_dw_smooth7x7"), ("11 × 11", "rf_dw_smooth11x11"), ("21 × 21 (about 210 m)", "rf_dw_smooth21x21")]
    blends = [("0.5 (standard hybrid)", "hybrid_dw_rfraw"), ("0.6", "hybrid_dw_unetweight0.6"),
              ("0.7", "hybrid_dw_unetweight0.7"), ("0.8", "hybrid_dw_unetweight0.8"),
              ("0.9", "hybrid_dw_unetweight0.9"), ("1.0 (U-Net alone)", "unet_dw")]
    doc.table(["RF majority-smoothing window", "Accuracy"], [[w, acc(r)] for w, r in windows], [6400, 3500])
    doc.para("Larger windows remove speckle but also spatial detail; their higher test accuracy does not justify choosing them.")
    doc.table(["U-Net weight in the hybrid", "Accuracy"], [[w, acc(r)] for w, r in blends], [6400, 3500])
    doc.para(
        "Accuracy rises or holds as the U-Net weight increases, so the sweep shows no advantage in adding RF on "
        "these points; it does not show RF could never help on other data."
    )

    # --- Limitations ---------------------------------------------------------------
    doc.section("Limitations")
    for lead, text in [
        ("Small classes. ", f"Bare land ({counts.get('bare', 0)} points) and water ({counts.get('water', 0)} points) "
                            "are reported but too small to rank methods on."),
        ("Unweighted accuracy. ", "The test sample over-represents vegetation, so model accuracies are not "
                                  "island-wide map accuracies."),
        ("Model-made labels. ", "Dynamic World is itself a model output, so models trained on it can inherit its "
                                "blind spots, and the hand labels and Dynamic World may share biases on mixed pixels."),
        ("Repeated testing. ", "Many comparisons reuse the same 300 points with no multiple-comparison "
                               "adjustment, so p-values near 0.05 are read with caution."),
    ]:
        doc.lead(lead, text)

    doc.section("Sources", numbered=False)
    doc.para(
        "Training: scripts/train_landcover_rf.py, notebooks/colab_training/train_unet.ipynb, "
        "scripts/build_landcover_hybrid.py. Evaluation: scripts/log_landcover_evaluation_mlflow.py (MLflow batch "
        "landcover_eval_final_2026-09-29c), scripts/evaluate_landcover_supplementary.py. Decision impact: "
        "scripts/build_landcover_swap_comparison.py and scripts/evaluate_landcover_supplementary.py "
        "(data/processed/landcover_label_source_swap.csv). Map comparison: scripts/compare_worldcover_dynamicworld.py and "
        "docs/dynamic_world_vs_worldcover_2026-09-27.md. Test points: "
        "data/interim/validation_sample/validation_sample_300_labeled.csv."
    )
