"""Evaluation strategy chapter: the framework every component chapter follows.
Methods only; results belong in the component chapters."""

TITLE = "Evaluation Strategy"
STANDALONE_TITLE = "Evaluation Strategy and Ablation Testing"
SUBTITLE = "Framework for the five pipeline components"


def build(doc, repo):
    doc.title(TITLE if doc.chapter else STANDALONE_TITLE, SUBTITLE)

    doc.section("Summary", numbered=False)
    doc.para(
        "The project turns satellite and census data into a ranking of URA subzones by cooling priority. "
        "It has five components: heat-map downscaling, land-cover classification, heat typology, the "
        "cooling-priority score, and scenario modelling. This chapter sets out one evaluation framework "
        "that all five component reports follow, so their results can be read side by side."
    )
    doc.para(
        "Each component is assessed at two levels. The first asks how good the component is on its own: "
        "its accuracy against an independent reference, or, where no reference exists, its internal "
        "consistency and robustness. The second asks whether it matters for the output: if the component "
        "or one of its design choices is changed, does the list of the 20 highest-priority subzones "
        "change by more than measurement noise alone would change it? This second level is the project's "
        "decision-focused evaluation."
    )

    doc.section("Kinds of test")
    doc.para(
        "Every result in the component reports is labelled as one of four kinds of test. The labels "
        "matter because they support different claims."
    )
    doc.table(
        ["Kind of test", "What it does", "What it can support"],
        [
            ["Evaluation",
             "Scores a final component against a reference it was not built from.",
             "How well the component performs."],
            ["Ablation or controlled comparison",
             "Changes one element (a feature group, a label source, a model, a weighting) and keeps "
             "everything else fixed, then scores both versions on the same test set.",
             "Whether that element contributes, and in which direction."],
            ["Decision-impact test",
             "Rebuilds the priority score with the element changed and counts how many of the "
             "top-20 subzones change, compared with the measurement-noise floor (see the decision-impact section).",
             "Whether the element changes planning priorities."],
            ["Exploratory sweep",
             "Tries a range of settings and compares them on the same test points.",
             "Sensitivity to a setting only. Not evidence for choosing a setting, because choosing "
             "the best score on the test points would tune the method to the test set."],
        ],
        [2300, 4300, 3300],
    )
    doc.para(
        "Several comparisons substitute one input for another (for example, one land-cover label "
        "source for another) instead of removing a part. They are reported as controlled comparisons "
        "and follow the same one-change-at-a-time rule as ablations."
    )

    doc.section("Reference data and independence")
    doc.para(
        "An evaluation is only meaningful against data the component did not learn from. Where no such "
        "reference exists, the reports say so and use internal-quality and robustness checks in its place."
    )
    doc.table(
        ["Component", "Reference used", "Independence and limits"],
        [
            ["Downscaling",
             "MODIS land surface temperature (329 subzones) and NEA weather-station air temperature "
             "(12 subzones).",
             "Different sensors from the Landsat input. NEA covers too few subzones to carry a "
             "conclusion; MODIS is the main reference. Both measure a different quantity or scale, "
             "so a constant offset is expected and is reported separately."],
            ["Land cover",
             "300 points hand-labelled on high-resolution imagery.",
             "Training excludes a 15 m buffer around every point. Two-labeller agreement on a "
             "20-point trial: kappa 0.676. The sample was drawn by WorldCover class, so the rarer "
             "classes (bare, water) have few points."],
            ["Heat typology",
             "No ground truth (unsupervised).",
             "Internal cluster quality, plus a coherence check against land-cover fractions that "
             "were not used as clustering inputs."],
            ["Cooling priority",
             "No ground truth: no record of where cooling was most needed.",
             "Robustness to measurement noise and to method choices. Face validity against "
             "published hotspot areas is a supporting check only."],
            ["Scenario modelling",
             "Held-out subzones (XGBoost) and held-out patches (CNN). No observed before-and-after "
             "greening interventions.",
             "Prediction accuracy is testable; the effect of a hypothetical change is not. "
             "Scenario outputs are therefore checked for consistency and plausibility and "
             "reported as indicative, not as forecasts."],
        ],
        [1800, 3700, 4400],
    )

    doc.section("Metrics")
    doc.table(
        ["Component", "Metrics", "Why these"],
        [
            ["Downscaling",
             "Systematic offset, spread after removing the offset, and Spearman rank correlation, "
             "each against MODIS and NEA.",
             "The score min-max scales temperature, so a constant offset cannot move any rank. RMSE "
             "is dominated by the offset and is not reported as accuracy."],
            ["Land cover",
             "Overall accuracy with 95% bootstrap interval, macro F1, per-class precision, recall "
             "and F1, confusion matrix; a vegetation versus non-vegetation score.",
             "Macro F1 stops the large built-up class from hiding failures on small classes. Only "
             "the vegetation share reaches the priority score, so the vegetation score shows the "
             "error that matters downstream."],
            ["Heat typology",
             "Silhouette and other internal quality scores; agreement between runs with different "
             "random seeds; land-cover coherence.",
             "Without labels, a typology is credible only if it is compact, repeatable and "
             "consistent with independent features."],
            ["Cooling priority",
             "PCA weights; Spearman correlation between rankings; number of top-20 subzones changed; "
             "bootstrap bands and each subzone's probability of staying in the top 20.",
             "The output is a ranking used to choose places, so the metrics describe the ranking, "
             "especially its top."],
            ["Scenario modelling",
             "R² and RMSE on held-out data; whether XGBoost and the CNN agree on the direction of "
             "change; share of edited subzones that cool and the median change; share of edits "
             "outside the range of real subzones.",
             "Accuracy shows the models learned the temperature pattern; agreement between two "
             "independent models and the extrapolation check show whether a what-if answer can "
             "be trusted."],
        ],
        [1800, 4300, 3800],
    )

    doc.section("Statistical testing")
    doc.rich([("Paired classifier comparisons. ", True),
          ("Two classifiers scored on the same points are compared with an exact McNemar test, which uses "
           "only the points where exactly one classifier is correct. Each comparison reports those two counts "
           "with the p-value, so the size of the difference is visible as well as its significance.", False)])
    doc.rich([("Confidence intervals. ", True),
          ("Accuracy intervals come from 4,000 bootstrap resamples of the test points. Rank-based results use "
           "Spearman correlation.", False)])
    doc.rich([("Reading the results. ", True),
          ("A non-significant difference does not show that two methods are equivalent; it shows the test "
           "set cannot separate them. Many comparisons reuse the same 300 points and no multiple-comparison "
           "adjustment is applied, so p-values near 0.05 are treated with caution. Classes with few test "
           "points (bare land, water) are reported but not used to rank methods.", False)])

    doc.section("Decision impact and the measurement-noise floor")
    doc.para(
        "A decision-impact test changes one choice, rebuilds the priority score with everything else "
        "fixed, and counts how many of the top-20 subzones are replaced. On its own, that count has no "
        "scale: a change of four subzones could be large or trivial. The noise floor supplies the scale."
    )
    doc.para(
        "The noise floor is the number of top-20 subzones that measurement noise alone replaces, on "
        "average. It comes from a bootstrap of the priority score in which the inputs are perturbed by "
        "error sizes the project has measured, not by assumed ranges. Two noise sources are used:"
    )
    doc.rich([("Heat exposure: ", True),
          ("the spread of the difference between Landsat and MODIS temperature across subzones, after "
           "removing the constant offset.", False)])
    doc.rich([("Greenery: ", True),
          ("the subzone-to-subzone disagreement between the two independent greenery estimates in the "
           "pipeline (land-cover map and NDVI threshold).", False)])
    doc.para(
        "A choice that changes fewer top-20 subzones than the noise floor cannot be distinguished from "
        "data noise for planning purposes. A choice that changes more is a real decision and has to be "
        "justified, not defaulted."
    )
    doc.para(
        "Three limits apply. The bootstrap covers measurement noise in heat and greenery only, not "
        "uncertainty in how the score is built, which is why formula choices are tested separately as "
        "decision-impact tests. The noise floor is an average over many draws, while each choice "
        "produces a single ranking. And the bands have not been tested for coverage, so they are "
        "described as based on validation error, not as calibrated intervals."
    )

    doc.section("Rules that keep the tests fair")
    for lead, text in [
        ("One change at a time. ", "Each ablation or decision-impact test changes a single element."),
        ("Same test set for every variant. ", "All variants of a component are scored on the same points or subzones."),
        ("Settings fixed before results. ",
         "A setting chosen by comparing results on the test set is labelled exploratory and is not "
         "counted as evidence for that setting."),
        ("One production version throughout. ",
         "All five component reports use the same production land-cover map and pipeline outputs. "
         "Mixing outputs from different versions would make the decision-impact counts meaningless."),
        ("Traceable numbers. ",
         "Every reported number is traced to the code version, the MLflow run that logged it, and the "
         "production land-cover map (identified by its SHA-256 checksum)."),
        ("Null results are reported. ",
         "A test that finds no effect, or cannot find one by construction, is reported as such."),
    ]:
        doc.rich([(lead, True), (text, False)])

    doc.section("Structure of each component report")
    doc.table(
        ["Section", "Content"],
        [
            ["Summary", "What the component does, its main results, and what they mean for the priority ranking."],
            ["Setup", "Purpose of the evaluation, reference data, metrics, significance tests, and the production version evaluated."],
            ["Evaluation", "Performance of the final component against its reference."],
            ["Ablations and controlled comparisons", "One-change tests on the same test set, with significance."],
            ["Decision impact", "Top-20 subzones changed by each choice, compared with the noise floor."],
            ["Exploratory analyses", "Sweeps and diagnostics, labelled as not supporting a choice of setting."],
            ["Limitations", "Sample sizes, dependence on the reference data, and untested assumptions."],
        ],
        [3000, 6900],
    )

    doc.section("Planned tests by component")
    doc.table(
        ["Component", "Evaluation", "Ablations and controlled comparisons", "Decision impact"],
        [
            ["Downscaling",
             "Agreement with MODIS and NEA (offset, spread, Spearman).",
             "Native 30 m vs bicubic 10 m vs regression 10 m; whether interpolation flattens the "
             "hottest areas; season window.",
             "Score rebuilt with each heat layer."],
            ["Land cover",
             "Accuracy, F1 and confusion matrix on the 300 points; vegetation vs non-vegetation.",
             "WorldCover vs Dynamic World labels; RF vs U-Net vs hybrid; texture features; "
             "smoothing. Blend-weight and window sweeps as exploratory.",
             "Score rebuilt with each greenery source, including the NDVI threshold."],
            ["Heat typology",
             "Internal cluster quality; land-cover coherence.",
             "K-means vs Gaussian mixture; number of clusters; feature set; stability across "
             "random seeds.",
             "Not applicable: the typology describes subzones and does not enter the score. The "
             "report shows how the top-20 subzones fall across the clusters."],
            ["Cooling priority",
             "Bootstrap bands, probability of staying in the top 20, noise floor.",
             "PCA vs equal weights; sensitivity-pillar formula; residents view vs all-places view; "
             "minimum-residents threshold (500).",
             "Every comparison in this row is itself a decision-impact test."],
            ["Scenario modelling",
             "XGBoost R² and RMSE on held-out subzones; CNN RMSE on held-out patches.",
             "Greening edit with and without moving NDVI and NDBI together; XGBoost with feature "
             "groups removed; XGBoost vs CNN direction agreement.",
             "Not applicable: scenario outputs answer what-if questions and do not change the "
             "ranking."],
        ],
        [1700, 2600, 3300, 2300],
    )

    doc.section("Sources", numbered=False)
    doc.para(
        "Evaluation code: validation/ (input_validation, landcover_validation, hotspots_validation, "
        "score_validation) and the scripts/diagnose_*, scripts/evaluate_* and scripts/build_* scripts. "
        "Noise floor: validation/score_validation/decision_impact.py and confidence_bands.py. "
        "Validation points: data/interim/validation_sample/validation_sample_300_labeled.csv."
    )

