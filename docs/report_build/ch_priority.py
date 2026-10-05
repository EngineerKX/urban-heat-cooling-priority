"""Cooling-priority chapter: the score itself, its robustness, and the
project-wide decision-impact summary. Numbers come from:
  data/processed/priority_score*.csv, priority_score*_confidence_bands.csv
  data/processed/weighting_comparison.csv, sensitivity_spec_comparison.csv,
      rank_impact_results_pca.csv, landcover_label_source_swap.csv, landcover_swap_comparison.csv
  data/processed/diagnostics/priority_*.csv (scripts/evaluate_priority_score.py)
  data/interim/adaptive_capacity_pillar.csv
"""

import pandas as pd

from helpers import pct

TITLE = "Cooling-priority Score"
STANDALONE_TITLE = "Cooling-priority Score Evaluation and Ablation Tests"
SUBTITLE = "Heat exposure, resident sensitivity and missing greenery, combined with data-derived weights"


def _range(lo, hi):
    return f"{int(lo)}" if int(lo) == int(hi) else f"{int(lo)}–{int(hi)}"


def _pair(a, b):
    return f"{int(a)} subzones each" if int(a) == int(b) else f"{int(a)} and {int(b)} subzones"


def build(doc, repo):
    P, D = repo / "data/processed", repo / "data/processed/diagnostics"
    pca = pd.read_csv(D / "priority_pca_structure.csv").set_index("pillar")
    corr = pd.read_csv(D / "priority_pillar_correlations.csv").set_index("pillar")
    abl = pd.read_csv(D / "priority_ablations.csv").set_index("change")
    thr = pd.read_csv(D / "priority_min_residents.csv")
    noise = pd.read_csv(D / "priority_noise_sources.csv").set_index("noise_sources")
    views = pd.read_csv(D / "priority_views.csv").iloc[0]
    face = pd.read_csv(D / "priority_face_validity.csv").set_index("view")
    bands = pd.read_csv(D / "priority_bands_summary.csv").set_index("view")
    weighting = pd.read_csv(P / "weighting_comparison.csv").set_index("variant").loc["lst_native30"]
    specs = pd.read_csv(P / "sensitivity_spec_comparison.csv")
    heat = pd.read_csv(P / "rank_impact_results_pca.csv").set_index("variant")
    lc = pd.read_csv(P / "landcover_label_source_swap.csv")
    ndvi = pd.read_csv(P / "landcover_swap_comparison.csv").set_index("greenery_source").loc["NDVI-threshold proxy"]
    greenery = pd.read_csv(repo / "data/interim/adaptive_capacity_pillar.csv")
    ranked = pd.read_csv(P / "priority_score.csv")
    greenery_effect = pd.read_csv(D / "priority_greenery_effect.csv").iloc[0]
    clip = pd.read_csv(D / "priority_clipping_check.csv").iloc[0]

    floor = bands.loc["Residents", "noise_floor"]
    names = {"exposure": "Heat exposure", "sensitivity": "Resident sensitivity", "adaptive_deficit": "Missing greenery"}
    low_green = (greenery.set_index("subzone_id").loc[ranked["subzone_id"], "greenery_fraction"] < 0.05).mean()
    alt_specs = specs[~specs["specification"].str.startswith("PRODUCTION") & (specs["spearman_vs_production"] < 1 - 1e-12)]
    heat_changed = [20 - round(heat.loc[v, "top20_overlap_vs_lst_native30"] * 20) for v in ("lst_bicubic10", "lst_regress10")]
    lc_alt = lc[~lc["is_reference"]]
    thr_alt = thr[~thr["is_production"]]
    w = pca["pca_weight"]

    doc.title(TITLE if doc.chapter else STANDALONE_TITLE, SUBTITLE)

    # --- Summary -----------------------------------------------------------------
    doc.section("Summary", numbered=False)
    doc.para(
        "The cooling-priority score ranks subzones by combining three pillars: heat exposure (Landsat surface "
        "temperature), resident sensitivity (population density and the share of elderly residents) and missing "
        "greenery (one minus the vegetation share from the land-cover map). The planning output is the list of "
        "the 20 highest-priority subzones."
    )
    doc.para(
        f"There is no record of where cooling was most needed, so the score is judged on its structure, its "
        f"robustness and whether its design choices matter. One index summarises the three pillars reasonably "
        f"(the first principal component carries {pct(pca['variance_explained_by_component'].iloc[0], 0)} of their "
        f"variation). Measurement noise alone replaces about {floor:.0f} of the top 20 in an average draw, and no "
        "top-20 subzone is certain. Against that noise floor, only removing the heat or the sensitivity pillar "
        f"changes the top 20 by more than noise ({_pair(abl.loc['Without exposure', 'top_changed'], abl.loc['Without sensitivity', 'top_changed'])}). Every method choice tested, including "
        "weighting, the sensitivity formula, the heat layer and the land-cover source, moves fewer subzones than "
        "noise does. The top 20 should therefore be read as a well-supported shortlist, not a precise order."
    )

    # --- Setup ---------------------------------------------------------------------
    doc.section("Setup")
    doc.table(
        ["Item", "Setting"],
        [
            ["Pillars", "Heat exposure: Landsat 30 m surface temperature, subzone mean. Resident sensitivity: "
                        "0.5 × population density + 0.5 × elderly share (the 50/50 split is a placeholder). "
                        "Missing greenery: 1 − vegetation share from the production land-cover map."],
            ["Scaling and weights", "Each pillar is min-max scaled to 0–1. Weights come from the first principal "
                                    "component of the standardised pillars: "
                                    f"{w['exposure']:.3f} / {w['sensitivity']:.3f} / {w['adaptive_deficit']:.3f} "
                                    "(exposure / sensitivity / missing greenery)."],
            ["Residents view (production)", f"{len(ranked)} subzones with at least 500 residents."],
            ["All-places view", f"All {int(views['n_all_places_view'])} subzones, heat and missing greenery only, "
                                "equal weights. It includes industrial and port areas but says nothing about how "
                                "many people are exposed."],
            ["Uncertainty", "1,000 bootstrap draws perturbing heat and greenery by measured error sizes: heat by "
                            "the spread of Landsat minus MODIS after removing the offset, greenery by the "
                            "disagreement between the land-cover and NDVI greenery estimates."],
        ],
        [2400, 7500],
    )

    # --- Evaluation ---------------------------------------------------------------
    doc.section("Evaluation")
    doc.subsection("Does one index summarise the pillars?")
    doc.table(
        ["Pillar", "PCA weight", "Correlation with heat", "with sensitivity", "with missing greenery",
         "Correlation with final score"],
        [[names[p], f"{w[p]:.3f}", f"{corr.loc[p, 'exposure']:.2f}", f"{corr.loc[p, 'sensitivity']:.2f}",
          f"{corr.loc[p, 'adaptive_deficit']:.2f}", f"{pca.loc[p, 'correlation_with_score']:.2f}"] for p in names],
        [2200, 1300, 1600, 1500, 1700, 1600],
    )
    ev = pca["variance_explained_by_component"].tolist()
    doc.para(
        f"The first principal component explains {pct(ev[0], 0)} of the pillars' variation (then {pct(ev[1], 0)} "
        f"and {pct(ev[2], 0)}), so the pillars share a common direction and a single weighted index is a reasonable "
        "summary. All three pillars correlate positively, and missing greenery is the most tightly linked to the "
        "final score. Because the pillars are min-max scaled before weighting, each pillar's influence depends on "
        "its spread as well as its weight."
    )
    doc.subsection("Robustness to measurement noise")
    r, a = bands.loc["Residents"], bands.loc["All places"]
    doc.table(
        ["View", "Subzones", "Mean 90% band width", "Point outside own band", "Noise floor (top 20)",
         "Top 20: chance of staying ≥ 0.9", "0.5–0.9", "< 0.5"],
        [[v, str(int(b["n"])), f"{b['mean_band_width']:.2f}", str(int(b["point_outside_own_band"])),
          f"{b['noise_floor']:.1f}", str(int(b["top20_p_ge_0.9"])), str(int(b["top20_p_0.5_to_0.9"])),
          str(int(b["top20_p_lt_0.5"]))] for v, b in (("Residents", r), ("All places", a))],
        [1300, 1100, 1300, 1300, 1300, 1400, 1100, 1100],
    )
    doc.para(
        f"Scores run from 0 to 1, so a band about {r['mean_band_width']:.2f} wide is large: neighbouring "
        "subzones in the ranking cannot be told apart. No top-20 subzone stays in the top 20 in 90% of draws; "
        f"{int(r['top20_p_0.5_to_0.9'])} stay in more than half. The noise floor is the yardstick for every "
        "decision-impact result in this report."
    )
    doc.table(
        ["Noise included", "Heat noise (°C)", "Greenery noise (share)", "Noise floor (top 20)"],
        [[lab, f"{row.exposure_noise_std_c:.2f}" if row.exposure_noise_std_c else "none",
          f"{row.greenery_noise_std:.3f}" if row.greenery_noise_std else "none", f"{row.noise_floor_top20:.1f}"]
         for lab, row in noise.iterrows()],
        [3400, 2000, 2200, 2300],
    )
    doc.para(
        "Heat measurement noise drives the uncertainty; greenery noise on its own moves about half as many "
        "subzones. The greenery-only figure is approximate (see Limitations)."
    )
    doc.subsection("Face validity against published hotspot areas")
    fr = face.loc["Residents"]
    doc.para(
        "The Singapore-ETH Centre's urban heat vulnerability analysis names seven planning areas as hotspots "
        "(Geylang, Jurong East, Marine Parade, Punggol, Sengkang, Serangoon, Woodlands). Its index differs from "
        "this one (air temperature, humidity, vegetation and demographics, at planning-area level, around 2020), so this is a "
        "supporting check only."
    )
    doc.table(
        ["View", "Top-20 subzones in those areas", "Expected by chance", "p (at least this many)", "Subzones"],
        [[v, str(int(f["top_in_hotspot_areas"])), f"{f['expected_by_chance']:.1f}", f"{f['p_value_at_least']:.2f}",
          f["subzones"].title()] for v, f in face.iterrows()],
        [1300, 1800, 1600, 1500, 3700],
    )
    doc.para(
        f"{int(fr['top_in_hotspot_areas'])} of the residents' top 20 lie in those areas against "
        f"{fr['expected_by_chance']:.1f} expected by chance, which is not distinguishable from chance. The two "
        "indices agree only weakly."
    )

    # --- Ablations ---------------------------------------------------------------------
    doc.section("Ablations and controlled comparisons")
    doc.para(
        "Each test rebuilds the score with one change and counts how many of the production top-20 subzones "
        f"leave, compared with the noise floor of {floor:.1f}."
    )
    doc.subsection("Pillars and aggregation")
    doc.table(
        ["Change", "Top-20 subzones changed", "Rank correlation with production", "Against noise floor"],
        [[c.replace("adaptive deficit", "missing greenery").replace("exposure", "heat exposure"),
          str(int(row.top_changed)), f"{row.spearman:.3f}",
          "Exceeds" if row.top_changed > floor else "Within"] for c, row in abl.iterrows()],
        [3900, 2000, 2400, 1600],
    )
    doc.para(
        "Heat and sensitivity each carry information the others do not: removing either changes more of the "
        "top 20 than noise does. Removing missing greenery leaves the top-20 list unchanged, but it does change "
        "the ranking:"
    )
    ge = greenery_effect
    doc.table(
        ["Without the missing-greenery pillar", "Result"],
        [["Top-20 subzones changed", str(int(ge["top20_changed"]))],
         ["Top-10 and top-50 subzones changed", f"{int(ge['top10_changed'])} and {int(ge['top50_changed'])}"],
         ["Rank shift within the top 20", f"{ge['mean_rank_shift_in_top20']:.1f} places on average, up to "
                                          f"{int(ge['max_rank_shift_in_top20'])}"],
         ["Rank shift across all ranked subzones", f"{ge['mean_rank_shift_all']:.1f} places on average; "
                                                   f"{int(ge['n_moving_10_or_more_places'])} of {int(ge['n_ranked'])} "
                                                   "move 10 or more places"]],
        [4900, 5000],
    )
    doc.para(
        "Greenery therefore shapes the order but not the cut at 20. The subzones that make the top 20 are "
        f"already the hot, densely populated ones, and they have little vegetation: {int(clip['n_top20_no_greenery'])} "
        f"of the top 20 have essentially none, and {pct(low_green, 0)} of all ranked subzones have under 5%. "
        "At the top of the list greenery reinforces what heat and sensitivity already show. The pillar stays in "
        "the score for its planning meaning, since greening is the intervention being prioritised. Standardising "
        "the pillars instead of min-max scaling changes "
        f"{int(abl.loc['z-score instead of min-max aggregation', 'top_changed'])} of the top 20."
    )
    doc.subsection("Weighting")
    doc.para(
        f"Equal weights instead of PCA weights change {int(weighting['n_top20_changed'])} of the top 20 (rank "
        f"correlation {weighting['spearman_pca_vs_equal']:.3f}). The data-derived weights are close to equal, so "
        "the choice has little effect."
    )
    doc.subsection("Sensitivity formula")
    doc.table(
        ["Alternative sensitivity formula", "Top-20 subzones changed", "Rank correlation with production"],
        [[r_.specification, str(int(r_.top20_changed)), f"{r_.spearman_vs_production:.3f}"] for r_ in alt_specs.itertuples()],
        [5300, 2100, 2500],
    )
    doc.para(
        f"How sensitivity is defined matters more than any other method choice ({int(alt_specs['top20_changed'].min())} "
        f"to {int(alt_specs['top20_changed'].max())} subzones), though every alternative stays within the noise "
        "floor. Population density was chosen over raw count (decided 19 September 2026); raw count is the main "
        "alternative reported here."
    )
    doc.subsection("Minimum-residents threshold")
    doc.table(
        ["Minimum residents", "Subzones ranked", "Top-20 subzones changed"],
        [[f"{int(t.min_residents)}" + (" (production)" if t.is_production else ""), str(int(t.n_ranked)),
          str(int(t.top_changed))] for t in thr.itertuples()],
        [3300, 3300, 3300],
    )
    t = thr.set_index("min_residents")["top_changed"]
    doc.para(
        f"Thresholds from 250 to 1,000 residents give almost the same top 20 (at most "
        f"{int(t.loc[[250, 1000]].max())} change). Lower thresholds change more ({int(t.loc[0])} with none, "
        f"{int(t.loc[100])} at 100), because subzones with a handful of residents get extreme sensitivity values; "
        f"a high threshold of 2,000 changes {int(t.loc[2000])}."
    )
    doc.subsection("Residents view against all-places view")
    doc.para(
        f"Only {int(views['top20_in_both'])} subzones are in both views' top 20; "
        f"{int(views['all_places_top20_not_ranked_in_residents_view'])} of the all-places top 20 have too few "
        "residents to be ranked in the residents view (industrial estates and the port). On the subzones both "
        f"views rank, their orders correlate at {views['spearman_on_common_subzones']:.2f}. The two views answer "
        "different questions and are reported separately."
    )

    # --- Decision impact summary ----------------------------------------------------------
    doc.section("Decision impact across the project")
    doc.para(
        "This table collects every decision-impact test in the report: each method choice, the range of top-20 "
        "subzones it changes across the alternatives tested, and the comparison with measurement noise."
    )
    summary = [
        ("Measurement noise alone (yardstick)", "1,000 bootstrap draws", f"{floor:.1f} on average", "—"),
        ("Remove a pillar", "3 pillars", f"{int(abl.iloc[:3]['top_changed'].min())}–{int(abl.iloc[:3]['top_changed'].max())}",
         "Heat and sensitivity exceed"),
        ("Sensitivity formula", f"{len(alt_specs)} alternatives",
         f"{int(alt_specs['top20_changed'].min())}–{int(alt_specs['top20_changed'].max())}", "Within"),
        ("Greenery source (land-cover model or NDVI)", f"{len(lc_alt) + 1} alternatives",
         f"{int(lc_alt['top20_changed'].min())}–{int(max(lc_alt['top20_changed'].max(), ndvi['top20_changed']))}", "Within"),
        ("Minimum-residents threshold", f"{len(thr_alt)} alternatives",
         f"{int(thr_alt['top_changed'].min())}–{int(thr_alt['top_changed'].max())}", "Within"),
        ("Weighting (PCA vs equal)", "1 alternative", str(int(weighting["n_top20_changed"])), "Within"),
        ("Aggregation (min-max vs z-score)", "1 alternative",
         str(int(abl.loc["z-score instead of min-max aggregation", "top_changed"])), "Within"),
        ("Heat layer (30 m vs 10 m)", "2 alternatives", _range(min(heat_changed), max(heat_changed)), "Within"),
    ]
    doc.table(["Choice", "Tested", "Top-20 subzones changed", "Against noise floor"], [list(s) for s in summary],
              [3700, 1900, 2200, 2100])
    doc.para(
        "Only the score's structure, which pillars it contains, changes the top 20 by more than measurement "
        "noise. None of the method choices does, so for planning purposes the ranking is robust to them, and "
        "each choice can be justified on its own grounds without changing the shortlist. The largest gain in ranking "
        "confidence would come from reducing heat measurement error, not from refining any one method."
    )

    # --- Limitations --------------------------------------------------------------------------
    doc.section("Limitations")
    for lead, text in [
        ("No ground truth. ", "There is no record of where cooling was most needed, so the score is tested for "
                              "robustness and consistency, not accuracy."),
        ("Bands are not calibrated. ", "They reflect measured error sizes in two pillars, but their coverage was "
                                       "never tested, and the heat error (Landsat against 1 km MODIS) is an upper "
                                       "bound."),
        ("Greenery noise is clipped at zero. ", (
            "Perturbed greenery is capped to the 0–1 range, so for the "
            f"{int(clip['n_subzones_no_greenery'])} subzones with essentially no vegetation the noise can only "
            "push greenery up, which pulls their noisy scores down. The effect is small: rerun without the cap, "
            f"the noise floor is {clip['noise_floor_unclipped']:.1f} instead of {clip['noise_floor_clipped']:.1f}, "
            "those subzones' average chance of being in the top 20 is "
            f"{clip['mean_p_top20_no_greenery_unclipped']:.2f} instead of {clip['mean_p_top20_no_greenery_clipped']:.2f}, "
            f"and no subzone's chance changes by more than {clip['max_abs_p_top20_change']:.2f}. It is most visible "
            "when greenery noise acts alone, so that floor is approximate. Adding the noise on a scale that respects "
            "the 0–1 range would remove it.")),
        ("Placeholder settings. ", "The 50/50 population-elderly split in the sensitivity pillar has not been "
                                   "justified; alternatives are tested above."),
        ("Residents only. ", "Sensitivity counts residents, not workers or visitors; the all-places view covers "
                             "non-residential areas but without any people term."),
    ]:
        doc.lead(lead, text)

    doc.section("Sources", numbered=False)
    doc.para(
        "Score: scripts/build_priority_score.py, src/priority_score/. Bands and noise floor: "
        "scripts/build_priority_score_confidence_bands.py, validation/score_validation/confidence_bands.py, "
        "decision_impact.py. Evaluation: scripts/evaluate_priority_score.py, "
        "validation/score_validation/score_evaluation.py, sensitivity_specs.py, rank_impact.py, landcover_impact.py "
        "(outputs in data/processed/ and data/processed/diagnostics/priority_*.csv)."
    )
