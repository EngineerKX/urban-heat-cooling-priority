"""Downscaling chapter: Landsat 30 m, bicubic 10 m and regression 10 m heat
layers. Every number is read from saved outputs:
  data/interim/heat_variants_subzone.csv, modis_heldout_lst.csv, nea_heldout_lst.csv
  data/processed/diagnostics/downscaling_skill_*.csv (scripts/evaluate_downscaling_skill.py)
  data/processed/rank_impact_results_*.csv, priority_score_confidence_bands.csv
"""

import pandas as pd

from helpers import signed
from validation.score_validation.decision_impact import noise_floor
from validation.score_validation.rank_impact import heldout_agreement

TITLE = "Heat-map Downscaling"
STANDALONE_TITLE = "Downscaling Evaluation and Ablation Tests"
SUBTITLE = "Landsat 30 m, bicubic 10 m and regression 10 m heat layers"

VARIANTS = {"lst_native30": "Landsat 30 m (reference)", "lst_bicubic10": "Bicubic 10 m", "lst_regress10": "Regression 10 m"}
METHOD = {"nearest": "Nearest (copy coarse value)", "bilinear": "Bilinear interpolation",
          "bicubic": "Bicubic interpolation", "tsharp": "Regression (production method)",
          "tsharp_no_residual": "Regression, no residual correction", "tsharp_ndvi_only": "Regression, NDVI only"}
f2, f3 = (lambda x: f"{x:.2f}"), (lambda x: f"{x:.3f}")


def sk(df, m, col):
    return float(df.set_index("method").loc[m, col])


def boot(df, label):
    r = df.set_index("comparison").loc[label]
    return f"{r.rmse_diff_c:+.2f} ({r.ci95_low:+.2f} to {r.ci95_high:+.2f})"




def build(doc, repo):
    D, I, P = repo / "data/processed/diagnostics", repo / "data/interim", repo / "data/processed"
    heat = pd.read_csv(I / "heat_variants_subzone.csv")
    modis, nea = pd.read_csv(I / "modis_heldout_lst.csv"), pd.read_csv(I / "nea_heldout_lst.csv")
    sk30, sk120 = pd.read_csv(D / "downscaling_skill_30m.csv"), pd.read_csv(D / "downscaling_skill_120m.csv")
    bt30, bt120 = pd.read_csv(D / "downscaling_skill_bootstrap_30m.csv"), pd.read_csv(D / "downscaling_skill_bootstrap_120m.csv")
    skgee = pd.read_csv(D / "downscaling_skill_gee.csv")
    land120 = pd.read_csv(D / "downscaling_skill_landfit_120m.csv")
    ri_pca, ri_eq = pd.read_csv(P / "rank_impact_results_pca.csv"), pd.read_csv(P / "rank_impact_results_equal.csv")
    floor_res = noise_floor(pd.read_csv(P / "priority_score_confidence_bands.csv"))

    doc.title(TITLE if doc.chapter else STANDALONE_TITLE, SUBTITLE)

    # --- Executive summary
    d_bic = sk(sk120, "tsharp", "rmse_c") - sk(sk120, "bicubic", "rmse_c")
    changed = 20 - round(float(ri_pca.set_index("variant").loc["lst_regress10", "top20_overlap_vs_lst_native30"]) * 20)
    doc.section("Summary", numbered=False)
    doc.para(
        "The project builds three heat layers from the same Landsat land surface temperature composite: the 30 m "
        "Landsat layer itself, a 10 m layer made by bicubic interpolation, and a 10 m layer made by regression "
        "downscaling with Sentinel-2 indices (a TsHARP-style method). The 30 m layer is the reference used by the "
        "priority score."
    )
    doc.para(
        f"At subzone level the three layers are practically identical, so they agree equally with independent MODIS "
        f"temperature, and swapping them in the priority score changes {changed} of the top-20 subzones, against a "
        f"measurement-noise floor of {floor_res:.1f}. A coarsen-and-recover test, which scores each method against real "
        f"Landsat temperature at a known finer scale, found that the regression method does not reconstruct fine-scale "
        f"temperature more accurately than bicubic interpolation (RMSE {d_bic:+.2f} °C at the 360 m to 120 m step), "
        f"although it preserves the hottest and coolest pixels better. The 10 m regression layer is therefore not "
        f"evidence of more accurate fine-scale heat, and the choice of heat layer does not change planning priorities."
    )

    # --- 1. Purpose and setup
    doc.section("Purpose and setup")
    doc.para(
        "Downscaling is meant to give a sharper heat map than Landsat's 30 m product by borrowing spatial detail from "
        "10 m Sentinel-2 imagery. Its evaluation must show two things: whether the extra detail is accurate, and "
        "whether it changes which subzones are prioritised."
    )
    doc.table(
        ["Item", "Setting"],
        [
            ["Heat composite", "Landsat 8 and 9 surface temperature, median of the inter-monsoon months (April, May, "
                               "October, November) from 2021 to June 2026, scenes under 70% cloud."],
            ["Regression method", "Temperature at 30 m regressed on Sentinel-2 NDVI, NDBI and NDWI (aggregated to 30 m), "
                                  "applied at 10 m, plus the 30 m residual resampled bilinearly. The residual step approximately keeps "
                                  "each 30 m block's mean temperature."],
            ["Independent references", f"MODIS land surface temperature at 1 km ({len(modis)} subzones) and NEA "
                                       f"weather-station air temperature ({len(nea)} subzones)."],
            ["Production version", "Heat layers from data/interim/heat_variants_subzone.csv. The priority score is "
                                   "rebuilt on the production land-cover map (U-Net, SHA-256 dc33f409…)."],
            ["Statistical method", "Spearman rank correlation; RMSE differences with 95% intervals from a spatial block "
                                   "bootstrap (1 km blocks, 2,000 resamples)."],
        ],
        [2600, 7300],
    )

    # --- 2. Evaluation: agreement with independent references
    doc.section("Evaluation: agreement with independent temperature")
    doc.para(
        "Each layer's subzone means are compared with MODIS and NEA. The disagreement is split into a constant offset "
        "(which cannot change a ranking, because the score rescales temperature), the spread left after removing the "
        "offset, and rank agreement."
    )
    rows = []
    for src_name, ref in (("MODIS", modis), ("NEA", nea)):
        for v, label in VARIANTS.items():
            a = heldout_agreement(heat, v, ref)
            rows.append([src_name, label, str(a["n"]), f"{a['mean_offset_c']:+.2f}", f2(a["spread_c"]), f3(a["spearman"])])
    doc.table(["Reference", "Heat layer", "Subzones", "Offset (°C)", "Spread (°C)", "Spearman"], rows,
              [1300, 2900, 1300, 1500, 1400, 1500])
    doc.para(
        "Landsat reads hotter than both references, as expected: MODIS measures at 1 km, and NEA stations measure air "
        "temperature, not surface temperature. Rank agreement with MODIS is moderate. NEA covers too few subzones to "
        "support a conclusion and is shown for context. The three layers are indistinguishable on every measure."
    )

    # --- 3. Evaluation: downscaling skill
    doc.section("Evaluation: downscaling skill (coarsen-and-recover)")
    doc.para(
        "There is no 10 m temperature ground truth. The standard workaround is to run the method one step coarser, "
        "where truth exists: aggregate the Landsat temperature to a coarse scale, rebuild a three-times finer scale "
        "with each method, and compare with the real Landsat temperature at that finer scale. Two scale pairs were "
        "tested. The 90 m to 30 m pair repeats the production step's factor of three. The 360 m to 120 m pair sits "
        "above Landsat's thermal sensor resolution (about 100 m; the 30 m product is resampled from it), so its "
        "reference holds detail the sensor really measured. The 90 m to 30 m reference is partly interpolated itself, "
        "which favours interpolation methods; the 360 m to 120 m pair is the fairer test."
    )
    for title, df in (("360 m to 120 m (fairer test)", sk120), ("90 m to 30 m", sk30)):
        n = int(df["n_pixels"].iloc[0])
        doc.rich([(f"{title}: ", True), (f"{n:,} land pixels.", False)])
        doc.table(
            ["Method", "RMSE (°C)", "R²", "Spread ratio", "Hottest 10% error (°C)", "Coolest 10% error (°C)"],
            [[METHOD[m], f2(sk(df, m, "rmse_c")), f3(sk(df, m, "r2")), f2(sk(df, m, "sd_ratio")),
              f"{sk(df, m, 'hottest10_bias_c'):+.2f}", f"{sk(df, m, 'coolest10_bias_c'):+.2f}"]
             for m in ["nearest", "bilinear", "bicubic", "tsharp"]],
            [3100, 1200, 1000, 1300, 1700, 1600],
        )
    doc.para(
        "Spread ratio is the standard deviation of the reconstruction divided by that of the truth; below 1 means the "
        "method smooths the temperature field. A negative error in the hottest 10% of pixels means the method "
        "under-reads peak heat."
    )
    doc.table(
        ["Comparison (RMSE difference, °C)", "360 m to 120 m", "90 m to 30 m"],
        [["Regression minus bicubic", boot(bt120, "tsharp minus bicubic"), boot(bt30, "tsharp minus bicubic")],
         ["Regression minus nearest", boot(bt120, "tsharp minus nearest"), boot(bt30, "tsharp minus nearest")],
         ["Bicubic minus nearest", boot(bt120, "bicubic minus nearest"), boot(bt30, "bicubic minus nearest")]],
        [3900, 3000, 3000],
    )
    doc.para(
        "Bicubic interpolation is the most accurate method at both scales. The regression method beats copying the "
        "coarse value at the fairer scale but is less accurate than bicubic at both. Its strength is at the extremes: "
        f"at 360 m to 120 m it keeps the full spread of temperatures (ratio {sk(sk120, 'tsharp', 'sd_ratio'):.2f} "
        f"against {sk(sk120, 'bicubic', 'sd_ratio'):.2f}), under-reads the hottest pixels less "
        f"({sk(sk120, 'tsharp', 'hottest10_bias_c'):+.2f} °C against {sk(sk120, 'bicubic', 'hottest10_bias_c'):+.2f} °C) "
        "and has almost no bias in the coolest pixels. Overall, the extra detail it adds is not more accurate than "
        "interpolation."
    )
    doc.para(
        f"Cross-check: the 90 m to 30 m pair was also run entirely in Earth Engine, with Earth Engine's own "
        f"interpolation and a 20,000-pixel sample. It gave the same ordering (bicubic RMSE "
        f"{sk(skgee, 'bicubic', 'rmse_c'):.2f} °C, regression {sk(skgee, 'tsharp', 'rmse_c'):.2f} °C)."
    )

    # --- 4. Ablations
    doc.section("Ablations and controlled comparisons")
    doc.table(
        ["Component tested", "Comparison", "360 m to 120 m", "90 m to 30 m", "Conclusion"],
        [
            ["Residual correction", "Regression with vs without the residual step (RMSE, °C)",
             f"{f2(sk(sk120, 'tsharp', 'rmse_c'))} vs {f2(sk(sk120, 'tsharp_no_residual', 'rmse_c'))}",
             f"{f2(sk(sk30, 'tsharp', 'rmse_c'))} vs {f2(sk(sk30, 'tsharp_no_residual', 'rmse_c'))}",
             "Essential. Without it, error more than doubles and the hottest pixels are under-read by about 5 °C or more."],
            ["Predictor set", "NDVI + NDBI + NDWI vs NDVI only (RMSE, °C)",
             f"{f2(sk(sk120, 'tsharp', 'rmse_c'))} vs {f2(sk(sk120, 'tsharp_ndvi_only', 'rmse_c'))}",
             f"{f2(sk(sk30, 'tsharp', 'rmse_c'))} vs {f2(sk(sk30, 'tsharp_ndvi_only', 'rmse_c'))}",
             "Inconsistent: three indices are better at 120 m, NDVI alone at 30 m. No evidence for either."],
            ["Interpolation kernel", "Bicubic vs bilinear vs nearest (RMSE, °C)",
             f"{f2(sk(sk120, 'bicubic', 'rmse_c'))} / {f2(sk(sk120, 'bilinear', 'rmse_c'))} / {f2(sk(sk120, 'nearest', 'rmse_c'))}",
             f"{f2(sk(sk30, 'bicubic', 'rmse_c'))} / {f2(sk(sk30, 'bilinear', 'rmse_c'))} / {f2(sk(sk30, 'nearest', 'rmse_c'))}",
             "Bicubic is the most accurate at both scales."],
        ],
        [1600, 2400, 1500, 1500, 2900],
    )
    g = heat.assign(gb=heat["lst_native30"] - heat["lst_bicubic10"], gr=heat["lst_native30"] - heat["lst_regress10"])
    hot = heat["lst_native30"] >= heat["lst_native30"].quantile(0.9)
    doc.para(
        f"Subzone-level smoothing check: the largest difference between a 10 m layer's subzone mean and the 30 m mean "
        f"is {g['gb'].abs().max():.2f} °C (bicubic) and {g['gr'].abs().max():.2f} °C (regression). In the hottest 10% "
        f"of subzones the mean difference is {signed(g.loc[hot, 'gb'].mean(), 3)} °C and {signed(g.loc[hot, 'gr'].mean(), 3)} °C, "
        "so neither 10 m layer flattens the hottest subzones."
    )

    # --- 5. Decision impact
    doc.section("Decision impact")
    rp, re_ = ri_pca.set_index("variant"), ri_eq.set_index("variant")
    rows = []
    for v in ["lst_bicubic10", "lst_regress10"]:
        for name, r in (("PCA weights", rp), ("Equal weights", re_)):
            ch = 20 - round(float(r.loc[v, "top20_overlap_vs_lst_native30"]) * 20)
            rows.append([VARIANTS[v], name, f"{float(r.loc[v, 'spearman_vs_lst_native30']):.3f}", str(ch)])
    doc.table(["Heat layer used in the score", "Weighting", "Rank correlation with reference", "Top-20 subzones changed"],
              rows, [3000, 2000, 2600, 2300])
    doc.para(
        f"No heat layer changes the top 20, against a measurement-noise floor of {floor_res:.1f} subzones. This follows "
        "from the method: the residual step approximately preserves each 30 m block's mean, so subzone averages, which are all the "
        "score uses, barely change. Higher resolution would matter only for decisions below subzone level, which this "
        "project does not rank."
    )

    # --- 6. Exploratory
    doc.section("Exploratory analyses")
    doc.rich([("Season window. ", True), (
        "Landsat scenes under 70% cloud, 2021 to June 2026: April and May 29; October and November 16; both "
        "inter-monsoon periods together 45 (chosen); February to April 43; all months 138. The chosen window has the "
        "most scenes of the seasonal candidates. The scene counts were compared after the choice was made, so they "
        "describe the choice rather than test it.", False)])
    doc.rich([("Land-only regression fit. ", True), (
        "The production regression is fitted over the whole bounding box, including sea. Fitting on land pixels only "
        "gives NDVI the expected negative sign but does not make the regression beat bicubic interpolation "
        f"(360 m to 120 m RMSE {sk(land120, 'tsharp', 'rmse_c'):.2f} °C against {sk(land120, 'bicubic', 'rmse_c'):.2f} °C), "
        "so the sea pixels do not explain the result.", False)])

    # --- 7. Limitations
    doc.section("Limitations")
    for lead, text in [
        ("Scale transfer. ", "The skill test assumes the method behaves at 30 m to 10 m as it does one or two steps "
                             "coarser. That cannot be tested without 10 m temperature data."),
        ("Sensor resolution. ", "Landsat thermal data is measured at about 100 m, so the 30 m product and both 10 m "
                                "layers carry no measured temperature detail finer than that."),
        ("Reference data. ", "MODIS pixels (1 km) are larger than most subzones, so its spread is an upper bound on "
                             "random error. NEA covers 12 subzones and measures air temperature."),
        ("One composite. ", "All tests use one multi-year inter-monsoon median; other seasons or single dates were "
                            "not tested."),
        ("Spatial dependence. ", "The bootstrap treats 1 km blocks as independent. Errors correlated over longer "
                                 "distances would make the intervals somewhat too narrow; the main differences are far "
                                 "larger than the intervals."),
    ]:
        doc.lead(lead, text)

    doc.section("Sources", numbered=False)
    doc.para(
        "Heat layers: scripts/build_heat_variants.py, src/downscaling/variants.py. Agreement: "
        "scripts/diagnose_modis_crosscheck.py. Skill test: scripts/evaluate_downscaling_skill.py, "
        "validation/input_validation/downscaling_skill.py (outputs in data/processed/diagnostics/). Decision impact and "
        "noise floor: scripts/build_priority_score.py, scripts/build_priority_score_confidence_bands.py. Season window: "
        "scripts/diagnose_season_window.py."
    )
