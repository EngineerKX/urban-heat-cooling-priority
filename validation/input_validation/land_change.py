"""Land-change diagnostic (from land_change_diagnostic.ipynb): checks
whether the multi-year composite window is blending genuinely different
land-cover states in any subzone (new development, reclaimed land, cleared
vegetation) rather than averaging noise around a stable value. A
`.median()` composite doesn't distinguish those cases — if land changed
partway through the window the result is a biased mix of two real states,
and that bias lands specifically in the subzones most likely to matter for
a cooling-priority tool (redevelopment zones).

Uses Dynamic World (validation-only for S3) to compare the dominant
land-cover class between an early and a late period, per subzone.
"""

import ee
import pandas as pd

from src.ingest.gee import date_filter_for_years_months


def build_dominant_class_composites(sg_bbox, early_start, early_end, late_start, late_end):
    """`ee.Reducer.mode()` = most frequently observed class per pixel across
    all images in the period — the "typical" class, robust to a single
    misclassified scene. No season-month restriction here: an earlier check
    (season_window.py's sensitivity test, generalized from this notebook's
    LC.1b cell) confirmed the season filter isn't the main constraint on
    Dynamic World image counts, and land-cover *class* — unlike LST — isn't
    seasonally distorted, so full years are used directly.
    """
    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(sg_bbox).select("label")
    dw_early = dw.filterDate(early_start, early_end)
    dw_late = dw.filterDate(late_start, late_end)

    n_early, n_late = dw_early.size().getInfo(), dw_late.size().getInfo()
    print(f"Dynamic World images — early ({early_start} to {early_end}): {n_early}")
    print(f"Dynamic World images — late ({late_start} to {late_end}): {n_late}")
    if n_early == 0 or n_late == 0:
        raise RuntimeError("Zero images in one period — cannot compute a meaningful change comparison.")
    if n_early < 20 or n_late < 20:
        print("⚠️  Low image count in at least one period — treat resulting flags as indicative, not definitive.")

    early_class = dw_early.reduce(ee.Reducer.mode()).rename("early_class").clip(sg_bbox)
    late_class = dw_late.reduce(ee.Reducer.mode()).rename("late_class").clip(sg_bbox)
    return early_class, late_class, n_early, n_late


def overall_change_fraction(early_class, late_class, sg_bbox, scale) -> float:
    """A single island-wide number: what fraction of Singapore's land area
    shows a different dominant class between the two periods. Doesn't say
    WHERE — that's what the per-subzone flags below are for."""
    changed_mask = early_class.neq(late_class).rename("changed")
    stats = changed_mask.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
        geometry=sg_bbox, scale=scale, maxPixels=1e10, bestEffort=True,
    )
    fraction = stats.get("changed_mean").getInfo()
    count = stats.get("changed_count").getInfo()
    print(f"Overall map-wide change: {fraction * 100:.1f}% of {count:,} valid pixels changed dominant class.")
    return fraction


def zonal_land_change(early_class, late_class, subzones_fc, id_property, scale) -> pd.DataFrame:
    """Per-subzone change fraction, with per-period valid-pixel counts so a
    coverage gap can be pinned to "early", "late", or "both" instead of
    showing up as one opaque low-pixel-count number."""
    changed_mask = early_class.neq(late_class).rename("changed")
    diagnostic_img = changed_mask.addBands(early_class.rename("early")).addBands(late_class.rename("late"))

    zonal = diagnostic_img.reduceRegions(
        collection=subzones_fc,
        reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
        scale=scale, tileScale=4,
    )
    records = zonal.map(lambda f: ee.Feature(None, {
        "subzone_id": f.get(id_property),
        "change_fraction": f.get("changed_mean"),
        "n_valid_pixels": f.get("changed_count"),
        "n_valid_pixels_early": f.get("early_count"),
        "n_valid_pixels_late": f.get("late_count"),
    })).getInfo()

    df = pd.DataFrame([r["properties"] for r in records["features"]])
    n_null = df["change_fraction"].isna().sum()
    print(f"Zonal reduction: {len(df)} subzones, {n_null} with null (no valid pixels in either period).")
    return df


def flag_unstable_subzones(
    lc_df: pd.DataFrame, heat_subzone_ids: pd.Series,
    change_fraction_threshold: float, min_valid_pixels: int,
) -> pd.DataFrame:
    """Every heat-CSV subzone gets an EXPLICIT status — 'flagged', 'stable',
    or 'insufficient_data' — rather than letting subzones with zero valid
    Dynamic World pixels silently disappear. A downstream left-join treating
    "we don't know" the same as "confirmed stable" would be a materially
    false claim, not a harmless simplification.
    """
    heat_ids = set(heat_subzone_ids.astype(str))
    lc_df = lc_df[lc_df["subzone_id"].astype(str).isin(heat_ids)].copy()
    lc_df = lc_df.dropna(subset=["change_fraction"])

    lc_df["land_change_flag"] = (
        (lc_df["change_fraction"] > change_fraction_threshold) & (lc_df["n_valid_pixels"] >= min_valid_pixels)
    )
    lc_df["low_sample_size"] = lc_df["n_valid_pixels"] < min_valid_pixels

    full = pd.DataFrame({"subzone_id": sorted(heat_ids)})
    full = full.merge(
        lc_df[["subzone_id", "change_fraction", "n_valid_pixels", "n_valid_pixels_early",
               "n_valid_pixels_late", "land_change_flag", "low_sample_size"]],
        on="subzone_id", how="left",
    )

    def _status(row):
        if pd.isna(row["change_fraction"]) or row["low_sample_size"]:
            return "insufficient_data"
        return "flagged" if row["land_change_flag"] else "stable"

    full["status"] = full.apply(_status, axis=1)
    print("Status breakdown:\n" + full["status"].value_counts().to_string())
    return full[["subzone_id", "change_fraction", "n_valid_pixels", "n_valid_pixels_early",
                 "n_valid_pixels_late", "status"]]
