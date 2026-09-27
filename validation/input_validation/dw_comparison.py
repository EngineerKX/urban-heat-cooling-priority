"""Dynamic World vs WorldCover (checkpoint_DynamicWorld): (1) how much land
changed between the WorldCover year and the latest imagery, and (2) how well
each map agrees with the hand labels.

Both maps are collapsed to the project's 4 classes first. Change is measured
with Dynamic World on both sides (a WorldCover-vs-Dynamic-World difference
would mostly be a method difference, not change), and the same measure between
the two halves of the baseline window is the noise floor.

The validation sample was drawn by WorldCover class, so raw agreement over-weights
whatever WorldCover class was oversampled. `area_weighted_agreement` re-weights
each stratum to its real area share.
"""

import ee
import numpy as np
import pandas as pd

from config.settings import DW_TO_BUCKET_FROM, DW_TO_BUCKET_TO, RANDOM_SEED

DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
VEG, BUILT, BARE = 1, 2, 3
TRANSITIONS = {
    "veg_to_built": (VEG, BUILT), "built_to_veg": (BUILT, VEG),
    "veg_to_bare": (VEG, BARE), "bare_to_veg": (BARE, VEG),
    "bare_to_built": (BARE, BUILT), "built_to_bare": (BUILT, BARE),
}


def dw_bucket_image(sg_bbox, start: str, end: str) -> "ee.Image":
    """Most common Dynamic World class per pixel in [start, end), as bucket ids
    (band `c`; masked where there is no class, e.g. snow/ice)."""
    col = ee.ImageCollection(DW_COLLECTION).filterBounds(sg_bbox).select("label").filterDate(start, end)
    n_images = col.size().getInfo()
    print(f"Dynamic World {start} to {end}: {n_images} images")
    if n_images == 0:
        raise RuntimeError(f"No Dynamic World images between {start} and {end}.")
    bucket = col.reduce(ee.Reducer.mode()).remap(DW_TO_BUCKET_FROM, DW_TO_BUCKET_TO, 0)
    return bucket.updateMask(bucket.neq(0)).rename("c").clip(sg_bbox)


def zonal_change_table(baseline, half_1, half_2, recent, subzones_fc, id_property, scale) -> pd.DataFrame:
    """Per subzone: share of pixels whose class changed baseline -> recent
    (`chg`), the same between the two baseline halves (`chg_noise`), vegetation
    and built-up shares in each window, and the main class-to-class
    transitions. Only pixels valid in all four windows are counted (`n_valid`)."""
    bands = {
        "chg": baseline.neq(recent), "chg_noise": half_1.neq(half_2),
        "veg_base": baseline.eq(VEG), "veg_recent": recent.eq(VEG),
        "veg_half1": half_1.eq(VEG), "veg_half2": half_2.eq(VEG),
        "built_base": baseline.eq(BUILT), "built_recent": recent.eq(BUILT),
    }
    for name, (a, b) in TRANSITIONS.items():
        bands[name] = baseline.eq(a).And(recent.eq(b))

    stack = ee.Image.cat([img.rename(name) for name, img in bands.items()])
    valid = baseline.mask().And(recent.mask()).And(half_1.mask()).And(half_2.mask())
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
    records = stack.updateMask(valid).reduceRegions(
        collection=subzones_fc, reducer=reducer, scale=scale, tileScale=4,
    ).getInfo()["features"]

    rows = []
    for f in records:
        props = f["properties"]
        row = {"subzone_id": props.get(id_property), "n_valid": props.get("chg_count")}
        row.update({name: props.get(f"{name}_mean") for name in bands})
        rows.append(row)
    return pd.DataFrame(rows)


def sample_dw_at_points(baseline, recent, points_df: pd.DataFrame, scale) -> pd.DataFrame:
    """Dynamic World bucket at each point in both windows. Points with no valid
    pixel are missing from the result."""
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"point_id": r.point_id}) for r in points_df.itertuples()
    ])
    sampled = ee.Image.cat([baseline.rename("dw_baseline"), recent.rename("dw_recent")]).sampleRegions(
        collection=fc, scale=scale, geometries=False,
    ).getInfo()["features"]
    return pd.DataFrame([
        {"point_id": s["properties"]["point_id"], "dw_baseline": s["properties"].get("dw_baseline"),
         "dw_recent": s["properties"].get("dw_recent")}
        for s in sampled
    ])


def island_summary(zonal_df: pd.DataFrame) -> dict:
    """Island-wide share for every measure in `zonal_change_table`, weighting
    each subzone by its valid pixel count."""
    valid = zonal_df.dropna(subset=["chg"])
    weights = valid["n_valid"]
    measures = [c for c in valid.columns if c not in ("subzone_id", "n_valid")]
    return {c: float((valid[c] * weights).sum() / weights.sum()) for c in measures}


def _stratum_agreement(correct: pd.DataFrame, strata: pd.Series, shares: dict) -> dict:
    """Area-weighted mean of `correct` (bool columns) across strata; strata with
    no points drop out and the remaining shares are renormalised."""
    per_stratum = correct.groupby(strata.values).mean()
    weights = pd.Series({k: v for k, v in shares.items() if k in per_stratum.index})
    weights = weights / weights.sum()
    return (per_stratum.loc[weights.index].mul(weights, axis=0)).sum().to_dict()


def area_weighted_agreement(
    points_df: pd.DataFrame, area_shares: dict, maps: dict, strata_col: str = "worldcover_class",
    true_col: str = "true_bucket", n_boot: int = 2000, seed: int = RANDOM_SEED,
):
    """Agreement of each map with the hand labels, re-weighted so each stratum
    counts by its real area share, with a 95% interval from resampling points
    within each stratum.

    `maps` is {name: column of predicted bucket ids in `points_df`}. Returns
    (summary [map, agreement, ci_low, ci_high], boot: one row per resample, one
    column per map -- subtract two columns for a paired difference)."""
    correct = pd.DataFrame({name: points_df[col] == points_df[true_col] for name, col in maps.items()})
    strata = points_df[strata_col]
    estimate = _stratum_agreement(correct, strata, area_shares)

    rng = np.random.default_rng(seed)
    groups = {k: np.flatnonzero(strata.values == k) for k in strata.unique()}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in groups.values()])
        boots.append(_stratum_agreement(correct.iloc[idx], strata.iloc[idx], area_shares))
    boot = pd.DataFrame(boots)

    summary = pd.DataFrame({
        "map": list(maps), "agreement": [estimate[m] for m in maps],
        "ci_low": [boot[m].quantile(0.025) for m in maps], "ci_high": [boot[m].quantile(0.975) for m in maps],
    })
    return summary, boot
