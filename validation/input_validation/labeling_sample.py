"""Stratified validation-sample generation (from generate_validation_sample.ipynb,
Track B / SB1): 200 points, stratified on WorldCover class proportions
(vegetation oversampled), restricted to Singapore's real land boundary and a
valid Sentinel-2 data footprint.

Scope note carried over from the notebook: this produces the VALIDATION
sample only. WorldCover here is used purely to stratify which points get
picked for hand-labeling — it is never the answer key (see
`worldcover_class` in the output: reference context for the labeler, not a
label). Validation itself uses Dynamic World + the resulting hand-labeled
sample, never WorldCover directly.
"""

from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.ingest.gee import fetch_sentinel2_collection
from src.ingest.worldcover import BUCKET_NAMES, get_worldcover_bucket_image

OVERSAMPLE_BUCKETS = [1]  # BUCKET_VEGETATION — see config.settings.BUCKET_VEGETATION


def build_valid_data_mask(sg_bbox, sg_boundary, years, months, cloud_prob_max, crs, target_scale):
    """Lightweight Sentinel-2 composite built ONLY to derive a valid-data
    mask (no bands from it are used as classifier input here)."""
    s2_masked = fetch_sentinel2_collection(sg_bbox, years, months, cloud_prob_max)
    composite = s2_masked.select("B4").median().clip(sg_boundary)
    composite = composite.reproject(crs=crs, scale=target_scale)
    return composite.mask()


def class_area_histogram(wc_sampling_frame, boundary, scale) -> dict:
    """Single reduceRegion + single .getInfo() — pull once, count locally."""
    raw = wc_sampling_frame.select("wc_class").reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(), geometry=boundary, scale=scale,
        maxPixels=1e13, bestEffort=True,
    ).get("wc_class").getInfo()
    class_hist = {int(k): v for k, v in raw.items()}
    total_px = sum(class_hist.values())
    if total_px == 0:
        raise RuntimeError("No valid pixels in sampling frame — check the composite/mask before proceeding.")
    for cls, px in sorted(class_hist.items(), key=lambda kv: -kv[1]):
        name = BUCKET_NAMES.get(cls, f"class_{cls}")
        print(f"  {cls:>3} {name:<20} {px:>10,} px  ({px / total_px * 100:5.1f}%)")
    return class_hist


def allocate_sample_sizes(
    class_hist: dict, total_points: int, veg_classes=OVERSAMPLE_BUCKETS,
    oversample_factor: float = 2.0, min_per_class: int = 8,
) -> dict:
    """Weight by raw area share (vegetation boosted by `oversample_factor`),
    renormalize to `total_points`, floor any present class at
    `min_per_class`, rebalance +/-1 at a time so the total still matches."""
    total_px = sum(class_hist.values())
    weights = {}
    for cls, px in class_hist.items():
        share = px / total_px
        if cls in veg_classes:
            share *= oversample_factor
        weights[cls] = share

    weight_sum = sum(weights.values())
    raw_alloc = {cls: (w / weight_sum) * total_points for cls, w in weights.items()}
    floored = {cls: max(min_per_class, round(n)) for cls, n in raw_alloc.items()}

    diff = total_points - sum(floored.values())
    if diff != 0:
        order = sorted(floored, key=lambda c: floored[c], reverse=(diff < 0))
        i = 0
        while diff != 0:
            cls = order[i % len(order)]
            if diff > 0:
                floored[cls] += 1
                diff -= 1
            elif floored[cls] > 1:
                floored[cls] -= 1
                diff += 1
            i += 1

    for cls, n in sorted(floored.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:>3} {BUCKET_NAMES.get(cls, f'class_{cls}'):<20} {n:>4} pts")
    return floored


def draw_stratified_sample(wc_sampling_frame, boundary, scale, class_alloc: dict, seed: int):
    """`stratifiedSample` can silently return FEWER points than requested
    for a class that doesn't have enough valid pixels in-region — it
    doesn't raise. Shortfalls are counted explicitly, not assumed away."""
    class_values = list(class_alloc.keys())
    class_points = [class_alloc[c] for c in class_values]

    samples_fc = wc_sampling_frame.stratifiedSample(
        numPoints=0, classBand="wc_class", region=boundary, scale=scale,
        classValues=class_values, classPoints=class_points, seed=seed,
        geometries=True, dropNulls=True, tileScale=4,
    )
    sample_records = samples_fc.getInfo()["features"]
    print(f"Points drawn: {len(sample_records)} (requested {sum(class_points)})")

    drawn_counts = {}
    for f in sample_records:
        cls = int(f["properties"]["wc_class"])
        drawn_counts[cls] = drawn_counts.get(cls, 0) + 1

    shortfalls = {cls: (req, drawn_counts.get(cls, 0)) for cls, req in class_alloc.items() if drawn_counts.get(cls, 0) < req}
    if shortfalls:
        print("⚠️  Shortfalls (requested vs actually drawn):")
        for cls, (req, got) in shortfalls.items():
            print(f"   {BUCKET_NAMES.get(cls, f'class_{cls}')}: requested {req}, got {got}")
    else:
        print("✅ No shortfalls — every class hit its requested allocation.")

    return sample_records, drawn_counts, shortfalls


def build_labeling_table(sample_records) -> pd.DataFrame:
    from config.settings import ORIGINAL_WC_NAMES

    rows = []
    for i, f in enumerate(sample_records, start=1):
        lon, lat = f["geometry"]["coordinates"]
        bucket = int(f["properties"]["wc_class"])
        raw_code = int(f["properties"]["wc_raw"])
        rows.append({
            "point_id": f"P{i:04d}",
            "lon": lon,
            "lat": lat,
            "worldcover_class": bucket,
            "worldcover_class_name": BUCKET_NAMES.get(bucket, f"class_{bucket}"),
            "worldcover_class_raw": raw_code,
            "worldcover_class_raw_name": ORIGINAL_WC_NAMES.get(raw_code, f"class_{raw_code}"),
            # filled in during labeling — the WorldCover columns above are
            # reference context only, never the answer.
            "agreed_label": "",
            "confidence": "",
            "notes": "",
        })
    return pd.DataFrame(rows)


def export_labeling_table(labeling_df: pd.DataFrame, out_dir: Path, prefix: str = "validation_sample_200"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = out_dir / f"{prefix}.geojson"
    csv_path = out_dir / f"{prefix}.csv"

    gdf = gpd.GeoDataFrame(
        labeling_df, geometry=[Point(xy) for xy in zip(labeling_df["lon"], labeling_df["lat"])], crs="EPSG:4326",
    )
    gdf.to_file(geojson_path, driver="GeoJSON")
    labeling_df.to_csv(csv_path, index=False)
    print(f"Wrote {len(gdf)} points to:\n  {geojson_path}\n  {csv_path}")
    print("Class distribution:\n" + labeling_df["worldcover_class_name"].value_counts().to_string())
    return geojson_path, csv_path
