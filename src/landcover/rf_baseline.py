"""Random Forest land-cover baseline (Track B / RF1): Sentinel-2 bands +
NDVI/NDBI/NDWI -> 4-class WorldCover-trained classifier -> full-Singapore
classified raster.

Non-circularity is enforced, not assumed: training pixels are drawn from a
region that explicitly excludes a buffer around every validation point (see
`build_training_region`), with a printed sanity check on how much area that
actually excluded.

Caching: `ee.Classifier` objects aren't locally serializable the way a
scikit-learn model is, so instead of a local file cache, a trained
classifier is persisted as a GEE asset (`ee.batch.Export.classifier.toAsset`)
and reloaded from there on the next run instead of retraining. This is
optional (`use_asset_cache=True` by default) and fails soft — if your GEE
project doesn't have a usable asset root configured yet, it just retrains.
"""

from pathlib import Path

import ee
import pandas as pd

from config.settings import (
    ALL_FEATURE_BANDS,
    GEE_EXPORT_BUCKET,
    GEE_PROJECT_ID,
    PROCESSED_DIR,
    RF_BAG_FRACTION,
    RF_MIN_LEAF_POPULATION,
    RF_NUM_TREES,
    S2_FEATURE_BANDS,
    S2_UTM_CRS,
    TARGET_SCALE_M,
    TRAINING_POINTS_PER_CLASS,
    VALIDATION_EXCLUSION_BUFFER_M,
)
from src.ingest.gee import add_spectral_indices, export_geotiff_to_gcs, fetch_sentinel2_collection
from src.ingest.worldcover import BUCKET_NAMES, get_worldcover_bucket_image

RF_CLASSIFIER_ASSET_ID = f"projects/{GEE_PROJECT_ID}/assets/rf_landcover_classifier"
RF_RASTER_PATH = PROCESSED_DIR / "landcover" / "rf_landcover.tif"
RF_PROB_RASTER_PATH = PROCESSED_DIR / "landcover" / "rf_landcover_prob.tif"


def build_feature_image(sg_bbox, boundary, years, months, cloud_prob_max, crs=S2_UTM_CRS, scale=TARGET_SCALE_M):
    """Season-controlled Sentinel-2 composite + NDVI/NDBI/NDWI, the exact
    feature set both RF and U-Net train on (kept identical between the two
    so the model-choice comparison isn't confounded by input drift)."""
    s2_masked = fetch_sentinel2_collection(sg_bbox, years, months, cloud_prob_max)
    composite_bands = s2_masked.select(S2_FEATURE_BANDS).median().clip(boundary)
    composite_bands = composite_bands.reproject(crs=crs, scale=scale)

    with_indices = add_spectral_indices(composite_bands)
    feature_image = with_indices.select(ALL_FEATURE_BANDS)
    valid_mask = feature_image.select("B4").mask()
    return feature_image, valid_mask


def build_training_region(boundary, validation_df: pd.DataFrame, buffer_m=VALIDATION_EXCLUSION_BUFFER_M):
    """Singapore boundary minus a buffer around every validation point —
    what makes the no-circularity claim checkable, not just asserted."""
    val_geoms = [ee.Geometry.Point([row.lon, row.lat]) for row in validation_df.itertuples()]
    val_points_fc = ee.FeatureCollection([ee.Feature(g) for g in val_geoms])
    val_buffer = val_points_fc.geometry().buffer(buffer_m)

    training_region = boundary.difference(val_buffer, ee.ErrorMargin(1))

    sg_area = boundary.area(1).getInfo()
    training_area = training_region.area(1).getInfo()
    excluded_area = sg_area - training_area
    print(f"Singapore boundary area: {sg_area / 1e6:,.2f} km²")
    print(f"Training region area (post-exclusion): {training_area / 1e6:,.2f} km²")
    print(f"Excluded around validation points: {excluded_area:,.0f} m²")
    if excluded_area < 1000:
        print("⚠️  Exclusion area suspiciously small — check validation_df before trusting this.")
    else:
        print("✅ Validation points are spatially excluded from the training region.")
    return training_region, excluded_area


def extract_training_samples(
    feature_image, wc_bucket_image, training_region, scale=TARGET_SCALE_M,
    points_per_class=TRAINING_POINTS_PER_CLASS, seed=42,
):
    """Stratified sample, capped per class. Server-side histogram for the
    per-class counts instead of pulling every feature locally — Earth Engine
    caps synchronous FeatureCollection pulls at 5000 elements, and 4 classes
    x up to `points_per_class` each can exceed that."""
    training_image = ee.Image.cat([feature_image, wc_bucket_image.select("wc_class")])
    class_values = list(BUCKET_NAMES.keys())
    class_points = [points_per_class] * len(class_values)

    training_fc = training_image.stratifiedSample(
        numPoints=0, classBand="wc_class", region=training_region, scale=scale,
        classValues=class_values, classPoints=class_points, seed=seed,
        geometries=False, dropNulls=True, tileScale=8,
    )
    n_training = training_fc.size().getInfo()
    print(f"Training samples drawn: {n_training} (requested up to {sum(class_points)})")

    train_counts_raw = training_fc.aggregate_histogram("wc_class").getInfo()
    train_counts = {int(k): v for k, v in train_counts_raw.items()}
    for cls, n in sorted(train_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cls} {BUCKET_NAMES.get(cls, f'class_{cls}'):<12} {n}")
    return training_fc, train_counts


def _classifier_asset_exists(asset_id: str) -> bool:
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def train_rf_classifier(
    training_fc, feature_bands=ALL_FEATURE_BANDS, num_trees=RF_NUM_TREES,
    min_leaf_population=RF_MIN_LEAF_POPULATION, bag_fraction=RF_BAG_FRACTION,
    seed=42, use_asset_cache: bool = True,
):
    """Train (or reload from a GEE asset cache) the RF classifier."""
    if use_asset_cache and _classifier_asset_exists(RF_CLASSIFIER_ASSET_ID):
        print(f"Loading cached classifier from asset: {RF_CLASSIFIER_ASSET_ID}")
        return ee.Classifier.load(RF_CLASSIFIER_ASSET_ID)

    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=num_trees, minLeafPopulation=min_leaf_population,
        bagFraction=bag_fraction, seed=seed,
    ).train(features=training_fc, classProperty="wc_class", inputProperties=feature_bands)

    schema = classifier.schema().getInfo()
    print(f"RF trained. Input properties: {schema}")

    if use_asset_cache:
        try:
            task = ee.batch.Export.classifier.toAsset(
                classifier=classifier, description="rf_landcover_classifier", assetId=RF_CLASSIFIER_ASSET_ID,
            )
            task.start()
            print(f"Persisting trained classifier to asset (async): {RF_CLASSIFIER_ASSET_ID}")
        except Exception as e:
            print(f"⚠️  Could not export classifier to a GEE asset ({e}) — will retrain next run.")

    return classifier


def classify(feature_image, classifier, boundary):
    return feature_image.classify(classifier).rename("rf_class").clip(boundary)


def classify_probability(feature_image, classifier, boundary, valid_mask, class_names=BUCKET_NAMES):
    """Same feature_image, MULTIPROBABILITY output mode instead of hard
    labels. `classifier` MUST be one trained fresh in the current session --
    empirically confirmed that a classifier round-tripped through
    Export.classifier.toAsset / ee.Classifier.load() only supports
    CLASSIFICATION mode; calling setOutputMode("MULTIPROBABILITY") on a
    loaded asset classifier fails with "Trainer only supports the mode
    'CLASSIFICATION'". See scripts/train_landcover_rf.py's
    effective_asset_cache handling.

    Band order verified empirically (not just per GEE's docs, which just say
    "in class order" without specifying which): sampled a mixed-landcover
    region and confirmed argmax(these 4 bands) matches the same classifier's
    own classify() output at ~97% of pixels, with the ~3% disagreement
    concentrated in built_up/bare near-ties (spectrally similar classes) --
    the signature of genuine close votes, not a band-order bug. Re-verify
    (download_small_image against a small region, compare to classify())
    if this function or GEE's classifier behavior ever changes.

    Adds an explicit `valid_mask` band from build_feature_image's own mask
    (not inferred later from nodata) so downstream ensemble alignment never
    has to guess which pixels are real — a probability of exactly 0.0 for a
    class at a genuinely valid pixel is legitimate and must not be confused
    with "outside the mask".
    """
    class_values = sorted(class_names.keys())
    band_names = [f"prob_{class_names[v]}" for v in class_values]
    prob_classifier = classifier.setOutputMode("MULTIPROBABILITY")
    prob_image = feature_image.classify(prob_classifier).arrayFlatten([band_names])
    prob_image = prob_image.addBands(valid_mask.rename("valid_mask")).toFloat()
    return prob_image.clip(boundary)


def informal_accuracy_check(classified_image, validation_df: pd.DataFrame, scale=TARGET_SCALE_M):
    """Sanity check only — NOT the formal RF-vs-U-Net-vs-ensemble evaluation."""
    validation_df = validation_df.copy()
    validation_df["agreed_label"] = validation_df["agreed_label"].fillna("").astype(str)
    valid_rows = validation_df[validation_df["agreed_label"].isin(BUCKET_NAMES.values())].copy()
    n_skipped = len(validation_df) - len(valid_rows)
    if n_skipped:
        print(f"Skipping {n_skipped} validation point(s) with blank/uncertain labels.")

    val_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([row.lon, row.lat]), {"point_id": row.point_id})
        for row in valid_rows.itertuples()
    ])
    sampled = classified_image.sampleRegions(collection=val_fc, scale=scale, geometries=False, tileScale=8)
    sampled_records = sampled.getInfo()["features"]
    pred_by_id = {f["properties"]["point_id"]: f["properties"].get("rf_class") for f in sampled_records}

    bucket_name_to_id = {v: k for k, v in BUCKET_NAMES.items()}
    valid_rows["rf_pred_bucket"] = valid_rows["point_id"].map(pred_by_id)
    valid_rows["rf_pred_name"] = valid_rows["rf_pred_bucket"].map(BUCKET_NAMES)
    valid_rows["true_bucket"] = valid_rows["agreed_label"].map(bucket_name_to_id)

    scored = valid_rows.dropna(subset=["rf_pred_bucket"])
    accuracy = (scored["rf_pred_bucket"] == scored["true_bucket"]).mean()
    crosstab = pd.crosstab(scored["agreed_label"], scored["rf_pred_name"])
    print(f"Informal RF accuracy on {len(scored)} validation points: {accuracy * 100:.1f}%")
    print("(Sanity check only — run the formal evaluation once U-Net + ensemble also exist.)")
    return accuracy, crosstab


def export_classified_raster(classified_image, boundary, scale=TARGET_SCALE_M, crs=S2_UTM_CRS,
                              out_path=RF_RASTER_PATH, bucket=GEE_EXPORT_BUCKET):
    """Export the full-Singapore classified raster via Cloud Storage and
    download it straight to local disk (no Drive hop, no geemap dependency —
    see `src/ingest/gee.py::export_geotiff_to_gcs` for why).

    description/prefix are derived from out_path's stem (not hardcoded) so
    that calling this twice in one run with different out_paths -- as
    scripts/train_landcover_rf.py does for the hard-label vs. probability
    rasters -- targets two distinct GCS objects instead of both silently
    overwriting the same blob."""
    stem = Path(out_path).stem
    return export_geotiff_to_gcs(
        classified_image, description=stem, bucket=bucket,
        prefix=f"rf_landcover/{stem}", region=boundary, scale=scale, crs=crs, out_path=out_path,
    )
