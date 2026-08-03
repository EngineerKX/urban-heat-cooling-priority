#!/usr/bin/env python
"""Standalone verification for
validation/landcover_validation/classifier_evaluation.py against a small
synthetic raster + hand-built validation points -- no GEE/GPU required.
Same "runnable script, printed pass/fail" convention as
tests/test_landcover_ensemble.py (see that file for why no pytest).

Usage: python tests/test_classifier_evaluation.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import Transformer

from validation.landcover_validation.classifier_evaluation import (
    evaluate_classifier,
    sample_raster_at_points,
    score_predictions,
)

CRS = "EPSG:32648"


def _write_label_raster(path, label_array, transform):
    with rasterio.open(
        path, "w", driver="GTiff", height=label_array.shape[0], width=label_array.shape[1],
        count=1, dtype=label_array.dtype, crs=CRS, transform=transform,
    ) as dst:
        dst.write(label_array, 1)


def _pixel_center_lonlat(transform, to_lonlat, row, col):
    x, y = transform * (col + 0.5, row + 0.5)
    return to_lonlat.transform(x, y)


def test_sample_raster_at_points(tmp_dir):
    transform = Affine(10.0, 0.0, 100000.0, 0.0, -10.0, 200000.0)
    label_array = np.array([[1, 2], [3, 4]], dtype=np.uint8)  # veg, built_up / bare, water
    path = tmp_dir / "labels.tif"
    _write_label_raster(path, label_array, transform)

    to_lonlat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    rows = []
    for r in range(2):
        for c in range(2):
            lon, lat = _pixel_center_lonlat(transform, to_lonlat, r, c)
            rows.append({"point_id": f"pt_{r}_{c}", "lon": lon, "lat": lat})
    validation_df = pd.DataFrame(rows)

    sampled = sample_raster_at_points(path, validation_df)
    expected = {"pt_0_0": 1, "pt_0_1": 2, "pt_1_0": 3, "pt_1_1": 4}
    for _, row in sampled.iterrows():
        assert row["pred_bucket"] == expected[row["point_id"]], (
            f"{row['point_id']}: expected {expected[row['point_id']]}, got {row['pred_bucket']}"
        )
    print("PASS: sample_raster_at_points recovers exact pixel values at pixel-center coordinates")


def test_score_predictions():
    df = pd.DataFrame({
        "true_bucket": [1, 1, 2, 2, 3, 4],
        "pred_bucket": [1, 2, 2, 2, 3, 3],  # 1 veg misclassified as built_up, 1 water misclassified as bare
    })
    result = score_predictions(df, "true_bucket", "pred_bucket")
    assert abs(result["accuracy"] - 4 / 6) < 1e-9, result["accuracy"]

    cm = result["confusion_matrix"]
    assert cm.loc["true:vegetation", "pred:built_up"] == 1
    assert cm.loc["true:built_up", "pred:built_up"] == 2
    assert cm.loc["true:bare", "pred:bare"] == 1
    assert cm.loc["true:water", "pred:bare"] == 1

    per_class = result["per_class_metrics"].set_index("class")
    assert per_class.loc["water", "f1"] == 0.0, "water was never predicted correctly -> F1 must be 0"
    assert per_class.loc["built_up", "recall"] == 1.0, "both built_up points were recalled correctly"
    print("PASS: score_predictions confusion matrix + per-class F1")


def test_evaluate_classifier_end_to_end(tmp_dir):
    transform = Affine(10.0, 0.0, 100000.0, 0.0, -10.0, 200000.0)
    label_array = np.array([[1, 2], [3, 0]], dtype=np.uint8)  # last pixel nodata
    path = tmp_dir / "labels2.tif"
    _write_label_raster(path, label_array, transform)

    to_lonlat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    labels = ["vegetation", "built_up", "bare", "water"]  # "water" point sits on the nodata pixel
    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rows = []
    for (r, c), true_label in zip(coords, labels):
        lon, lat = _pixel_center_lonlat(transform, to_lonlat, r, c)
        rows.append({"point_id": f"pt_{r}_{c}", "lon": lon, "lat": lat, "agreed_label": true_label})
    validation_df = pd.DataFrame(rows)

    result = evaluate_classifier(path, validation_df, "test_model")
    assert result["n_scored"] == 3, "the nodata (0) pixel's validation point must be dropped, not scored"
    assert abs(result["accuracy"] - 1.0) < 1e-9, "the 3 scoreable points all match exactly"
    print("PASS: evaluate_classifier drops nodata points and scores the rest correctly")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        test_sample_raster_at_points(tmp_dir)
        test_score_predictions()
        test_evaluate_classifier_end_to_end(tmp_dir)
    print("\nAll classifier_evaluation.py checks passed.")


if __name__ == "__main__":
    main()
