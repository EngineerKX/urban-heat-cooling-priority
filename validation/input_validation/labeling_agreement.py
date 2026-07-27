"""Inter-annotator agreement check (from urban_heat_sg_week1_gates.ipynb's
G5): both team members independently label the same shared points, then
Cohen's kappa quantifies whether the labeling protocol is consistent enough
to scale up (originally used for a 20-point trial before the full 300-point
hand-labeled sample; reusable for any future labeling QA round).
"""

import ee
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

KAPPA_THRESHOLD = 0.60  # "substantial agreement" (Landis & Koch, 1977)


def generate_shared_points(sg_land_geom, n_points: int, seed: int) -> pd.DataFrame:
    """Fixed-seed random points on Singapore's actual landmass (not sea) —
    both members MUST load this same file/list rather than regenerating
    independently, to guarantee identical points."""
    random_points_fc = ee.FeatureCollection.randomPoints(region=sg_land_geom, points=n_points, seed=seed, maxError=10)
    coords = random_points_fc.geometry().coordinates().getInfo()
    return pd.DataFrame({
        "point_id": [f"pt_{i + 1:02d}" for i in range(n_points)],
        "lon": [c[0] for c in coords],
        "lat": [c[1] for c in coords],
    })


def kappa_interpretation(k: float) -> str:
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def agreement_report(labels_a: list, labels_b: list, point_ids: list, class_names: dict) -> dict:
    """Percent agreement + Cohen's kappa + confusion matrix + disagreement
    breakdown between two annotators' labels for the SAME points, in order."""
    assert len(labels_a) == len(labels_b) == len(point_ids), "Label lists and point_ids must be the same length."
    class_values = sorted(class_names.keys())
    assert all(v in class_values for v in list(labels_a) + list(labels_b)), \
        f"All labels must be one of {class_values}."

    import numpy as np

    agree = np.array(labels_a) == np.array(labels_b)
    pct_agreement = float(agree.mean())
    kappa = float(cohen_kappa_score(labels_a, labels_b))
    interpretation = kappa_interpretation(kappa)

    disagreements = pd.DataFrame({
        "point_id": point_ids,
        "member_a": [class_names[l] for l in labels_a],
        "member_b": [class_names[l] for l in labels_b],
        "agree": agree,
    })
    cm = confusion_matrix(labels_a, labels_b, labels=class_values)
    cm_df = pd.DataFrame(
        cm, index=[f"A:{class_names[i]}" for i in class_values], columns=[f"B:{class_names[i]}" for i in class_values],
    )

    print(f"Raw percent agreement: {pct_agreement * 100:.1f}% ({agree.sum()}/{len(point_ids)} points)")
    print(f"Cohen's kappa: {kappa:.3f} ({interpretation})")

    passed = kappa >= KAPPA_THRESHOLD
    if passed:
        print(f"✅ kappa >= {KAPPA_THRESHOLD} — labeling protocol is consistent enough to scale up.")
    else:
        print(f"⚠️  kappa < {KAPPA_THRESHOLD} — review disagreement cases together and clarify class "
              f"definitions before scaling to the full sample.")

    return {
        "pct_agreement": pct_agreement,
        "kappa": kappa,
        "interpretation": interpretation,
        "passed": passed,
        "disagreements": disagreements[~disagreements["agree"]],
        "confusion_matrix": cm_df,
    }
