"""Land-cover classifier swap: does choosing a different land-cover model
change which subzones are prioritised? The classifiers are judged on
accuracy/F1 (validation/landcover_validation/); this asks the decision-level
question the same way rank_impact.py does for heat layers and
sensitivity_specs.py does for the sensitivity formula -- swap ONE input (the
greenery fraction that feeds the adaptive-capacity pillar), hold everything
else fixed (exposure, sensitivity, PCA weighting), and count how many of the
top-N subzones move versus the production (hybrid) score.

Worth knowing when reading the result: only the VEGETATION share of each
raster reaches the score, so a classifier's errors on other classes (e.g.
U-Net's zero recall on "bare") matter only insofar as they move pixels into or
out of vegetation. And a constant misclassification rate is an affine
distortion of greenery that the score's min-max normalisation (and the
standardised PCA) cancel exactly -- only errors that vary from subzone to
subzone can move a rank.
"""

import pandas as pd
from scipy.stats import spearmanr

from config.settings import REFERENCE_VARIANT, TOP_N
from src.priority_score.score import build_score


def run_landcover_swap(
    df: pd.DataFrame, greenery_columns: dict, reference_label: str,
    exposure_col: str = REFERENCE_VARIANT, top_n: int = TOP_N,
) -> pd.DataFrame:
    """`greenery_columns` maps a label (e.g. "RF") to the column of `df` holding
    that source's per-subzone greenery fraction; `reference_label` names the
    source the others are compared against (the production one). `df` also needs
    subzone_id, `exposure_col` and sensitivity_raw.

    Subzones with a missing value in ANY compared column are dropped from every
    row (n_subzones says how many are left), so all rows rank the same set.

    Returns one row per source, reference first: how similar its greenery is
    to the reference's (Spearman), how similar the resulting priority ranking
    is (Spearman), how many top-N subzones changed, and which ones left/entered.
    """
    if reference_label not in greenery_columns:
        raise ValueError(f"reference_label '{reference_label}' not in {sorted(greenery_columns)}.")

    needed = list(greenery_columns.values()) + [exposure_col, "sensitivity_raw"]
    common = df.dropna(subset=needed).reset_index(drop=True)

    scores = {
        label: build_score(common, exposure_col, "pca", adaptive_capacity_col=col)[0]
        for label, col in greenery_columns.items()
    }

    def _top(score):
        return set(common.loc[score.sort_values(ascending=False).index[:top_n], "subzone_id"].astype(str))

    reference_col, reference_score, reference_top = (
        greenery_columns[reference_label], scores[reference_label], _top(scores[reference_label]),
    )

    ordered = [reference_label] + [label for label in greenery_columns if label != reference_label]
    rows = []
    for label in ordered:
        col, top = greenery_columns[label], _top(scores[label])
        # The verdict columns come first: a wide table is clipped on the right in the dashboard.
        rows.append({
            "greenery_source": label,
            f"top{top_n}_changed": top_n - len(top & reference_top),
            "score_spearman_vs_reference": float(spearmanr(scores[label], reference_score)[0]),
            "greenery_spearman_vs_reference": float(spearmanr(common[col], common[reference_col])[0]),
            "greenery_mean": float(common[col].mean()),
            "n_subzones": len(common),
            "is_reference": label == reference_label,
            "left_top": "; ".join(sorted(reference_top - top)),
            "entered_top": "; ".join(sorted(top - reference_top)),
        })

    result = pd.DataFrame(rows)
    print(result.drop(columns=["left_top", "entered_top"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return result
