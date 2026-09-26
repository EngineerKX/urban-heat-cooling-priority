"""The two views ("lenses") of the priority score, defined once so the build
scripts and the app can't drift apart.

- residents: heat + resident sensitivity + missing greenery (PCA-weighted).
  Ranks only subzones with at least config.settings.MIN_RESIDENTS_FOR_RANKING
  residents, because a resident-based sensitivity can't be estimated below that.
- all_places: heat + missing greenery only (equal weights), for EVERY subzone,
  so hot industrial and port areas are included. It has no population term:
  it says where it is hottest and least green, not how many people or workers
  are exposed.
"""

from config.settings import PROCESSED_DIR

LENSES = {
    "residents": {
        "label": "Residents — heat, people and greenery",
        "description": "Ranks subzones where heat, residents and missing greenery coincide. Subzones with too few "
                       "residents are not ranked.",
        "weighting": "pca",
        "include_unranked": False,
        "score_csv": PROCESSED_DIR / "priority_score.csv",
        "bands_csv": PROCESSED_DIR / "priority_score_confidence_bands.csv",
    },
    "all_places": {
        "label": "All places — heat and greenery only",
        "description": "Ranks every subzone, including industrial and port areas, on heat and missing greenery only. "
                       "It has no population or workforce term, so it does not say how many people are exposed.",
        "weighting": "heat_greenery",
        "include_unranked": True,
        "score_csv": PROCESSED_DIR / "priority_score_all_places.csv",
        "bands_csv": PROCESSED_DIR / "priority_score_all_places_confidence_bands.csv",
    },
}
DEFAULT_LENS = "residents"
