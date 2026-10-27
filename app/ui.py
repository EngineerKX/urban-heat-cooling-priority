"""Display helpers shared by the app pages: one set of names, labels and
colours, so the same pillar or class reads the same on every page. Display
only -- the data keeps its raw keys (e.g. upper-case URA subzone names), and
pages pass those as widget values and format them for the screen.
"""

import re

# The three pillars of the priority score, keyed as in src/priority_score/score.py.
PILLAR_LABELS = {"exposure": "Heat", "sensitivity": "Vulnerable residents", "adaptive_deficit": "Missing greenery"}
PILLAR_COLORS = {"exposure": "#d6604d", "sensitivity": "#f2994a", "adaptive_deficit": "#27ae60"}

# Land-cover buckets, keyed as in src/ingest/worldcover.py's BUCKET_NAMES.
LANDCOVER_LABELS = {"vegetation": "Vegetation", "built_up": "Built-up", "bare": "Bare", "water": "Water"}
LANDCOVER_COLORS = {"vegetation": "#27ae60", "built_up": "#8c8c8c", "bare": "#c9a35d", "water": "#2f80ed"}

# Hotspot clusters (S4), named from data/processed/hotspot_cluster_profile.csv.
# Cluster ids are only stable for a given clustering run: re-check these names
# against the profile whenever scripts/build_hotspot_clusters.py is re-run.
CLUSTER_NAMES = {
    0: "Mixed, moderate",
    1: "Cool and green",
    2: "Low greenery, dense city core",
    3: "Hot, dense built-up",
}

_LOWER_WORDS = {"of", "and", "the"}
_UPPER_WORDS = {"mp"}


def subzone_name(subzone_id) -> str:
    """'NATIONAL UNIVERSITY OF S'PORE' -> "National University of S'pore",
    'MARINA EAST (MP)' -> 'Marina East (MP)'. Non-strings pass through."""
    if not isinstance(subzone_id, str):
        return subzone_id

    def _word(match):
        word = match.group(0).lower()
        if word in _UPPER_WORDS:
            return word.upper()
        if word in _LOWER_WORDS and match.start() > 0:
            return word
        return word[0].upper() + word[1:]

    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", _word, subzone_id)


def cluster_name(cluster_id) -> str:
    return CLUSTER_NAMES.get(int(cluster_id), f"Cluster {int(cluster_id)}")
