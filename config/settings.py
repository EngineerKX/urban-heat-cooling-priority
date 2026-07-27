"""Single source of truth for constants that used to be copy-pasted across
every Colab notebook (AOI, season window, CRS, dataset IDs, class scheme,
seeds). If a value needs to change, change it here — nowhere else should
redefine it.

Migration note: `S2_CLOUD_PROB_MAX` was `40` in gee_heat_variants.ipynb but
`70` everywhere else (generate_validation_sample, train_rf_baseline,
train_unet all agree on 70, and one of them even comments "same as Track A"
even though Track A's own value had drifted to 40). Consolidated to `70`
here; override if that's wrong.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Every script/module prints status with UTF-8 symbols (checkmarks, warning
# signs). On Windows, the console's default codepage (e.g. cp950, cp1252)
# often can't encode those and crashes the print — not a pipeline bug, just
# a terminal encoding mismatch. Every script imports config.settings before
# printing anything, so reconfiguring stdout/stderr here fixes it everywhere
# at once instead of patching each print site.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = REPO_ROOT / "models"

# Per-source raw caches (mirrors the data/raw/* folders already in the repo).
RAW_URA_SUBZONES_DIR = RAW_DIR / "ura_subzones"
RAW_NEA_STATIONS_DIR = RAW_DIR / "nea_stations"
RAW_SINGSTAT_DIR = RAW_DIR / "singstat"
RAW_LANDSAT_DIR = RAW_DIR / "landsat"
RAW_SENTINEL2_DIR = RAW_DIR / "sentinel2"
RAW_WORLDCOVER_DIR = RAW_DIR / "worldcover"
RAW_DYNAMIC_WORLD_DIR = RAW_DIR / "dynamic_world"

DIAGNOSTICS_DIR = PROCESSED_DIR / "diagnostics"

# ---------------------------------------------------------------------------
# Google Earth Engine
# ---------------------------------------------------------------------------
GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID", "nus-iss-urban-heat-sg")
GEE_SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT", "")
GEE_PRIVATE_KEY_PATH = os.environ.get("GEE_PRIVATE_KEY_PATH", "")

# GCS bucket used only for the U-Net TFRecord patch export/import (GEE's
# patch-export mechanism requires an async Export task to Drive or Cloud
# Storage — there's no direct synchronous "download to local disk" for it).
# Using GCS instead of Drive keeps this off Colab entirely; the result is
# downloaded to data/interim/unet_patches/ right after export completes.
GEE_EXPORT_BUCKET = os.environ.get("GEE_EXPORT_BUCKET", "")

# ---------------------------------------------------------------------------
# AOI
# ---------------------------------------------------------------------------
# Singapore bounding box — cheap prefilter only. The real sampling/analysis
# boundary is the dissolved URA subzone polygon (see src/ingest/subzones.py).
SG_BBOX = (103.55, 1.15, 104.10, 1.48)  # (minLon, minLat, maxLon, maxLat)
SG_CENTER = (1.3521, 103.8198)  # (lat, lon)

# ---------------------------------------------------------------------------
# Season window (C4 — locked via season_window_diagnostic.ipynb)
# ---------------------------------------------------------------------------
# Both inter-monsoon periods (weak winds, low cloud, high insolation ==
# peak-heat conditions) — confirmed to give the most usable Landsat scenes
# of the candidates tested (see validation/input_validation/season_window.py).
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
DRY_SEASON_MONTHS = [4, 5, 10, 11]

LANDSAT_CLOUD_COVER_MAX = 70  # scene-metadata prefilter (locked Week-1 gate value)
S2_CLOUD_PROB_MAX = 70  # s2cloudless per-pixel probability threshold (see drift note above)

# ---------------------------------------------------------------------------
# Resolution / CRS
# ---------------------------------------------------------------------------
NATIVE_SCALE_M = 30  # Landsat thermal
TARGET_SCALE_M = 10  # Sentinel-2 / downscaled products
S2_UTM_CRS = "EPSG:32648"  # UTM Zone 48N — covers Singapore

# ---------------------------------------------------------------------------
# URA subzones (data.gov.sg)
# ---------------------------------------------------------------------------
SUBZONE_DATASET_ID = "d_8594ae9ff96d0c708bc2af633048edfb"  # MP19 Subzone Boundary (No Sea)
SUBZONE_ID_PROPERTY = "SUBZONE_N"

# ---------------------------------------------------------------------------
# WorldCover (training labels only — never the validation answer key)
# ---------------------------------------------------------------------------
WORLDCOVER_ASSET = "ESA/WorldCover/v200/2021"

WC_TREE, WC_SHRUB, WC_GRASS, WC_CROP = 10, 20, 30, 40
WC_BUILTUP, WC_BARE, WC_SNOWICE, WC_WATER = 50, 60, 70, 80
WC_WETLAND, WC_MANGROVE, WC_MOSSLICHEN = 90, 95, 100

ORIGINAL_WC_NAMES = {
    WC_TREE: "tree_cover", WC_SHRUB: "shrubland", WC_GRASS: "grassland",
    WC_CROP: "cropland", WC_BUILTUP: "built_up", WC_BARE: "bare_sparse_veg",
    WC_SNOWICE: "snow_ice", WC_WATER: "water", WC_WETLAND: "herbaceous_wetland",
    WC_MANGROVE: "mangroves", WC_MOSSLICHEN: "moss_lichen",
}

# 4-class gate-review scheme (matches the labeling tool's dropdown exactly).
BUCKET_VEGETATION, BUCKET_BUILTUP, BUCKET_BARE, BUCKET_WATER = 1, 2, 3, 4
BUCKET_NAMES = {
    BUCKET_VEGETATION: "vegetation",
    BUCKET_BUILTUP: "built_up",
    BUCKET_BARE: "bare",
    BUCKET_WATER: "water",
}

# Raw WorldCover code -> bucket id. Snow/ice -> sentinel 0 (masked out; not
# present in Singapore, not one of the 4 official classes).
WC_TO_BUCKET_FROM = [
    WC_TREE, WC_SHRUB, WC_GRASS, WC_CROP, WC_BUILTUP,
    WC_BARE, WC_SNOWICE, WC_WATER, WC_WETLAND, WC_MANGROVE, WC_MOSSLICHEN,
]
WC_TO_BUCKET_TO = [
    BUCKET_VEGETATION, BUCKET_VEGETATION, BUCKET_VEGETATION, BUCKET_VEGETATION,
    BUCKET_BUILTUP, BUCKET_BARE, 0, BUCKET_WATER,
    BUCKET_VEGETATION, BUCKET_VEGETATION, BUCKET_VEGETATION,
]

# ---------------------------------------------------------------------------
# Land-cover classifier feature bands
# ---------------------------------------------------------------------------
S2_FEATURE_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
INDEX_BANDS = ["NDVI", "NDBI", "NDWI"]
ALL_FEATURE_BANDS = S2_FEATURE_BANDS + INDEX_BANDS

VALIDATION_EXCLUSION_BUFFER_M = 15  # ~1.5x pixel, errs generous — non-circularity enforcement

# ---------------------------------------------------------------------------
# SingStat population (sensitivity pillar)
# ---------------------------------------------------------------------------
POP_DATASET_ID = "d_d95ae740c0f8961a0b10435836660ce0"  # Resident Population by Planning Area/Subzone
ELDERLY_AGE_COLUMNS = [
    "Total_65_69", "Total_70_74", "Total_75_79",
    "Total_80_84", "Total_85_89", "Total_90andOver",
]
TOTAL_POP_COLUMN = "Total_Total"
SINGSTAT_NAME_COLUMN = "Number"  # this dataset's subzone/planning-area name field

# ---------------------------------------------------------------------------
# Priority-score pillar placeholders — PRESERVE, do not silently "fix".
# These are explicit open decisions for whoever owns S6, carried over as-is
# from the notebooks that first flagged them.
# ---------------------------------------------------------------------------
NDVI_VEGETATION_THRESHOLD = 0.35  # placeholder proxy until real S3 land-cover fraction exists
SENSITIVITY_POPULATION_WEIGHT = 0.5  # placeholder 50/50 split vs elderly_proportion
SENSITIVITY_ELDERLY_WEIGHT = 0.5

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Rank-impact / ablation
# ---------------------------------------------------------------------------
TOP_N = 20
REFERENCE_VARIANT = "lst_native30"
VARIANT_COLUMNS = ["lst_native30", "lst_bicubic10", "lst_regress10"]

# ---------------------------------------------------------------------------
# Land-change diagnostic
# ---------------------------------------------------------------------------
LC_EARLY_START, LC_EARLY_END = "2018-01-01", "2021-01-01"
LC_LATE_START, LC_LATE_END = "2022-01-01", "2025-01-01"
LC_CHANGE_FRACTION_THRESHOLD = 0.15
LC_MIN_VALID_PIXELS = 30

# ---------------------------------------------------------------------------
# RF / U-Net hyperparameters
# ---------------------------------------------------------------------------
TRAINING_POINTS_PER_CLASS = 3000
RF_NUM_TREES = 200
RF_MIN_LEAF_POPULATION = 1
RF_BAG_FRACTION = 0.5

UNET_PATCH_SIZE = 128
UNET_BATCH_SIZE = 8
UNET_EPOCHS = 30
UNET_LEARNING_RATE = 1e-3
UNET_BASE_FILTERS = 32
UNET_EARLY_STOP_PATIENCE = 5
UNET_TRAIN_VAL_SPLIT = 0.85

for _dir in (
    RAW_URA_SUBZONES_DIR, RAW_NEA_STATIONS_DIR, RAW_SINGSTAT_DIR,
    RAW_LANDSAT_DIR, RAW_SENTINEL2_DIR, RAW_WORLDCOVER_DIR, RAW_DYNAMIC_WORLD_DIR,
    INTERIM_DIR, PROCESSED_DIR, DIAGNOSTICS_DIR, MODELS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
