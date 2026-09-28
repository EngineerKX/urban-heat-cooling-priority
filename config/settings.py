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
RAW_CCI_MODIS_LST_DIR = RAW_DIR / "cci_modis_lst"

DIAGNOSTICS_DIR = PROCESSED_DIR / "diagnostics"

# ---------------------------------------------------------------------------
# Google Earth Engine
# ---------------------------------------------------------------------------
GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID", "nus-iss-urban-heat-sg")
GEE_SERVICE_ACCOUNT = os.environ.get("GEE_SERVICE_ACCOUNT", "")
GEE_PRIVATE_KEY_PATH = os.environ.get("GEE_PRIVATE_KEY_PATH", "")

# GCS bucket used for the U-Net TFRecord patch export/import (GEE's
# patch-export mechanism requires an async Export task to Drive or Cloud
# Storage — there's no direct synchronous "download to local disk" for it).
# The result is downloaded to data/interim/unet_patches/ right after export
# completes.
GEE_EXPORT_BUCKET = os.environ.get("GEE_EXPORT_BUCKET", "")

# Trained-model sync (Colab trains -> pushes here -> scripts/pull_models.py
# downloads locally) and Colab training-run summaries both reuse the export
# bucket under new prefixes by default — same already-configured bucket,
# no new bucket/IAM setup needed. Override via GCS_MODEL_BUCKET only if you
# want model artifacts on a different lifecycle policy than raw exports.
GCS_MODEL_BUCKET = os.environ.get("GCS_MODEL_BUCKET") or GEE_EXPORT_BUCKET

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

# Wet-season complement for S4's dry/wet hotspot-typology clustering (NOT a
# second C4 composite — this is a clustering FEATURE, C4 itself is still
# just the single dry-season composite above). NE monsoon, Singapore's
# climatologically wettest months. Checked against real Landsat scene counts
# before locking (validation/input_validation/season_window.py's
# count_landsat_scenes): 30 usable scenes at LANDSAT_CLOUD_COVER_MAX, well
# above MIN_DEFENSIBLE_SCENES=8 (vs. 45 for the dry window) — no widening
# needed.
WET_SEASON_MONTHS = [12, 1, 2]

# Fixed (exclusive) end date for every satellite composite and the Dynamic
# World training labels. YEARS runs to 2026, so without a cap each new month
# of imagery (e.g. Oct-Nov 2026 dry season, Dec 2026 wet season) would
# silently change every composite, and Dynamic World labels would change with
# every newly released image -- runs could not be reproduced. Applied in
# src/ingest/gee.py::date_filter_for_years_months, the single choke point all
# Landsat / Sentinel-2 / MODIS composites go through. "2026-07-01" keeps
# everything through June 2026; the last dry-season data in use is May 2026.
DATA_END_DATE = "2026-07-01"

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
# Land-cover training labels
# ---------------------------------------------------------------------------
# Which product the land-cover classifiers learn from: "dynamicworld" (the
# production choice since 2026-09-29) or "worldcover" (the original source,
# kept as the comparison). Neither is ever the validation answer key -- that is
# the 300 hand-labelled points. Dynamic World agreed with those points far
# better (area-weighted 81.9% vs 69.5%) and retraining on it lifted every
# model significantly; see docs/dynamic_world_vs_worldcover_2026-09-27.md and
# the MLflow batch dynamicworld_eval_2026-09-29b. Resolved to a label image by
# src/landcover/labels.py.
LANDCOVER_LABEL_SOURCE = "dynamicworld"

# WorldCover (training labels when LANDCOVER_LABEL_SOURCE = "worldcover")
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

# Population term of the sensitivity pillar: "density" (residents per km²) or
# "count" (raw residents). DECIDED 2026-09-19: density -- a per-area measure like
# the other two pillars (exposure is a subzone mean, greenery a fraction), and
# the Singapore heat-health work this project cites (Cities paper "Urban heat
# health risk assessment in Singapore...", abstract-level evidence only) weights
# risk by elderly DENSITY and PROPORTION. It is a planning judgement, not noise:
# it still moves several top-20 subzones, so the alternative is reported rather
# than hidden (validation/score_validation/sensitivity_specs.py, dashboard
# "Sensitivity specification" table). The 50/50 split above is still a placeholder.
SENSITIVITY_POPULATION_MEASURE = "density"

# A subzone needs at least this many residents to be RANKED. Below it the
# sensitivity pillar is undefined (NaN) and the subzone is left out of the score,
# the bands and every top-N. Why: the elderly share of a tiny population is
# unstable (SE ~ sqrt(p(1-p)/n): +-1.6 pp at 500 residents against an island share
# of 15%, +-5 pp at 50) and resident counts say nothing about workers -- before
# this rule Loyang West (220 residents, 86% elderly) was ranked #1 and Tuas North
# (30 residents) #20. 121 of 332 subzones fall under 500 (parks, reserves,
# industrial estates, water). Decided 2026-09-25; the top-20 is the same set of
# leavers at 500 and 1000. Set to 0 to rank everything (old behaviour).
MIN_RESIDENTS_FOR_RANKING = 500

# Which greenery-fraction source feeds the adaptive-capacity pillar's
# canonical `greenery_fraction` column: "landcover" (the validated RF/U-Net
# hybrid, see src/landcover/zonal.py) or "ndvi" (the older NDVI-threshold
# proxy above). Both get computed and compared (Spearman correlation
# printed by scripts/build_adaptive_capacity_pillar.py) regardless of this
# setting -- it only decides which one becomes `greenery_fraction`.
ADAPTIVE_CAPACITY_SOURCE = "landcover"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# S6 — confidence bands (bootstrap over validation error)
# ---------------------------------------------------------------------------
PRIORITY_SCORE_BOOTSTRAP_ITERATIONS = 1000
PRIORITY_SCORE_BAND_QUANTILES = (0.05, 0.50, 0.95)

# Which held-out LST source is the project's independent reference for Landsat
# exposure -- it sets the exposure noise level in the S6 bootstrap AND is the
# source behind the held-out agreement columns of the rank-impact table
# (src/priority_score/io.py::load_heldout):
# "modis" (MOD11A2 -- same physical quantity as Landsat LST, ~329 subzones)
# or "nea" (weather-station AIR temperature, only ~12 subzones). Either way
# the noise is the std of (Landsat - held-out) residuals AFTER removing their
# mean offset: a constant offset can't move any rank, so only the spread
# around it is noise. Still an upper bound on random error, because it also
# contains the mismatch between the held-out footprint (MODIS is 1km) and a
# subzone. See validation/score_validation/confidence_bands.py.
EXPOSURE_NOISE_SOURCE = "modis"

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
# Dynamic World vs WorldCover comparison (checkpoint_DynamicWorld)
# ---------------------------------------------------------------------------
# Same product on both sides, so differences between products don't count as
# "change". Windows are 18 months; the baseline is centred on the WorldCover year
# and its two halves give the noise floor (little real change within 2021).
DW_BASELINE_START, DW_BASELINE_END = "2020-10-01", "2022-04-01"
DW_BASELINE_SPLIT = "2021-07-01"
DW_RECENT_START, DW_RECENT_END = "2025-01-01", "2026-07-01"
# Dynamic World label -> bucket id: 0 water, 1 trees, 2 grass, 3 flooded vegetation,
# 4 crops, 5 shrub, 6 built, 7 bare, 8 snow/ice (masked; not in Singapore)
DW_TO_BUCKET_FROM = [0, 1, 2, 3, 4, 5, 6, 7, 8]
DW_TO_BUCKET_TO = [
    BUCKET_WATER, BUCKET_VEGETATION, BUCKET_VEGETATION, BUCKET_VEGETATION, BUCKET_VEGETATION,
    BUCKET_VEGETATION, BUCKET_BUILTUP, BUCKET_BARE, 0,
]

# Dynamic World training-label window: the mode class over this whole span,
# covering the SAME period as the satellite feature composite (YEARS, capped
# at DATA_END_DATE) rather than a single snapshot year like WorldCover's -- so
# a label reflects "the typical class over the period the model's features are
# drawn from". Land-cover class isn't seasonally distorted the way LST is (see
# validation/input_validation/land_change.py), so this uses full years, not
# DRY_SEASON_MONTHS. The end is fixed (DATA_END_DATE) so newly released images
# can't change the labels between runs. (The 2026-09-27/28 trial models used
# an open end of 2027-01-01, i.e. every image up to the day they were trained.)
DW_TRAIN_START, DW_TRAIN_END = f"{YEARS[0]}-01-01", DATA_END_DATE

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

# Moved out of src/landcover/unet.py (now unet_model.py/unet_data.py/
# unet_train.py/unet_infer.py) during the TF->PyTorch migration, matching
# this file's "only place constants live" convention — CNN_MODEL_SAVE_PATH
# below already lived here, U-Net's didn't, which was the inconsistency.
UNET_MODEL_SAVE_PATH = MODELS_DIR / "unet_landcover.keras"
UNET_CLASSIFIED_RASTER_PATH = PROCESSED_DIR / "landcover" / "unet_landcover.tif"
UNET_PROB_RASTER_PATH = PROCESSED_DIR / "landcover" / "unet_landcover_prob.tif"
UNET_TRAIN_PATCH_DIR = INTERIM_DIR / "unet_patches" / "train"
UNET_INFERENCE_PATCH_DIR = INTERIM_DIR / "unet_patches" / "inference"

# The production land-cover map: the ONE raster every downstream consumer
# reads (greenery pillar, hotspot land-cover fractions -> XGBoost features,
# the CNN heat model's land-cover input channels, the counterfactual tool).
# "unet" since 2026-09-29: on Dynamic World labels it scored highest (83.1%
# accuracy, macro F1 0.774); the RF+U-Net hybrid (81.8%) and RF (72.6%) remain
# as reported comparisons. The U-Net-vs-hybrid gap is within noise (McNemar
# p=0.29) and swapping them moves 1 of the top-20 subzones.
LANDCOVER_DIR = PROCESSED_DIR / "landcover"
LANDCOVER_PRODUCTION_MODEL = "unet"
LANDCOVER_RASTER_PATHS = {
    "rf": LANDCOVER_DIR / "rf_landcover.tif",
    "unet": UNET_CLASSIFIED_RASTER_PATH,
    "hybrid": LANDCOVER_DIR / "hybrid_landcover.tif",
}
if LANDCOVER_PRODUCTION_MODEL not in LANDCOVER_RASTER_PATHS:
    raise ValueError(f"LANDCOVER_PRODUCTION_MODEL must be one of {sorted(LANDCOVER_RASTER_PATHS)}.")
LANDCOVER_RASTER_PATH = LANDCOVER_RASTER_PATHS[LANDCOVER_PRODUCTION_MODEL]

# GCS prefixes: models/ mirrors MODELS_DIR (trained weights, source of
# truth after Colab training); unet_train_patches/ and
# unet_inference_patches/ mirror the GEE export prefixes already in use,
# now doubling as a persistent cross-session cache (see
# src/landcover/unet_data.py) since Colab's local disk is ephemeral and
# can't rely on the old "skip if already on disk" check alone.
UNET_MODEL_GCS_PREFIX = "models/unet_landcover"
UNET_TRAIN_PATCHES_GCS_PREFIX = "unet_train_patches"
UNET_INFERENCE_PATCHES_GCS_PREFIX = "unet_inference_patches"

# ---------------------------------------------------------------------------
# S5 — XGBoost + CNN predictive heat model (C2)
# ---------------------------------------------------------------------------
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8

# Same values as the UNET_* block above -- the CNN regressor reuses U-Net's
# backbone (src/heat_model/cnn_model.py::build_cnn_regressor wraps
# src/landcover/unet_model.py::build_unet_backbone), so both models are
# trained with identical hyperparameters today. Literal duplicates, not a
# shared UNET_* reference, so cnn_train.py and its notebook read
# unambiguously as CNN settings and the two can be tuned independently.
CNN_BATCH_SIZE = 8
CNN_EPOCHS = 30
CNN_LEARNING_RATE = 1e-3
CNN_BASE_FILTERS = 32
CNN_EARLY_STOP_PATIENCE = 5
CNN_TRAIN_VAL_SPLIT = 0.85

CNN_MODEL_SAVE_PATH = MODELS_DIR / "heat_cnn.keras"
CNN_MODEL_GCS_PREFIX = "models/heat_cnn"

# The one CNN training input Colab can't regenerate itself (needs local
# U-Net inference + RF combined via build_landcover_hybrid.py) — pushed
# manually after a local hybrid build, pulled by the CNN Colab notebook.
HYBRID_RASTER_GCS_PREFIX = "training_inputs/hybrid_raster"
# Where the production land-cover map (LANDCOVER_RASTER_PATH) is pushed for the
# CNN Colab notebook, which can't regenerate it. Separate from the hybrid
# prefix above so the old hybrid upload can't be picked up by mistake.
LANDCOVER_RASTER_GCS_PREFIX = "training_inputs/landcover_raster"

# The hand-labeled validation sample also can't be regenerated in Colab (it
# requires the Streamlit labeling app, a human, and the joint-labeling
# process) — pushed manually after relabeling, pulled by train_unet.ipynb
# to build the training region's exclusion zone.
VALIDATION_SAMPLE_GCS_PREFIX = "training_inputs/validation_sample_300_labeled"

# ---------------------------------------------------------------------------
# MODIS LST secondary cross-check (proposal's data plan table)
# ---------------------------------------------------------------------------
MODIS_LST_ASSET = "MODIS/061/MOD11A2"
MODIS_LST_SCALE_M = 1000

for _dir in (
    RAW_URA_SUBZONES_DIR, RAW_NEA_STATIONS_DIR, RAW_SINGSTAT_DIR,
    RAW_LANDSAT_DIR, RAW_SENTINEL2_DIR, RAW_WORLDCOVER_DIR, RAW_DYNAMIC_WORLD_DIR, RAW_CCI_MODIS_LST_DIR,
    INTERIM_DIR, PROCESSED_DIR, DIAGNOSTICS_DIR, MODELS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
