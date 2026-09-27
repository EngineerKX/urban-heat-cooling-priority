# Dynamic World vs WorldCover — checkpoint notes (2026-09-27)

Branch: `snapshot/checkpoint_DynamicWorld`. This is a checkpoint taken **before**
any decision to retrain the land-cover models on Dynamic World labels. Nothing
about the trained models or the priority score changed on this branch.

## Why this exists

The land-cover models (RF, U-Net, hybrid) are trained on ESA WorldCover v200
(2021) and agree with the hand labels about 66–70% of the time. Two questions
came up: (1) is WorldCover out of date (imagery window is 2021–2026), and (2) is
Dynamic World a better label source? The original label protocol was WorldCover
for training, Dynamic World + hand labels for validation. In practice scoring
uses the hand labels only; Dynamic World was used only in the land-change
diagnostic (`scripts/diagnose_land_change.py`), which compares Dynamic World with
itself (2018–2020 vs 2022–2024) and does not test WorldCover's age.

## How to reproduce

```
python scripts/compare_worldcover_dynamicworld.py [--force]
python tests/test_dw_comparison.py
```

Earth Engine results are cached in `data/interim/dw_vs_worldcover/` (gitignored).
Windows and the Dynamic World → 4-class mapping are in `config/settings.py`
(`DW_*`). Code: `validation/input_validation/dw_comparison.py`.

Method: Dynamic World's most common class per pixel, collapsed to the 4 classes,
over 18-month windows — baseline 2020-10 → 2022-04 (21 images, centred on 2021)
and recent 2025-01 → 2026-07 (25 images). Same product on both sides, so method
differences between products do not count as change. Noise floor = the same
measure between the two halves of the baseline (11 and 10 images, so if anything
overstated).

## Result 1 — land change ~2021 → 2025/26

| Measure | Value |
|---|---|
| Pixels that changed class, island-wide | 12.5% (noise floor 7.0%) |
| Vegetation share | 30.8% → 32.1% (noise: 30.9% vs 31.3%) |
| Built-up share | 52.5% → 52.4% |
| Subzones changing > 15% | 60 of 332 (noise alone would give 17) |
| Top 20 priority subzones | median change 0.4%, max 8.4% (Sungei Road), none above 15% |

Net change is small: vegetation→built-up (2.4% of pixels) and built-up→vegetation
(3.0%) roughly cancel. The top-20 inputs are not sensitive to post-2021 change.
The score was **not** re-run with updated land cover; this is judged from how much
the top-20 subzones changed.

## Result 2 — agreement with the 300 hand labels

Re-weighted to Singapore's real WorldCover area shares (the sample was drawn by
WorldCover class), 95% interval from resampling within each stratum, n = 293:

| Map | Agreement |
|---|---|
| WorldCover 2021 | 69.5% (65–74%) |
| Hybrid model | 72.4% (67–77%) |
| Dynamic World, baseline (~2021) | 81.9% (77–86%) |
| Dynamic World, recent (2025–26) | 85.5% (82–89%) |

Dynamic World minus WorldCover: +12.4 points (8 to 17). Dynamic World minus the
hybrid: +9.5 points (5 to 14). Recent minus baseline Dynamic World: +3.6 points
(0 to 7) — borderline, not clearly different.

An ad hoc check (not in the script) found the gain is concentrated where
WorldCover calls built-up land vegetation: of the hand-labelled built-up points,
Dynamic World got 78% right, WorldCover 47%, the hybrid 55%.

## Caveats

- Dynamic World is itself a model-made map; its "change" includes its own flips.
- The sample was drawn by WorldCover class; only 10 bare and 22 water points, so
  those classes say almost nothing.
- The hand labels and Dynamic World may share a bias (e.g. mixed pixels labelled
  built-up).
- This shows Dynamic World is the better *map* on these points, not that training
  on it gives a better *model*.
- WorldCover disagreement with the hand labels is mostly not explained by change:
  of 104 disagreement points, Dynamic World shows change on 33 and the new class
  matches the hand label on 20 (ad hoc check).

## Not in this branch

Models (`models/*.keras`), rasters and evaluation tables under `data/processed/`
are gitignored, so this branch holds code only. Copy `models/` and
`data/processed/landcover/` to a `_backup_*` folder before retraining if the
current numbers might be needed again.

## Next step (not done)

Retrain RF/U-Net on Dynamic World labels (same 15 m exclusion around validation
points) and score against the same hand labels. Only that shows whether the
model improves.
