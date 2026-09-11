# Cloud-threshold figures

Code to regenerate the figures showing why the S1 cloud threshold matters:
too loose lets cloud + haze through into the composite; too strict punches
real data out. Rendered straight from Earth Engine using the pipeline's own
masking helpers (`src/ingest/gee.py`), so the panels reflect real S1
behaviour.

## Run

From the repo root, with the venv active and `.env` GEE credentials set
(see `SETUP.md` §3-4):

```
python figures/cloud_threshold/generate_cloud_threshold_figures.py
```

Writes three montage PNGs into `figures/cloud_threshold/out/` (gitignored).
Re-running skips any figure whose PNG already exists — pass `--force` to
rebuild.

## What it produces

| file | shows |
|---|---|
| `figure1_single_scene.png` | one partly-cloudy Sentinel-2 scene: raw / s2cloudless probability / after masking `prob < 40`. A single scene can't be salvaged by a threshold — hence multi-scene compositing. |
| `figure2_threshold_sweep.png` | dry-season median RGB at `no mask` + several s2cloudless probability cutoffs (the `S2_CLOUD_PROB_MAX` knob). Loose = milky haze; project setting (70) = clean; very strict (20) = salt-and-pepper data holes. |
| `figure3_landsat_lst.png` | Landsat land-surface-temperature median with the QA cloud/shadow mask **off vs on**. Unmasked, cold cloud tops drag the median down to a blue, gap-ridden layer and the urban-heat gradient disappears. |

## Useful flags

```
--only sweep                       # just one figure (single | sweep | lst, space-separated)
--thresholds 90 70 40              # probability cutoffs for the sweep
--year 2023                        # composite year (one year on purpose; more years median clouds away)
--months 4 5 10 11                 # season window (defaults to config.settings.DRY_SEASON_MONTHS)
--scene-cloud-max 70               # CLOUDY_PIXEL_PERCENTAGE scene prefilter for the sweep
--aoi 103.55 1.15 104.10 1.48      # bounding box to render
--dimensions 1200                  # thumbnail long edge, px
--out-dir data/processed/diagnostics/cloud_threshold
--force
```

## Notes

- Presentation-only constants (the render AOI, RGB stretch, LST palette)
  live in this script, not `config/settings.py` — they're figure framing,
  not pipeline parameters.
- `config/settings.py` records that `S2_CLOUD_PROB_MAX` drifted between
  `40` and `70` across the original notebooks before being consolidated to
  `70`; `figure2` is the visual case for that.
