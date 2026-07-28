# Reading guide — original Colab notebooks

These are the original development notebooks, kept here as historical
reference. **If you just want to run the pipeline, you don't need these
at all** — the maintained, consolidated version lives in `src/`,
`validation/`, `scripts/`, and `app/` at the repo root; see
[`../../SETUP.md`](../../SETUP.md) for that. This guide is for actually
opening and re-running these specific notebooks (e.g. to understand how a
decision was originally reached, or to redo a diagnostic).

## Reading order

There's a real dependency chain here, not an arbitrary one — each
notebook after the first reads a file another one exported.

**0. Foundational feasibility check** (read first):
- `urban_heat_sg_week1_gates.ipynb` — Week 1, gates G1-G5: confirms GEE
  composites, the NEA API, and the downscaling approach are all viable

**Track A — heat layer & scoring** (read top to bottom):
1. `season_window_diagnostic.ipynb` — decides the season window every
   later notebook locks onto
2. `gee_heat_variants.ipynb` (S2) — the core one: builds the 3 LST
   variants (native30/bicubic10/regress10), exports
   `heat_variants_subzone.csv`, which almost everything after this reads
3. `diagnostic_heat_variants.ipynb` — optional side investigation into
   that CSV's output; not required to move forward
4. `adaptive_capacity_pillar.ipynb`, `sensitivity_pillar.ipynb`,
   `land_change_diagnostic.ipynb`, `nea_heldout_lst.ipynb` — any order
   relative to each other; each reads step 2's CSV independently and adds
   one piece (greenery, population, land-change stability, NEA cross-check)
5. `rank_impact.ipynb` (S6) — pulls steps 2 and 4 together into the
   actual PCA-weighted score plus the ablation table; read last

**Track B — land cover** (separate, parallel track):
1. `generate_validation_sample.ipynb` (SB1) — draws the 200-point
   stratified sample
2. `label_points_interactive.ipynb` (SB2) — the labeling tool, consumes
   SB1's output
3. `train_rf_baseline.ipynb` (RF1) — trains Random Forest, needs the
   *labeled* output from SB2
4. `train_unet.ipynb` (UN1) — trains U-Net, same dependency as RF1,
   deliberately mirrors it for a fair comparison

## Getting a notebook into Colab

These `.ipynb` files live in this repo/on your local machine — Colab
doesn't read this folder directly, so you need to get a copy into Colab
first:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook → Browse** → pick the `.ipynb` file from this
   folder
   (alternative: upload the file to your own Google Drive first, then
   right-click it in Drive → **Open with → Google Colaboratory**)
3. Once it's open, run cells top to bottom starting with the "Setup"
   section — that's where the `ee.Authenticate()` browser popup happens
   (see below)

A couple of things to know:
- **Uploading creates a separate copy inside Colab** — it does not stay
  in sync with this folder. If you make changes in Colab you want to
  keep, download it again afterward (**File → Download → Download
  .ipynb**) and manually replace the file here.
- **If a notebook expects a CSV another notebook exported** (see
  "Reading order" above), that file needs to exist in the *same Colab
  session* too — either re-run the earlier notebook's export cell there
  first, or upload/mount that CSV yourself via Colab's file browser
  (left sidebar) or Google Drive.

## Before you run these: Google account setup (⚠️ different from the pipeline)

**These notebooks and the migrated pipeline authenticate to Earth Engine
in two completely different ways — don't confuse them:**

| | This folder (notebooks) | The pipeline (`src/`, `scripts/`) |
|---|---|---|
| Auth method | **Your own personal Google login**, via an interactive browser popup (`ee.Authenticate()`) | A shared **GCP service account** — a machine identity, no browser, no popup |
| Where credentials live | Cached by Colab/your browser session, tied to whichever Google account you're signed in as | `GEE_SERVICE_ACCOUNT` / `GEE_PRIVATE_KEY_PATH` in `.env` (see `SETUP.md`) |
| Who it "is" | You, personally | A robot identity anyone with the key file can act as |

**To run these notebooks, you need:**
1. Your own Google account registered for Earth Engine access — if it
   isn't yet, the first `ee.Authenticate()` call will walk you through
   registering it (or do it ahead of time at
   [code.earthengine.google.com/register](https://code.earthengine.google.com/register))
2. Access to the shared GCP project (`nus-iss-urban-heat-sg`) — you
   should already have this if you were added as a principal for the
   pipeline setup (see `SETUP.md`); if not, ask whoever set that up to
   grant you access via IAM
3. When you run each notebook's "Setup 2 — Authenticate" cell, a browser
   tab will open asking you to sign in and grant permissions — that's
   normal, sign in with your own account and approve it

None of this touches or requires the service-account key file used by
the pipeline — they're independent. You don't need the `.json` key file
from `credentials/` to run these notebooks at all.

## Caching — heads up, most of these don't have any

One of the known inconsistencies in this original set of notebooks (fixed
in the migrated pipeline, not in these files) is that **almost every
notebook re-fetches and rebuilds everything from scratch on every single
run** — expect re-running most of these to take a few minutes each time,
even if nothing actually changed since your last run.

- **No caching** (re-downloads the subzone boundary / rebuilds GEE
  composites every run): `urban_heat_sg_week1_gates.ipynb`,
  `season_window_diagnostic.ipynb`, `gee_heat_variants.ipynb`,
  `adaptive_capacity_pillar.ipynb`, `sensitivity_pillar.ipynb`,
  `land_change_diagnostic.ipynb`, `nea_heldout_lst.ipynb`,
  `generate_validation_sample.ipynb`, `train_rf_baseline.ipynb`
- **Has real caching**: `train_unet.ipynb` (cell UN1.2) — checks for a
  cached subzone GeoJSON file before hitting data.gov.sg again. This was
  the one prototype of the caching pattern the migrated pipeline
  generalized (see `src/utils/caching.py`).
- **No GEE/network calls at all** (reads only local CSVs already
  exported by other notebooks, so caching doesn't apply):
  `diagnostic_heat_variants.ipynb`, `rank_impact.ipynb`
- **Different kind of "cache"** — `label_points_interactive.ipynb`
  auto-saves your labeling *progress* to a CSV as you go, so you can
  close the notebook and resume later without losing work. That's not
  about avoiding re-fetching data; it's a checkpoint for a long manual
  task. Intentional and fine as-is.

If a notebook feels slow to re-run, that's expected — it's not your
setup, it's this known gap in the original notebooks.
