# TensorFlow → PyTorch + Colab-only training migration

_Saved snapshot of the approved implementation plan, 2026-08-05. Working copy lives at `C:\Users\kxloo\.claude\plans\take-note-of-what-cheeky-lighthouse.md` during execution; this file is the durable record in the repo._

## Context

`main` currently has a working TF/Keras U-Net (land-cover S3) and CNN heat model (S5), both requiring WSL2 GPU because native Windows TensorFlow is broken in this repo's `.venv`. The user is about to share this workspace with a partner who has no GPU at all, so the plan is: eliminate WSL2 entirely, rewrite both deep-learning models in PyTorch (CPU-friendly for local inference), and move **all training** exclusively to Google Colab using the official VS Code↔Colab extension (ships Nov 2025 — connects a local `.ipynb` directly to a real Colab GPU kernel, free T4 or Pro tier). Trained model artifacts sync Colab → GCS → local, with GCS as the source of truth (already-configured bucket/service-account, no new infra).

A `snapshot/tensorflow` branch already exists, frozen at the current commit — this is a full in-place replacement on `main`, not a side-by-side toggle. PyTorch cannot load Keras weights, so retraining both models from scratch is required and already expected by the user.

Confirmed decisions: repo is **public** on GitHub (Colab `git clone` needs no credentials), and the app's CNN counterfactual page should move from precomputed canned examples to **live CPU inference** as part of this migration (the only reason for canned examples — broken native Windows TF — goes away). Google Drive was considered as an additional sync layer and rejected: it requires interactive OAuth (breaks the service-account-only auth convention), fragments the shared GCS source of truth into a per-person path, reintroduces a hard 15GB quota GCS doesn't have, and only offers marginal savings on an already-cheap (<$1/month) egress cost.

## Recommended approach

### 1. Split each model into `model` / `data` / `train` / `infer` modules

Today `src/landcover/unet.py` (446 lines) and `src/heat_model/cnn_patches.py` (216 lines) each mix architecture, GEE/GCS patch I/O, training loop, and inference in one file — confirmed coupled, not modular. Split so Colab only needs `model+data+train`, local inference only needs `model+data+infer`:

```
src/landcover/
  unet_model.py   — ConvBlock, UNetBackbone, UNet (pure torch.nn.Module, zero I/O deps)
  unet_data.py    — export_training_patches, export_inference_patches, fingerprint helpers,
                     read_feature_patches() [NEW shared low-level TFRecord reader], parse_training_patches
  unet_train.py   — train_unet (manual PyTorch loop via src/utils/torch_train.py)
  unet_infer.py   — load_unet(), run_inference_and_reconstruct, informal_accuracy_check

src/heat_model/
  cnn_model.py    — CNNRegressor (wraps unet_model.UNetBackbone + linear head), N_LANDCOVER_CLASSES
  cnn_data.py     — build_local_feature_target_patches (calls unet_data.read_feature_patches)
  cnn_train.py    — train_cnn_regressor
  cnn_infer.py    — load_cnn_regressor(), run_cnn_inference, predict_patch(), locate_patch_and_pixel
```

Two things to write once instead of three/twice while doing this split:
- **`unet_data.py::read_feature_patches()`** — the `tf.io.parse_single_example`/`TFRecordDataset` block is currently duplicated 3x across `unet.py` and `cnn_patches.py`. Write it once using the pure-Python `tfrecord` PyPI package (`pip install 'tfrecord[torch]'` — no tensorflow dependency, PyTorch-native `TFRecordDataset`, supports GEE's GZIP-compressed patch export format). **Validate this substitution first, standalone**, against real `.tfrecord.gz` shards already on disk, before touching any other code — it's the one piece with real "might not just work" risk.
- **`src/utils/torch_train.py::train_loop(model, train_loader, val_loader, loss_fn, optimizer, epochs, patience, device)`** — Keras's `.fit()+EarlyStopping(restore_best_weights=True)` has no PyTorch equivalent; hand-write the epoch/batch loop, per-pixel-weighted loss (`reduction="none"` + mask multiply, not the class-`weight=` kwarg), and best-checkpoint tracking once, shared by both `unet_train.py` and `cnn_train.py`. Auto-select `cuda` if available (Colab) else `cpu`.

Other required edits from this split:
- `src/heat_model/counterfactual.py` — replace the two `.predict(..., verbose=0)` calls (Keras-specific) with `cnn_infer.predict_patch(model, patch_X)` (wraps `model.eval(); torch.no_grad(): model(x)`); update its `N_LANDCOVER_CLASSES` import source to `cnn_model`.
- `src/utils/seed.py` — `tf.random.set_seed` → `torch.manual_seed` (+ guarded `torch.cuda.manual_seed_all`, matters for Colab reproducibility).
- **`map_location="cpu"` is mandatory** in both `load_unet()`/`load_cnn_regressor()` — weights are always saved from a Colab CUDA session; forgetting this breaks the very first local/partner inference attempt.
- Save format: PyTorch `state_dict` `.pt` files (`models/unet_landcover.pt`, `models/heat_cnn.pt`), replacing `.keras`. Keep the existing sha256-of-validation-CSV fingerprint sidecar convention for "does this model need retraining," unchanged in mechanism.

### 2. `src/utils/gcs.py` — extract + extend the GCS layer

`src/ingest/gee.py::_gcs_client()` is today's only GCS integration point, with no general upload helper. Extract into a new leaf module (depends only on `config.settings`, so `gee.py → utils.gcs → config.settings` stays one-directional, no circular import):

```python
def get_client(): ...                                              # extracted from gee.py::_gcs_client
def download_blob(bucket, blob_name, out_path) -> Path: ...        # extracted
def download_blobs_with_prefix(bucket, prefix, out_dir) -> list[Path]: ...  # extracted
def upload_file(local_path, bucket, blob_name) -> str: ...         # NEW
def blob_exists(bucket, blob_name) -> bool: ...                    # NEW
def download_text(bucket, blob_name) -> str | None: ...            # NEW — sidecar hash read
def upload_text(text, bucket, blob_name) -> None: ...              # NEW — sidecar hash write
```

`gee.py`'s `export_patches_to_gcs`/`export_geotiff_to_gcs` call these instead of duplicating the client/download logic; delete `_gcs_client()`. Reword (don't remove) the two WSL2-specific comments around `download_blob_to_file` — the defensive choice itself still applies everywhere, only the WSL2 framing goes stale once WSL2 is gone from the project.

New/changed `config/settings.py` constants (the *only* place constants live, per existing convention): `UNET_MODEL_SAVE_PATH`, `UNET_CLASSIFIED_RASTER_PATH`, `UNET_PROB_RASTER_PATH`, `UNET_TRAIN_PATCH_DIR`, `UNET_INFERENCE_PATCH_DIR` (moved out of `unet.py`, fixing the existing inconsistency where U-Net's save path lived outside settings.py while CNN's didn't); `CNN_MODEL_SAVE_PATH` extension `.keras`→`.pt`; `UNET_MODEL_GCS_PREFIX`, `CNN_MODEL_GCS_PREFIX`, `GCS_MODEL_BUCKET` (optional override, defaults to reusing the already-configured `GEE_EXPORT_BUCKET` under a `models/` prefix — no new bucket needed). Add `GCS_MODEL_BUCKET` as a commented-optional line in `.env.example`.

### GCS bucket layout (current + proposed additions)

Checked the real bucket (`gs://nus-iss-urban-heat-sg-exports`, already configured via `GEE_EXPORT_BUCKET`) directly — current contents, ~796 MB total across 8 objects:

```
rf_landcover/rf_landcover.tif                       1.7 MB
rf_landcover/rf_landcover_classified.tif           140.3 MB
rf_landcover/rf_landcover_prob.tif                 139.9 MB
unet_train_patches/unet_train.tfrecord.gz          165.3 MB
unet_train_patches/unet_train.json                 ~0 MB   (GEE mixer/metadata file)
unet_inference_patches/unet_inference.tfrecord.gz  164.3 MB
unet_inference_patches/unet_inference.json         ~0 MB
heat_model/lst_bicubic10_full.tif                  184.6 MB
```

**Storage capacity is not a real constraint**: GCS is metered pay-as-you-go (~US$0.02/GB/month for STANDARD storage), not a fixed quota like Google Drive's 15 GB — a bucket scales to petabytes with no size cap to hit. After this migration's additions (below), total footprint lands around ~1.1 GB, costing roughly 2 cents/month. The one thing I *can't* verify via the storage API (would need Billing IAM, not granted to the pipeline service account) is whether the GCP project is running on a time-limited free-trial credit rather than a standing billing account — if you're unsure, check GCP Console → Billing once; it's a $/expiry question, not a storage-size one.

Proposed additions — minimal diff from today, reusing the existing bucket (no new bucket/IAM setup needed, your partner already has access):

```
rf_landcover/                          [unchanged]
unet_train_patches/{fingerprint}/      [existing prefix, now nested one level under the training-data
                                         fingerprint — see item 3 below; today it's a flat, always-
                                         overwritten path with no versioning]
unet_inference_patches/                [unchanged — stays existence-keyed like today]
heat_model/                            [unchanged]
training_inputs/
  ensemble_raster.tif                  [NEW — pushed manually after a local build_landcover_ensemble.py
                                         run; the one input the CNN Colab notebook can't self-generate]
models/
  unet_landcover.pt                    [NEW — trained PyTorch weights, GCS source of truth]
  unet_landcover.sha256
  heat_cnn.pt
  heat_cnn.sha256
training_runs/
  {model_name}_{timestamp}.json        [NEW — MLflow run summaries exported from Colab, re-imported
                                         into the local mlflow.db by pull_models.py; see item 5]
```

`UNET_MODEL_GCS_PREFIX="models/unet_landcover"`, `CNN_MODEL_GCS_PREFIX="models/heat_cnn"` in `config/settings.py` point `pull_models.py`/`push_models.py` at these exact paths.

**Estimated monthly cost** (current GCS pricing, US standard storage: $0.020/GB-month; network egress: $0.12/GB for the first 1 TB/month, storage inbound/upload is always free):

- **Storage**: ~1.1 GB total after this migration's additions → **~$0.02/month**. Negligible regardless of usage pattern — GCS scales to petabytes, storage is never the cost driver here.
- **Egress is the real (and only meaningful) variable cost**, and it's driven by Colab sessions, not local use: Colab's disk is ephemeral, so even with the fingerprint cache from item 3 (which avoids re-running the *GEE export*), every Colab training session still has to *download* the already-exported patches from GCS to the Colab VM fresh. Per-session download: ~165 MB (U-Net training) or ~490 MB (CNN training: ensemble raster + lst_bicubic10 + inference patches). Local `pull_models.py` only pulls the small `.pt` files (tens of MB) — trivial.
  - Occasional retraining (~2-4 sessions/month total, both models): ~1-1.5 GB egress → **~$0.15/month**
  - Active development sprint (~15-20 sessions/month while tuning): ~5-6 GB egress → **~$0.65/month**
  - Heavy daily iteration (~40 sessions/month, an upper bound unlikely to be sustained): ~12 GB → **~$1.50/month**
- **Realistic range: well under $1/month** for this project's actual usage pattern (periodic retrains, not continuous iteration), with storage itself essentially free.

I can't see your actual GCP billing history or confirm whether the project is on a free-trial credit — the service account has no Billing IAM role, so this is a formula-based estimate, not a pulled invoice. If you want ground truth, GCP Console → Billing → Reports shows real historical spend.

Sources: [CloudZero — GCS Pricing Guide 2026](https://www.cloudzero.com/blog/gcp-storage-pricing/), [Eon — Google Cloud Pricing 2026](https://www.eon.io/blog/google-cloud-pricing)

### 3. Cache exported GEE patches in GCS too (avoid re-exporting every Colab run)

Today's `export_training_patches`/`export_inference_patches` already skip recompute if the output exists **on local disk** — but Colab's VM disk is empty every session. Without a fix, every single Colab training run (including a same-data, hyperparameter-only rerun) would re-trigger the full multi-minute-plus `ee.batch.Export.image.toCloudStorage` job again, wasting time and GEE quota. Since exports already round-trip through GCS on the way to local disk, extend that same GCS layer into a persistent, deterministic cache shared across both partners' machines and every Colab session:

- Use a **fixed, fingerprint-keyed GCS prefix** for training patches (e.g. `unet_patches/train/{training_fingerprint}/...`) instead of a timestamp/run-unique prefix — verify during implementation whether the current `description`/`prefix` generation is already timestamp-randomized (if so, fix that first, it'd defeat this cache).
- Before calling the GEE export, check via `gcs.blob_exists`/`list_blobs` whether that prefix already has data. If yes: skip the export entirely, `download_blobs_with_prefix` straight from GCS to wherever we're running (local disk or Colab's `/content`). If no: run the export as today, which naturally populates the cache for next time.
- `export_inference_patches` gets the same treatment keyed by existence rather than a fingerprint (its inputs — season windows, region — have no equivalent hash source today; existence-check just makes the current local-cache behavior GCS-shared instead of per-machine).
- A new `--force-export` flag (alongside the existing `--force` retrain flag) lets you deliberately bypass this cache — e.g. if the underlying GEE imagery changed in a way the validation-CSV fingerprint wouldn't catch.

This makes GCS double as both "trained-model source of truth" (below) *and* "exported-training-data source of truth": one run does the real GEE export; every subsequent run — yours, your partner's, or a retrain — pulls from GCS in seconds instead of minutes.

### 4. `scripts/pull_models.py` + `scripts/push_models.py`

Deliberately **not** reusing the existing `{model_stem}_fingerprint.txt` sidecar — that file means "hash of the training-data CSV," a different concept from "has the GCS-hosted weights file changed." New sidecar: `models/{stem}.sha256`.

`pull_models.py --model {unet,cnn,all} [--force]`: read the small remote `.sha256` text blob first (cheap, no need to pull full weights just to check) → compare to local sidecar → skip if match and `--force` not set → else download, recompute hash, assert it matches (catches truncated downloads), write verified hash locally. Matches the existing "standalone CLI, checks its own output, prints status" convention.

`push_models.py` (same flags, inverse direction) is what the Colab notebooks call at the end of training (`!python scripts/push_models.py --model unet`) — since the repo is cloned onto the Colab VM, it's just another runnable script, no notebook-specific upload code needed.

For the CNN notebook's training-input dependency (`training_inputs/ensemble_raster.tif` — see bucket layout above), reuse `gcs.upload_file()`/`download_blob()` directly as one-off calls rather than building a dedicated sync script — this is a large deterministic build output consciously re-pushed on change, not something needing fingerprint sophistication yet.

### 5. Colab notebooks, via the VS Code Colab extension

`notebooks/colab_training/train_unet.ipynb` and `train_heat_cnn.ipynb` replace `scripts/train_landcover_unet.py`/`train_heat_model_cnn.py` as the training entry points. Opened in VS Code → "Select Kernel" → Colab → GPU (free T4) → connects to a real Colab VM; the `.ipynb` file itself stays in the repo, normally git-tracked.

Both notebooks follow the same opening sequence: (1) idempotent `git clone`/`git pull` of the public repo onto `/content`, add to `sys.path`; (2) `pip install -q -r requirements-colab.txt`; (3) auth — read the service-account JSON content from a Colab Secret (`google.colab.userdata`, one-time setup per Google account, not per-session), write it to a local file, set `GEE_PRIVATE_KEY_PATH`/`GEE_SERVICE_ACCOUNT`/`GEE_PROJECT_ID`/`GEE_EXPORT_BUCKET` env vars **before** `config.settings` is imported; (4) `init_ee()`. This stays service-account-only, consistent with the project's existing no-interactive-auth convention.

`train_unet.ipynb` then: rebuilds the feature stack (same GEE calls as today's script), exports training patches (now GCS-cache-aware per item 3 — only a real GEE job on the first run or after `--force-export`), trains via `unet_train.train_unet` (GPU auto-used through `torch_train.train_loop`), prints val metrics, `push_models.py --model unet`. **Full-Singapore inference does NOT run in Colab** — that's local-only (see below), both because that's the stated local-inference requirement and because it avoids redundantly re-running a multi-minute GEE export inside Colab.

`train_heat_cnn.ipynb` has one real, pre-existing dependency wrinkle made newly visible by this migration: CNN training needs the land-cover **ensemble raster**, which can only be produced after U-Net is trained (Colab) *and* run through local CPU inference *and* combined with RF (`build_landcover_ensemble.py`, local). Colab can't regenerate this raster itself. So this notebook pulls `training_inputs/ensemble_raster.tif` and `heat_model/lst_bicubic10_full.tif` from GCS rather than recomputing, re-exports U-Net's inference patches (same GCS-cache-aware call — cheap after the first run), builds CNN patches, trains, pushes.

`requirements-colab.txt` (new, trimmed subset) — **excludes** `torch` (rely on Colab's pre-installed CUDA-matched build; blindly installing the local CPU pin would risk silently losing GPU accel) and `tensorflow` (being deleted); excludes app-only packages (`streamlit`, `folium`, `plotly`, etc.) and non-DL packages (`xgboost`, `scikit-learn`, `mlflow`); includes `earthengine-api`, `google-cloud-storage`, `google-auth`, `python-dotenv`, `rasterio`, `geopandas`, `pyproj`, `shapely`, `affine`, `tfrecord[torch]`.

**MLflow tracking is preserved, not dropped** (user confirmed this is required): Colab's local `mlflow.db` is ephemeral, so writes there vanish with the session — but each notebook still calls the existing `experiment_tracking.start_run()` unchanged (useful for in-session introspection), and additionally exports the run's params/per-epoch metrics as a small JSON (`experiment_tracking.export_run_summary()`), pushed to GCS under `training_runs/{model_name}_{timestamp}.json`. Locally, a new `experiment_tracking.import_remote_runs()` — called automatically at the end of `pull_models.py` — downloads any not-yet-imported run JSONs and re-logs them into the shared local `mlflow.db` via the plain `mlflow.log_params`/`mlflow.log_metric(..., step=i)` API (replaying the full epoch curve, not just the final number), tracked idempotently via a small local manifest file (`mlflow_imported_runs.txt`, gitignored) so repeated pulls don't duplicate runs. End result: U-Net/CNN training runs show up in the same `mlflow ui --backend-store-uri sqlite:///mlflow.db` store as RF/XGBoost, just with an extra sync hop instead of the old same-machine-shared-file assumption.

### 6. Remove WSL2, restore live CNN inference in the app

- Delete `scripts/train_landcover_unet.py`, `scripts/train_heat_model_cnn.py` (training moves to Colab). Add new local-only `scripts/run_landcover_unet_inference.py` (the former's inference half): checks `UNET_MODEL_SAVE_PATH` exists (else "run `pull_models.py` first"), exports inference patches, `unet_infer.load_unet()`, `run_inference_and_reconstruct`. CPU is fine here — U-Net inference was never the GPU bottleneck, only 30-epoch training was.
- Strip/reword WSL2 references in `scripts/diagnose_heat_model.py`, `scripts/run_counterfactual.py` (also swap `tf.keras.models.load_model` → `cnn_infer.load_cnn_regressor`), and the two comments in `src/ingest/gee.py`.
- `requirements.txt`: remove `tensorflow`, add `tfrecord`; `torch==2.13.0` is already pinned (currently unused — becomes load-bearing).
- `SETUP.md`: add a new "Training on Colab" section (VS Code Colab extension install, Colab Secrets setup for the service-account JSON, `pull_models.py`/`push_models.py` usage) — WSL2 was never documented there, so this is net-new content, not a removal.
- `app/pages/4_Counterfactual_Greening.py`: replace the precomputed canned-example gallery with a live code path calling `cnn_infer.predict_patch()` on slider interaction (per user's choice); `5_Validation_Dashboard.py`: update the "precomputed via WSL2" explanatory text since it's no longer accurate. Both need a real Playwright screenshot verification pass before considered done (per this repo's established convention: AppTest catches "doesn't crash," not "looks right").
- Leave `notebooks/Colab Notebooks/` (the legacy pre-migration notebooks, already preserved via a dedicated reading-guide commit) untouched, and leave all framework-free scripts (RF, XGBoost, hotspots, priority score, `tests/*` — grepped, zero TF/torch/keras references) untouched.

### 7. Retraining & downstream impact (already expected by the user)

Both models retrain from scratch in Colab — Keras weights don't port to PyTorch. GEE-exported TFRecord patches don't need re-export (same data, only the parsing library changes). Once retrained, everything downstream needs regenerating: land-cover ensemble (S3), hotspot clusters (S4), heat model + counterfactuals (S5), confidence bands (S6) — the existing fingerprint-based caching will correctly detect the change and force recompute, same blast radius as a normal full-pipeline rebuild. Expect new real accuracy/RMSE numbers that differ from the current TF ones; document them, don't assume parity.

## Avoiding charges after the project ends

Two time horizons — a cheap safety net now, and a real teardown once the project is graded/submitted:

**Now (during the project) — low-risk, do this early:**
- Set a **GCP Billing Budget alert** on the `nus-iss-urban-heat-sg` project (Console → Billing → Budgets & alerts), e.g. alert at $5/month. Given the ~$0.15-1.50/month estimate above, this would only ever fire if something is genuinely misbehaving (e.g. a runaway export loop) — cheap insurance, reversible, doesn't affect the pipeline.
- Optional: a GCS **Object Lifecycle Management** rule on the bucket that auto-deletes objects after a fixed calendar date (e.g. "project submission date + 30 days"), rather than a rolling age window — a rolling window (e.g. "delete after 90 days") risks deleting the cached training patches / trained models mid-project if development runs longer than expected, which would silently reintroduce the "re-export from GEE" cost this migration is designed to avoid. A fixed end-date is safer than a rolling one here.

**At true project end (after grading/submission) — do this deliberately, not automatically:**
1. Delete the bucket's contents (or the bucket itself) via Console or `gsutil -m rm -r gs://nus-iss-urban-heat-sg-exports` — stops the ~2 cents/month storage charge.
2. Revoke/delete the service account key (`credentials/nus-iss-urban-heat-sg-*.json`) via Console → IAM → Service Accounts — closes the access path entirely, not just the bucket.
3. **Cleanest guarantee of $0 going forward**: delete the entire `nus-iss-urban-heat-sg` GCP project (Console → IAM & Admin → Settings → Shut down). GCP stops all billing for a deleted project (30-day recovery window, then permanent) — this is the only step that guarantees zero charges from *anything* in the project, not just this bucket. Only do this once both you and your partner confirm nothing else is needed from it (GEE access, other exports, etc.) — this is a real, hard-to-reverse action, not something to do reflexively at the first lull in work.

None of this is part of the implementation work below — it's a checklist for later, noted here so it doesn't get forgotten once the project wraps.

## Sequencing

1. Standalone validation of `tfrecord` package against real on-disk `.tfrecord.gz` shards (highest-uncertainty step, isolated from everything else).
2. `src/utils/gcs.py` + `gee.py` refactor (extraction, zero behavior change) **plus** the fingerprint-keyed export cache from item 3 — same module, land together. Independently testable/useful immediately, before any PyTorch code exists.
3. Module split + PyTorch rewrite (`unet_*`, `cnn_*`, `torch_train.py`) + a new model-free `tests/test_model_architectures.py` (random-init forward pass, assert output shapes `(128,128,4)`/`(128,128,1)`) — validates the architecture port without needing a trained model or Colab yet. Delete old `unet.py`/`cnn_patches.py`, fix the two confirmed cross-file imports (`build_landcover_ensemble.py`, `evaluate_landcover_classifiers.py`).
4. `pull_models.py`/`push_models.py` + new settings constants — testable via a dummy-file round trip before any real model exists.
5. `train_unet.ipynb` end-to-end in Colab → first run does the real GEE patch export (populating the GCS cache from step 2) → real `unet_landcover.pt` pushed to GCS.
6. Local: `pull_models.py --model unet`, new `run_landcover_unet_inference.py`, `build_landcover_ensemble.py` → real ensemble raster.
7. `train_heat_cnn.ipynb` (depends on step 6's raster) → `heat_cnn.pt` pushed.
8. Local CNN inference/counterfactual wiring, live Streamlit page, full pipeline re-run top to bottom against the real PyTorch models.
9. Cleanup: delete stale `.keras` files, finish `SETUP.md`/`.env.example`/docstring updates.

Any *later* retrain (yours, or your partner's) re-runs step 5/7's notebooks but hits the GCS patch cache instead of re-exporting from GEE — only a genuine `--force-export` or a fingerprint change (new/relabeled validation CSV) triggers a fresh GEE job.

(Rollback safety throughout is the frozen `snapshot/tensorflow` branch — there's no partial-migration state on `main` where old `.keras` models keep working, since PyTorch can't load them.)

## Verification

- `tests/test_model_architectures.py` (new, plain-assert style matching repo convention) — shape/wiring check, no trained weights needed.
- Manual `tfrecord`-parsing smoke test against real on-disk patch shards before the module split begins.
- `pull_models.py`/`push_models.py` round-trip against the real GCS bucket with a dummy file.
- After real Colab training: `scripts/evaluate_landcover_classifiers.py` and `scripts/diagnose_heat_model.py` re-run against the new PyTorch models for real accuracy/RMSE numbers.
- Launch the Streamlit app locally, click through pages 4 and 5, confirm live CNN inference renders correctly with a real screenshot (Playwright) — not just `AppTest`.
- Confirm `map_location="cpu"` loading works by running local inference without any GPU present (simulates the partner's machine).
