# Training instructions — full retrain + MLflow verification walkthrough

Manual, step-by-step sequence to retrain every model in the pipeline
(RF, K-means/GMM hotspot clustering, XGBoost, U-Net, CNN) end-to-end and
see the results land in MLflow. Start the MLflow UI first so you can
refresh it as you go:

```
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Opens at `http://localhost:5000`. Three experiments will show up:
`landcover_classifiers` (RF/U-Net/ensemble training + evaluation),
`hotspot_clusters_s4` (K-means/GMM), `heat_model_s5` (XGBoost/CNN +
diagnostics).

Steps below are ordered by actual dependency, not by convenience — each
step only waits on the steps it genuinely needs:

- Ensemble (Step 3) needs RF (Step 1) + U-Net (Step 2).
- Formal evaluation (Step 5) needs RF + U-Net + ensemble — nothing else,
  so it runs right after the ensemble, not at the end.
- Hotspot clustering (Step 6)'s land-cover coherence check is *optional*
  and only benefits from the ensemble existing (Step 3/4) — the
  clustering itself has no hard dependency on RF/U-Net at all.
- XGBoost (Step 7) needs hotspot clustering's `primary_cluster` feature
  (Step 6).
- CNN (Step 8) needs the ensemble raster pushed to GCS (Step 4).
- The heat-model diagnostic (Step 10) needs XGBoost (Step 7) + CNN
  (Step 9).

---

## Step 1: Retrain RF locally

```
python scripts/train_landcover_rf.py --with-probabilities
```

`--with-probabilities` is required (not just nice-to-have) — the ensemble
needs RF's per-class probability raster, and it forces a fresh classifier
train regardless of the GEE asset cache (a cached classifier only
supports hard-label output, not probabilities).

**Check MLflow now**: `landcover_classifiers` experiment, newest `rf` run.

**If this fails with `EEException: Computation timed out.` at
`informal_accuracy_check`**: this is a known GEE synchronous-compute
timeout, not a quota problem (a quota rejection would be a 429, not this
400). It's already mitigated (`tileScale=8` in
`src/landcover/rf_baseline.py::informal_accuracy_check`, matching the
`tileScale=8` already used in `extract_training_samples`'s
`stratifiedSample`), but if it recurs, just rerun the command — it's
retrying an expensive lazy-graph evaluation, not a deterministic failure.

## Step 2: Retrain U-Net on Colab

1. Open `notebooks/colab_training/train_unet.ipynb` in VS Code, connect to
   a Colab GPU kernel (**Select Kernel** → **Colab** → **New Colab
   Server** → **GPU**).
2. Run every cell top to bottom. At the auth cell, paste your
   service-account JSON contents when prompted.
3. It's already set to `force_retrain=True`, so it'll genuinely retrain
   (not silently reload a cached model) regardless of whether you're
   reusing a Colab session from an earlier run.
4. The last cell pushes the trained model + its MLflow run summary to GCS.

## Step 3: Pull U-Net locally, run inference, rebuild the ensemble

```
python scripts/pull_models.py --model unet
python scripts/run_landcover_unet_inference.py
python scripts/build_landcover_ensemble.py --force
```

`--force` is required here — the ensemble script skips recompute if
`ensemble_landcover.tif` already exists on disk, which it will if you've
ever built it before, and that would silently leave it built from the
*old* RF/U-Net rasters instead of the ones you just retrained.

**Check MLflow now**: refresh `http://localhost:5000`, open the
`landcover_classifiers` experiment, click into the newest `unet` run. You
should see **10 params** and **5 metrics** — including `train_accuracy`
and `best_val_loss`, not just the 3 metrics it used to log.

## Step 4: Push the fresh ensemble raster (the CNN notebook needs it)

```
python -c "from src.utils import gcs; from config.settings import GCS_MODEL_BUCKET, ENSEMBLE_RASTER_GCS_PREFIX; gcs.upload_file('data/processed/landcover/ensemble_landcover.tif', GCS_MODEL_BUCKET, f'{ENSEMBLE_RASTER_GCS_PREFIX}.tif')"
```

## Step 5: Run the formal land-cover evaluation

```
python scripts/evaluate_landcover_classifiers.py
```

Runs now, right after the ensemble, since RF + U-Net + ensemble are all
it needs — no reason to wait for hotspot clustering/XGBoost below.

**Check MLflow**: 3 new runs — `evaluate_rf`, `evaluate_unet`,
`evaluate_ensemble` (tag `stage=evaluation`) — each with `accuracy`,
`macro_f1`, `weighted_f1`, `n_scored`, and per-class F1
(`vegetation_f1`, `built_up_f1`, `bare_f1`, `water_f1`).

## Step 6: Rebuild hotspot clusters (K-means/GMM)

```
python scripts/build_hotspot_clusters.py --force
```

`--force` for the same reason as Step 3 — `hotspot_clusters.csv` already
existing on disk would otherwise skip the rebuild. This sweeps both
K-means and GMM over k=2..8, keeps whichever wins on silhouette as
`primary_cluster`, and (since the ensemble raster from Step 3 now exists)
runs the land-cover coherence sanity check against it.

**Check MLflow now**: new `hotspot_clusters_s4` experiment, three runs —
`kmeans` and `gmm` (each with the full k-sweep logged as a metric curve,
`stage=clustering`), and `primary_cluster` (`stage=selection`, which
method won and the final quality/coherence numbers).

## Step 7: Retrain XGBoost

```
python scripts/train_heat_model_xgboost.py --force-retrain
```

`--force-retrain` — same skip-if-exists caching pattern again.
`primary_cluster` from Step 6 is one of its input features, so this must
run *after* hotspot clustering.

**Check MLflow**: newest `xgboost` run in `heat_model_s5` — params +
`test_rmse`/`test_r2`/`ndvi_vegetation_slope` metrics.

## Step 8: Retrain CNN on Colab

1. Open `notebooks/colab_training/train_heat_cnn.ipynb`, connect to a
   Colab GPU kernel (same as Step 2).
2. Run every cell top to bottom (auth cell again needs your JSON pasted).
3. The last cell pushes the model + its full run summary.

## Step 9: Pull CNN locally

```
python scripts/pull_models.py --model cnn
```

**Check MLflow**: newest `cnn` run in the `heat_model_s5` experiment
should show **9 params** and **5 metrics** — `train_rmse` alongside
`val_rmse`, plus `best_val_loss`.

## Step 10: Run the heat-model diagnostic

```
python scripts/diagnose_heat_model.py
```

**Check MLflow**: 2 new runs in `heat_model_s5` — `diagnose_xgboost`
(`full_table_rmse_informal`) and `diagnose_cnn_xgboost_crosscheck`
(`n_subzones_checked`, `n_agreements`, `agreement_rate`, plus a
per-subzone `xgb_delta_lst_*`/`cnn_delta_lst_*` pair for each of the 3
demo subzones).
