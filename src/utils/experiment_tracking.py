"""Thin MLflow wrapper so scripts/train_landcover_rf.py and the U-Net/CNN
training code log to the same experiment consistently, without duplicating
tracking-setup boilerplate in each place.

Local SQLite-backed tracking (a single mlflow.db file under REPO_ROOT) --
still no separate server needed, but MLflow 3.x puts the plain flat-file
store ("./mlruns") into maintenance mode and refuses to use it without an
explicit opt-out flag, so SQLite is the simpler path going forward rather
than fighting that deprecation. Browse runs with
`mlflow ui --backend-store-uri sqlite:///mlflow.db` from the repo root.

U-Net/CNN training now happens in Colab (GPU), whose `/content` filesystem
is ephemeral -- `mlflow.db` written there vanishes when the session ends, so
it can no longer just be "the same physical directory" the way WSL2's
/mnt/c mount used to make true for RF vs. U-Net on the same machine.
`export_run_summary()` + `import_remote_runs()` bridge that gap via GCS:
each Colab notebook still calls `start_run()` as normal for in-session
visibility, then also uploads a small JSON summary of what it logged;
`import_remote_runs()` (called from `scripts/pull_models.py`) re-logs those
into the local shared store, so Colab-trained runs end up queryable
alongside RF/XGBoost's, just with an extra sync hop.
"""

import json
from pathlib import Path

import mlflow

from config.settings import REPO_ROOT

EXPERIMENT_NAME = "landcover_classifiers"
HEAT_MODEL_EXPERIMENT_NAME = "heat_model_s5"
RUNS_GCS_PREFIX = "training_runs"
_IMPORTED_RUNS_MANIFEST = REPO_ROOT / "mlflow_imported_runs.txt"

mlflow.set_tracking_uri(f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")


def start_run(model_name: str, experiment_name: str = EXPERIMENT_NAME, **tags):
    """`experiment_name` defaults to EXPERIMENT_NAME so every existing call
    site (scripts/train_landcover_rf.py, notebooks/colab_training/train_unet.ipynb)
    is unaffected. S5's training code passes HEAT_MODEL_EXPERIMENT_NAME
    explicitly instead of this module growing a second hardcoded default."""
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=model_name, tags={"model_type": model_name, **tags})


def log_artifact_safe(local_path) -> None:
    """`mlflow.log_artifact()` given an absolute Windows path (drive letter
    + colon, e.g. `C:\\Users\\...`) has a real, confirmed bug on this
    platform: some version in this project's dependency chain misreads the
    drive letter as a URI scheme and recreates the ENTIRE absolute path as
    nested folders inside the artifact store instead of just copying the
    file in -- caught the hard way (2026-08-06): a ~23MB model file ended
    up buried under a literal (Unicode-substituted, since NTFS forbids a
    real ':' in a filename) "C:" folder sitting at the repo root instead of
    inside `mlruns/`. Sidestepped entirely by cd-ing into the file's own
    directory and logging just its basename, so `mlflow.log_artifact()`
    never sees a drive-letter path in the first place. Use this instead of
    calling `mlflow.log_artifact()` directly with any absolute path."""
    import os

    local_path = Path(local_path)
    old_cwd = os.getcwd()
    os.chdir(local_path.parent)
    try:
        mlflow.log_artifact(local_path.name)
    finally:
        os.chdir(old_cwd)


def export_run_summary(run_name: str, experiment_name: str, params: dict, metrics_history: dict,
                        final_metrics: dict | None = None, tags: dict | None = None) -> dict:
    """Build the JSON-able summary a Colab notebook uploads to GCS after
    training, so the run survives the session ending. `metrics_history`
    values must be lists (one entry per epoch, e.g. `history["val_loss"]`
    from `src/utils/torch_train.py::train_loop`) so `import_remote_runs`
    can replay the full curve rather than just the final number.
    `final_metrics` is for single-value summary metrics logged without a
    step during the live session (e.g. `mlflow.log_metric("best_val_loss",
    min(history["val_loss"]))`) -- kept separate from `metrics_history`
    since they're not part of the per-epoch curve."""
    return {
        "run_name": run_name,
        "experiment_name": experiment_name,
        "tags": tags or {},
        "params": {k: str(v) for k, v in params.items()},
        "metrics_history": metrics_history,
        "final_metrics": {k: float(v) for k, v in (final_metrics or {}).items()},
    }


def import_remote_runs(bucket: str, prefix: str = RUNS_GCS_PREFIX) -> list[str]:
    """Pull any training-run summaries pushed from Colab that haven't been
    imported into the local mlflow.db yet, and log them as real MLflow runs.
    Idempotent: already-imported blob names are tracked in a local manifest
    file so re-running `pull_models.py` doesn't duplicate runs. Returns the
    list of newly-imported blob names (empty if nothing new)."""
    from src.utils import gcs

    imported = set(_IMPORTED_RUNS_MANIFEST.read_text().splitlines()) if _IMPORTED_RUNS_MANIFEST.exists() else set()
    client = gcs.get_client()
    newly_imported = []

    for blob in client.list_blobs(bucket, prefix=prefix):
        if blob.name in imported:
            continue
        summary = json.loads(blob.download_as_text())
        with start_run(summary["run_name"], experiment_name=summary["experiment_name"], **summary.get("tags", {})):
            mlflow.log_params(summary["params"])
            for metric_name, values in summary["metrics_history"].items():
                for step, value in enumerate(values):
                    mlflow.log_metric(metric_name, value, step=step)
            for metric_name, value in summary.get("final_metrics", {}).items():
                mlflow.log_metric(metric_name, value)
        imported.add(blob.name)
        newly_imported.append(blob.name)

    if newly_imported:
        _IMPORTED_RUNS_MANIFEST.write_text("\n".join(sorted(imported)))
        print(f"Imported {len(newly_imported)} Colab training run(s) into the local MLflow store: {newly_imported}")
    return newly_imported
