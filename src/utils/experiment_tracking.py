"""Thin MLflow wrapper so scripts/train_landcover_rf.py and
scripts/train_landcover_unet.py log to the same experiment consistently,
without duplicating tracking-setup boilerplate in both places.

Local SQLite-backed tracking (a single mlflow.db file under REPO_ROOT) --
still no separate server needed, but MLflow 3.x puts the plain flat-file
store ("./mlruns") into maintenance mode and refuses to use it without an
explicit opt-out flag, so SQLite is the simpler path going forward rather
than fighting that deprecation. The two training scripts run in different
environments (RF on native Windows, U-Net in WSL2), but REPO_ROOT resolves
to the same physical directory in both (WSL2 reaches the repo via the
/mnt/c mount), so pinning the tracking URI to an absolute path here keeps
both writing to the one shared store regardless of which OS invokes the
script. Browse runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db`
from the repo root.
"""

import mlflow

from config.settings import REPO_ROOT

EXPERIMENT_NAME = "landcover_classifiers"
HEAT_MODEL_EXPERIMENT_NAME = "heat_model_s5"

mlflow.set_tracking_uri(f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")


def start_run(model_name: str, experiment_name: str = EXPERIMENT_NAME, **tags):
    """`experiment_name` defaults to EXPERIMENT_NAME so every existing call
    site (scripts/train_landcover_rf.py, scripts/train_landcover_unet.py)
    is unaffected. S5's training scripts pass HEAT_MODEL_EXPERIMENT_NAME
    explicitly instead of this module growing a second hardcoded default."""
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=model_name, tags={"model_type": model_name, **tags})
