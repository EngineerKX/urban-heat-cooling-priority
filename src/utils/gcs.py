"""Cloud Storage client + generic blob helpers, authenticated with the same
service-account key already configured for Earth Engine (`GEE_PRIVATE_KEY_PATH`).

Extracted out of `src/ingest/gee.py` (which originally had its own private
`_gcs_client()`) because GCS is no longer only a GEE-export destination —
`scripts/pull_models.py`/`push_models.py` and the Colab training notebooks
also need plain upload/download/existence-check access to the bucket, for
syncing trained model artifacts and cached training patches. This module has
no dependency on `src.ingest.gee` (only on `config.settings`), so `gee.py`
imports from here, not the other way around.
"""

from pathlib import Path

from config.settings import GEE_PRIVATE_KEY_PATH, GEE_PROJECT_ID


def get_client():
    """`google-cloud-storage` is a separate credential system from `ee` and
    does not reuse `init_ee()`'s credentials automatically — left to its
    default behavior, it looks for Application Default Credentials (only
    present if you've separately run `gcloud auth application-default
    login`), which nothing in this project sets up. Passing the
    service-account file explicitly avoids that dependency entirely."""
    from google.cloud import storage
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(GEE_PRIVATE_KEY_PATH)
    return storage.Client(project=GEE_PROJECT_ID, credentials=credentials)


def download_blob(bucket_name: str, blob_name: str, out_path) -> Path:
    """Single blob -> local file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = get_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    # blob.download_to_filename() also calls os.utime() afterward to match
    # the blob's cloud mtime — this raised PermissionError under WSL2's
    # Windows-mounted filesystem (/mnt/c/...) and is avoided everywhere in
    # this project as a defensive habit, not just for WSL2's sake.
    with open(out_path, "wb") as f:
        client.download_blob_to_file(blob, f)
    return out_path


def download_blobs_with_prefix(bucket_name: str, prefix: str, out_dir) -> list[Path]:
    """Every blob under `prefix` -> flat files in `out_dir` (named by the
    blob's basename, matching how GEE's patch exports lay out shards)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = get_client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    downloaded = []
    for blob in blobs:
        local_path = out_dir / Path(blob.name).name
        with open(local_path, "wb") as f:
            client.download_blob_to_file(blob, f)
        downloaded.append(local_path)
    return downloaded


def upload_file(local_path, bucket_name: str, blob_name: str) -> str:
    """Local file -> GCS. Returns the `gs://` URI written to."""
    client = get_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{blob_name}"


def blob_exists(bucket_name: str, blob_name: str) -> bool:
    client = get_client()
    return client.bucket(bucket_name).blob(blob_name).exists()


def blobs_exist_with_prefix(bucket_name: str, prefix: str) -> bool:
    """True if at least one blob is stored under `prefix` — used to decide
    whether an expensive GEE export can be skipped in favor of a GCS
    download (see `src/landcover/unet_data.py`)."""
    client = get_client()
    return next(iter(client.list_blobs(bucket_name, prefix=prefix, max_results=1)), None) is not None


def download_text(bucket_name: str, blob_name: str) -> str | None:
    """Small text blob (e.g. a sha256 sidecar) -> str, or None if missing."""
    client = get_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists():
        return None
    return blob.download_as_text()


def upload_text(text: str, bucket_name: str, blob_name: str) -> None:
    client = get_client()
    client.bucket(bucket_name).blob(blob_name).upload_from_string(text)
