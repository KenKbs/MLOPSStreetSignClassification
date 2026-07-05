"""Storage helpers for API monitoring records."""

import os
from datetime import UTC, datetime
from pathlib import Path

from google.cloud import storage

DEFAULT_MONITORING_PREFIX = "production"
DEFAULT_REFERENCE_FEATURES_BLOB_NAME = "reference/datadrift_reference_features.csv"
MONITORING_BUCKET_ENV = "MONITORING_BUCKET"
MONITORING_PREFIX_ENV = "MONITORING_PREFIX"


# looks funny, but did so so tests can use a lightweight client with monkeypatch
def _storage_client() -> storage.Client:
    """Create a Google Cloud Storage client."""
    return storage.Client()


def production_record_blob_name(local_path: Path, prefix: str = DEFAULT_MONITORING_PREFIX) -> str:
    """Build the GCS object name for a production monitoring record.
    So inside the GCP buckets monitor requests are organized in
    subfolders by date!"""
    date_partition = datetime.now(UTC).date().isoformat()
    return f"{prefix}/date={date_partition}/{local_path.name}"


def upload_file_to_gcs(local_path: Path, bucket_name: str, blob_name: str, content_type: str) -> str:
    """Upload a local file to Google Cloud Storage.

    Args:
        local_path: Local file path to upload.
        bucket_name: GCS bucket name without the ``gs://`` prefix.
        blob_name: Object name inside the GCS bucket.
        content_type: MIME type for the uploaded object.

    Returns:
        GCS URI for the uploaded object.
    """
    client = _storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path), content_type=content_type)
    return f"gs://{bucket_name}/{blob_name}"


def upload_reference_features(
    local_path: Path,
    bucket_name: str | None = None,
    blob_name: str = DEFAULT_REFERENCE_FEATURES_BLOB_NAME,
) -> str | None:
    """Upload generated reference feature data to GCS when configured.

    Args:
        local_path: Local reference feature CSV path.
        bucket_name: Optional GCS bucket name. Falls back to ``MONITORING_BUCKET``.
        blob_name: Object name inside the GCS bucket.

    Returns:
        GCS URI for the uploaded object, or ``None`` when no bucket is configured.
    """
    configured_bucket = bucket_name or os.getenv(MONITORING_BUCKET_ENV)
    if not configured_bucket:
        return None

    return upload_file_to_gcs(
        local_path=local_path,
        bucket_name=configured_bucket,
        blob_name=blob_name,
        content_type="text/csv",
    )


def upload_production_record(
    local_path: Path,
    bucket_name: str | None = None,
    prefix: str | None = None,
) -> str | None:
    """Upload a local production monitoring record to GCS when configured.

    Args:
        local_path: Local JSONL record path.
        bucket_name: Optional GCS bucket name. Falls back to ``MONITORING_BUCKET``.
        prefix: Optional object prefix. Falls back to ``MONITORING_PREFIX`` or ``production``.

    Returns:
        GCS URI for the uploaded object, or ``None`` when no bucket is configured.
    """
    configured_bucket = bucket_name or os.getenv(MONITORING_BUCKET_ENV)
    # Do nothing if no configured bucket is present
    if not configured_bucket:
        return None

    # Get model prefix and build blob_name to save file under
    configured_prefix = prefix or os.getenv(MONITORING_PREFIX_ENV, DEFAULT_MONITORING_PREFIX)
    blob_name = production_record_blob_name(local_path=local_path, prefix=configured_prefix)

    # upload file
    return upload_file_to_gcs(
        local_path=local_path,
        bucket_name=configured_bucket,
        blob_name=blob_name,
        content_type="application/json",
    )
