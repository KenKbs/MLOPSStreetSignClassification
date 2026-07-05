"""Build Evidently data drift reports from cloud monitoring data."""

import json
import os
from io import StringIO

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from street_sign_project.monitoring.storage import (
    DEFAULT_MONITORING_PREFIX,
    DEFAULT_REFERENCE_FEATURES_BLOB_NAME,
    MONITORING_BUCKET_ENV,
    MONITORING_PREFIX_ENV,
    download_blob_text,
    list_blob_names,
)

DEFAULT_MONITORING_BUCKET = "mlops-street-signs-prod-data"
IMAGE_FEATURE_COLUMNS = [
    "width",
    "height",
    "aspect_ratio",
    "brightness_mean",
    "brightness_std",
    "contrast",
    "sharpness",
    "red_mean",
    "green_mean",
    "blue_mean",
]


def _monitoring_bucket_name(bucket_name: str | None = None) -> str:
    """Return the bucket used for reading monitoring data."""
    return bucket_name or os.getenv(MONITORING_BUCKET_ENV, DEFAULT_MONITORING_BUCKET)


def load_reference_features_from_gcs(
    bucket_name: str | None = None,
    blob_name: str = DEFAULT_REFERENCE_FEATURES_BLOB_NAME,
) -> pd.DataFrame:
    """Load reference image features from a GCS CSV object.

    Args:
        bucket_name: Optional GCS bucket name. Falls back to ``MONITORING_BUCKET`` or the production bucket.
        blob_name: Reference feature object name inside the bucket.

    Returns:
        Reference feature data.
    """
    csv_text = download_blob_text(bucket_name=_monitoring_bucket_name(bucket_name), blob_name=blob_name)
    return pd.read_csv(StringIO(csv_text))


def load_production_records_from_gcs(
    bucket_name: str | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Load production monitoring records from GCS JSONL objects.

    Args:
        bucket_name: Optional GCS bucket name. Falls back to ``MONITORING_BUCKET`` or the production bucket.
        prefix: Optional production records prefix. Falls back to ``MONITORING_PREFIX`` or ``production``.

    Returns:
        Production monitoring records.

    Raises:
        ValueError: If no production JSONL records are found.
    """
    configured_bucket = _monitoring_bucket_name(bucket_name)
    configured_prefix = prefix or os.getenv(MONITORING_PREFIX_ENV, DEFAULT_MONITORING_PREFIX)
    blob_names = [
        blob_name
        for blob_name in list_blob_names(bucket_name=configured_bucket, prefix=f"{configured_prefix}/date=")
        if blob_name.endswith(".jsonl")
    ]
    if not blob_names:
        raise ValueError(f"No production monitoring records found in gs://{configured_bucket}/{configured_prefix}/")

    rows = []
    for blob_name in blob_names:
        jsonl_text = download_blob_text(bucket_name=configured_bucket, blob_name=blob_name)
        rows.extend(json.loads(line) for line in jsonl_text.splitlines() if line.strip())
    return pd.DataFrame(rows)


def select_image_feature_columns(
    reference_data: pd.DataFrame, production_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select shared numeric image feature columns for drift reporting.

    Args:
        reference_data: Reference feature data.
        production_data: Production feature data.

    Returns:
        Reference and production data with only shared numeric image feature columns.

    Raises:
        ValueError: If no shared image feature columns are available.
    """
    shared_columns = [
        column
        for column in IMAGE_FEATURE_COLUMNS
        if column in reference_data.columns and column in production_data.columns
    ]
    if not shared_columns:
        raise ValueError("No shared image feature columns found for drift reporting")

    reference_features = reference_data[shared_columns].apply(pd.to_numeric, errors="coerce")
    production_features = production_data[shared_columns].apply(pd.to_numeric, errors="coerce")
    numeric_columns = [
        column
        for column in shared_columns
        if not reference_features[column].isna().all() and not production_features[column].isna().all()
    ]
    if not numeric_columns:
        raise ValueError("No shared numeric image feature columns found for drift reporting")

    return reference_features[numeric_columns], production_features[numeric_columns]


def build_evidently_drift_report_html(reference_data: pd.DataFrame, production_data: pd.DataFrame) -> str:
    """Build an Evidently data drift report as HTML.

    Args:
        reference_data: Reference feature data.
        production_data: Production feature data.

    Returns:
        Evidently report HTML.
    """
    reference_features, production_features = select_image_feature_columns(
        reference_data=reference_data,
        production_data=production_data,
    )
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_features, current_data=production_features)
    return report.get_html()


def build_cloud_evidently_drift_report_html(bucket_name: str | None = None) -> str:
    """Load cloud monitoring data and build an Evidently drift report as HTML.

    Args:
        bucket_name: Optional GCS bucket name. Falls back to ``MONITORING_BUCKET`` or the production bucket.

    Returns:
        Evidently report HTML.
    """
    reference_data = load_reference_features_from_gcs(bucket_name=bucket_name)
    production_data = load_production_records_from_gcs(bucket_name=bucket_name)
    return build_evidently_drift_report_html(reference_data=reference_data, production_data=production_data)
