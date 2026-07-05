from pathlib import Path

import pytest
from street_sign_project.monitoring import storage


class _FakeBlob:
    """Fake GCS blob capturing uploads."""

    def __init__(self, name: str, uploaded: list[tuple[str, str, str]]) -> None:
        """Create a fake blob."""
        self.name = name
        self.uploaded = uploaded

    def upload_from_filename(self, filename: str, content_type: str) -> None:
        """Capture an upload call."""
        self.uploaded.append((self.name, filename, content_type))


class _FakeBucket:
    """Fake GCS bucket returning fake blobs."""

    def __init__(self, uploaded: list[tuple[str, str, str]]) -> None:
        """Create a fake bucket."""
        self.uploaded = uploaded

    def blob(self, name: str) -> _FakeBlob:
        """Return a fake blob."""
        return _FakeBlob(name=name, uploaded=self.uploaded)


class _FakeClient:
    """Fake GCS client returning fake buckets."""

    def __init__(self, uploaded: list[tuple[str, str, str]]) -> None:
        """Create a fake client."""
        self.uploaded = uploaded

    def bucket(self, bucket_name: str) -> _FakeBucket:
        """Return a fake bucket."""
        return _FakeBucket(uploaded=self.uploaded)


def test_upload_production_record_returns_none_without_bucket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that GCS upload is skipped when no monitoring bucket is configured."""
    monkeypatch.delenv(storage.MONITORING_BUCKET_ENV, raising=False)
    local_path = tmp_path / "record.jsonl"
    local_path.write_text("{}\n", encoding="utf-8")

    assert storage.upload_production_record(local_path) is None


def test_upload_production_record_uploads_to_configured_bucket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that production records are uploaded to the configured GCS bucket."""
    uploaded = []
    local_path = tmp_path / "request-123.jsonl"
    local_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(storage.MONITORING_BUCKET_ENV, "monitoring-bucket")
    monkeypatch.setenv(storage.MONITORING_PREFIX_ENV, "api-records")
    monkeypatch.setattr(storage, "_storage_client", lambda: _FakeClient(uploaded=uploaded))

    uri = storage.upload_production_record(local_path)

    assert uri is not None
    assert uri.startswith("gs://monitoring-bucket/api-records/date=")
    assert uri.endswith("/request-123.jsonl")
    assert uploaded == [
        (
            uri.removeprefix("gs://monitoring-bucket/"),
            str(local_path),
            "application/json",
        )
    ]


def test_upload_reference_features_uses_stable_blob_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that reference features upload to the stable reference object path."""
    uploaded = []
    local_path = tmp_path / "datadrift_reference_features.csv"
    local_path.write_text("width,height\n1,2\n", encoding="utf-8")
    monkeypatch.setenv(storage.MONITORING_BUCKET_ENV, "monitoring-bucket")
    monkeypatch.setattr(storage, "_storage_client", lambda: _FakeClient(uploaded=uploaded))

    uri = storage.upload_reference_features(local_path)

    assert uri == "gs://monitoring-bucket/reference/datadrift_reference_features.csv"
    assert uploaded == [
        (
            "reference/datadrift_reference_features.csv",
            str(local_path),
            "text/csv",
        )
    ]
