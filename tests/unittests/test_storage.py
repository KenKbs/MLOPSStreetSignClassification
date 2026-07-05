from pathlib import Path

import pytest
from street_sign_project.monitoring import storage


class _FakeBlob:
    """Fake GCS blob capturing uploads."""

    def __init__(
        self,
        name: str,
        uploaded: list[tuple[str, str, str]],
        contents: dict[str, str] | None = None,
    ) -> None:
        """Create a fake blob."""
        self.name = name
        self.uploaded = uploaded
        self.contents = contents or {}

    def upload_from_filename(self, filename: str, content_type: str) -> None:
        """Capture an upload call."""
        self.uploaded.append((self.name, filename, content_type))

    def download_as_text(self) -> str:
        """Return fake text contents."""
        return self.contents[self.name]


class _FakeBucket:
    """Fake GCS bucket returning fake blobs."""

    def __init__(self, uploaded: list[tuple[str, str, str]], contents: dict[str, str] | None = None) -> None:
        """Create a fake bucket."""
        self.uploaded = uploaded
        self.contents = contents or {}

    def blob(self, name: str) -> _FakeBlob:
        """Return a fake blob."""
        return _FakeBlob(name=name, uploaded=self.uploaded, contents=self.contents)


class _FakeListedBlob:
    """Fake listed GCS blob."""

    def __init__(self, name: str) -> None:
        """Create a fake listed blob."""
        self.name = name


class _FakeClient:
    """Fake GCS client returning fake buckets."""

    def __init__(
        self,
        uploaded: list[tuple[str, str, str]],
        contents: dict[str, str] | None = None,
        blob_names: list[str] | None = None,
    ) -> None:
        """Create a fake client."""
        self.uploaded = uploaded
        self.contents = contents or {}
        self.blob_names = blob_names or []

    def bucket(self, bucket_name: str) -> _FakeBucket:
        """Return a fake bucket."""
        return _FakeBucket(uploaded=self.uploaded, contents=self.contents)

    def list_blobs(self, bucket_name: str, prefix: str) -> list[_FakeListedBlob]:
        """Return fake listed blobs matching a prefix."""
        return [_FakeListedBlob(name) for name in self.blob_names if name.startswith(prefix)]


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


def test_download_blob_text_reads_gcs_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that text objects can be downloaded from GCS."""
    monkeypatch.setattr(
        storage,
        "_storage_client",
        lambda: _FakeClient(uploaded=[], contents={"reference/features.csv": "width,height\n4,3\n"}),
    )

    text = storage.download_blob_text(bucket_name="monitoring-bucket", blob_name="reference/features.csv")

    assert text == "width,height\n4,3\n"


def test_list_blob_names_returns_sorted_prefix_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that GCS object names are listed and sorted under a prefix."""
    monkeypatch.setattr(
        storage,
        "_storage_client",
        lambda: _FakeClient(
            uploaded=[],
            blob_names=[
                "production/date=2026-07-06/request-2.jsonl",
                "reference/datadrift_reference_features.csv",
                "production/date=2026-07-05/request-1.jsonl",
            ],
        ),
    )

    blob_names = storage.list_blob_names(bucket_name="monitoring-bucket", prefix="production/date=")

    assert blob_names == [
        "production/date=2026-07-05/request-1.jsonl",
        "production/date=2026-07-06/request-2.jsonl",
    ]
