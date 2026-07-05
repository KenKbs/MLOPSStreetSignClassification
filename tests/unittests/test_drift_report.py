import pandas as pd
import pytest
from street_sign_project.monitoring import drift_report


def test_load_reference_features_from_gcs_reads_cloud_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that reference features are loaded from the configured GCS object."""
    calls = []

    def fake_download_blob_text(bucket_name: str, blob_name: str) -> str:
        """Return a small reference CSV."""
        calls.append((bucket_name, blob_name))
        return "width,height,brightness_mean\n4,3,20.0\n"

    monkeypatch.setenv(drift_report.MONITORING_BUCKET_ENV, "monitoring-bucket")
    monkeypatch.setattr(drift_report, "download_blob_text", fake_download_blob_text)

    reference_data = drift_report.load_reference_features_from_gcs()

    assert calls == [("monitoring-bucket", "reference/datadrift_reference_features.csv")]
    assert reference_data.to_dict(orient="records") == [{"width": 4, "height": 3, "brightness_mean": 20.0}]


def test_load_production_records_from_gcs_reads_cloud_jsonl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that production records are loaded only from GCS JSONL objects."""
    downloaded_blob_names = []

    def fake_list_blob_names(bucket_name: str, prefix: str) -> list[str]:
        """Return production object names under date partitions."""
        assert bucket_name == "monitoring-bucket"
        assert prefix == "production/date="
        return [
            "production/date=2026-07-05/request-1.jsonl",
            "production/date=2026-07-05/notes.txt",
            "production/date=2026-07-06/request-2.jsonl",
        ]

    def fake_download_blob_text(bucket_name: str, blob_name: str) -> str:
        """Return small production JSONL objects."""
        assert bucket_name == "monitoring-bucket"
        downloaded_blob_names.append(blob_name)
        return '{"request_id": "request-id", "width": 4, "height": 3, "brightness_mean": 20.0}\n'

    monkeypatch.setenv(drift_report.MONITORING_BUCKET_ENV, "monitoring-bucket")
    monkeypatch.setattr(drift_report, "list_blob_names", fake_list_blob_names)
    monkeypatch.setattr(drift_report, "download_blob_text", fake_download_blob_text)

    production_data = drift_report.load_production_records_from_gcs()

    assert downloaded_blob_names == [
        "production/date=2026-07-05/request-1.jsonl",
        "production/date=2026-07-06/request-2.jsonl",
    ]
    assert list(production_data["request_id"]) == ["request-id", "request-id"]


def test_load_production_records_from_gcs_rejects_empty_cloud_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that missing production records fail clearly."""
    monkeypatch.setattr(drift_report, "list_blob_names", lambda bucket_name, prefix: [])

    with pytest.raises(ValueError, match="No production monitoring records found"):
        drift_report.load_production_records_from_gcs(bucket_name="monitoring-bucket")


def test_select_image_feature_columns_keeps_only_shared_numeric_features() -> None:
    """Test that drift inputs are limited to shared numeric image feature columns."""
    reference_data = pd.DataFrame(
        {
            "width": [4],
            "height": [3],
            "brightness_mean": ["20.0"],
            "prediction_count": [1],
            "reference_only": [10],
        }
    )
    production_data = pd.DataFrame(
        {
            "width": [5],
            "height": [4],
            "brightness_mean": ["25.0"],
            "prediction_count": [2],
            "production_only": [11],
        }
    )

    reference_features, production_features = drift_report.select_image_feature_columns(reference_data, production_data)

    assert list(reference_features.columns) == ["width", "height", "brightness_mean"]
    assert list(production_features.columns) == ["width", "height", "brightness_mean"]
    assert reference_features["brightness_mean"].tolist() == [20.0]
    assert production_features["brightness_mean"].tolist() == [25.0]


def test_build_evidently_drift_report_html_uses_selected_feature_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Evidently receives only selected image feature columns."""
    captured_columns = {}

    class _FakeReport:
        """Small Evidently Report replacement."""

        def __init__(self, metrics: list[object]) -> None:
            """Store metrics."""
            self.metrics = metrics

        def run(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
            """Capture report input columns."""
            captured_columns["reference"] = list(reference_data.columns)
            captured_columns["current"] = list(current_data.columns)

        def get_html(self) -> str:
            """Return fake HTML."""
            return "<html>drift</html>"

    monkeypatch.setattr(drift_report, "DataDriftPreset", lambda: "data-drift")
    monkeypatch.setattr(drift_report, "Report", _FakeReport)
    reference_data = pd.DataFrame({"width": [4], "height": [3], "prediction_count": [1]})
    production_data = pd.DataFrame({"width": [5], "height": [4], "prediction_count": [2]})

    html = drift_report.build_evidently_drift_report_html(reference_data, production_data)

    assert html == "<html>drift</html>"
    assert captured_columns == {"reference": ["width", "height"], "current": ["width", "height"]}
