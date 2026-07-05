import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from street_sign_project.monitoring.production_records import (
    create_production_record,
    summarize_predictions,
    write_monitoring_record,
    write_production_record,
)


class _Scalar:
    """Small tensor-like scalar for fake prediction boxes."""

    def __init__(self, value: float) -> None:
        """Store a scalar value."""
        self.value = value

    def item(self) -> float:
        """Return the scalar value."""
        return self.value


class _FakeBox:
    """Fake Ultralytics box with class and confidence fields."""

    def __init__(self, class_id: int, confidence: float) -> None:
        """Create a fake prediction box."""
        self.cls = [_Scalar(class_id)]
        self.conf = [_Scalar(confidence)]


class _FakePrediction:
    """Fake Ultralytics prediction result."""

    def __init__(self, boxes: list[_FakeBox] | None) -> None:
        """Create a fake prediction result."""
        self.boxes = boxes


def _write_test_image(path: Path) -> None:
    """Write a small valid test image."""
    image = np.full((3, 4, 3), 20, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_summarize_predictions_handles_multiple_boxes() -> None:
    """Test that prediction summaries flatten class and confidence data."""
    prediction = _FakePrediction([_FakeBox(class_id=12, confidence=0.5), _FakeBox(class_id=47, confidence=0.9)])

    summary = summarize_predictions(prediction)

    assert summary == {
        "prediction_count": 2,
        "mean_confidence": pytest.approx(0.7),
        "max_confidence": 0.9,
        "predicted_classes": "12,47",
    }


def test_summarize_predictions_handles_no_boxes() -> None:
    """Test that empty predictions still produce numeric monitoring fields."""
    summary = summarize_predictions(_FakePrediction([]))

    assert summary == {
        "prediction_count": 0,
        "mean_confidence": 0.0,
        "max_confidence": 0.0,
        "predicted_classes": "",
    }


def test_create_production_record_combines_metadata_image_and_prediction_features(tmp_path: Path) -> None:
    """Test that production records contain request metadata and feature columns."""
    image_path = tmp_path / "input.jpg"
    output_image_path = tmp_path / "output.jpg"
    _write_test_image(image_path)

    record = create_production_record(
        image_path=image_path,
        prediction_summary={
            "prediction_count": 1,
            "mean_confidence": 0.8,
            "max_confidence": 0.8,
            "predicted_classes": "4",
        },
        request_id="request-123",
        model_name="model.pt",
        output_image_path=output_image_path,
        original_filename="upload.jpg",
    )

    assert record["request_id"] == "request-123"
    assert record["model_name"] == "model.pt"
    assert record["image_name"] == "upload.jpg"
    assert record["width"] == 4
    assert record["height"] == 3
    assert record["prediction_count"] == 1
    assert record["predicted_classes"] == "4"
    assert "timestamp_utc" in record


def test_write_production_record_writes_one_json_line(tmp_path: Path) -> None:
    """Test that production records are written as one JSONL file per request."""
    record = {"request_id": "request-123", "prediction_count": 0}

    output_path = write_production_record(record=record, output_dir=tmp_path)

    assert output_path == tmp_path / "request-123.jsonl"
    assert json.loads(output_path.read_text(encoding="utf-8")) == record


def test_write_monitoring_record_creates_local_jsonl_record(tmp_path: Path) -> None:
    """Test that the background monitoring helper creates and writes a record."""
    image_path = tmp_path / "input.jpg"
    output_image_path = tmp_path / "output.jpg"
    output_dir = tmp_path / "monitoring"
    _write_test_image(image_path)

    output_path = write_monitoring_record(
        image_path=image_path,
        prediction_summary={
            "prediction_count": 0,
            "mean_confidence": 0.0,
            "max_confidence": 0.0,
            "predicted_classes": "",
        },
        request_id="request-123",
        model_name="model.pt",
        output_image_path=output_image_path,
        output_dir=output_dir,
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path == output_dir / "request-123.jsonl"
    assert record["request_id"] == "request-123"
    assert record["width"] == 4


def test_write_monitoring_record_keeps_local_record_when_upload_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that GCS upload failures do not prevent local monitoring writes."""
    image_path = tmp_path / "input.jpg"
    output_image_path = tmp_path / "output.jpg"
    output_dir = tmp_path / "monitoring"
    _write_test_image(image_path)

    def fail_upload(local_path: Path) -> None:
        """Raise an upload failure."""
        raise RuntimeError("upload failed")

    monkeypatch.setattr("street_sign_project.monitoring.production_records.upload_production_record", fail_upload)

    output_path = write_monitoring_record(
        image_path=image_path,
        prediction_summary={
            "prediction_count": 0,
            "mean_confidence": 0.0,
            "max_confidence": 0.0,
            "predicted_classes": "",
        },
        request_id="request-123",
        model_name="model.pt",
        output_image_path=output_image_path,
        output_dir=output_dir,
    )

    assert output_path.exists()
