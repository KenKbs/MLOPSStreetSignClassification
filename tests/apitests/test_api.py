from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from street_sign_project import fast_api

app = fast_api.app

client = TestClient(app)


def test_openapi_schema_includes_image_input_route() -> None:
    """Test that the API exposes the image upload route."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/image_input/" in response.json()["paths"]
    assert "/monitoring/" in response.json()["paths"]


def test_monitoring_route_returns_evidently_html(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the monitoring route returns the generated Evidently HTML report."""
    monkeypatch.setattr(fast_api, "build_cloud_evidently_drift_report_html", lambda: "<html>drift</html>")

    response = client.get("/monitoring/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == "<html>drift</html>"


class _Scalar:
    """Small tensor-like scalar for fake prediction boxes."""

    def __init__(self, value: float) -> None:
        """Store a scalar value."""
        self.value = value

    def item(self) -> float:
        """Return the scalar value."""
        return self.value


class _FakeBox:
    """Fake Ultralytics box for API endpoint tests."""

    xyxy = [[0, 0, 2, 2]]

    def __init__(self, class_id: int, confidence: float) -> None:
        """Create a fake prediction box."""
        self.cls = [_Scalar(class_id)]
        self.conf = [_Scalar(confidence)]


class _FakePrediction:
    """Fake Ultralytics prediction result."""

    boxes = [_FakeBox(class_id=4, confidence=0.8)]


class _FakeModel:
    """Fake API model returning one prediction."""

    def predict(self, new_data: Path) -> _FakePrediction:
        """Return a fake prediction."""
        return _FakePrediction()


def test_image_input_schedules_monitoring_background_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that image uploads schedule a production monitoring record."""
    scheduled_records = []

    def fake_write_monitoring_record(**kwargs) -> None:
        """Capture scheduled monitoring write arguments."""
        scheduled_records.append(kwargs)

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(fast_api, "INPUT_DIR", input_dir)
    monkeypatch.setattr(fast_api, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(fast_api, "model", _FakeModel(), raising=False)
    monkeypatch.setattr(fast_api, "_model_name", lambda: "model.pt")
    monkeypatch.setattr(fast_api, "write_monitoring_record", fake_write_monitoring_record)

    image = np.full((4, 4, 3), 50, dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)

    response = client.post(
        "/image_input/",
        files={"data": ("upload.jpg", encoded_image.tobytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(scheduled_records) == 1
    assert scheduled_records[0]["model_name"] == "model.pt"
    assert scheduled_records[0]["original_filename"] == "upload.jpg"
    assert scheduled_records[0]["prediction_summary"] == {
        "prediction_count": 1,
        "mean_confidence": 0.8,
        "max_confidence": 0.8,
        "predicted_classes": "4",
    }
