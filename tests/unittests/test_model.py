from pathlib import Path

import pytest
from street_sign_project.model import YOLOv26


class DummyResult:
    pass


class FakeYOLO:
    """Fake Ultralytics YOLO object used to test the project wrapper.
    Mainly stores, what get's passed in during construction and for predict / save method calls"""

    def __init__(self, model_path: str) -> None:
        """Store the model path requested by the wrapper."""
        self.model_path = model_path
        self.predict_source = None
        self.predict_conf = None
        self.saved_path = None

    def save(self, filename: str | Path) -> None:
        """Store the path requested by the wrapper."""
        self.saved_path = filename

    def predict(self, source: str | Path | None = None, conf: float = 0.25) -> list[DummyResult]:
        """store the inputs"""
        self.predict_source = source
        self.predict_conf = conf
        return [DummyResult()]


def test_yolov26_predict_rejects_missing_or_non_yaml_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that YOLOv26 prediction validates the dataset path format.
    'Fail on purpose to pass' test, checks if errors really get raised for wrong input"""
    monkeypatch.setattr("street_sign_project.model.YOLO", FakeYOLO)  # mock the YOLO class
    model = YOLOv26(model_size="n")

    # model.predict with new_data=None should produce error
    with pytest.raises(RuntimeError, match="Datenverweis muss als .yaml oder .jpg"):
        model.predict(new_data=None)

    # same for model.predict with new_data = image.jpg should produce error
    with pytest.raises(RuntimeError, match="Datenverweis muss als .yaml oder .jpg"):
        model.predict(new_data="image.png")


def test_yolov26_save_model_requires_pt_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that YOLOv26 only saves models to .pt files."""
    monkeypatch.setattr("street_sign_project.model.YOLO", FakeYOLO)
    model = YOLOv26(model_size="n")
    output_path = tmp_path / "model.pt"  # arrange

    model.save_model(output_path)  # Act

    # Assert that the save path was passed through to YOLO
    assert model.model.saved_path == output_path, "Model save path was not passed through to YOLO"

    # Check if wrong model ending really fails
    with pytest.raises(ValueError, match='file path needs to be a ".pt" file'):
        model.save_model(tmp_path / "model.txt")


def test_yolov26_invalid_model_size_falls_back_to_n(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that invalid YOLO model sizes fall back to the nano model."""
    monkeypatch.setattr("street_sign_project.model.YOLO", FakeYOLO)

    model = YOLOv26(model_size="invalid")

    assert model.model_size == "n", "Invalid model sizes should fall back to n"
    assert model.model.model_path == "yolo26n.pt", "Fallback model should load the nano checkpoint"
