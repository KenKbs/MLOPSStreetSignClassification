from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from street_sign_project.monitoring.reference_features import generate_reference_features


def _write_test_image(path: Path, value: int) -> None:
    """Write a small valid test image."""
    image = np.full((3, 4, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_generate_reference_features_writes_expected_csv(tmp_path: Path) -> None:
    """Test that reference feature generation writes one row per supported image."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_test_image(image_dir / "first.jpg", 10)
    _write_test_image(image_dir / "second.png", 20)
    (image_dir / "skip.txt").write_text("not an image", encoding="utf-8")
    output_path = tmp_path / "reference_features.csv"

    generated_path = generate_reference_features(image_dir=image_dir, output_path=output_path)

    reference_data = pd.read_csv(generated_path)
    assert generated_path == output_path
    assert list(reference_data["image_name"]) == ["first.jpg", "second.png"]
    assert set(reference_data.columns) == {
        "image_name",
        "image_path",
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
    }


def test_generate_reference_features_rejects_missing_image_dir(tmp_path: Path) -> None:
    """Test that missing reference image directories fail clearly."""
    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        generate_reference_features(image_dir=tmp_path / "missing", output_path=tmp_path / "features.csv")


def test_generate_reference_features_rejects_empty_image_dir(tmp_path: Path) -> None:
    """Test that directories without supported images fail clearly."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    with pytest.raises(ValueError, match="No supported images found"):
        generate_reference_features(image_dir=image_dir, output_path=tmp_path / "features.csv")
