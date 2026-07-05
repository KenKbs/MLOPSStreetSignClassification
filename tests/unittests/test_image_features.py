import numpy as np
import pytest
from street_sign_project.monitoring.image_features import extract_image_features


def test_extract_image_features_returns_expected_schema_for_color_image() -> None:
    """Test that color image features use stable names and numeric values."""
    # Create synthetic image
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30

    # Extract Features (Act)
    features = extract_image_features(image)

    # Assert if features exist
    assert set(features) == {
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

    # Assert if features have correct values!
    assert features["width"] == 4
    assert features["height"] == 2
    assert features["aspect_ratio"] == 2.0
    assert features["red_mean"] == 30.0
    assert features["green_mean"] == 20.0
    assert features["blue_mean"] == 10.0


def test_extract_image_features_returns_expected_values_for_grayscale_image() -> None:
    """Test that grayscale images produce deterministic monitoring features."""
    image = np.array([[0, 10], [20, 30]], dtype=np.uint8)

    features = extract_image_features(image)

    assert features["width"] == 2
    assert features["height"] == 2
    assert features["aspect_ratio"] == 1.0
    assert features["brightness_mean"] == 15.0
    assert features["brightness_std"] == pytest.approx(11.1803398875)
    assert features["contrast"] == 30.0
    assert features["red_mean"] == 15.0
    assert features["green_mean"] == 15.0
    assert features["blue_mean"] == 15.0


def test_extract_image_features_rejects_empty_image() -> None:
    """Test that empty images fail with a clear validation error."""
    with pytest.raises(ValueError, match="Image is empty"):
        extract_image_features(np.array([], dtype=np.uint8))


def test_extract_image_features_rejects_unsupported_channel_count() -> None:
    """Test that images with unsupported channel counts fail validation."""
    with pytest.raises(ValueError, match="Unsupported channel count"):
        extract_image_features(np.zeros((2, 2, 4), dtype=np.uint8))
