"""Image feature extraction for API monitoring."""

from typing import TypeAlias

import cv2
import numpy as np

ImageFeatureValue: TypeAlias = float | int


def extract_image_features(image: np.ndarray) -> dict[str, ImageFeatureValue]:
    """Extract tabular monitoring features from an image.

    Args:
        image: Image array as loaded by OpenCV. Color images are expected in BGR channel order
            (BGR = default for OpenCV).

    Returns:
        Dictionary of numeric image features for monitoring and drift detection.

    Raises:
        ValueError: If the image is empty or has an unsupported shape.
    """
    # Sanity check image dimension and channels
    if image.size == 0:
        raise ValueError("Image is empty")
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(f"Unsupported channel count: {image.shape[2]}")

    height, width = image.shape[:2]
    aspect_ratio = width / height

    if image.ndim == 2:  # image = (height,width) --> only greyscale image
        gray_image = image
        red_channel = image
        green_channel = image
        blue_channel = image
    else:  # image = (height,width,3) --> color image
        blue_channel, green_channel, red_channel = cv2.split(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # extract features
    gray_float = gray_image.astype(np.float64)
    sharpness = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    return {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": float(aspect_ratio),
        "brightness_mean": float(gray_float.mean()),
        "brightness_std": float(gray_float.std()),
        "contrast": float(gray_float.max() - gray_float.min()),
        "sharpness": float(sharpness),
        "red_mean": float(red_channel.mean()),
        "green_mean": float(green_channel.mean()),
        "blue_mean": float(blue_channel.mean()),
    }
