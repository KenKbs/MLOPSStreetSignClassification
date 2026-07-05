"""Generate reference image feature datasets for data drift monitoring."""

from pathlib import Path

import cv2
import pandas as pd
import typer

from street_sign_project.monitoring.image_features import extract_image_features
from street_sign_project.utils import project_root

app = typer.Typer()
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGE_DIR = project_root() / "data" / "preprocessed" / "train" / "images"
DEFAULT_OUTPUT_PATH = project_root() / "reports" / "datadrift_reference_features.csv"


# Helper function for clear image path in csv
def _display_image_path(image_path: Path) -> str:
    """Return a stable image path for the generated feature CSV."""
    try:
        return str(image_path.relative_to(project_root()))
    except ValueError:
        return str(image_path)


# main function
def generate_reference_features(image_dir: Path = DEFAULT_IMAGE_DIR, output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Generate a CSV of image features from a reference image directory.

    Args:
        image_dir: Directory containing reference images.
        output_path: CSV path where extracted features should be written.

    Returns:
        Path to the generated CSV file.

    Raises:
        FileNotFoundError: If the image directory does not exist.
        ValueError: If the image directory contains no supported images or an image cannot be loaded.
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise ValueError(f"No supported images found in {image_dir}")

    rows = []
    # Loop through all images
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Image could not be loaded: {image_path}")

        # Append features to the row list
        rows.append(
            {
                "image_name": image_path.name,
                "image_path": _display_image_path(image_path),
                **extract_image_features(image),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


# Typer entry point
@app.command()
def generate(
    image_dir: Path = DEFAULT_IMAGE_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Generate reference image features for data drift monitoring."""
    generated_path = generate_reference_features(image_dir=image_dir, output_path=output_path)
    print(f"Saved reference features to {generated_path}")


if __name__ == "__main__":
    app()
