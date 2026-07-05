"""Production monitoring records for FastAPI prediction requests."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from street_sign_project.monitoring.image_features import ImageFeatureValue, extract_image_features
from street_sign_project.utils import project_root

# Default Output dir (inside container / wherever API runs)
MONITORING_OUTPUT_DIR = project_root() / "API_uploads" / "monitoring" / "production"
ProductionRecordValue = ImageFeatureValue | str


def _display_path(path: Path) -> str:
    """Return a stable path string for monitoring records."""
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return str(path)


def summarize_predictions(prediction: Any) -> dict[str, ProductionRecordValue]:
    """Summarize a model prediction into flat monitoring fields.
    e.g. the model predicted 2 boxes, with mean confidence etc.

    Args:
        prediction: Ultralytics prediction result with a ``boxes`` attribute.

    Returns:
        Flat prediction summary fields for one API request.
    """
    boxes = getattr(prediction, "boxes", None)
    if boxes is None:
        return {
            "prediction_count": 0,
            "mean_confidence": 0.0,
            "max_confidence": 0.0,
            "predicted_classes": "",
        }

    confidences = []
    classes = []
    for box in boxes:
        confidences.append(float(box.conf[0].item()))
        classes.append(int(box.cls[0].item()))

    if not confidences:
        return {
            "prediction_count": 0,
            "mean_confidence": 0.0,
            "max_confidence": 0.0,
            "predicted_classes": "",
        }

    return {
        "prediction_count": len(confidences),
        "mean_confidence": float(sum(confidences) / len(confidences)),
        "max_confidence": float(max(confidences)),
        "predicted_classes": ",".join(str(class_id) for class_id in classes),
    }


def create_production_record(
    image_path: Path,
    prediction_summary: dict[str, ProductionRecordValue],
    request_id: str,
    model_name: str,
    output_image_path: Path,
    original_filename: str | None = None,
) -> dict[str, ProductionRecordValue]:
    """Create one production monitoring record from image and prediction features.
    So it loads the saved image path, and saved prediction summary, and combines them into one
    dictonary, which then can be written to jsonl object as a file.

    Args:
        image_path: Saved input image path.
        prediction_summary: Flat prediction fields from ``summarize_predictions``.
        request_id: Unique request identifier.
        model_name: Model name used to serve the prediction.
        output_image_path: Saved annotated output image path.
        original_filename: Optional filename from the upload request.

    Returns:
        Flat monitoring record ready to serialize as JSONL.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image could not be loaded: {image_path}")

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "model_name": model_name,
        "image_name": original_filename or image_path.name,
        "input_image_path": _display_path(image_path),
        "output_image_path": _display_path(output_image_path),
        **extract_image_features(image),
        **prediction_summary,
    }


# Here we would later change the path to some GCP bucket
def write_production_record(
    record: dict[str, ProductionRecordValue],
    output_dir: Path = MONITORING_OUTPUT_DIR,
) -> Path:
    """Write one production monitoring record as a local JSONL file.

    Args:
        record: Flat production monitoring record.
        output_dir: Directory where the JSONL record should be written.

    Returns:
        Path to the written JSONL file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{record['request_id']}.jsonl"
    with output_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True))
        file.write("\n")
    return output_path


def write_monitoring_record(
    image_path: Path,
    prediction_summary: dict[str, ProductionRecordValue],
    request_id: str,
    model_name: str,
    output_image_path: Path,
    original_filename: str | None = None,
    output_dir: Path = MONITORING_OUTPUT_DIR,
) -> Path:
    """Create and write one production monitoring record."""
    record = create_production_record(
        image_path=image_path,
        prediction_summary=prediction_summary,
        request_id=request_id,
        model_name=model_name,
        output_image_path=output_image_path,
        original_filename=original_filename,
    )
    return write_production_record(record=record, output_dir=output_dir)
