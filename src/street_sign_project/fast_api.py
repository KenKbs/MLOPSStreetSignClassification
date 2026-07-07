# file for setting up the API access to our model's predictions

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import cv2
import yaml
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from street_sign_project.model import YOLOv26
from street_sign_project.monitoring.drift_report import build_cloud_evidently_drift_report_html
from street_sign_project.monitoring.production_records import summarize_predictions, write_monitoring_record
from street_sign_project.utils import project_root

DEFAULT_MODEL_NAME = "YOLO_eps420_bs8_lr0.005_fr10_x.pt"
DATASET_YAML_PATH = project_root() / "data" / "dataset.yaml"
INPUT_DIR = project_root() / "API_uploads" / "input"
OUTPUT_DIR = project_root() / "API_uploads" / "output"

model: YOLOv26
class_name_by_id: dict[int, str] = {}


def _load_class_name_mapping(dataset_yaml_path: Path) -> dict[int, str]:
    """Load class-id-to-name mapping from a YOLO dataset yaml file."""
    if not dataset_yaml_path.exists():
        return {}

    with dataset_yaml_path.open(encoding="utf-8") as file:
        dataset_cfg = yaml.safe_load(file) or {}

    names = dataset_cfg.get("names", {})
    if not isinstance(names, dict):
        return {}

    return {int(class_id): str(class_name) for class_id, class_name in names.items()}


def _model_name() -> str:
    """Return the configured API model name."""
    return os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the YOLO model when the FastAPI application starts."""
    print("Starting Application")
    global model, class_name_by_id
    model = YOLOv26(local_model_name=_model_name())
    class_name_by_id = _load_class_name_mapping(DATASET_YAML_PATH)
    yield
    print("Closing Application")
    del model


app = FastAPI(lifespan=lifespan)


@app.post("/image_input/")
async def cv_model(background_tasks: BackgroundTasks, data: UploadFile = File(...)) -> FileResponse:  # noqa: B008
    """Predict street signs on an uploaded image and return an annotated image."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    request_id = uuid4().hex
    input_image_path = INPUT_DIR / f"image_{request_id}.jpg"
    output_image_path = OUTPUT_DIR / f"image_{request_id}.jpg"

    # TODO Put into separate "load_image" function
    with input_image_path.open("wb") as image:
        content = await data.read()
        image.write(content)
    img = cv2.imread(str(input_image_path))
    if img is None:
        raise ValueError("Bild konnte nicht geladen werden")
    pred = model.predict(input_image_path)
    if pred.boxes is None:
        raise ValueError("pred.boxes is None, but should be a list of prediction objects")

    # Get prediction summary for data drift
    prediction_summary = summarize_predictions(pred)

    # TODO put into separate "draw_image" function or so
    for box in pred.boxes:
        class_id = int(box.cls[0].item())
        class_name = class_name_by_id.get(
            class_id, str(class_id)
        )  # take name of class ID. If class ID not in mapping, take the Calss ID as Name
        xyxy = box.xyxy[0]
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        lbl = f"{class_name}: {round(box.conf[0].item(), 2)}"
        (text_width, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_COMPLEX, 0.7, 1)
        cv2.rectangle(
            img, (int(xyxy[0]), int(xyxy[1]) - 15), (int(xyxy[0] + text_width), int(xyxy[1]) + 5), (0, 0, 0), -1
        )
        cv2.putText(img, lbl, (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)

    # Save the image
    cv2.imwrite(str(output_image_path), img)

    # Add a background task to not increase latency:
    # Writes the jsonl file which is later used by data drift
    background_tasks.add_task(
        write_monitoring_record,
        image_path=input_image_path,
        prediction_summary=prediction_summary,
        request_id=request_id,
        model_name=_model_name(),
        output_image_path=output_image_path,
        original_filename=data.filename,
    )

    return FileResponse(str(output_image_path), media_type="image/jpeg")


@app.get("/monitoring/", response_class=HTMLResponse)
def monitoring_report() -> HTMLResponse:
    """Build and return an Evidently drift report from cloud monitoring data."""
    try:
        report_html = build_cloud_evidently_drift_report_html()
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return HTMLResponse(content=report_html)
