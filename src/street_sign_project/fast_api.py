# file for setting up the API access to our model's predictions

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import cv2
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from street_sign_project.model import YOLOv26
from street_sign_project.monitoring.production_records import summarize_predictions, write_monitoring_record
from street_sign_project.utils import project_root

DEFAULT_MODEL_NAME = "YOLO_eps420_bs8_lr0.005_fr10_x.pt"
INPUT_DIR = project_root() / "API_uploads" / "input"
OUTPUT_DIR = project_root() / "API_uploads" / "output"

model: YOLOv26


def _model_name() -> str:
    """Return the configured API model name."""
    return os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the YOLO model when the FastAPI application starts."""
    print("Starting Application")
    global model
    model = YOLOv26(local_model_name=_model_name())
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
        xyxy = box.xyxy[0]
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        lbl = f"{int(box.cls[0].item())}: {round(box.conf[0].item(), 2)}"
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[0] + 80), int(xyxy[1] + 12)), (0, 0, 0), -1)
        cv2.putText(img, lbl, (int(xyxy[0]), int(xyxy[1] + 12)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

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
